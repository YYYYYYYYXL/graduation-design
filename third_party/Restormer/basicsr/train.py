import argparse
import datetime
import logging
import math
import random
import time
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

import numpy as np
import os
import json
from copy import deepcopy
from pathlib import Path

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='恢复当前实验自己的最近训练断点')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)
    opt['explicit_resume'] = args.resume
    if opt['path'].get('pretrain_network_g') is not None:
        raise ValueError('禁止加载外部预训练权重：path.pretrain_network_g必须为null')
    if opt['path'].get('resume_state') and not args.resume:
        raise ValueError('恢复已有训练状态必须显式加--resume')
    if set(opt['datasets']) != {'train', 'val'}:
        raise ValueError('训练配置只能包含train/val，不能加入test')
    for dataset in opt['datasets'].values():
        if dataset['type'] != 'FixedPairedDataset':
            raise ValueError('必须使用FixedPairedDataset读取已有噪声图，禁止在线高斯加噪读取器')
    if opt['datasets']['train']['dataset'] != opt['datasets']['val']['dataset']:
        raise ValueError('训练与验证必须属于同一个数据集')
    train_data = opt['datasets']['train']
    if train_data['gt_size'] != 128 or any(size != 128 for size in train_data['gt_sizes']):
        raise ValueError('现有训练块固定128×128，不能启用更大尺寸的渐进训练')
    if train_data['batch_size_per_gpu'] < 1 or any(
        size != train_data['batch_size_per_gpu'] for size in train_data['mini_batch_sizes']
    ):
        raise ValueError('固定尺寸方案中mini_batch_sizes须与batch_size_per_gpu一致且为正')
    if sum(train_data['iters']) != opt['train']['total_iter']:
        raise ValueError('datasets.train.iters之和必须等于train.total_iter')
    if sum(opt['train']['scheduler']['periods']) != opt['train']['total_iter']:
        raise ValueError('学习率调度总迭代数必须与训练总迭代数一致')
    if opt['train']['mixing_augs'].get('mixup', False):
        raise ValueError('固定噪声方案关闭mixup，避免改变原配对与噪声强度')
    if not opt['val'].get('use_image', False):
        raise ValueError('统一按8-bit图像计算验证指标，val.use_image必须为true')
    root = Path(opt['path']['experiments_root']).resolve()
    project = Path(opt['path']['root']).resolve()
    if project not in root.parents or (project / 'data') == root or (project / 'data') in root.parents:
        raise ValueError('实验目录必须位于当前项目内、data目录之外')
    downloaded = project / 'checkpoints' / 'restormer_official'
    if downloaded == root or downloaded in root.parents:
        raise ValueError('不能把训练结果写入已下载的官方预训练权重目录')
    if opt['num_gpu'] and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请使用denoise310，或在配置中设置num_gpu: 0')

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # 官方自动续训改为显式--resume，避免新实验意外加载已有模型。
    resume_state = None
    if opt['explicit_resume']:
        state_directory = Path(opt['path']['training_states']).resolve()
        chosen = opt['path'].get('resume_state')
        if chosen is None:
            states = [path for path in state_directory.glob('*.state') if path.stem.isdigit()]
            if not states:
                raise FileNotFoundError(f'没有可恢复的训练状态：{state_directory}')
            chosen = max(states, key=lambda path: int(path.stem))
        chosen = Path(chosen).resolve()
        if chosen.parent != state_directory:
            raise ValueError('只能恢复当前实验training_states目录内的断点')
        opt['path']['resume_state'] = str(chosen)
        resume_state = torch.load(chosen, map_location='cpu', weights_only=True)

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result
    training, validation = train_loader.dataset, val_loader.dataset
    if len(train_loader) == 0:
        raise ValueError('训练批量大于数据集，无法产生任何batch')
    if training.opt['dataset'] == 'div2k' and training.source_ids & validation.source_ids:
        raise ValueError('DIV2K训练集和验证集存在原图交叉')
    if training.opt['dataset'] == 'sidd':
        logger.warning('沿用现有SIDD划分：训练/测试存在实际场景重叠，官方验证块缺少原图映射。')
    train_options = deepcopy(opt['datasets']['train'])
    for performance_option in ('num_worker_per_gpu', 'prefetch_mode', 'pin_memory'):
        train_options.pop(performance_option, None)
    signature = dict(train_fingerprint=training.fingerprint, val_fingerprint=validation.fingerprint,
                     network=deepcopy(opt['network_g']), train=deepcopy(opt['train']),
                     train_data=train_options,
                     val_metrics=deepcopy(opt['val']['metrics']), seed=opt['manual_seed'], num_gpu=opt['num_gpu'])
    opt['run_signature'] = json.dumps(signature, sort_keys=True)
    if resume_state and resume_state.get('run_signature') != opt['run_signature']:
        raise ValueError('断点的数据版本、噪声配对或训练配置不一致，不能继续此实验')
    with open(osp.join(opt['path']['experiments_root'], 'data_info.json'), 'w', encoding='utf-8') as stream:
        json.dump(signature, stream, ensure_ascii=False, indent=2)

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        resume_skip = resume_state.get('batch_in_epoch', 0)
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0
        resume_skip = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):

    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)

    scale = opt['scale']

    epoch = start_epoch
    while current_iter < total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()
        batch_in_epoch = 0
        # 数据集没有在线随机变换，恢复时跳过已完成的批次，不重复本轮前缀。
        for _ in range(resume_skip):
            if train_data is None:
                raise ValueError('断点的轮内批次位置与数据集长度不一致')
            train_data = prefetcher.next()
            batch_in_epoch += 1
        resume_skip = 0

        while train_data is not None and current_iter < total_iters:
            data_time = time.time() - data_time

            current_iter += 1
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            
            ### ------Progressive learning ---------------------
            j = ((current_iter>groups) !=True).nonzero()[0]
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]
            
            if logger_j[bs_j]:
                logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(mini_gt_size, mini_batch_size*max(1, opt['num_gpu'])))
                logger_j[bs_j] = False

            lq = train_data['lq']
            gt = train_data['gt']

            if mini_batch_size < batch_size:
                indices = random.sample(range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]

            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:,:,x0:x1,y0:y1]
                gt = gt[:,:,x0*scale:x1*scale,y0*scale:y1*scale]
            ###-------------------------------------------

            
            model.feed_train_data({'lq': lq, 'gt':gt})
            model.optimize_parameters(current_iter)
            batch_in_epoch += 1
            opt['batch_in_epoch'] = batch_in_epoch

            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # validation
            if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                model.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'], rgb2bgr, use_image )

            # 先验证更新best阈值，再保存可恢复状态。
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter
        epoch += 1

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    if opt.get('val') is not None and current_iter % opt['val']['val_freq'] != 0:
        model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'], opt['val'].get('rgb2bgr', True),
                         opt['val'].get('use_image', True))
    model.save(epoch=epoch, current_iter=current_iter)
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
