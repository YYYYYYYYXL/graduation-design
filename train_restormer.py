"""完整Restormer配对监督训练，从随机初始化开始，不在线加噪或裁剪。

python train_restormer.py --dataset div2k
python train_restormer.py --dataset sidd
python train_restormer.py --dataset div2k --resume

DIV2K直接读取manifest中的三档固定噪声；SIDD直接读取真实噪声/GT。
默认最多200轮、batch1、CUDA混合精度、AdamW、L1监督、每50轮学习率减半。
训练规模按数据遍历次数与DnCNN对齐，不代表计算耗时相同。
--precision fp32关闭混合精度；CPU固定使用FP32。旧BasicSR断点不能续训。
至少60轮后，验证PSNR连续25轮未累计提高0.01dB则提前停止；学习率减半时重置等待计数。
验证采用与DnCNN相同的8-bit量化、RGB PSNR/SSIM，并按PSNR选best。
每轮保存last.pth及训练状态，--resume恢复最近完整轮次。
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from training_utils import early_stopping_status, save_checkpoint, write_history, validate
from restormer.model import ARCHITECTURE, create_model
from paired_dataset import DenoisingDataset
from training_runtime import configure_runtime, loader_options


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=("div2k", "sidd"))
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--performance", choices=("reproducible", "fast"), default="reproducible")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-step", type=int, default=50)
    parser.add_argument("--early-stop-min-epochs", type=int, default=60,
                        help="允许提前停止的最早轮次，默认60")
    parser.add_argument("--early-stop-patience", type=int, default=25,
                        help="验证PSNR无显著提升的等待轮数，0表示禁用提前停止")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.01,
                        help="重置等待计数所需的累计PSNR提升，单位dB，默认0.01")
    parser.add_argument("--precision", choices=("amp", "fp32"), default="amp")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.val_batch_size, args.lr_step) < 1 or args.workers < 0:
        parser.error("轮数、batch、lr-step须为正整数，workers不能为负")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.seed < 2**32:
        parser.error("学习率须为有限正数，seed须在[0,2**32)内")
    if args.early_stop_min_epochs < 1 or args.early_stop_patience < 0:
        parser.error("早停最少轮数须为正整数，等待轮数须为非负整数")
    if not math.isfinite(args.early_stop_min_delta) or args.early_stop_min_delta < 0:
        parser.error("早停PSNR提升阈值须为有限非负数")
    return args


def train_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total, count = 0.0, 0
    with tqdm(loader, desc=f"Epoch {epoch} 训练", unit="batch", dynamic_ncols=True,
              mininterval=0.5) as progress:
        for step, (noisy, gt, _) in enumerate(progress, 1):
            noisy, gt = noisy.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                prediction = model(noisy)
                loss = F.l1_loss(prediction, gt)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Epoch {epoch} 第{step}批损失非有限，停止本轮")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # 先还原梯度尺度，再按官方阈值裁剪。
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, error_if_nonfinite=not scaler.is_enabled())
            scaler.step(optimizer)  # AMP溢出时跳过更新并降低尺度；FP32非有限梯度直接报错。
            scaler.update()
            total += float(loss.detach()) * noisy.size(0)
            count += noisy.size(0)
            progress.set_postfix(L1=f"{total / count:.6f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2g}", refresh=False)
    return total / count


def main():
    args = parse_args()
    early_stop_settings = dict(min_epochs=args.early_stop_min_epochs, patience=args.early_stop_patience,
                               min_delta=args.early_stop_min_delta, lr_step=args.lr_step)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，请使用denoise310；需要CPU时显式加--device cpu")
    use_amp = device.type == "cuda" and args.precision == "amp"
    data_root = args.data_root.resolve()
    output = (args.output_dir or PROJECT_ROOT / "checkpoints" / f"restormer_{args.dataset}_scratch").resolve()
    if output == data_root or data_root in output.parents:
        raise ValueError("训练输出不能位于数据目录内")
    state = None
    if args.resume:
        state = torch.load(output / "last.pth", map_location="cpu", weights_only=True)
        if state.get("format_version") != 2 or state.get("architecture") != ARCHITECTURE:
            raise ValueError("断点格式不匹配，只能恢复本训练器保存的last.pth")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空：{output}。续训加--resume，新实验换--output-dir")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    configure_runtime(args.performance, device)
    training = DenoisingDataset(data_root, args.dataset, "train")
    validation = DenoisingDataset(data_root, args.dataset, "val")
    if args.dataset == "div2k" and training.source_ids & validation.source_ids:
        raise ValueError("DIV2K训练和验证存在原图交叉")
    generator = torch.Generator()
    common = dict(loader_options(args.workers, device, args.performance), drop_last=False)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(validation, batch_size=args.val_batch_size, shuffle=False, **common)

    model = create_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    config = dict(architecture=ARCHITECTURE, dataset=args.dataset, seed=args.seed,
                  batch_size=args.batch_size, lr=args.lr, lr_step=args.lr_step, lr_gamma=0.5,
                  optimizer="AdamW", weight_decay=1e-4, loss="L1",
                  precision="AMP_FP16" if use_amp else "FP32", max_grad_norm=0.01, initialization="random",
                  train_fingerprint=training.fingerprint, val_fingerprint=validation.fingerprint,
                  online_noise=False, online_crop=False, online_augmentation=False,
                  metric="8bit RGB PSNR and channelwise Gaussian11 SSIM",
                  sigmas=[15, 25, 50] if args.dataset == "div2k" else None,
                  noise_type="fixed_gaussian_png" if args.dataset == "div2k" else "real")
    history, best_psnr, start_epoch = [], -math.inf, 1
    if args.performance == "fast":
        config["performance"] = "fast_tf32_cudnn_benchmark"
    if state is not None:
        if state["config"] != config:
            raise ValueError("断点数据版本或训练设置不同，恢复原设置或另开输出目录")
        model.load_state_dict(state["params"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best_psnr, start_epoch = state["history"], state["best_psnr"], state["epoch"] + 1
        previous_early_stop = state.get("early_stopping", {}).get("settings")
        if previous_early_stop is not None and previous_early_stop != early_stop_settings:
            print("早停设置已调整，将按当前阈值重新判断已有验证历史。", flush=True)
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        rng = state["numpy_rng"]
        np.random.set_state((rng[0], rng[1].numpy().astype(np.uint32), *rng[2:]))
        del state
        if start_epoch > args.epochs:
            print(f"已完成{start_epoch - 1}轮，目标{args.epochs}轮，无需继续。")
            return
        if early_stopping_status(history, **early_stop_settings)["should_stop"]:
            print(f"第{start_epoch - 1}轮已满足提前停止条件，保留best.pth。"
                  "如需继续观察，可增加--early-stop-patience或设为0。", flush=True)
            return

    output.mkdir(parents=True, exist_ok=True)
    run_info = dict(config, epochs=args.epochs, workers=args.workers, val_batch_size=args.val_batch_size,
                    early_stopping=early_stop_settings,
                    data_root=str(data_root), device=str(device), torch_version=str(torch.__version__))
    (output / "config.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    if history:
        write_history(history, output / "history.csv")
    print(f"Restormer | {args.dataset} | 训练{len(training)}对 | 验证{len(validation)}对 | 输出{output}", flush=True)
    print(f"最多{args.epochs}轮 | batch={args.batch_size} | {config['precision']} | 早停设置：{early_stop_settings}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        generator.manual_seed(args.seed + epoch)
        lr = optimizer.param_groups[0]["lr"]
        train_l1 = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        metrics = validate(model, val_loader, device, epoch=epoch)
        scheduler.step()
        row = dict(epoch=epoch, lr=lr, train_l1=train_l1, **metrics, seconds=time.perf_counter() - started)
        history.append(row)
        improved = metrics["val_psnr"] > best_psnr
        if improved:
            best_psnr = metrics["val_psnr"]
        stopping = early_stopping_status(history, **early_stop_settings)
        payload = dict(format_version=2, architecture=ARCHITECTURE, epoch=epoch, params=model.state_dict(),
                       config=config, metrics=metrics, best_psnr=best_psnr,
                       early_stopping=dict(settings=early_stop_settings, status=stopping))
        if improved:
            save_checkpoint(payload, output / "best.pth")
        rng = np.random.get_state()
        payload.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), history=history,
                       scaler=scaler.state_dict(), python_rng=random.getstate(),
                       numpy_rng=(rng[0], torch.from_numpy(rng[1].astype(np.int64)), *rng[2:]),
                       torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
        save_checkpoint(payload, output / "last.pth")
        write_history(history, output / "history.csv")
        print(f"Epoch {epoch}/{args.epochs} | L1 {train_l1:.6f} | PSNR {metrics['val_psnr']:.3f} dB"
              f" | SSIM {metrics['val_ssim']:.4f} | {row['seconds']:.1f}秒"
              + (f" | 早停等待 {stopping['bad_epochs']}/{args.early_stop_patience}"
                 if stopping["enabled"] else " | 早停已关闭")
              + (" | 更新best.pth" if improved else ""), flush=True)
        if stopping["should_stop"]:
            print(f"第{epoch}轮提前停止：验证PSNR连续{stopping['bad_epochs']}轮未累计提高"
                  f"{args.early_stop_min_delta:g} dB。最佳PSNR {best_psnr:.3f} dB，"
                  "请使用best.pth；last.pth保留本轮完整状态。", flush=True)
            break
    else:
        print(f"已达到训练上限{args.epochs}轮，最佳PSNR {best_psnr:.3f} dB，请使用best.pth。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断，可加--resume从最近保存的完整轮次恢复。")
