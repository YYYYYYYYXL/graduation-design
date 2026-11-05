"""P2N 初始化模型：DIV2K 合成噪声或 SIDD 真实噪声分别做 Noise -> GT 有监督预训练。

在项目根目录、denoise310 环境中运行：
    python train_p2n_pretrain.py
    python train_p2n_pretrain.py --dataset sidd
    python train_p2n_pretrain.py --resume
    python train_p2n_pretrain.py --resume --no-amp
    python train_p2n_pretrain.py --resume --no-amp --resume-lr 0.0001
    python train_p2n_pretrain.py --batch-size 4 --val-batch-size 2

默认 100 轮仅为项目起始配置，不是论文公开的预训练超参数。
每轮验证三档 sigma；按三档平均 PSNR 选择 best.pth。
last.pth 保存最近完整轮次，可续训。此脚本不执行 P2N 单图自监督适应。
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from p2n.dataset import DIV2KPretrainDataset, SIGMAS
from p2n.model import ARCHITECTURE, P2NUNet
from paired_dataset import DenoisingDataset
from training_utils import validate as validate_paired
from training_runtime import configure_runtime, loader_options


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--dataset", choices=("div2k", "sidd"), default="div2k")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--performance", choices=("reproducible", "fast"), default="reproducible")
    parser.add_argument("--epochs", type=int, default=100, help="目标总轮数，续训时不是追加轮数")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2, help="Windows 加载进程异常时可设为 0")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-step", type=int, default=30, help="每隔多少轮将学习率乘以 0.5")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="关闭 CUDA 混合精度；也可用于恢复原 AMP 断点")
    parser.add_argument("--resume", action="store_true", help="从输出目录 last.pth 继续")
    parser.add_argument("--resume-lr", type=float, help="显式调整续训当前学习率，后续仍按原间隔衰减")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.val_batch_size, args.lr_step) < 1 or args.workers < 0:
        parser.error("轮数、batch size、lr-step 必须为正整数，workers 不能为负数")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.seed < 2**32:
        parser.error("lr 必须为有限正数，seed 必须在 [0,2**32) 内")
    if args.resume_lr is not None and (
        not args.resume or not math.isfinite(args.resume_lr) or args.resume_lr <= 0
    ):
        parser.error("resume-lr 必须为有限正数，且需要同时指定 --resume")
    return args


def save_checkpoint(payload, path):
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_history(history, path):
    temporary = path.with_suffix(".csv.part")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def train_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss, samples = 0.0, 0
    for step, (noisy, gt, _) in enumerate(loader, 1):
        noisy = noisy.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            prediction = model(noisy)
            # 训练不 clamp，保留超出 [0,1] 部分的梯度。
            loss = loss_fn(prediction, gt)
        if not torch.isfinite(loss) and scaler.is_enabled():
            # GradScaler 处理反向传播的梯度溢出，不能修复前向传播已产生的 NaN/Inf。
            # 本网络无 BatchNorm/Dropout，重算不会额外更新统计量；不跳过该批数据。
            print(f"Epoch {epoch} | batch {step} 混合精度损失非有限，改用 FP32 重算该批", flush=True)
            del prediction, loss
            with torch.autocast(device_type=device.type, enabled=False):
                loss = loss_fn(model(noisy.float()), gt.float())
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"第 {epoch} 轮第 {step} 批次在 FP32 下仍出现非有限损失；"
                "本批未更新参数，请保留断点并检查模型和输入"
            )
        scaler.scale(loss).backward()
        # 先取消梯度缩放，再限制梯度范数，避免个别批次产生过大的更新。
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not scaler.is_enabled() and not torch.isfinite(grad_norm):
            raise RuntimeError(f"第 {epoch} 轮第 {step} 批次出现非有限梯度，本批未更新参数")
        # AMP 的非有限梯度由 GradScaler 跳过更新并降低缩放因子。
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * noisy.shape[0]
        samples += noisy.shape[0]
        if step == 1 or step % 100 == 0 or step == len(loader):
            print(f"Epoch {epoch} | {step}/{len(loader)} | train MSE {total_loss / samples:.6f}", flush=True)
    return total_loss / samples


@torch.no_grad()
def validate(model, loader, device, dataset="div2k"):
    if dataset == "sidd":
        metrics = validate_paired(model, loader, device)
        metrics["val_psnr_mean"] = metrics["val_psnr"]
        return metrics
    model.eval()
    totals = {sigma: {"count": 0, "mse": 0.0, "psnr": 0.0, "noisy_psnr": 0.0} for sigma in SIGMAS}
    for noisy, gt, sigmas in loader:
        noisy, gt = noisy.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        # 验证统一使用 float32；PSNR 按逐张 RGB 图计算后取平均。
        prediction = model(noisy)
        raw_mse = (prediction - gt).square().mean(dim=(1, 2, 3))
        mse = (prediction.clamp(0, 1) - gt).square().mean(dim=(1, 2, 3))
        noisy_mse = (noisy - gt).square().mean(dim=(1, 2, 3))
        if not torch.isfinite(raw_mse).all():
            raise RuntimeError("验证出现非有限输出，停止保存该轮模型")
        # 极小误差下上限为 120 dB，防止 log(0) 导致无穷值。
        psnr = -10 * torch.log10(mse.clamp_min(1e-12))
        noisy_psnr = -10 * torch.log10(noisy_mse.clamp_min(1e-12))
        for sigma, loss, score, baseline in zip(sigmas.tolist(), raw_mse.tolist(), psnr.tolist(), noisy_psnr.tolist()):
            group = totals[sigma]
            group["count"] += 1
            group["mse"] += loss
            group["psnr"] += score
            group["noisy_psnr"] += baseline
    result = {}
    for sigma, group in totals.items():
        if group["count"] == 0:
            raise ValueError(f"验证集缺少 sigma={sigma}")
        for metric in ("mse", "psnr", "noisy_psnr"):
            result[f"val_{metric}_{sigma}"] = group[metric] / group["count"]
    result["val_psnr_mean"] = sum(result[f"val_psnr_{s}"] for s in SIGMAS) / len(SIGMAS)
    return result


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请使用 denoise310 的 CUDA 版 PyTorch，或显式加 --device cpu")
    default_output = "p2n_pretrain" if args.dataset == "div2k" else "p2n_pretrain_sidd"
    output = (args.output_dir or PROJECT_ROOT / "checkpoints" / default_output).resolve()
    data_root = args.data_root.resolve()
    if output == data_root or data_root in output.parents:
        raise ValueError("训练输出不能位于数据目录内")
    if args.resume:
        if not (output / "last.pth").is_file():
            raise FileNotFoundError(f"没有可恢复的断点：{output / 'last.pth'}")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空：{output}。续训用 --resume；新实验用另一个 --output-dir")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    configure_runtime(args.performance, device)

    if args.dataset == "div2k":
        training = DIV2KPretrainDataset(data_root, "train")
        validation = DIV2KPretrainDataset(data_root, "val")
    else:
        training = DenoisingDataset(data_root, "sidd", "train")
        validation = DenoisingDataset(data_root, "sidd", "val")
    if args.dataset == "div2k" and training.source_ids & validation.source_ids:
        raise ValueError("训练集和验证集存在原图交叉")
    generator = torch.Generator()
    common = loader_options(args.workers, device, args.performance)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                              generator=generator, drop_last=False, **common)
    val_loader = DataLoader(validation, batch_size=args.val_batch_size, shuffle=False, **common)

    model = P2NUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and not args.no_amp)
    # 续训禁止更换数据清单或关键训练设置；允许延长总轮数或显式切换 AMP。
    config = {
        "architecture": ARCHITECTURE, "seed": args.seed, "batch_size": args.batch_size,
        "lr": args.lr, "lr_step": args.lr_step, "lr_gamma": 0.5,
        "optimizer": "Adam", "loss": "MSE", "amp": scaler.is_enabled(),
        "train_manifest_sha256": (training.manifest_sha256 if args.dataset == "div2k" else training.fingerprint),
        "val_manifest_sha256": (validation.manifest_sha256 if args.dataset == "div2k" else validation.fingerprint),
        "sigmas": list(SIGMAS) if args.dataset == "div2k" else None, "input_range": [0.0, 1.0],
    }
    if args.dataset == "sidd":
        config["dataset"] = "sidd"
        config["metric"] = "8bit RGB PSNR and channelwise Gaussian11 SSIM"
    if args.performance == "fast":
        config["performance"] = "fast_tf32_cudnn_benchmark"
    history, best_psnr, start_epoch = [], -math.inf, 1
    if args.resume:
        state = torch.load(output / "last.pth", map_location="cpu", weights_only=True)
        saved_config = {key: value for key, value in state["config"].items() if key != "amp"}
        current_config = {key: value for key, value in config.items() if key != "amp"}
        if saved_config != current_config:
            raise ValueError("断点的数据清单或训练设置不匹配，请恢复原设置；新实验请使用新输出目录")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if args.resume_lr is not None:
            # 保留已完成轮次及调度进度；同时调整 base_lrs，供后续衰减使用。
            base_lr = args.resume_lr / (scheduler.gamma ** (scheduler.last_epoch // scheduler.step_size))
            scheduler.base_lrs = [base_lr for _ in optimizer.param_groups]
            for group in optimizer.param_groups:
                group["lr"] = args.resume_lr
                group["initial_lr"] = base_lr
            print(f"续训当前学习率调整为 {args.resume_lr:g}", flush=True)
        if scaler.is_enabled() and state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        if state["config"]["amp"] != config["amp"]:
            print(f"续训精度切换：AMP={state['config']['amp']} -> AMP={config['amp']}", flush=True)
        history, best_psnr = state["history"], state["best_psnr"]
        start_epoch = state["epoch"] + 1
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if start_epoch > args.epochs:
            print(f"已完成 {start_epoch - 1} 轮；目标总轮数为 {args.epochs}，无需继续。")
            return

    output.mkdir(parents=True, exist_ok=True)
    run_info = dict(config, epochs=args.epochs, data_root=str(args.data_root.resolve()),
                    val_batch_size=args.val_batch_size, workers=args.workers, device=str(device),
                    resume_lr=args.resume_lr, max_grad_norm=1.0,
                    torch_version=str(torch.__version__), numpy_version=np.__version__)
    (output / "config.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    if history:
        write_history(history, output / "history.csv")
    print(f"P2N U-Net | 参数 {sum(p.numel() for p in model.parameters()):,} | {device}")
    print(f"训练 {len(training)} 对 | 验证 {len(validation)} 对 | 输出 {output}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        # 无在线随机增强；每轮固定独立种子，续训仍获得该轮相同的样本顺序。
        generator.manual_seed(args.seed + epoch)
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        metrics = validate(model, val_loader, device, args.dataset)
        scheduler.step()
        row = dict(epoch=epoch, lr=lr, train_mse=train_loss, **metrics,
                   seconds=time.perf_counter() - started)
        history.append(row)
        improved = metrics["val_psnr_mean"] > best_psnr
        if improved:
            best_psnr = metrics["val_psnr_mean"]
        checkpoint = {
            "format_version": 1, "architecture": ARCHITECTURE, "epoch": epoch,
            "model": model.state_dict(), "config": config,
            "best_psnr": best_psnr, "metrics": metrics,
        }
        if improved:
            save_checkpoint(checkpoint, output / "best.pth")
        checkpoint.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                          scaler=scaler.state_dict(), history=history,
                          torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
        save_checkpoint(checkpoint, output / "last.pth")
        write_history(history, output / "history.csv")
        scores = (" | ".join(f"σ{s}: {metrics[f'val_psnr_{s}']:.2f} dB" for s in SIGMAS)
                  if args.dataset == "div2k" else f"SIDD: {metrics['val_psnr']:.2f} dB")
        print(f"Epoch {epoch}/{args.epochs} | {scores} | 平均 {metrics['val_psnr_mean']:.2f} dB"
              f" | {row['seconds']:.1f}s" + (" | 更新最佳权重" if improved else ""), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断；可用 --resume 从最近已保存的完整轮次继续。")
