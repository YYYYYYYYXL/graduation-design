"""Restormer 配对监督训练入口，支持 DIV2K 和 SIDD。"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from paired_dataset import DenoisingDataset, validate
from restormer.model import ARCHITECTURE, create_model

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("div2k", "sidd"), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("轮数、batch-size 和学习率必须为正数")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境没有可用的 CUDA")
    device = torch.device(device_name)
    output = (args.output_dir or ROOT / "checkpoints" / f"restormer_{args.dataset}_batch16").resolve()
    if output.exists():
        raise FileExistsError(f"目录已存在，请用 --output-dir 指定新目录：{output}")
    if output.is_relative_to(args.data_root.resolve()):
        parser.error("输出目录不能放在数据目录内")

    torch.manual_seed(20261022)
    training = DenoisingDataset(args.data_root, args.dataset, "train")
    validation = DenoisingDataset(args.data_root, args.dataset, "val")
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=0)
    model = create_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_function = nn.L1Loss()
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    config = dict(dataset=args.dataset, architecture=ARCHITECTURE, epochs=args.epochs,
                  batch_size=args.batch_size, lr=args.lr, optimizer="AdamW", loss="L1",
                  seed=20261022, device=device_name, data_root=str(args.data_root.resolve()),
                  precision="AMP_FP16" if use_amp else "FP32")
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Restormer / {args.dataset}，训练 {len(training)} 对，验证 {len(validation)} 对", flush=True)
    best_psnr = -math.inf
    fields = ["epoch", "lr", "train_l1", "val_psnr", "val_ssim", "seconds"]
    with (output / "history.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            model.train()
            total_loss = 0.0
            for noisy, clean in train_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    prediction = model(noisy)
                    loss = loss_function(prediction, clean)
                if not torch.isfinite(loss):
                    raise ValueError("训练损失出现无效数值")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, error_if_nonfinite=not use_amp)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * noisy.size(0)

            metrics = validate(model, val_loader, device)
            scheduler.step()
            row = dict(epoch=epoch, lr=learning_rate, train_l1=total_loss / len(training),
                       **metrics, seconds=time.perf_counter() - started)
            state = dict(architecture=ARCHITECTURE, epoch=epoch, params=model.state_dict(),
                         config=config, metrics=metrics)
            torch.save(state, output / "last.pth")
            if metrics["val_psnr"] > best_psnr:
                best_psnr = metrics["val_psnr"]
                torch.save(state, output / "best.pth")
            writer.writerow(row)
            stream.flush()
            print(f"Epoch {epoch}/{args.epochs}，L1={row['train_l1']:.6f}，"
                  f"PSNR={metrics['val_psnr']:.3f}，SSIM={metrics['val_ssim']:.4f}", flush=True)


if __name__ == "__main__":
    main()
