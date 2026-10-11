"""DnCNN 正式训练入口。用 python -m src.train.train_dncnn 从项目根目录运行。"""

import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.sidd_dataset import SIDDDataset
from src.datasets.sidd_train_dataset import SIDDTrainDataset
from src.models.dncnn import DnCNN

EPOCHS = 20  # 总目标轮数；已有第 20 轮断点时不会重新训练
IMAGES_PER_BATCH = 2
PATCHES_PER_IMAGE = 4
PATCH_SIZE = 128
LEARNING_RATE = 0.001  # 仅新建优化器时使用；断点恢复以保存的优化器为准
NUM_WORKERS = 2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVE_DIR = PROJECT_ROOT / "checkpoints"
BEST_PATH = SAVE_DIR / "dncnn_sidd_best.pth"
LAST_PATH = SAVE_DIR / "dncnn_sidd_last.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备：", device)

    model = DnCNN(in_channels=3, features=64, depth=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    start_epoch, best_val_loss = 0, float("inf")

    if LAST_PATH.exists():
        # 只加载你自己可信的训练断点；本文件需要完整优化器状态。
        checkpoint = torch.load(LAST_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best_val_loss = float(checkpoint["best_val_loss"])
        print(f"恢复训练：已完成 {start_epoch} 个 Epoch，学习率 {optimizer.param_groups[0]['lr']}")

    # 检查目标轮数，避免在已完成时重新加载耗时的训练数据。
    if start_epoch >= EPOCHS:
        print(f"已完成 {start_epoch} 个 Epoch；总目标 EPOCHS={EPOCHS}，无需继续训练。")
        return

    train_dataset = SIDDTrainDataset(
        patch_size=PATCH_SIZE, patches_per_image=PATCHES_PER_IMAGE
    )
    val_dataset = SIDDDataset(split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=IMAGES_PER_BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=1 if NUM_WORKERS else None,
        persistent_workers=bool(NUM_WORKERS),
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print("训练 Patch 数量：", len(train_dataset) * PATCHES_PER_IMAGE)
    print("验证样本数：", len(val_dataset))
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, EPOCHS):
        started = time.perf_counter()
        model.train()
        total_train_loss, train_count = 0.0, 0

        for idx, (noisy, gt) in enumerate(train_loader, 1):
            noisy = noisy.flatten(0, 1).to(device, non_blocking=True)
            gt = gt.flatten(0, 1).to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, gt)
            loss.backward()
            optimizer.step()
            batch_count = noisy.size(0)
            total_train_loss += loss.item() * batch_count
            train_count += batch_count
            if idx % 10 == 0 or idx == len(train_loader):
                print(f"Epoch {epoch + 1} | Batch {idx}/{len(train_loader)} | Loss: {loss.item():.6f}", flush=True)

        train_loss = total_train_loss / train_count
        model.eval()
        val_loss_sum, val_psnr_sum, val_count = 0.0, 0.0, 0
        with torch.inference_mode():
            for noisy, gt in val_loader:
                noisy, gt = noisy.to(device), gt.to(device)
                output = model(noisy)
                val_loss_sum += criterion(output, gt).item()  # 历史口径：未 clamp
                mse = torch.mean((output.clamp(0, 1) - gt) ** 2).item()
                val_psnr_sum += -10 * math.log10(max(mse, 1e-12))
                val_count += 1

        val_loss = val_loss_sum / val_count
        val_psnr = val_psnr_sum / val_count
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_PATH)
            print("已保存最佳模型")
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            LAST_PATH,
        )
        elapsed = time.perf_counter() - started
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.2f} dB")
        print(f"Epoch {epoch + 1} 总耗时：{elapsed:.2f} 秒")
    print("训练程序执行结束！")


if __name__ == "__main__":
    main()
