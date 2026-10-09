
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from src.models.dncnn import DnCNN
from src.datasets.sidd_dataset import SIDDDataset
from src.datasets.sidd_grouped_dataset import SIDDGroupedDataset



# ==========================================
# 1. 训练参数配置
# ==========================================

EPOCHS = 4
BATCH_SIZE = 8
LEARNING_RATE = 0.001
PATCH_SIZE = 128

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVE_DIR = PROJECT_ROOT / "checkpoints"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BEST_PATH = SAVE_DIR / "dncnn_sidd_best.pth"
LAST_PATH = SAVE_DIR / "dncnn_sidd_last.pth"


def main():

    # ==========================================
    # 2. 选择计算设备
    # ==========================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("当前设备：", device)

    # ==========================================
    # 3. 加载训练集和验证集
    # ==========================================

    train_dataset = SIDDGroupedDataset(
    patch_size=PATCH_SIZE,
    patches_per_image=4
    )

    val_dataset = SIDDDataset(split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print("训练 Patch 数量：", len(train_dataset) * 4)
    print("验证样本数：", len(val_dataset))

    # ==========================================
    # 4. 创建 DnCNN 模型
    # ==========================================

    model = DnCNN(
        in_channels=3,
        features=64,
        depth=20
    ).to(device)

    # ==========================================
    # 5. 定义损失函数和优化器
    # ==========================================

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ==========================================
    # 6. 恢复之前的训练进度
    # ==========================================

    start_epoch = 0
    best_val_loss = float("inf")

    if LAST_PATH.exists():

        checkpoint = torch.load(
            LAST_PATH,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]

        print(f"恢复训练：已完成 {start_epoch} 个 Epoch")

    # ==========================================
    # 7. 开始训练
    # ==========================================

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.perf_counter()

        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for batch_idx, (noisy, gt) in enumerate(train_loader, start=1):
            noisy = noisy.flatten(0, 1)
            gt = gt.flatten(0, 1)
            # 数据移动到 GPU
            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            # 清空上一轮梯度
            optimizer.zero_grad()

            # 前向传播
            denoised = model(noisy)

            # 计算损失
            loss = criterion(denoised, gt)

            # 反向传播
            loss.backward()

            # 更新网络参数
            optimizer.step()

            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                print(
                    f"Epoch {epoch + 1} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.6f}",
                    flush=True
                )

            batch_size = noisy.size(0)

            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / train_count

        # ==========================================
        # 8. 验证模型
        # ==========================================

        model.eval()

        val_loss_sum = 0.0
        val_psnr_sum = 0.0
        val_count = 0

        with torch.no_grad():

            for noisy, gt in val_loader:

                noisy = noisy.to(device)
                gt = gt.to(device)

                denoised = model(noisy)

                # 验证损失
                loss = criterion(denoised, gt)

                val_loss_sum += loss.item()

                # PSNR 指标
                denoised_clamped = denoised.clamp(0, 1)

                mse = torch.mean(
                    (denoised_clamped - gt) ** 2
                ).item()

                psnr = -10 * math.log10(
                    max(mse, 1e-12)
                )

                val_psnr_sum += psnr
                val_count += 1

        val_loss = val_loss_sum / val_count
        val_psnr = val_psnr_sum / val_count

        # ==========================================
        # 9. 保存最佳模型
        # ==========================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                BEST_PATH
            )

            print("已保存最佳模型")

        # ==========================================
        # 10. 保存训练断点
        # ==========================================

        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss
            },
            LAST_PATH
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val PSNR: {val_psnr:.2f} dB"
        )
        elapsed = time.perf_counter() - epoch_start

        print(f"Epoch {epoch + 1} 总耗时：{elapsed:.2f} 秒")
        print("训练程序执行结束！")


if __name__ == "__main__":
    main()
