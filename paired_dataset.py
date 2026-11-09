"""读取现有的 Noisy/GT 图像对，供 DnCNN 和 Restormer 训练使用。"""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from denoising_metrics import calculate_psnr, calculate_ssim, to_uint8


class DenoisingDataset(Dataset):
    def __init__(self, data_root, dataset, split):
        if dataset not in ("div2k", "sidd") or split not in ("train", "val"):
            raise ValueError("请选择 div2k/sidd 数据集与 train/val 划分")
        root = Path(data_root).resolve()
        folder = root / f"{dataset}_{split}"
        self.pairs = []

        if dataset == "div2k":
            # 清单包含训练图的固定噪声，以及验证图的 15/25/50 三档噪声。
            with (folder / "noise_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    if row["split"] != split:
                        raise ValueError("加噪清单的划分与所选集合不一致")
                    noisy = root / row["noise_path"].replace("\\", "/")
                    clean = root / row["gt_path"].replace("\\", "/")
                    self.pairs.append((noisy, clean))
        else:
            # SIDD 训练块与验证块使用不同目录，同名文件组成一对。
            if split == "train":
                noisy_folder, clean_folder = folder / "noisy", folder / "gt"
            else:
                noisy_folder, clean_folder = folder / "input_crops", folder / "target_crops"
            noisy_files = {path.name: path for path in noisy_folder.glob("*.png")}
            clean_files = {path.name: path for path in clean_folder.glob("*.png")}
            if noisy_files.keys() != clean_files.keys():
                raise ValueError("SIDD 的带噪图片和干净图片没有完整配对")
            self.pairs = [(noisy_files[name], clean_files[name]) for name in sorted(noisy_files)]

        if not self.pairs:
            raise ValueError(f"没有找到配对图片：{folder}")
        for noisy, clean in self.pairs:
            if not noisy.is_file() or not clean.is_file():
                raise FileNotFoundError(f"图片对缺失：{noisy} / {clean}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        noisy_path, clean_path = self.pairs[index]
        images = []
        for path in (noisy_path, clean_path):
            with Image.open(path) as image:
                pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
            images.append(torch.from_numpy(pixels).permute(2, 0, 1).contiguous())
        noisy, clean = images
        if noisy.shape != clean.shape or min(clean.shape[-2:]) < 11:
            raise ValueError(f"配对图尺寸不一致或尺寸过小：{noisy_path}")
        return noisy, clean


@torch.no_grad()
def validate(model, loader, device):
    """按图片平均 PSNR、SSIM，与网页使用相同的 8-bit RGB 计算方法。"""
    model.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    for noisy, clean in loader:
        prediction = model(noisy.to(device))
        if not torch.isfinite(prediction).all():
            raise ValueError("验证输出出现无效数值")
        for output, target in zip(prediction, clean):
            output_pixels, target_pixels = to_uint8(output), to_uint8(target)
            total_psnr += calculate_psnr(output_pixels, target_pixels)
            total_ssim += calculate_ssim(output_pixels, target_pixels)
            count += 1
    return {"val_psnr": total_psnr / count, "val_ssim": total_ssim / count}
