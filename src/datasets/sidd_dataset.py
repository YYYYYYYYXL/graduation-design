"""SIDD Medium sRGB CSV 读取；验证集使用固定裁剪缓存。"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SIDDDataset(Dataset):
    """从 sidd_split.csv 按 train/val/test 加载成对的 RGB 图像。"""

    def __init__(
        self,
        split="train",
        patch_size=128,
        eval_patch_size=256,
        patches_per_image=1,
        project_root=None,
        use_val_cache=True,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"不支持的数据集类型：{split}")
        if patches_per_image < 1:
            raise ValueError("patches_per_image 必须 >= 1")
        self.split = split
        self.patch_size = patch_size
        self.eval_patch_size = eval_patch_size
        self.patches_per_image = patches_per_image
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.val_cache_dir = self.project_root / "data" / "cache" / "sidd_val"
        self.use_val_cache = use_val_cache
        csv_path = self.project_root / "data" / "splits" / "sidd_split.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"找不到数据划分 CSV：{csv_path}")

        self.samples = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                if row["split"] != split:
                    continue
                noisy_path = self._resolve_path(row["noisy_path"])
                gt_path = self._resolve_path(row["gt_path"])
                if not noisy_path.is_file():
                    raise FileNotFoundError(noisy_path)
                if not gt_path.is_file():
                    raise FileNotFoundError(gt_path)
                self.samples.append((noisy_path, gt_path))
        if not self.samples:
            raise ValueError(f"{split} 没有找到样本")

    def _resolve_path(self, value):
        path = Path(value.replace("\\", "/"))
        return path if path.is_absolute() else self.project_root / path

    def __len__(self):
        if self.split == "train":
            return len(self.samples) * self.patches_per_image
        return len(self.samples)

    def __getitem__(self, index):
        if self.split == "train":
            index //= self.patches_per_image
        noisy_path, gt_path = self.samples[index]

        if self.split == "val" and self.use_val_cache and self.val_cache_dir.is_dir():
            noisy_cache = self.val_cache_dir / f"val_{index:04d}_noisy.npy"
            gt_cache = self.val_cache_dir / f"val_{index:04d}_gt.npy"
            if not noisy_cache.is_file() or not gt_cache.is_file():
                raise FileNotFoundError(f"验证缓存不完整：{index}；请运行缓存检查脚本")
            noisy = np.load(noisy_cache, allow_pickle=False)
            gt = np.load(gt_cache, allow_pickle=False)
            expected = (self.eval_patch_size, self.eval_patch_size, 3)
            if noisy.shape != expected or gt.shape != expected:
                raise ValueError(f"验证缓存形状不符，索引 {index}")
            if noisy.dtype != np.uint8 or gt.dtype != np.uint8:
                raise ValueError(f"验证缓存必须是 uint8，索引 {index}")
            return self._to_tensor(noisy), self._to_tensor(gt)

        with Image.open(noisy_path) as image:
            noisy = image.copy()
        with Image.open(gt_path) as image:
            gt = image.copy()
        if noisy.size != gt.size:
            raise ValueError(f"NOISY/GT 尺寸不一致：{noisy_path}")
        if noisy.mode != "RGB" or gt.mode != "RGB":
            raise ValueError(f"图像必须是 RGB：{noisy_path}")

        if self.split == "train":
            noisy, gt = self._crop_pair(noisy, gt, self.patch_size, random_crop=True)
            noisy, gt = self._augment_pair(noisy, gt)
        else:
            noisy, gt = self._crop_pair(noisy, gt, self.eval_patch_size, random_crop=False)
        return self._to_tensor(noisy), self._to_tensor(gt)

    @staticmethod
    def _crop_pair(noisy, gt, size, random_crop):
        width, height = noisy.size
        if min(width, height) < size:
            raise ValueError(f"图片尺寸小于裁剪尺寸 {size}")
        if random_crop:
            x = random.randint(0, width - size)
            y = random.randint(0, height - size)
        else:
            x, y = (width - size) // 2, (height - size) // 2
        box = (x, y, x + size, y + size)
        return noisy.crop(box), gt.crop(box)

    @staticmethod
    def _augment_pair(noisy, gt):
        if random.random() < 0.5:
            noisy = noisy.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            noisy = noisy.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            gt = gt.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        k = random.choice((0, 90, 180, 270))
        if k:
            noisy = noisy.rotate(k)
            gt = gt.rotate(k)
        return noisy, gt

    @staticmethod
    def _to_tensor(image):
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1).copy())
