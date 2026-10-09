
import random
from pathlib import Path

import numpy as np
import torch

from src.datasets.sidd_dataset import SIDDDataset


class SIDDGroupedDataset(SIDDDataset):

    def __init__(
        self,
        patch_size=128,
        patches_per_image=4,
        project_root=None
    ):
        super().__init__(
            split="train",
            patch_size=patch_size,
            patches_per_image=patches_per_image,
            project_root=project_root
        )

        # 完整图像解码缓存目录
        self.cache_dir = (
            self.project_root
            / "data"
            / "cache"
            / "sidd_full"
        )

        if not self.cache_dir.is_dir():
            raise FileNotFoundError(
                f"缓存目录不存在：{self.cache_dir}"
            )

        # 检查缓存是否齐全
        for i in range(len(self.samples)):
            for kind in ("noisy", "gt"):
                path = self.cache_dir / f"test_{i:04d}_{kind}.npy"

                if not path.is_file():
                    raise FileNotFoundError(path)

    def __len__(self):
        # 每个索引对应一对原始图像
        return len(self.samples)

    def __getitem__(self, index):

        # 1. 内存映射读取解码缓存
        noisy = np.load(
            self.cache_dir / f"test_{index:04d}_noisy.npy",
            mmap_mode="r"
        )

        gt = np.load(
            self.cache_dir / f"test_{index:04d}_gt.npy",
            mmap_mode="r"
        )

        if noisy.shape != gt.shape:
            raise ValueError("NOISY 与 GT 尺寸不一致")

        if noisy.ndim != 3 or noisy.shape[2] != 3:
            raise ValueError("缓存不是 RGB 三通道图像")

        h, w, _ = noisy.shape

        if h < self.patch_size or w < self.patch_size:
            raise ValueError("原图小于 Patch 尺寸")

        noisy_patches = []
        gt_patches = []

        # 2. 每对原图随机生成4对Patch
        for _ in range(self.patches_per_image):

            y = random.randint(0, h - self.patch_size)
            x = random.randint(0, w - self.patch_size)

            n = noisy[
                y:y+self.patch_size,
                x:x+self.patch_size
            ]

            g = gt[
                y:y+self.patch_size,
                x:x+self.patch_size
            ]

            # 3. 同步随机增强
            if random.random() < 0.5:
                n = np.flip(n, axis=1)
                g = np.flip(g, axis=1)

            if random.random() < 0.5:
                n = np.flip(n, axis=0)
                g = np.flip(g, axis=0)

            k = random.randrange(4)

            if k:
                n = np.rot90(n, k)
                g = np.rot90(g, k)

            # 4. 转换为float32 Tensor
            n = np.ascontiguousarray(n.transpose(2, 0, 1))
            g = np.ascontiguousarray(g.transpose(2, 0, 1))

            n = torch.from_numpy(n).float() / 255.0
            g = torch.from_numpy(g).float() / 255.0

            noisy_patches.append(n)
            gt_patches.append(g)

        # 5. 返回4对Patch
        return (
            torch.stack(noisy_patches),
            torch.stack(gt_patches)
        )
