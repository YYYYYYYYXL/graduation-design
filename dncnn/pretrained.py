"""KAIR 彩色盲去噪模型的推理适配；发布权重已合并 BatchNorm。"""

import torch
from torch import nn


class ColorBlindDnCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(18):
            layers.extend([nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True)])
        layers.append(nn.Conv2d(64, 3, 3, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, image):
        return image - self.model(image)


def load_pretrained(path, device="cpu"):
    model = ColorBlindDnCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()
