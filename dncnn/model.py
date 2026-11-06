"""
同一网络分别从零训练DIV2K混合15/25/50和SIDD两个模型。
"""

import torch
from torch import nn


ARCHITECTURE = "dncnn_rgb_depth20_channels64_v1"


class DnCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(18):

            layers.extend([nn.Conv2d(64, 64, 3, padding=1, bias=False),
                           nn.BatchNorm2d(64), nn.ReLU(inplace=True)])
        layers.append(nn.Conv2d(64, 3, 3, padding=1))
        self.network = nn.Sequential(*layers) #顺序网络
        for module in self.modules(): # 遍历当前模型中的所有模块，包括里面的卷积层、批归一化层、ReLU 等。
            if isinstance(module, nn.Conv2d): # 是卷积层？
                nn.init.orthogonal_(module.weight) #卷积层的权重参数进行正交初始化
                if module.bias is not None:
                    nn.init.zeros_(module.bias) #偏执归零

    def predict_noise(self, noisy):
        # ndim是图片的维度数量，比如图片数量啊，通道数啊，宽度高度啊
        # 模型的输入数据的维度是四个[图片数量，通道数，宽，高]
        if noisy.ndim != 4 or noisy.shape[1] != 3 or min(noisy.shape[-2:]) < 1:
            raise ValueError("DnCNN输入必须为[N,3,H,W] RGB张量")
        return self.network(noisy)

    def forward(self, noisy):
        return noisy - self.predict_noise(noisy) #带噪图-预测噪声


def load_trained(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("architecture") != ARCHITECTURE:
        raise ValueError("不是本项目DnCNN结构的权重")
    model = DnCNN()
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()
