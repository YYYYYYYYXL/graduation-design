"""20层、64通道的RGB DnCNN，预测噪声残差后从输入中减去。

参考DnCNN论文/作者仓库的Conv-ReLU、Conv-BN-ReLU、Conv残差结构：
https://github.com/cszn/DnCNN
这是项目实现，不声称是作者训练程序或复现其数据协议。
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
        self.network = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def predict_noise(self, noisy):
        if noisy.ndim != 4 or noisy.shape[1] != 3 or min(noisy.shape[-2:]) < 1:
            raise ValueError("DnCNN输入必须为[N,3,H,W] RGB张量")
        return self.network(noisy)

    def forward(self, noisy):
        return noisy - self.predict_noise(noisy)


def load_trained(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("architecture") != ARCHITECTURE:
        raise ValueError("不是本项目DnCNN结构的权重")
    model = DnCNN()
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()
