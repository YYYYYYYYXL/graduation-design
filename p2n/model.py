"""用于高斯去噪预训练、后续 P2N 单图适应的同一个 U-Net。

结构依据：P2N 论文 4.1 节及 MASH 的 UNet_n2n_un。
https://arxiv.org/html/2412.16460v2#S4.SS1
https://github.com/hamadichihaoui/mash/blob/master/model.py
这是项目内独立实现；参数名称不同，不保证兼容第三方权重文件。
"""

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = "p2n_n2n_unet_rgb_v1"


def conv_activation(channels_in, channels_out):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )


class P2NUNet(nn.Module):
    """RGB 输入/输出；输出是干净图估计，不是噪声残差。

    五次下采样，编码器 48 通道、解码器 96 通道，无 BatchNorm。
    预训练输入为 [0,1]；forward 不限制数值范围，供后续 P2N 再加噪使用。
    非 32 倍数的尺寸在右侧/下侧复制边缘补齐，输出恢复原尺寸。
    """

    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(conv_activation(3, 48), conv_activation(48, 48))
        self.down = nn.ModuleList([conv_activation(48, 48) for _ in range(4)])
        self.bottom = conv_activation(48, 48)
        self.up = nn.ModuleList([
            nn.Sequential(conv_activation(channels, 96), conv_activation(96, 96))
            for channels in (96, 144, 144, 144)
        ])
        self.output = nn.Sequential(
            conv_activation(99, 64), conv_activation(64, 32),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, noisy):
        if noisy.ndim != 4 or noisy.shape[1] != 3:
            raise ValueError("输入必须是 [N,3,H,W] RGB 张量")
        height, width = noisy.shape[-2:]
        if height < 1 or width < 1:
            raise ValueError("图片尺寸必须大于零")
        padded = F.pad(noisy, (0, (-width) % 32, 0, (-height) % 32), mode="replicate")
        features = F.max_pool2d(self.entry(padded), 2)
        skips = [features]
        for index, block in enumerate(self.down):
            features = F.max_pool2d(block(features), 2)
            if index < 3:
                skips.append(features)
        features = self.bottom(features)
        for block, skip in zip(self.up, reversed(skips)):
            features = F.interpolate(features, scale_factor=2, mode="nearest")
            features = block(torch.cat((features, skip), dim=1))
        features = F.interpolate(features, scale_factor=2, mode="nearest")
        clean = self.output(torch.cat((features, padded), dim=1))
        return clean[..., :height, :width]


def load_pretrained(path, device="cpu"):
    """加载本项目 best.pth 或 last.pth，返回 eval 模式的网络。

    后续 P2N 对每张图片重新调用此函数，再 model.train()，避免跨图累积适应。
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("权重的网络结构标识与 P2NUNet 不一致")
    model = P2NUNet()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()
