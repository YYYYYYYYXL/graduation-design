
import torch.nn as nn


class DnCNN(nn.Module):

    def __init__(self, in_channels=3, features=64, depth=20):
        super().__init__()

        if depth < 3:
            raise ValueError("depth 必须至少为 3")

        layers = []

        # 第一部分：输入层
        layers.append(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        )
        layers.append(nn.ReLU(inplace=True))

        # 第二部分：中间隐藏层
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(features, features, kernel_size=3, padding=1)
            )
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # 第三部分：输出层
        layers.append(
            nn.Conv2d(features, in_channels, kernel_size=3, padding=1)
        )

        # 将所有层组合成网络
        self.net = nn.Sequential(*layers)

    def forward(self, noisy):
        # 网络预测噪声
        predicted_noise = self.net(noisy)

        # 残差学习：输入图像减去预测噪声
        denoised = noisy - predicted_noise

        return denoised
