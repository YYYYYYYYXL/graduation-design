"""P2N 单图自监督适应：RDC + DCS，不接收或使用 GT。

依据 https://arxiv.org/html/2412.16460v2 的公式 (3)~(11)、4.1 节。
这是项目内的论文实现，不是作者官方代码。明确的实现选择：
* 每步使用当前网络估计 clean/noise；RDC 在 no_grad 下构造训练输入。
* 每个正/负分支各采样一个全图标量 N(1, scale_std)，不截断或取绝对值。
  直接保留公式中的正态采样，因此标量也可能为负；不额外改变分布。
* 两个 DCS 分支均反向传播；不裁剪再加噪输入，不使用增强、EMA 或 GT。
* AdamW weight_decay=0.01；FP32、梯度范数上限 1 为项目的稳定性设置。
* 论文学习率为 1e-4。当前自训初始化在小验证样本上用该值出现退化，
  因此项目暂用保守的 1e-6；这不是完整验证集调参或论文效果复现。
* 默认 100 步为项目起始设置，后续只能用验证集确定，不能按测试 GT 选步数。
"""

import math
import time
from dataclasses import dataclass

import torch

from p2n.model import load_pretrained


@dataclass
class DenoiseResult:
    image: torch.Tensor  # [1,3,H,W]，在运行设备上，已限制到 [0,1]
    seconds: float  # 重置参数、创建优化器、全部适应步骤和最终推理；排除读写/传图
    losses: tuple


def construct_pair(noisy, clean, scale_std, generator):
    """RDC：预测图和残差均作为固定数据；每步重新估计，不累积计算图。"""
    clean = clean.detach()
    noise = noisy.detach() - clean
    scales = 1.0 + scale_std * torch.randn(2, generator=generator)
    positive = clean + scales[0].to(noisy.device) * noise
    negative = clean - scales[1].to(noisy.device) * noise
    return positive, negative


def consistency_loss(positive, negative, step, steps):
    """DCS：gamma 随步骤从 2 线性降到 1.5；一步时使用起点 2。"""
    gamma = 2.0 - 0.5 * step / max(steps - 1, 1)
    return ((positive - negative).abs() + 1e-8).pow(gamma).mean()


class P2NDenoiser:
    """权重只读加载一次；每次 denoise 都重置模型并新建单图优化器。"""

    def __init__(self, weights, device="cuda"):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；请检查 denoise310 环境或指定 device='cpu'")
        self.model = load_pretrained(weights, device=self.device).float()
        self.initial_state = {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def denoise(self, noisy, steps=100, lr=1e-6, scale_std=0.75, seed=20261021):
        """只输入一张噪声图；steps=0 为预训练网络直接推理的对照。

        noisy 为 RGB float 张量 [1,3,H,W]，范围 [0,1]。
        使用局部随机生成器，不改变调用者的全局随机状态。
        返回最终一步的输出，不按 GT 或自监督损失挑选中间结果。
        """
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("steps 必须为非负整数")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr 必须为有限正数")
        if not math.isfinite(scale_std) or scale_std < 0:
            raise ValueError("scale_std 必须为有限非负数，与图像的噪声 sigma 不同")
        if not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed 必须为 [0,2**63) 内的整数")
        if noisy.ndim != 4 or noisy.shape[:2] != (1, 3) or min(noisy.shape[-2:]) < 1:
            raise ValueError("输入必须为单张 RGB 图片 [1,3,H,W]")
        if not noisy.is_floating_point() or not torch.isfinite(noisy).all():
            raise ValueError("输入必须为有限浮点数")
        if noisy.min() < 0 or noisy.max() > 1:
            raise ValueError("原始噪声图输入应归一化到 [0,1]")
        noisy = noisy.detach().to(device=self.device, dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        self._synchronize()
        started = time.perf_counter()
        self.model.load_state_dict(self.initial_state, strict=True)
        self.model.zero_grad(set_to_none=True)
        losses = []
        if steps:
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    clean = self.model(noisy)
                    positive, negative = construct_pair(noisy, clean, scale_std, generator)
                # 两个分支都需要梯度；不能将其中一个当作固定 GT。
                loss = consistency_loss(self.model(positive), self.model(negative), step, steps)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"P2N 第 {step + 1} 步损失非有限，停止当前图片")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError(f"P2N 第 {step + 1} 步梯度非有限，未执行该步更新")
                optimizer.step()
                losses.append(float(loss.detach()))

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            output = self.model(noisy)
            if not torch.isfinite(output).all():
                raise RuntimeError("P2N 最终输出存在非有限数值")
            output = output.clamp(0, 1)
        self._synchronize()
        return DenoiseResult(output, time.perf_counter() - started, tuple(losses))
