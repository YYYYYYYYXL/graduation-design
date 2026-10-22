"""统一的逐张8-bit彩色图像指标；三个通道等权，不混合颜色通道。

用于训练期间验证；图像在评分前统一限制到[0,1]并量化到8-bit。
SSIM为11×11高斯窗口，sigma=1.5，总体协方差，RGB逐通道平均。
"""

import numpy as np
from skimage.metrics import structural_similarity


def to_uint8(tensor):
    """单张 [3,H,W] Tensor -> [H,W,3] uint8；不读写图片。"""
    pixels = tensor.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.rint(pixels * 255).astype(np.uint8)


def calculate_psnr(prediction, target, **kwargs):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[2] != 3:
        raise ValueError("指标输入必须为同尺寸H×W×3的8-bit彩色图")
    if prediction.dtype != np.uint8 or target.dtype != np.uint8:
        raise ValueError("指标输入必须已量化为uint8，不能通过像素最大值猜测动态范围")
    error = np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2)
    return float(-10 * np.log10(max(error / (255.0 ** 2), 1e-12)))


def calculate_ssim(prediction, target, **kwargs):
    if prediction.shape != target.shape or prediction.dtype != np.uint8 or target.dtype != np.uint8:
        raise ValueError("SSIM要求同尺寸uint8图像")
    return float(structural_similarity(target, prediction, data_range=255, channel_axis=-1,
                                       gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
