"""实验页面的无参考指标；使用本地评分文件，不训练、不联网。"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from skimage.restoration import estimate_sigma

ASSETS = Path(__file__).resolve().parent / "third_party/brisque"


@lru_cache(maxsize=1)
def brisque_evaluator():
    # 先用 Python 读取文本，兼容 Windows 中文目录和特殊字符路径。
    flags = cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY
    model_file = cv2.FileStorage((ASSETS / "brisque_model_live.yml").read_text(), flags)
    range_file = cv2.FileStorage((ASSETS / "brisque_range_live.yml").read_text(), flags)
    try:
        model = cv2.ml.SVM_create()
        model.read(model_file.getFirstTopLevelNode())
        return cv2.quality.QualityBRISQUE_create(model, range_file.getNode("range").mat())
    finally:
        model_file.release()
        range_file.release()


def no_reference_metrics(pixels):
    """BRISQUE 越低通常越好；噪声估计仅作辅助，不能奖励过度平滑。"""
    result = dict(brisque=None, estimated_noise=None, quality_notes=[])
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    if min(gray.shape) < 32:
        result["quality_notes"].append("图片边长不足 32 像素，不计算无参考评分。")
        return result
    if np.ptp(gray) == 0:
        result["quality_notes"].append("画面没有亮度变化，不计算 BRISQUE。")
    else:
        try:
            score = float(brisque_evaluator().compute(gray)[0])
            if np.isfinite(score):
                result["brisque"] = score
            else:
                result["quality_notes"].append("BRISQUE 无法得到有效数值。")
        except (cv2.error, OSError, AttributeError) as exc:
            # 评分失败不丢掉已经完成的去噪图片，也不填入虚假的分数。
            import logging
            logging.getLogger(__name__).warning("BRISQUE 计算失败: %s", exc)
            result["quality_notes"].append("BRISQUE 暂不可用，请检查评分依赖和本地评分文件。")
    # 逐通道估计，常量通道直接取 0，避免小波估计空样本导致 NaN。
    estimates = [0.0 if np.ptp(channel) == 0 else float(estimate_sigma(channel)) * 255
                 for channel in (pixels.astype(np.float32) / 255).transpose(2, 0, 1)]
    value = float(np.mean(estimates))
    if np.isfinite(value):
        result["estimated_noise"] = value
    else:
        result["quality_notes"].append("本图无法得到有效的噪声估计。")
    return result
