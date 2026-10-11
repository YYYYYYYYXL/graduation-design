"""DnCNN 推理：支持分块处理大图；仅推理，不修改训练权重。"""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.dncnn import DnCNN


def load_dncnn(weight_path, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DnCNN(in_channels=3, features=64, depth=20).to(device)
    state = torch.load(Path(weight_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def denoise_image(image, model, tile_size=512, halo=24, fp16=False):
    """对 PIL RGB 图片分块推理，返回去噪 PIL 图像。

    DnCNN depth=20, 3x3 卷积的感受野半径为 20 像素。halo>=20 可保证
    在 FP32 下分块内侧结果与整图卷积边界条件一致（浮点误差除外）。
    """
    if tile_size < 1 or halo < 20:
        raise ValueError("tile_size 必须为正数；当前 20 层 DnCNN 的 halo 不应小于 20")
    image = image.convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)
    _, height, width = tensor.shape
    result = torch.empty_like(tensor)
    device = next(model.parameters()).device
    model.eval()
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if fp16 and device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), ctx:
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom, right = min(top + tile_size, height), min(left + tile_size, width)
                y0, x0 = max(0, top - halo), max(0, left - halo)
                y1, x1 = min(height, bottom + halo), min(width, right + halo)
                patch = tensor[:, y0:y1, x0:x1].unsqueeze(0).to(device)
                output = model(patch).clamp_(0, 1)[0]
                result[:, top:bottom, left:right] = (
                    output[:, top - y0:bottom - y0, left - x0:right - x0]
                    .float().cpu()
                )
    output_array = (
        result.mul_(255).round_().byte().permute(1, 2, 0).numpy()
    )
    return Image.fromarray(output_array, mode="RGB")
