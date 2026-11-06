"""四种方法的只读推理入口，不调用任何训练或单图适应代码。"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
MODELS = {item["id"]: item for item in json.loads((ROOT / "models.json").read_text(encoding="utf-8"))}
# 普通照片使用现有 SIDD 版本；实验页面逐个选择五个模型和 BM3D。
DEFAULT_MODEL = "restormer_sidd_batch16"
_cached_key = None
_cached_model = None


def model_list():
    return [dict(item, available=item["path"] is None or (ROOT / item["path"]).is_file())
            for item in MODELS.values()]


def load_model(item, device):
    global _cached_key, _cached_model
    path = ROOT / item["path"]
    key = (item["id"], device, path.stat().st_mtime_ns)
    if key == _cached_key:
        return _cached_model
    _cached_model = None
    _cached_key = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    method = item["method"]
    if method == "dncnn":
        if item.get("weights_format") == "fused_bn":
            from dncnn.pretrained import load_pretrained
            model = load_pretrained(path, device)
        else:
            from dncnn.model import load_trained
            model = load_trained(path, device)
    elif method == "p2n":
        from p2n.model import load_pretrained
        model = load_pretrained(path, device)
    else:
        from restormer.model import Restormer
        state = torch.load(path, map_location="cpu", weights_only=True)
        params = state.get("params_ema", state.get("params"))
        if not isinstance(params, dict):
            raise ValueError("Restormer 权重需要 params 或 params_ema 字典")
        # 按权重实际包含的参数选择 LayerNorm，兼容不同来源的快照。
        norm = "WithBias" if any("norm1.body.bias" in key for key in params) else "BiasFree"
        model = Restormer(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
                          num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                          bias=False, LayerNorm_type=norm, dual_pixel_task=False)
        model.load_state_dict(params, strict=True)
        model = model.to(device).eval()
    model.requires_grad_(False)
    _cached_key, _cached_model = key, model
    return model


def neural_image(model, pixels, device):
    """256 像素核心块加 32 像素边界；输出保持原尺寸，限制显存占用。"""
    height, width = pixels.shape[:2]
    image = torch.from_numpy(pixels.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
    output = np.empty_like(pixels)
    with torch.inference_mode():
        for top in range(0, height, 256):
            for left in range(0, width, 256):
                bottom, right = min(top + 256, height), min(left + 256, width)
                y0, x0 = max(0, top - 32), max(0, left - 32)
                y1, x1 = min(height, bottom + 32), min(width, right + 32)
                patch = image[:, :, y0:y1, x0:x1].to(device)
                patch = F.pad(patch, (0, (-(x1-x0)) % 8, 0, (-(y1-y0)) % 8), mode="replicate")
                clean = model(patch)
                clean = clean[0, :, top-y0:bottom-y0, left-x0:right-x0]
                if not torch.isfinite(clean).all():
                    raise ValueError("模型输出包含非有限数值")
                values = clean.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                output[top:bottom, left:right] = np.rint(values * 255).astype(np.uint8)
    return output


def denoise(model_id, input_path, output_path, sigma=None, device="auto"):
    item = MODELS.get(model_id)
    if item is None:
        raise ValueError("未知模型")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("设备必须为 auto、cpu 或 cuda")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用，请选择 CPU 或自动")
    if item["method"] == "bm3d":
        from denoise_bm3d import denoise as bm3d_denoise
        actual_sigma, seconds = bm3d_denoise(input_path, output_path, sigma, verbose=False)
        return dict(seconds=seconds, load_seconds=0, device="cpu", sigma=actual_sigma,
                    tiled=False, parameters=None, weights_bytes=0)
    if not (ROOT / item["path"]).is_file():
        raise ValueError("该模型权重尚未提供")
    started = time.perf_counter()
    model = load_model(item, device)
    if device == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    with Image.open(input_path) as image:
        pixels = np.array(image.convert("RGB"))
    started = time.perf_counter()
    output = neural_image(model, pixels, device)
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    Image.fromarray(output).save(output_path)
    return dict(seconds=seconds, load_seconds=load_seconds, device=device, sigma=sigma,
                tiled=max(pixels.shape[:2]) > 256,
                parameters=sum(parameter.numel() for parameter in model.parameters()),
                weights_bytes=(ROOT / item["path"]).stat().st_size)
