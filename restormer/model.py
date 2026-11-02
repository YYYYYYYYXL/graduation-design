"""Restormer网络与权重加载。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

# 只加载网络文件，不导入BasicSR训练框架，也不修改全局模块搜索路径。
_source = Path(__file__).resolve().parents[1] / "third_party/Restormer/basicsr/models/archs/restormer_arch.py"
_spec = spec_from_file_location("restormer._architecture", _source)
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
Restormer = _module.Restormer


ARCHITECTURE = "official_restormer_68dc6ac_rgb_biasfree"


def create_model():
    """创建随机初始化的完整RGB去噪模型；训练与推理共用同一份结构配置。"""
    return Restormer(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
                      num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                      bias=False, LayerNorm_type="BiasFree", dual_pixel_task=False)


def load_trained(path, device="cpu"):
    """加载本训练器best/last.pth或旧版net_g_*.pth；不用于初始化训练。"""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "params" not in state:
        raise ValueError("权重必须包含params字典，并与当前Restormer结构匹配")
    if state.get("architecture", ARCHITECTURE) != ARCHITECTURE:
        raise ValueError("权重结构标识与当前Restormer不匹配")
    model = create_model()
    model.load_state_dict(state["params"], strict=True)
    return model.to(device).eval()
