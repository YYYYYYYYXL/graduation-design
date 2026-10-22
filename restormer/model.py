"""直接导出项目内的官方Restormer结构，不再维护另一份自写网络。"""

import torch

from restormer.official import use_official_source

use_official_source()
from basicsr.models.archs.restormer_arch import Restormer


ARCHITECTURE = "official_restormer_68dc6ac_rgb_biasfree"


def load_trained(path, device="cpu"):
    """推理时加载官方训练程序输出的net_g_*.pth；训练入口不调用此函数。"""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "params" not in state:
        raise ValueError("要求官方训练格式的params字典，旧版项目自写权重不适用于此结构")
    model = Restormer(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
                      num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                      bias=False, LayerNorm_type="BiasFree", dual_pixel_task=False)
    model.load_state_dict(state["params"], strict=True)
    return model.to(device).eval()
