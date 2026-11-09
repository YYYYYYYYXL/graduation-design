"""Restormer 网络：归一化、Transformer 模块、完整编码解码网络。"""
# Restormer: Efficient Transformer for High-Resolution Image Restoration
# Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan and Ming-Hsuan Yang
# https://arxiv.org/abs/2111.09881

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """对每个像素位置的特征通道做归一化。"""

    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        self.with_bias = LayerNorm_type != "BiasFree"
        # body 保留现有权重中的参数名称，例如 norm1.body.weight。
        self.body = nn.ParameterDict({"weight": nn.Parameter(torch.ones(dim))})
        if self.with_bias:
            self.body["bias"] = nn.Parameter(torch.zeros(dim))

    def forward(self, features):
        values = features.permute(0, 2, 3, 1)  # [图片数, 高, 宽, 通道]
        variance = values.var(dim=-1, keepdim=True, unbiased=False)
        if self.with_bias:
            mean = values.mean(dim=-1, keepdim=True)
            values = values - mean
        values = values / torch.sqrt(variance + 1e-5) * self.body["weight"]
        if self.with_bias:
            values = values + self.body["bias"]
        return values.permute(0, 3, 1, 2)


class TransformerBlock(nn.Module):
    """一个模块的两步处理：注意力与残差相加，前馈与残差相加。"""

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        # attn 和 ffn 只存放层及参数，具体计算直接写在本类的 forward 中。
        self.attn = nn.Module()
        self.attn.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.attn.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.attn.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.attn.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

        hidden = int(dim * ffn_expansion_factor)
        self.ffn = nn.Module()
        self.ffn.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.ffn.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.ffn.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, features):
        # 第一步：归一化，再按注意力头分组，计算特征通道之间的关系。
        residual = features
        normalized = self.norm1(features)
        qkv = self.attn.qkv(normalized)
        qkv = self.attn.qkv_dwconv(qkv)
        query, key, value = qkv.chunk(3, dim=1)
        batch, channels, height, width = features.shape
        query = query.reshape(batch, self.num_heads, channels // self.num_heads, height * width)
        key = key.reshape(batch, self.num_heads, channels // self.num_heads, height * width)
        value = value.reshape(batch, self.num_heads, channels // self.num_heads, height * width)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        attention = query @ key.transpose(-2, -1)
        attention = (attention * self.attn.temperature).softmax(dim=-1)
        attended = attention @ value
        attended = attended.reshape(batch, channels, height, width)
        attended = self.attn.project_out(attended)
        features = residual + attended

        # 第二步：再次归一化，用卷积与门控处理特征，再做残差相加。
        residual = features
        normalized = self.norm2(features)
        expanded = self.ffn.project_in(normalized)
        expanded = self.ffn.dwconv(expanded)
        first, second = expanded.chunk(2, dim=1)
        gated = F.gelu(first) * second
        refined = self.ffn.project_out(gated)
        return residual + refined


def make_downsample(channels):
    layer = nn.Module()
    layer.body = nn.Sequential(
        nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False), nn.PixelUnshuffle(2)
    )
    return layer


def make_upsample(channels):
    layer = nn.Module()
    layer.body = nn.Sequential(
        nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False), nn.PixelShuffle(2)
    )
    return layer


class Restormer(nn.Module):
    """编码三层、瓶颈一层，再逐步解码并细化，输出图像修正量。"""

    def __init__(self, inp_channels=3, out_channels=3, dim=48, num_blocks=(4, 6, 6, 8),
                 num_refinement_blocks=4, heads=(1, 2, 4, 8), ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type="WithBias", dual_pixel_task=False):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(inp_channels, dim, 3, padding=1, bias=False)

        # 每一阶段放指定数量的 TransformerBlock。层名对应现有权重中的名称。
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[0])])
        self.down1_2 = make_downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[1])])
        self.down2_3 = make_downsample(dim * 2)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[2])])
        self.down3_4 = make_downsample(dim * 4)
        self.latent = nn.Sequential(*[
            TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[3])])

        self.up4_3 = make_upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[2])])
        self.up3_2 = make_upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[1])])
        self.up2_1 = make_upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_refinement_blocks)])

        self.dual_pixel_task = dual_pixel_task
        if dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.output = nn.Conv2d(dim * 2, out_channels, 3, padding=1, bias=bias)

    def forward(self, image):
        # 初始卷积：RGB 三通道转换为 dim 个特征通道，高宽相同。
        initial = self.patch_embed.proj(image)

        # 编码：每次下采样，高宽各减半、通道数加倍。
        encoded1 = self.encoder_level1(initial)
        smaller = self.down1_2.body(encoded1)
        encoded2 = self.encoder_level2(smaller)
        smaller = self.down2_3.body(encoded2)
        encoded3 = self.encoder_level3(smaller)
        smaller = self.down3_4.body(encoded3)
        deepest = self.latent(smaller)

        # 解码：放大后拼接对应的编码特征，再用卷积整理通道数。
        decoded3 = self.up4_3.body(deepest)
        decoded3 = torch.cat([decoded3, encoded3], dim=1)
        decoded3 = self.reduce_chan_level3(decoded3)
        decoded3 = self.decoder_level3(decoded3)
        decoded2 = self.up3_2.body(decoded3)
        decoded2 = torch.cat([decoded2, encoded2], dim=1)
        decoded2 = self.reduce_chan_level2(decoded2)
        decoded2 = self.decoder_level2(decoded2)
        decoded1 = self.up2_1.body(decoded2)
        decoded1 = torch.cat([decoded1, encoded1], dim=1)
        decoded1 = self.decoder_level1(decoded1)

        # 细化后转回 RGB，修正量与输入图相加得到去噪结果。
        refined = self.refinement(decoded1)
        if self.dual_pixel_task:
            refined = refined + self.skip_conv(initial)
            return self.output(refined)
        correction = self.output(refined)
        return correction + image


ARCHITECTURE = "official_restormer_68dc6ac_rgb_biasfree"


def create_model():
    """创建训练使用的 RGB Restormer。"""
    return Restormer(inp_channels=3, out_channels=3, dim=48, num_blocks=(4, 6, 6, 8),
                     num_refinement_blocks=4, heads=(1, 2, 4, 8), ffn_expansion_factor=2.66,
                     bias=False, LayerNorm_type="BiasFree", dual_pixel_task=False)


def load_trained(path, device="cpu"):
    """读取 params 权重，严格匹配网络参数。"""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "params" not in state:
        raise ValueError("权重必须包含params字典，并与当前Restormer结构匹配")
    if state.get("architecture", ARCHITECTURE) != ARCHITECTURE:
        raise ValueError("权重结构标识与当前Restormer不匹配")
    model = create_model()
    model.load_state_dict(state["params"], strict=True)
    return model.to(device).eval()
