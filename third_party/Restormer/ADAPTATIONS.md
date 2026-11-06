# Restormer 来源与项目适配

来源：https://github.com/swz30/Restormer

固定提交：`68dc6ac472db26f16361150cb7a96a1bc87da93f`。
保留 `LICENSE.md`；`UPSTREAM.json` 是最初引入文件的上游哈希清单，部分文件现已移除。

## 网络

只保留 `basicsr/models/archs/restormer_arch.py` 网络源码及其学习注释。
完整 RGB 网络为48通道、[4,6,6,8]块、[1,2,4,8]头、4个精炼块、BiasFree归一化。
`restormer/model.py` 直接加载这个文件，不导入BasicSR、不修改sys.path。

## 当前用途

当前系统只加载网络和已有权重进行推理。历史训练入口、训练数据加载器和归档代码已按项目清理要求移除，模型权重继续保留。本目录保留网络源码、来源信息及许可证。
