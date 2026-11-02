# Restormer 来源与项目适配

来源：https://github.com/swz30/Restormer

固定提交：`68dc6ac472db26f16361150cb7a96a1bc87da93f`。
保留 `LICENSE.md`；`UPSTREAM.json` 是最初引入文件的上游哈希清单，部分文件现已移除。

## 网络

只保留 `basicsr/models/archs/restormer_arch.py` 网络源码及其学习注释。
完整 RGB 网络为48通道、[4,6,6,8]块、[1,2,4,8]头、4个精炼块、BiasFree归一化。
`restormer/model.py` 直接加载这个文件，不导入BasicSR、不修改sys.path。

## 训练

训练入口为根目录 `train_restormer.py`，参数通过命令行调整，不再使用YAML。
DIV2K、SIDD分别随机初始化；不加载下载的官方预训练权重。
数据共用 `paired_dataset.py`，仅读取现有train/val，不重新裁剪、增强、加噪。

训练规模与DnCNN按数据遍历次数对齐：最多200轮，每50轮学习率减半；
至少60轮后，验证PSNR连续25轮未累计提高0.01dB则早停，降学习率时重置等待计数。
这些是项目预算设置，不保证两种模型耗时相同或均已充分收敛。
Restormer保留L1、AdamW(lr=3e-4, weight_decay=1e-4)与梯度范数上限0.01。
默认batch=1、CUDA AMP；CPU使用FP32，`--precision fp32`可关闭AMP。
每轮完整验证，使用与DnCNN相同的8-bit RGB PSNR及逐通道SSIM。
训练和验证均显示tqdm进度条。

默认输出为 `checkpoints/restormer_div2k_scratch/`、`checkpoints/restormer_sidd_scratch/`，
与旧训练结果分开；包含best.pth、last.pth、history.csv、config.json。
`--resume`只恢复本训练器最近完整轮次，校验数据指纹、结构和关键训练设置，
并恢复优化器、调度器、AMP scaler和随机状态。中断的半轮需要重做。
数据指纹记录配对路径、文件大小与清单，不是逐像素哈希。
旧版含params的网络权重仍能通过load_trained用于推理，但旧BasicSR训练状态不能续训。

通用BasicSR训练器、分布式入口、动态注册、预取器、重复损失/指标封装和YAML已移除。
这是项目训练协议，不是官方论文训练流程的完整复现。
