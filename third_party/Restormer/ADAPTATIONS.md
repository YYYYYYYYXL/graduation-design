# 官方源码版本与项目适配

来源：https://github.com/swz30/Restormer

固定提交：`68dc6ac472db26f16361150cb7a96a1bc87da93f`。
`UPSTREAM.json`记录下载时各文件的原始SHA256；许可证保留在`LICENSE.md`。
这里只引入RGB去噪训练所需的官方源码，没有引入演示、其他任务数据集或模型权重。

## 保留的官方部分

- `basicsr/models/archs/restormer_arch.py`保持官方文件原样。
- 使用官方`basicsr/train.py`迭代式训练程序，以及官方模型、L1损失、AdamW和
  `CosineAnnealingRestartCyclicLR`。网络为48通道、[4,6,6,8]块、[1,2,4,8]头、
  4个refinement块、BiasFree归一化，训练梯度范数上限保持官方的0.01。
- 两阶段调度保持92000次固定学习率、208000次余弦下降，共300000次参数更新。

## 数据和硬件适配

- 新增`data/fixed_paired_dataset.py`，调用项目根目录`paired_dataset.py`。
  直接读取已经生成的Noisy/GT；没有在线高斯加噪、随机裁剪或图像增强。
  未引入官方`Dataset_GaussianDenoising`，避免重复加噪。
- 两份项目YAML位于项目根目录`options/`，各自读取DIV2K和SIDD的train/val。
  DIV2K三档sigma混合训练一个模型；验证保留每个sigma的独立分组。
- 使用官方提供的固定128×128尺寸配置路径，不启用渐进增大训练块尺寸。
  关闭MixUp，保持已有噪声方案。默认单卡、batch=1、FP32；这不是官方8卡配置的完整复现。
- 训练入口显式拒绝外部预训练路径；只有`--resume`才能恢复当前实验自己的权重、
  优化器和调度器。恢复时校验训练配置、清单/配对路径/文件大小的指纹，
  保存轮内批次位置并跳过已完成前缀。指纹不是逐文件像素哈希。

## 运行环境、记录和文件处理适配

- 通过项目入口将本目录放到Python搜索路径，新增`basicsr/__init__.py`，防止误用环境里的BasicSR。
- 调整`utils/options.py`的UTF-8配置读取及实验根目录，结果放在项目`checkpoints/`。
  `utils/misc.py`不再自动重命名已有实验目录；非空目录须显式续训或换输出位置。
- 移除未引入的LMDB构建工具、NIQE导入；TorchVision仅在可选的多图网格导出时延迟导入，
  单图验证不需要它，环境信息使用包元数据读取版本。
- 网络权重和训练状态先写`.part`再替换；补充非有限损失/梯度/输出检查。
  保存验证集PSNR最佳的`models/net_g_best.pth`、最终`net_g_latest.pth`及训练断点。
  默认只保留最近两个数字编号的权重/状态组合，避免完整300k训练积累大量断点。
- 验证调用项目`denoising_metrics.py`，与DnCNN统一使用8-bit量化输出、RGB PSNR和
  逐通道二维高斯窗口SSIM。官方源码中的三维SSIM函数保留，但项目配置不调用。
  `validation.csv`记录整体和sigma/真实噪声分组；best始终按PSNR选择。
- 修正原训练循环结束时迭代计数多一、最终验证参数不一致的问题。

这些改动是正式训练适配；不能把本项目的数据、单卡预算或验证协议称为完整复现论文实验。
