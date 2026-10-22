# DnCNN与Restormer正式训练

在项目根目录、Conda `denoise310`环境运行。四个实验分别训练，输出互不覆盖。
源码默认从随机初始化开始，下载的两个官方Restormer权重不参与训练。

## 数据协议

两种模型共用`paired_dataset.py`，仅读取现有训练集与验证集，不读取测试集。

| 数据集 | 训练输入 | 训练GT | 验证 |
| --- | --- | --- | --- |
| DIV2K | `data/div2k_train/noise/`，5250张 | `data/div2k_train/gt/` | 400个GT分别对应sigma15/25/50，共1200对 |
| SIDD | `data/sidd_train/noisy/`，5184张 | `data/sidd_train/gt/` | 官方`input_crops/target_crops`，1280对 |

- 训练块128×128，验证块256×256，RGB归一化到[0,1]。
- DIV2K训练sigma15/25/50各1750对。一个混合噪声模型覆盖这三个档位，模型输入不需要sigma。
- 直接读取现有噪声PNG；不重新添加高斯噪声，不更改原图或现有CSV，不再次裁剪/增强。
- 两种模型的验证指标统一按8-bit量化图像计算RGB PSNR、逐通道SSIM，选择权重仅用验证集。
- 沿用现有SIDD划分：训练/测试存在实际场景重叠；官方验证块缺少原始场景映射。
  因此不能据此声称验证/测试具有严格的未见场景独立性。

## Restormer

正式训练由项目内固定版本的官方`third_party/Restormer/basicsr/train.py`执行。
官方来源、提交号和必要改动见`third_party/Restormer/ADAPTATIONS.md`。

若缺少新引入的依赖，手动安装：

```powershell
python -m pip install -r requirements-restormer.txt
```

此依赖文件不安装或升级PyTorch、TorchVision；保留denoise310现有CUDA版PyTorch。

```powershell
python train_restormer.py --dataset div2k
python train_restormer.py --dataset sidd
```

两个命令依次运行。配置分别为`options/restormer_div2k.yml`、`options/restormer_sidd.yml`。
默认使用完整官方网络、FP32、单卡batch=1，固定128×128，L1、AdamW，学习率3e-4，
训练300000次参数更新，每5000次验证和保存状态。没有加载官方预训练权重。
这是正式完整训练配置，耗时由用户运行时的日志体现，尚未实测耗时或显存。

现有训练块固定128×128，因此关闭官方默认渐进增大裁剪尺寸与MixUp。
默认配置没有AMP、梯度累积或梯度检查点；之前项目自写训练器的这些设置不再生效。
300000是参数更新次数，不是300000轮，也不能直接与DnCNN的轮数等同。

若调整物理batch，须同时调整配置的`batch_size_per_gpu`与`mini_batch_sizes`。
Windows数据加载遇到进程问题时，可把`num_worker_per_gpu`改为0。
自定义配置可用`--config 路径`启动，不再使用旧训练器的`--epochs`等参数。

输出分别位于`checkpoints/restormer_div2k/`、`checkpoints/restormer_sidd/`：

- `models/net_g_best.pth`：验证集平均PSNR最高的权重，字典键为`params`。
- `models/net_g_latest.pth`：完整训练正常结束时保存的最终权重。
- `models/net_g_迭代数.pth`及`training_states/迭代数.state`：最近两个可恢复断点。
- `validation.csv`：验证整体及各sigma/真实噪声组的PSNR、SSIM。
- `data_info.json`和`train_*.log`：配置、配对指纹、训练损失和运行日志。

恢复训练：

```powershell
python train_restormer.py --dataset div2k --resume
python train_restormer.py --dataset sidd --resume
```

只恢复当前实验自己的断点；配置或数据发生关键变化时会拒绝续训。
新实验使用新配置并更改`path.experiments_root`，不会自动覆盖或移动已有实验。
异常中断恢复到最近成功保存的断点，未保存的迭代需重做。

## DnCNN

使用本项目的20层、64通道RGB DnCNN：预测噪声残差，输出为输入减去预测残差。
训练采用残差MSE、Adam、FP32；默认最多200轮、batch=8，学习率1e-3，每50轮减半。
这是项目训练方案，不称为DnCNN作者官方训练程序。

### DnCNN 提前停止规则

- 每轮用完整验证集计算PSNR、SSIM，使用平均验证PSNR判断是否继续训练。
- 至少训练60轮；连续25轮未累计提高0.01 dB时提前停止。
- 提升阈值相对上一次显著提升的参考PSNR计算，数次小提升可以累计；不要求每一轮都提高0.01 dB。
- 每50轮结束后学习率减半，并重置等待计数，给下一阶段保留观察时间。默认设置下，即使一直无提升，也不会早于第75轮停止。
- `best.pth`仍保存任何一次验证PSNR新高，不受0.01 dB阈值限制；触发早停的本轮状态完整保存到`last.pth`后再结束。
- 这些阈值是保守的初始策略，尚未经过全量训练验证，无法保证一定不损失后续潜在提升。达到200轮时结束，不自动延长。

早停可通过`--early-stop-min-epochs`、`--early-stop-patience`、`--early-stop-min-delta`调整。
`--early-stop-patience 0`关闭早停，训练到指定上限。
早停设置写入`config.json`和权重文件；续训从已保存的验证历史重建等待次数。
已满足早停条件的断点默认不再训练；如需继续，可在`--resume`时增加等待轮数或禁用早停。
旧DnCNN断点没有早停字段时，也按当前命令行阈值和已有历史判断。

### DnCNN 运行方式

```powershell
python train_dncnn.py --dataset div2k
python train_dncnn.py --dataset sidd
```

输出分别位于`checkpoints/dncnn_div2k/`、`checkpoints/dncnn_sidd/`，包括
`best.pth`、`last.pth`、`history.csv`、`config.json`。每轮验证并保存，best按验证PSNR选择。

```powershell
python train_dncnn.py --dataset div2k --resume
python train_dncnn.py --dataset sidd --resume
```

可配置`--epochs`、`--batch-size`、`--workers`等；续训须保留原关键配置。
新实验使用新的`--output-dir`。中途退出后从最近完整轮次继续，半轮内容会重新训练。

两类模型的训练预算、优化器和损失函数不同，实验记录应如实列出。
同一噪声输入上的最终测试指标和耗时需在训练后另行统一评测。
