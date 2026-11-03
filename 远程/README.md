# 远程训练文件

本目录按显卡分开代码与参数，共用一份完整裁剪数据和已有断点。生成代码以此前已同步到服务器的本地训练代码为基础，包含本次5090适配；不是重新从已关机服务器导出的全盘环境。

5090也提供单文件部署包：`5090/5090专用整包_代码数据断点.zip`。只上传这一个ZIP，解压一次即可获得代码、全部裁剪数据及最新两组断点，详见[整包使用说明](5090/整包使用说明.md)。

```text
远程/
├── 数据/裁剪数据.zip                  # 两个数据集完整train/val/test，无原始大图
├── 断点/checkpoints.tar.gz            # 桌面备份：DIV2K last75/best13，SIDD last33/best16
├── 4090D/
│   ├── 代码/                         # 可阅读、可上传的完整Restormer运行代码
│   ├── 参数.json
│   ├── 远程代码_4090D.zip
│   └── 使用说明.md
├── 5090/
│   ├── 代码/
│   ├── 参数.json
│   ├── 远程代码_5090.zip
│   └── 使用说明.md
└── 文件校验.json                     # 压缩包大小、SHA256、数据清单检查结果
```

## 参数

| 显卡 | 训练batch | 验证batch | 状态 |
| --- | --- | --- | --- |
| 4090D 24GB | 16 | 4 | 旧实例已实测；训练约157～160秒/轮，显存接近满载 |
| 5090 32GB | 20 | 8 | 待实测起始值；不是最大batch或最快配置 |

两者均为完整Restormer、128×128训练块、256×256验证块、FP16混合精度训练、FP32验证、AdamW、初始学习率0.0003、每50轮减半、最多200轮、4个数据加载进程，保留原早停规则。改batch时沿用last的实际学习率、优化器、调度器和GradScaler，不自动把学习率拉高。更大的batch减少每轮参数更新次数，不等同于提高最终去噪质量。

5090是32GB Blackwell显卡，选择PyTorch 2.8.0 / CUDA 12.8镜像，并运行包内`setup_remote.sh`。仅看到主机驱动显示CUDA版本不能证明Python里的PyTorch构建兼容5090。

## 上传与解压

上传对应显卡的代码ZIP、共用数据ZIP、断点tar.gz到`/root/autodl-tmp/`，不要上传原始大图。两个代码ZIP均解压为`graduation-design/`，分别用于不同实例，不在同一目录混合覆盖。

以5090为例，打开终端：

```bash
cd /root/autodl-tmp
python3 -m zipfile -e 远程代码_5090.zip .
python3 -m zipfile -e 裁剪数据.zip .
tar -xf codex_graduation_data_20261102.tar -C graduation-design
tar -xzf checkpoints.tar.gz -C graduation-design
cd graduation-design
bash setup_remote.sh
```

`setup_remote.sh`仅安装环境并检查数据，不启动训练。已用兼容镜像且环境就绪时可以跳过安装，设置`DENOISE_PYTHON`使用自己的Python绝对路径。

## 先试测5090

有卡后，确保同卡没有其他训练，再运行：

```bash
cd /root/autodl-tmp/graduation-design
/root/venvs/denoise/bin/python -B probe_restormer.py --dataset sidd
```

默认训练候选为16、18、20、22、24、28、32；验证候选为4、8、12、16、24、32、48。每档在独立进程中预热3批、测6批，包含实际前反向/优化器或FP32验证与PSNR/SSIM。使用迁移last的模型和优化器状态，仅在临时进程内更新，不修改best/last。显存不足时结束该阶段更大候选；其他异常停止，不把环境错误误报成显存不足。

分别报告候选中的最大安全batch和吞吐最高的推荐batch；要求至少约1GiB空闲显存并限制峰值reserved。输出`results/codex_batch_probe_*/codex_tuned_profile.json`，不会自动开始正式训练。短时间试测不是对所有整数batch的穷举；正式训练前两轮、尾批次及完整验证仍需观察。

## SIDD续训

```bash
mkdir -p logs
nohup bash resume_sidd.sh > logs/restormer_sidd.log 2>&1 < /dev/null &
tail -f logs/restormer_sidd.log
```

首次将旧SIDD的四个文件复制到新显卡目录后，从第34轮接着训练；原始第33轮last及best保持完整。4090D输出在`checkpoints/remote_4090d/`，5090输出在`checkpoints/remote_5090/`。已迁移后再次运行时直接恢复新目录的last。

若试测生成了推荐配置，给同一命令附加`--profile /完整路径/codex_tuned_profile.json`；试测配置继承相同输出目录和run_tag，因此可直接续训。若只想看启动命令，运行`bash run_gpu.sh sidd --print-command`。

## DIV2K新训练

```bash
nohup bash run_gpu.sh div2k > logs/restormer_div2k.log 2>&1 < /dev/null &
```

新目录从零训练，旧best13/last75不覆盖。这里仅改变显卡速度配置，没有宣称能改善原模型的纹理丢失；改善质量需要另行设计训练策略，并按验证集选择模型。测试集不参与batch试测、训练或选权重。

## 重新生成目录与压缩包

请修改项目根目录的源代码与`options/`配置，再运行`build_remote_packages.py`；不要直接修改生成的`代码/`目录。完整训练包不包含Git目录、凭据、原始大图、本地测试结果或第三方预训练成品权重。

已有分包后，运行`python -B build_remote_packages.py --bundle-5090-only`可重新生成5090单文件部署包。

官方兼容性参考：[PyTorch Blackwell/CUDA 12.8支持](https://pytorch.org/blog/pytorch-2-7/)、[PyTorch历史版本安装](https://pytorch.org/get-started/previous-versions/)、[RTX 5090规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)。
