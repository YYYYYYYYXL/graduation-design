# AutoDL 4090D 训练

项目：`/root/autodl-tmp/graduation-design`。环境：`/root/venvs/denoise`，Python 3.12、PyTorch 2.8.0+cu128。

## 拿卡后启动

先在 AutoDL 控制台正常有卡开机，然后运行任意一条命令。绝对路径入口不依赖终端当前目录。

```bash
bash /root/autodl-tmp/graduation-design/run_autodl.sh dncnn div2k
bash /root/autodl-tmp/graduation-design/run_autodl.sh dncnn sidd
bash /root/autodl-tmp/graduation-design/run_autodl.sh p2n div2k
bash /root/autodl-tmp/graduation-design/run_autodl.sh p2n sidd
bash /root/autodl-tmp/graduation-design/run_autodl.sh restormer div2k
bash /root/autodl-tmp/graduation-design/run_autodl.sh restormer sidd
```

运行全部六组训练：`bash /root/autodl-tmp/graduation-design/run_autodl.sh all`，依次执行，不同时挤占显存。断开 SSH 前应自行使用终端复用工具；本入口不会自动开启后台任务或关机。

同一命令加 `--resume` 续训，参数及数据必须与断点兼容。新实验必须换 `--output-root`，不要覆盖已有权重。

## 路径

- 数据默认位于项目的 `data/`，包含 `div2k_train/val/test`、`sidd_train/val/test` 及相应清单；直接使用已有裁剪和固定噪声数据，不需原始大图或重新预处理。
- 权重默认位于 `checkpoints/<模型>_<数据集>_fast_4090d/`，每组分别保存 `best.pth`、`last.pth`、配置和历史。
- `--data-root`、`--output-root` 的相对路径均以项目目录为基准，也可传 Linux 绝对路径。不要使用 Windows 盘符。
- P2N 单图适应另用 `denoise_p2n.py --weights` 显式指定对应数据集的 `best.pth`；本训练入口只执行监督训练。

## 速度配置与限制

`options/autodl_4090d.json` 是未实测的 24GB 起始配置，不是显存最大值或吞吐最优值。

| 模型 | 训练 batch | 验证 batch | 精度 | 最多轮数 |
| --- | --- | --- | --- | --- |
| DnCNN | 32 | 4 | BF16 | 200 |
| P2N 预训练 | 16 | 2 | AMP FP16 | 100 |
| Restormer | 8 | 1 | AMP FP16 | 200 |

每组使用 4 个数据加载进程、预取和持久进程；快速模式开启 cuDNN autotune 与 TF32，不保证逐位复现。默认直接训练脚本仍保留原有参数。学习率和衰减计划沿用现有各模型设置；DnCNN/Restormer保留早停规则。

更大的 batch 减少每轮更新次数，与原 batch 的同轮数训练不等价。混合精度和 TF32 也可能影响数值结果。各模型仍使用完整结构、128×128训练块和256×256验证块，不通过缩小网络或数据来提速。

可用 `--batch-size`、`--val-batch-size`、`--workers`、`--epochs` 覆盖起始配置。首次拿卡若显存不足，减半训练 batch；若验证阶段显存不足，减小验证 batch。没有 GPU 实测，不能承诺不会 OOM 或提升多少倍。

## 无卡检查

```bash
bash /root/autodl-tmp/graduation-design/run_autodl.sh all --check-only
bash /root/autodl-tmp/graduation-design/run_autodl.sh all --print-command
```

`--check-only` 检查训练/验证配对、全部文件路径和首尾样本解码，不训练，不代表完成 GPU 前反向验证。测试集不参与训练、选权重或调参。
