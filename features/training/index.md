Commit: 3adf61e154c3fe3fca428ad6bc3818b27a3b8291

# 模型训练与微调

## 能力概述

用户可通过命令行训练或微调 GPT 模型，支持从零训练、从检查点恢复、以及基于 GPT-2 预训练权重的微调。

## 触发方式

```bash
python train.py config/train_shakespeare_char.py   # 使用配置文件
python train.py --dataset=openwebtext --n_layer=12  # 命令行覆盖
```

## 行为与规则

### 初始化模式

| `init_from` 值 | 行为 |
|----------------|------|
| `scratch` | 从零创建 GPT，使用 `GPTConfig` 默认值 + 配置覆盖 |
| `resume` | 从 `out_dir/ckpt.pt` 恢复模型、优化器状态、迭代数、最佳验证损失 |
| `gpt2` / `gpt2-medium` / `gpt2-large` / `gpt2-xl` | 从 HuggingFace 加载 OpenAI GPT-2 预训练权重 |

### 训练过程

- 学习率调度：线性 warmup（`warmup_iters` 步）→ cosine decay → `min_lr` 下限
- 梯度累积：每 `gradient_accumulation_steps` 个 micro-batch 执行一次优化器步骤
- 混合精度：自动检测 CUDA bfloat16 / float16；float16 使用 GradScaler
- 梯度裁剪：`grad_clip`（默认 1.0）
- 评估：每 `eval_iters` 批次在 train/val 上平均损失
- 检查点：每 `eval_interval` 步评估，条件满足时保存到 `out_dir/ckpt.pt`
- wandb 日志：可选，通过 `wandb_log=True` 开启

### 检查点策略

- `always_save_checkpoint=True`：每次评估都保存
- `always_save_checkpoint=False`：仅在验证损失改善时保存
- 检查点内容：模型参数、优化器状态、模型配置、迭代数、最佳验证损失

### 分布式训练

- 单卡：直接运行
- 多卡 DDP：`torchrun --standalone --nproc_per_node=N train.py`
- 多节点：设置 `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK` 环境变量
- DDP 后端：`nccl`（默认），仅同步最后一个 micro-step 的梯度

### 数据准备

```bash
python data/shakespeare_char/prepare.py  # 字符级，输出 meta.pkl
python data/shakespeare/prepare.py       # BPE
python data/openwebtext/prepare.py       # BPE，大规模
```

输出 `train.bin` / `val.bin`（uint16 numpy 数组），训练时通过 memory-map 按需读取。字符级数据集额外输出 `meta.pkl`（包含 stoi/itos 映射）。

### 性能基准测试

```bash
python bench.py                           # 简单吞吐量测试
python bench.py --profile=True            # TensorBoard profiling
```

- 默认 10 步 burn-in + 20 步计时，报告每步耗时和 MFU
- `real_data=True` 使用真实数据，`False` 使用随机张量
- `profile=True` 导出 TensorBoard trace 到 `./bench_log`

## 关键状态

| 状态 | 触发条件 | 效果 |
|------|---------|------|
| 训练中 | `eval_only=False` | 正常训练循环 |
| 仅评估 | `eval_only=True` | 评估后退出 |
| 首次迭代 | `iter_num == 0` | 跳过 warmup 内的 MFU 估算 |
| 最佳模型 | `val_loss < best_val_loss` | 更新最佳损失并保存检查点 |

## 错误面

- 数据文件不存在：`train.bin` / `val.bin` 缺失时 memmap 失败
- CUDA OOM：batch_size / block_size 过大
- 检查点不兼容：`resume` 模式下配置与检查点中的 `model_args` 不一致

## 约束

- 不支持动态 batch size 或 sequence length
- 学习率调度参数与 `max_iters` 紧耦合
- 配置系统通过全局变量变异工作，无类型安全

## 相关逻辑模块

- [模型架构](../../agents/model/index.md)
- [训练与推理管线](../../agents/pipeline/index.md)
