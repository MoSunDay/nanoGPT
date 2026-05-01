Commit: 3adf61e154c3fe3fca428ad6bc3818b27a3b8291

# 训练与推理管线

## 职责

涵盖模型训练、文本采样、性能基准测试、配置管理和数据预处理。将 [模型架构](../model/index.md) 封装为完整的端到端工作流。

## 组成

| 文件 | 角色 |
|------|------|
| `train.py` | 主训练循环：单卡/多卡 DDP/多节点训练、检查点、评估、学习率调度、wandb 日志 |
| `sample.py` | 文本生成：从检查点或 GPT-2 预训练权重加载模型，自回归采样 |
| `bench.py` | 基准测试：测量训练吞吐量和 MFU，支持 TensorBoard profiling |
| `configurator.py` | 配置覆盖机制：通过 `exec()` 将配置文件和命令行参数注入调用方全局变量 |
| `config/` | 预置配置脚本（训练、微调、评估） |
| `data/` | 数据集预处理脚本（下载 + tokenize，输出 train.bin / val.bin） |

## 关键流程

### 训练 (train.py)

**入口**：`python train.py [config_file] [--key=value...]`

1. 定义默认配置为模块级全局变量（~40 个参数覆盖 I/O、模型、优化器、学习率、DDP、设备）
2. `exec(open('configurator.py').read())` 应用配置覆盖
3. DDP 初始化（多卡时设置进程组、设备、Rank）
4. 模型初始化（4 种模式）：
   - `scratch`：从零创建 GPT，`vocab_size` 从 `meta.pkl` 或默认 50304
   - `resume`：从 `out_dir/ckpt.pt` 恢复模型、优化器、迭代数、最佳验证损失
   - `gpt2` / `gpt2-medium` / `gpt2-large` / `gpt2-xl`：通过 `GPT.from_pretrained()` 加载
5. 可选 `torch.compile()` 编译模型
6. 训练循环（L255-333）：
   - 学习率：线性 warmup → cosine decay → min_lr 下限
   - 梯度累积：`gradient_accumulation_steps` 个 micro-batch 后更新一次
   - DDP：仅在最后一个 micro-step 同步梯度
   - 混合精度：自动检测 bfloat16 / float16，使用 GradScaler（float16）
   - 定期评估（`eval_interval`），保存最佳或始终保存检查点
   - 追踪 MFU（5 步 warmup 后）和每步耗时
   - wandb 可选日志

**数据加载** (get_batch, L116)：numpy memory-map 读取 `data/<dataset>/train.bin` 或 `val.bin`（uint16），转换为 int64 tensor。支持 CUDA 异步传输（pin_memory）。

**检查点格式** (`out_dir/ckpt.pt`)：`{'model', 'optimizer', 'model_args', 'iter_num', 'best_val_loss', 'config'}`

### 文本采样 (sample.py)

**入口**：`python sample.py [--key=value...]`

1. 加载模型（从 `out_dir/ckpt.pt` 或 GPT-2 预训练权重）
2. 解析编码方式：
   - 检查 checkpoint 中的 `meta.pkl`（字符级编码）→ 使用 `meta['stoi']` / `meta['itos']`
   - 否则使用 tiktoken GPT-2 BPE
3. 编码起始提示（`--start=FILE:path` 从文件读取）
4. 调用 `model.generate()` 生成 N 个样本
5. 解码并打印

### 基准测试 (bench.py)

**入口**：`python bench.py [--key=value...]`

两种模式：
- **简单基准**（默认）：10 步 burn-in + 20 步计时，报告每步耗时和 MFU
- **Profiling**（`profile=True`）：使用 `torch.profiler` 导出 TensorBoard trace 到 `./bench_log`

数据源：`real_data=True` 加载 OpenWebText（同 train.py 的 memmap 方式），`False` 使用随机张量。

### 配置系统 (configurator.py)

通过 `exec(open('configurator.py').read())` 在调用方的全局命名空间中执行：

- `sys.argv` 中无 `=` 的参数 → 视为配置文件路径，`exec()` 执行
- 有 `=` 的参数 → 解析为 `--key=value`，通过 `literal_eval()` 类型推断后覆盖对应全局变量
- 不作为模块导入，仅通过 `exec()` 修改调用方状态

### 数据预处理 (data/*/prepare.py)

独立运行的离线脚本，每个数据集一个目录：

| 数据集 | tokenizer | 输出 |
|--------|-----------|------|
| `shakespeare_char` | 字符级（65 词表） | train.bin, val.bin, meta.pkl |
| `shakespeare` | tiktoken GPT-2 BPE | train.bin, val.bin |
| `openwebtext` | tiktoken GPT-2 BPE | train.bin, val.bin |

输出格式：numpy uint16 数组（token ID），通过 memory-map 在训练时读取。

## 关键依赖

- [模型架构](../model/index.md)：`GPTConfig`, `GPT`
- `torch`, `numpy`, `tiktoken`
- `torch.distributed`（DDP 训练）
- `transformers`（预训练权重加载，间接通过 model.py）
- `wandb`（可选日志）
- `datasets`（OpenWebText 下载，仅 openwebtext/prepare.py）
