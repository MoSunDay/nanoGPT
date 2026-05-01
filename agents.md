Commit: 3adf61e154c3fe3fca428ad6bc3818b27a3b8291

# nanoGPT 逻辑结构

## 概述

nanoGPT 是一个用于训练和微调中等规模 GPT 模型的最小化仓库。基于 PyTorch 实现了完整的 pre-norm Transformer（GPT-2 架构），支持单卡、多卡 DDP 和多节点训练。项目已归档，推荐迁移至 nanochat。

## 模块索引

| 模块 | 核心文件 | 职责 |
|------|---------|------|
| [模型架构](agents/model/index.md) | `model.py` | GPT 模型定义：Transformer 块、注意力机制、权重初始化、预训练加载、自回归生成 |
| [训练与推理管线](agents/pipeline/index.md) | `train.py`, `sample.py`, `bench.py`, `configurator.py`, `config/`, `data/` | 训练循环、文本采样、基准测试、配置覆盖、数据预处理 |

## 依赖关系

```
configurator.py (exec 注入全局变量)
    ↑           ↑           ↑
train.py    sample.py    bench.py
    |           |           |
    └─────── model.py ──────┘
              (GPTConfig, GPT)
                  |
          transformers (可选，from_pretrained)
```

- `configurator.py` 不作为模块导入，而是通过 `exec()` 修改调用方的全局变量
- `train.py` / `sample.py` / `bench.py` 均依赖 `model.py` 导出 `GPTConfig` 和 `GPT`
- 数据预处理脚本 `data/*/prepare.py` 独立运行，生成 `train.bin` / `val.bin`

## 业务能力

参见 [features/index.md](features/index.md)
