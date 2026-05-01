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

## 模型 0→1 关键步骤与技术点

从零构建 GPT 模型的关键路径（详见 [agents/model/index.md](agents/model/index.md)）：

### 步骤 1：输入表示 — Token + Position Embedding

- **Token Embedding** (`wte`)：将离散 token ID 映射为 `n_embd` 维连续向量，shape `(vocab_size, n_embd)`
- **Position Embedding** (`wpe`)：为每个位置编码绝对位置信息，shape `(block_size, n_embd)`
- 两者相加后过 Dropout，得到 `(B, T, n_embd)` 的输入表示
- **技术决策**：`vocab_size=50304`（而非 GPT-2 原始的 50257），向上取整到 64 的倍数，使 GPU 计算（矩阵乘法）对齐更高效

### 步骤 2：Transformer Block — 核心计算单元

每个 Block 由两个子层组成，均采用 **Pre-Norm + Residual** 模式：

```
x = x + Attention(LayerNorm(x))    # 子层1：自注意力
x = x + MLP(LayerNorm(x))          # 子层2：前馈网络
```

**关键技术点**：

- **Pre-Norm**（非 Post-Norm）：LayerNorm 放在注意力/MLP 之前。训练更稳定，梯度流动更平滑，是 GPT-2 论文的选择
- **残差连接**：每个子层输出直接加回输入，避免深层网络梯度消失

### 步骤 3：Multi-Head Causal Self-Attention

这是 Transformer 的核心机制：

1. **QKV 投影**：用单个 Linear 层一次算出 Q/K/V（`c_attn: 3*n_embd`），再 split 拆分
2. **多头切分**：将 `n_embd` 维拆成 `n_head` 个 `n_embd//n_head` 维的头，并行计算注意力
3. **因果掩码**（Causal Mask）：下三角矩阵确保只能看到当前及之前的 token，防止信息泄露
4. **缩放点积注意力**：`softmax(QK^T / sqrt(d_k)) V`
5. **输出投影**：多头结果拼接后过 `c_proj` 映射回 `n_embd` 维

**性能优化 — Flash Attention**（`model.py:45`）：
- 自动检测 `torch.nn.functional.scaled_dot_product_attention`（PyTorch ≥ 2.0）
- 启用时利用 CUDA 内核，避免显式物化 `(T, T)` 注意力矩阵，节省显存 + 加速计算
- 不支持时 fallback 到手动实现（上三角填 `-inf` 再 softmax）

### 步骤 4：MLP（前馈网络）

```
Linear(n_embd → 4*n_embd) → GELU → Linear(4*n_embd → n_embd) → Dropout
```

- **4 倍扩展**：先升维到 4 倍（`n_embd → 4*n_embd`），提供足够的非线性表达能力，再投影回来
- **GELU 激活**：比 ReLU 更平滑，是 GPT-2/BERT 的标准选择
- **Dropout 正则化**：防止过拟合

### 步骤 5：输出层 — Final LayerNorm + LM Head

- **Final LayerNorm** (`ln_f`)：在所有 Block 之后做一次 LayerNorm，稳定输出分布
- **LM Head**：`Linear(n_embd, vocab_size)` 将隐藏状态映射回词表大小的 logits
- **权重共享**（Weight Tying，`model.py:138`）：`lm_head.weight = wte.weight`，减少参数量并提升泛化

**推理优化**（`model.py:190`）：推理时 `targets=None`，仅对最后一个位置计算 logits（`x[:, [-1], :]`），避免无用的全序列计算

### 步骤 6：权重初始化策略

```
默认：Linear / Embedding → N(0, 0.02)
特殊：残差投影层 c_proj → N(0, 0.02 / sqrt(2 * n_layer))
```

- **缩放初始化**：残差路径上的 `c_proj` 使用 `1/sqrt(2*n_layer)` 的额外缩放，补偿每层残差分支对方差的累积效应（GPT-2 论文方法），确保深层网络前向/反向信号不爆炸或消失

### 步骤 7：优化器配置 — 分组 AdamW

`configure_optimizers`（`model.py:263`）：

- **分组权重衰减**：2D 参数（权重矩阵、Embedding）施加 weight_decay；1D 参数（bias、LayerNorm）不施加
- **Fused AdamW**：CUDA 上自动使用融合内核版本，减少 kernel launch 开销

### 步骤 8：自回归生成

`generate`（`model.py:306`）：

1. 循环生成，每步截断输入到 `block_size`
2. 前向传播 → 取最后位置 logits → temperature 缩放
3. 可选 top-k 过滤（截断低概率词）
4. softmax → multinomial 采样 → 追加到序列

**技术要点**：`@torch.no_grad()` 装饰器关闭梯度计算；top-k 避免采样到极低概率 token

### 架构总览

```
idx (B, T)
  │
  ├── wte(idx) ──→ tok_emb (B, T, n_embd)
  ├── wpe(pos) ──→ pos_emb (T, n_embd)
  │
  └── tok_emb + pos_emb → Dropout
        │
        ▼
   ┌─ Block × n_layer ──────────────┐
   │  x = x + Attn(LN(x))          │
   │  x = x + MLP(LN(x))           │
   └────────────────────────────────┘
        │
        ▼
     LN_f → lm_head → logits (B, T, vocab_size)
                        │
                        ├── 训练: cross_entropy(logits, targets) → loss
                        └── 推理: logits[:, -1, :] (仅最后位置)
```

## 业务能力

参见 [features/index.md](features/index.md)
