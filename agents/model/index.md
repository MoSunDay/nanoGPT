Commit: 3adf61e154c3fe3fca428ad6bc3818b27a3b8291

# 模型架构 (model.py)

## 职责

定义完整的 GPT-2 模型：pre-norm Transformer 架构，包含词嵌入、位置编码、多头因果自注意力、前馈网络、权重初始化、预训练权重加载和自回归文本生成。

## 边界

- 仅负责模型的前向推理、权重管理和生成逻辑
- 不涉及训练循环、数据加载、优化器调度或分布式训练编排
- 不负责 token 编码/解码（由调用方处理）

## 关键抽象

### GPTConfig (dataclass, L108)

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `block_size` | 1024 | 最大上下文长度 |
| `vocab_size` | 50304 | 词表大小（GPT-2 为 50257，向上取整到 64 的倍数以提升效率） |
| `n_layer` | 12 | Transformer 层数 |
| `n_head` | 12 | 注意力头数 |
| `n_embd` | 768 | 嵌入维度 |
| `dropout` | 0.0 | Dropout 率 |
| `bias` | True | Linear/LayerNorm 是否使用偏置 |

### 层级结构

```
GPT
 ├── wte: Token Embedding (vocab_size, n_embd)
 ├── wpe: Position Embedding (block_size, n_embd)
 ├── h: ModuleList[Block] × n_layer
 │    └── Block
 │         ├── ln_1: LayerNorm → CausalSelfAttention → residual
 │         └── ln_2: LayerNorm → MLP → residual
 │              └── MLP: Linear → GELU → Linear → Dropout
 ├── ln_f: LayerNorm
 └── lm_head: Linear (与 wte 权重共享)
```

- **Pre-norm**：LayerNorm 在注意力/MLP 之前（非之后）
- **权重共享**：`lm_head.weight = wte.weight`
- **Flash Attention**：自动检测 `torch.nn.functional.scaled_dot_product_attention`，支持时自动启用

### 权重初始化 (_init_weights, L138)

- Linear 和 Embedding：`N(0, 0.02)`
- 残差投影层（`c_proj`）：`N(0, 0.02 / sqrt(2 * n_layer))`，补偿残差路径的方差累积

## 主要流程

### 前向传播 (forward, L140)

`forward(idx, targets=None) → (logits, loss)`

1. 输入 `idx` shape `(B, T)`，经过 `wte` + `wpe` 得到 `(B, T, n_embd)`
2. 依次通过 `n_layer` 个 `Block`，再经过 `ln_f`
3. 仅对最后一个位置计算 logits（推理优化）：`logits = lm_head(x[:, -1, :])` 或全部位置
4. 若提供 `targets`，计算交叉熵损失 `(logits, loss)`；否则 `(logits, None)`

### 预训练加载 (from_pretrained, L171)

`@classmethod from_pretrained(model_type, override_args=None)`

1. 根据字符串映射确定 GPT-2 配置（`gpt2` / `gpt2-medium` / `gpt2-large` / `gpt2-xl`）
2. 创建 `GPT` 实例，从 HuggingFace 加载 `GPT2LMHeadModel` 权重
3. 转换 HuggingFace 的 `Conv1D` 权重为 `Linear`（转置权重矩阵）
4. 支持 `override_args` 覆盖 `dropout` 和 `block_size`（`crop_block_size`）

### 自回归生成 (generate, L269)

`@torch.no_grad() generate(idx, max_new_tokens, temperature=1.0, top_k=None)`

1. 循环 `max_new_tokens` 次
2. 截断输入到 `block_size` 长度
3. 前向传播获取 logits（仅最后一个位置）
4. 除以 `temperature`，可选 `top_k` 过滤
5. 从 softmax 分布中采样，追加到序列

## 关键接口

| 方法 | 签名 | 用途 |
|------|------|------|
| `forward` | `(idx, targets=None) → (logits, loss)` | 前向推理 |
| `from_pretrained` | `(model_type, override_args) → GPT` | 加载 OpenAI GPT-2 权重 |
| `configure_optimizers` | `(weight_decay, lr, betas, device_type) → optimizer` | 创建分组 AdamW |
| `estimate_mfu` | `(fwdbwd_per_iter, dt) → float` | 估算 MFU |
| `generate` | `(idx, max_new_tokens, temperature, top_k) → idx` | 自回归生成 |
| `crop_block_size` | `(block_size)` | 缩小上下文窗口 |
| `get_num_params` | `(non_embedding=True) → int` | 统计参数量 |

## 依赖

- PyTorch (`torch`, `torch.nn`, `torch.nn.functional`)
- `transformers.GPT2LMHeadModel`（仅 `from_pretrained` 使用，可选依赖）
- `math`, `inspect`, `dataclasses`（标准库）

## 相关模块

- [训练与推理管线](../pipeline/index.md)：调用 `GPT` 进行训练、采样和基准测试
