Commit: 3adf61e154c3fe3fca428ad6bc3818b27a3b8291

# 文本生成

## 能力概述

用户可从训练好的模型或 GPT-2 预训练权重生成文本，支持温度采样和 top-k 过滤。

## 触发方式

```bash
python sample.py --out_dir=out --num_samples=5 --max_new_tokens=500
python sample.py --init_from=gpt2 --start="Hello world"
python sample.py --start=FILE:prompt.txt
```

## 行为与规则

### 模型加载

| `init_from` | 来源 |
|-------------|------|
| `resume`（默认） | 从 `out_dir/ckpt.pt` 加载 |
| `gpt2` / `gpt2-medium` / `gpt2-large` / `gpt2-xl` | 从 HuggingFace 加载预训练权重 |

加载后可选 `torch.compile()` 编译。

### 编码解析

1. 若 checkpoint 配置中有 `dataset` 字段且 `data/<dataset>/meta.pkl` 存在 → 字符级编码（使用 `meta['stoi']` / `meta['itos']`）
2. 否则 → tiktoken GPT-2 BPE 编码

### 起始提示

- 直接字符串：`--start="Once upon a time"`
- 从文件读取：`--start=FILE:prompt.txt`
- 默认：换行符 `\n`

### 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_samples` | 10 | 生成样本数量 |
| `max_new_tokens` | 500 | 每个样本的最大 token 数 |
| `temperature` | 0.8 | softmax 温度（越低越确定性） |
| `top_k` | 200 | top-k 过滤（None 表示不过滤） |
| `seed` | 1337 | 随机种子 |

### 生成过程

对每个样本：在 `torch.no_grad()` 和 autocast 上下文中调用 `model.generate()`，自回归地逐 token 生成，解码后输出到 stdout。

## 关键状态

- Compiled 模型检查点中的 state_dict 键名带有 `_orig_mod.` 前缀，加载时自动去除

## 约束

- 生成上下文受模型 `block_size` 限制，超出部分自动截断
- 不支持 nucleus sampling (top-p) 或 beam search
- 不支持流式输出（一次性生成完整样本后打印）

## 相关逻辑模块

- [模型架构](../../agents/model/index.md)（`GPT.generate()`）
- [训练与推理管线](../../agents/pipeline/index.md)（sample.py 流程）
