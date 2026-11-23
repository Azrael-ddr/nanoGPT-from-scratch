# nanoGPT-from-scratch

> 从零手写一个字符级 GPT：从 65×65 的 Bigram 表出发，一步步加进注意力机制和多层 Transformer，把预测错误率（val NLL）从 **2.48 降到 1.79**，并可视化注意力头的行为。

## 一句话

语言模型 = 给定前面的字符，预测下一个字符的概率分布。本项目用最少的代码，从「只看当前字符的 bigram」一路走到「完整的 Transformer」，亲手验证每一步为什么有效、为什么无效。

## Loss 阶梯（核心成果）

| 模型 | val NLL | 说明 |
|---|---|---|
| 均匀瞎猜 | 4.17 | ln(65)，无任何信息 |
| Bigram | 2.48 | 只看当前字符，无视上下文 |
| 因果平均 | 2.94 | 抹掉顺序，反而更差 |
| 单头注意力 | 2.46 | 学出来的权重，但力量有限 |
| **完整 Transformer（44K 参数）** | **1.79** | 多头 + FFN + 残差 + Norm 合力 |
| 参考 nanoGPT（10.65M 参数） | 1.72 | 240 倍参数只多 0.07，收益递减 |

## 项目结构

| 文件 | 内容 | 阶段 |
|---|---|---|
| `bigram.py` | numpy 手写 Bigram（65×65 转移表） | Phase 2 |
| `phase3_step1_bigram_torch.py` | torch 重写 Bigram（baseline） | Phase 3 |
| `phase3_step2_causal_avg.py` | 因果平均（反例：为什么顺序重要） | Phase 3 |
| `phase3_step3_attention.py` | 单头自注意力（Q/K/V） | Phase 3 |
| `phase4_gpt.py` | 完整 Transformer（多头 + FFN + 残差 + Norm） | Phase 4 |
| `phase5_attention_viz.py` | 注意力权重可视化 | Phase 5 |
| `项目总结.md` | 完整学习笔记 | 收尾 |

## 技术栈

Python 3.12 · NumPy · PyTorch 2.15 (CUDA 13.4, RTX 4070)

## 快速运行

```bash
# 数据准备（tiny Shakespeare，来自 karpathy/char-rnn）
# 若 data/shakespeare_char/input.txt 不存在，需先下载：
#   https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

pip install -r requirements.txt

python bigram.py                   # Phase 2：bigram 基线，val ≈ 2.48
python phase4_gpt.py               # Phase 4：训练完整 transformer，val ≈ 1.79
python phase5_attention_viz.py     # Phase 5：可视化注意力头（需先跑 phase4 生成 mini_gpt.pt）
```

## 关键收获

1. **Bigram 强，是因为「上一个字符」是主力信号。** 字符级文本里，预测下一个字符时上一个字符贡献了大部分信息。
2. **平均会毁掉顺序。** 因果平均（2.94）比 bigram（2.48）还差——因为 `mean([a,b,c]) == mean([c,b,a])`，抹掉了先后。必须用学出来的权重（注意力）让模型自己决定重视谁。
3. **力量来自堆叠。** 单头注意力 ≈ bigram（2.46），真正让 loss 跳最大的是「多头 + FFN + 残差 + Norm + 深度」一起上。
4. **参数规模收益递减。** 44K 参数到 1.79，10.65M 参数只到 1.72——字符级莎士比亚太简单。

## 参考

本项目受 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 启发，所有脚本均为从零手写，未直接使用其 `model.py` / `train.py`。
