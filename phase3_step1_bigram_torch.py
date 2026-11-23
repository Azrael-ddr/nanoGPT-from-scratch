# -*- coding: utf-8 -*-
r"""
phase3_step1_bigram_torch.py — Phase 3 第 1 步：用 torch 重写 Bigram（baseline）

目标：
  1. 熟悉 torch 训练套路（Embedding / AdamW / 训练循环 / cross_entropy）
  2. 在 torch 里复现 Bigram 的 val NLL ≈ 2.48，作为后续加注意力的对照基线

核心洞见：
  numpy 版 bigram 的模型是"65×65 转移表"；
  torch 版的等价物是  nn.Embedding(65, 65)  ——
  第 i 行的 65 个数，就是"当前字符=i 时，下一个字符的 65 个 logits"。
  参数量 65×65 = 4225，和 numpy 版那张表完全一样。

运行： .\.venv\Scripts\python.exe phase3_step1_bigram_torch.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 1. 读数据 + 建字符表（和 bigram.py 一致） ----------
text = open('data/shakespeare_char/input.txt', encoding='utf-8').read()
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
print(f"字符表大小 V = {V}")

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"train {len(train_data):,} / val {len(val_data):,} 个字符")

# ---------- 2. 模型：Bigram = 一张 Embedding 表 ----------
class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 第 i 行 = "当前字符 i 之后下一个字符" 的 65 个 logits
        self.emb = nn.Embedding(vocab_size, vocab_size)   # (65, 65)

    def forward(self, idx):              # idx: (B, T) 字符编号
        return self.emb(idx)             # -> (B, T, 65) logits

model = Bigram(V)
print(f"参数量：{sum(p.numel() for p in model.parameters()):,}（= 65×65 转移表）")

# ---------- 3. 超参 + 优化器 ----------
batch_size = 64
block_size = 16      # 每个样本看 16 个字符（bigram 只用当前1个，但这为注意力铺路）
lr = 1e-2
max_steps = 5000

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# ---------- 4. 取 batch：x/y 错位 1 格（"预测下一个字符"） ----------
def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size]      for i in ix])   # (64, 16) 输入
    y = torch.stack([d[i + 1:i + 1 + block_size] for i in ix]) # (64, 16) 目标=右移1格
    return x, y

# ---------- 5. 评估（不开梯度，节省显存/时间） ----------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        for _ in range(50):
            x, y = get_batch(split)
            logits = model(x)                            # (64, 16, 65)
            loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# ---------- 6. 训练循环：forward → backward → update ----------
for step in range(max_steps):
    x, y = get_batch('train')
    logits = model(x)                                    # forward：算 logits
    loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
    optimizer.zero_grad()                                # 清空上一轮梯度
    loss.backward()                                      # backward：算梯度
    optimizer.step()                                     # update：沿梯度反向更新权重
    if step % 300 == 0 or step == max_steps - 1:
        e = estimate_loss()
        print(f"step {step:4d}: train {e['train']:.4f}, val {e['val']:.4f}")

print("\n对照：numpy bigram val ≈ 2.48；均匀瞎猜 = 4.17；GPT ≈ 1.72")
