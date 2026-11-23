# -*- coding: utf-8 -*-
r"""
phase3_step2_causal_avg.py — Phase 3 第 2 步：朴素上下文（因果平均）

上一版 bigram 只用"当前这 1 个字符"预测下一个。
这一版给它开个"小窗"：位置 t 预测时，把 0..t 所有字符的向量取平均，
用这个平均向量来预测下一个字符。

关键洞见：
  wei @ x  就是"加权聚合"的形状 —— 这正是注意力要做的事！
  区别只是：这里的 wei 是"人手写死"的等权平均（第 t 行 = 1/(t+1)）；
  Step 3 会把 wei 换成"模型自己学出来的权重"（Q·Kᵀ → softmax）。

预期：val NLL 从 2.48 明显下降（≈ 2.2 附近）——这就是"看到上下文"的威力。

运行： .\.venv\Scripts\python.exe phase3_step2_causal_avg.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 1. 读数据（同 step1） ----------
text = open('data/shakespeare_char/input.txt', encoding='utf-8').read()
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# ---------- 2. 模型：embed → 因果平均 → linear ----------
C = 32   # 每个字符的向量维度（真 nanoGPT 用 384，这里取小跑得快）

class CausalAvg(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)   # (65, 32)：字符 → 向量
        self.lm_head = nn.Linear(n_embd, vocab_size)  # (32, 65)：向量 → logits

    def forward(self, idx):
        x = self.emb(idx)                       # (B, T, 32)
        B, T, C = x.shape
        # 因果平均权重：wei[t, s] = 1/(t+1)  当 s <= t；否则 0
        wei = torch.tril(torch.ones(T, T))            # (T, T) 下三角全是 1
        wei = wei / wei.sum(dim=1, keepdim=True)      # 每行归一化 = 平均
        xavg = wei @ x                          # (B, T, 32)：xavg[t] = mean(x[0..t])
        return self.lm_head(xavg)               # (B, T, 65)

model = CausalAvg(V, C)
print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")

# ---------- 3. 超参 + 优化器（同 step1） ----------
batch_size = 64
block_size = 16
lr = 1e-2
max_steps = 5000
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# ---------- 4. get_batch（同 step1） ----------
def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + block_size] for i in ix])
    return x, y

# ---------- 5. estimate_loss（同 step1） ----------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        for _ in range(50):
            x, y = get_batch(split)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# ---------- 6. 训练循环 ----------
for step in range(max_steps):
    x, y = get_batch('train')
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0 or step == max_steps - 1:
        e = estimate_loss()
        print(f"step {step:4d}: train {e['train']:.4f}, val {e['val']:.4f}")

print("\n对照：bigram val ≈ 2.48；本步应明显更低；GPT ≈ 1.72")
