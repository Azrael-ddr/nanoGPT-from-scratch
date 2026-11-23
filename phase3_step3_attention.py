# -*- coding: utf-8 -*-
r"""
phase3_step3_attention.py — Phase 3 第 3 步：真正的注意力（单头）

上一版因果平均（step2）比 bigram 还差（2.94 > 2.48），原因是：
平均是"顺序无关"的 —— mean([a,b,c]) == mean([c,b,a])，
它没法告诉你"最后一个字符是什么"，而"最后一个字符"恰恰是最强预测。

注意力修的就是这个：把"人手写死的等权平均"换成"模型自己学的权重"。
模型可以学会"把大部分权重放在最近的字符上，少部分给更早的"，于是：
既恢复了 bigram 的能力（知道上一个字符是谁），又能用更长的上下文。

单头注意力的完整公式（Phase 1 讲过，这里亲手实现）：
  Q = x @ Wq,  K = x @ Wk,  V = x @ Wv
  scores = Q @ Kᵀ / √d               （点积 = 匹配度）
  weights = softmax(causal(scores))  （因果掩码 + 归一化）
  out = weights @ V                  （加权聚合）

运行： .\.venv\Scripts\python.exe phase3_step3_attention.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 1. 读数据（同前） ----------
text = open('data/shakespeare_char/input.txt', encoding='utf-8').read()
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# ---------- 2. 模型：单头自注意力 ----------
C = 32   # 嵌入维度 = 头维度（单头，不拆分）

class SingleHeadAttention(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)      # 字符 → 向量 (B,T,C)
        self.Wq = nn.Linear(n_embd, n_embd, bias=False)  # 查询：我在找什么
        self.Wk = nn.Linear(n_embd, n_embd, bias=False)  # 键：我有什么标签
        self.Wv = nn.Linear(n_embd, n_embd, bias=False)  # 值：我携带的内容
        self.lm_head = nn.Linear(n_embd, vocab_size)     # 向量 → logits

    def forward(self, idx):
        x = self.emb(idx)                      # (B, T, C)
        B, T, C = x.shape

        q = self.Wq(x)                         # (B, T, C) 查询
        k = self.Wk(x)                         # (B, T, C) 键
        v = self.Wv(x)                         # (B, T, C) 值

        # ① 点积 = 匹配度：Q @ Kᵀ → (B, T, T)，位置 (t,s) = t 想关注 s 的程度
        wei = q @ k.transpose(-2, -1)          # (B, T, T)
        # ② ÷ √d：稳定梯度（d = C = 32）
        wei = wei * (1.0 / (C ** 0.5))
        # ③ 因果掩码：t 只能看 s<=t（把未来设成 -inf）
        tril = torch.tril(torch.ones(T, T))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        # ④ softmax：每行和=1，变成"权重"
        wei = F.softmax(wei, dim=-1)           # (B, T, T)
        # ⑤ 加权聚合：用权重对 V 求和
        out = wei @ v                          # (B, T, C)

        return self.lm_head(out)               # (B, T, V)

model = SingleHeadAttention(V, C)
print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")

# ---------- 3~6. 超参 / get_batch / estimate_loss / 训练循环（同前） ----------
batch_size = 64
block_size = 16
lr = 1e-2
max_steps = 5000  # 单头注意力只到 2.46；见 phase4_gpt.py 继续组装完整 Transformer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + block_size] for i in ix])
    return x, y

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

print("\n对照：bigram ≈ 2.48；因果平均 ≈ 2.94；本步应低于 2.48；GPT ≈ 1.72")
