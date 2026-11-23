# -*- coding: utf-8 -*-
r"""
phase5_attention_viz.py — Phase 5：注意力可视化（看模型"在看哪里"）

训练一个和 phase4 一样的迷你 GPT，把权重存成 mini_gpt.pt（以后可反复加载），
然后取一段文本，把最后一层 4 个注意力头的权重画成 ASCII 热力图，
亲眼看到每个头各自学会了什么。

看什么：
  · 对角线上亮 → 头在"看自己"
  · 紧挨对角线下面一条亮 → 头在"看上一个字符"（最常见，像 bigram）
  · 最左边一列亮 → 头在"看句子开头"
  · 整片均匀 → 头在"取平均"（像 step2 的因果平均）

ASCII 图例：@=最关注  #=较强  o=一般  .=几乎不关注

运行： .\.venv\Scripts\python.exe phase5_attention_viz.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 1. 数据 ----------
text = open('data/shakespeare_char/input.txt', encoding='utf-8').read()
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# ---------- 2. 超参（同 phase4） ----------
C = 32
nh = 4
hs = C // nh
n_layer = 3
block_size = 64

# ---------- 3. 模型（Head 额外存下注意力权重，供可视化） ----------
class Head(nn.Module):
    def __init__(self, hs):
        super().__init__()
        self.Wq = nn.Linear(C, hs, bias=False)
        self.Wk = nn.Linear(C, hs, bias=False)
        self.Wv = nn.Linear(C, hs, bias=False)
        self.attn_weights = None   # 存最近一次 forward 的注意力权重

    def forward(self, x):
        T = x.shape[1]
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        wei = q @ k.transpose(-2, -1) * (hs ** -0.5)
        wei = wei.masked_fill(torch.tril(torch.ones(T, T)) == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        self.attn_weights = wei.detach()   # 保存 (B,T,T)
        return wei @ v

class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head(hs) for _ in range(nh)])
        self.proj = nn.Linear(C, C, bias=False)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(C, 4 * C)
        self.proj = nn.Linear(4 * C, C)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = MultiHead()
        self.ln2 = nn.LayerNorm(C)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, C)
        self.pos_emb = nn.Embedding(block_size, C)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, V)

    def forward(self, idx):
        T = idx.shape[1]
        pos = torch.arange(T)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

model = GPT()
print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")

# ---------- 4. 训练（同 phase4） ----------
batch_size = 64
lr = 3e-3
max_steps = 5000
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + block_size] for i in ix])
    return x, y

for step in range(max_steps):
    x, y = get_batch('train')
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 1000 == 0 or step == max_steps - 1:
        print(f"step {step:4d}: loss {loss.item():.4f}")

# 存权重，以后可 torch.load 直接复用
torch.save(model.state_dict(), 'mini_gpt.pt')
print("\n已保存权重到 mini_gpt.pt")

# ---------- 5. 可视化：取前 24 个字符，看最后一层 4 个头 ----------
model.eval()
idx = data[:24].unsqueeze(0)          # (1, 24)
with torch.no_grad():
    model(idx)

T = 24
disp = [('⏎' if c == '\n' else ('·' if c == ' ' else c)) for c in (itos[i] for i in idx[0].tolist())]
print("\n=== 示例文本（前 24 字符）===")
print("位置: " + " ".join(f"{i:2d}" for i in range(T)))
print("字符: " + " ".join(f"{c:>2}" for c in disp))

last_layer_heads = model.blocks[-1].attn.heads
for h, head in enumerate(last_layer_heads):
    wei = head.attn_weights[0]        # (24, 24)
    print(f"\n--- 第 3 层 · 头 {h} ---")
    for r in range(T):
        line = ''.join('@' if p > 0.3 else '#' if p > 0.1 else 'o' if p > 0.04 else '.' for p in wei[r])
        print(f"{r:2d} {line}")

# ---------- 6. 量化总结：每个头"平均把权重放哪" ----------
print("\n=== 每个头关注点量化 ===")
print("（看自己 / 看上一个字符 / 看≥4步前 的平均权重；示例文本=First Citizen: 开头 24 字符）")
t = torch.arange(T)
s = torch.arange(T)
dist = t[:, None] - s[None, :]        # dist[t, s] = t - s
for h, head in enumerate(last_layer_heads):
    wei = head.attn_weights[0]
    self_w = wei.diagonal().mean().item()
    prev_w = wei.diagonal(-1).mean().item()
    far_w = wei[dist >= 4].mean().item()
    print(f"头 {h}: 看自己 {self_w:.3f} | 看上一位 {prev_w:.3f} | 看远处(≥4) {far_w:.3f}")
