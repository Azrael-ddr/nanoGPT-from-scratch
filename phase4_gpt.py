# -*- coding: utf-8 -*-
r"""
phase4_gpt.py — Phase 4：组装完整的 Transformer（迷你版 GPT）

Step 3 的单头注意力只把 val 从 2.48 压到 2.46——提升很小。
原因：一个注意力头只是"一次混合操作"，缺了 Transformer 的另外三大件：
  1. 多头（多个头各看不同模式）
  2. FFN / MLP（每个位置独立地"思考"，放大 4 倍再缩回）
  3. 残差 + LayerNorm（Pre-Norm，梯度高速路）+ 堆叠多层（深度）

这个脚本把 Phase 1 里读过的 model.py 的 Block/GPT 结构，
从零完整重拼一遍（小尺寸），看 loss 真正往下跳。

对应 model.py：
  Head        → CausalSelfAttention 的单头
  MultiHead   → CausalSelfAttention 的多头 + 输出投影
  MLP         → MLP
  Block       → Block（残差 + Pre-Norm）
  GPT         → GPT.forward（tok_emb + pos_emb → blocks → ln_f → lm_head）

运行： .\.venv\Scripts\python.exe phase4_gpt.py
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

# ---------- 2. 超参 ----------
C = 32          # 嵌入维度（model.py 里 n_embd=384，这里取小）
nh = 4          # 头数（model.py 里 n_head=6）
hs = C // nh    # 每头维度 = 8
n_layer = 3     # 堆叠层数（model.py 里 n_layer=6）
block_size = 64 # 上下文长度（model.py 里 256）

# ---------- 3. 模型组件 ----------
class Head(nn.Module):
    """单头：Q/K/V 各一套投影，输出 hs 维"""
    def __init__(self, hs):
        super().__init__()
        self.Wq = nn.Linear(C, hs, bias=False)
        self.Wk = nn.Linear(C, hs, bias=False)
        self.Wv = nn.Linear(C, hs, bias=False)

    def forward(self, x):
        T = x.shape[1]
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)      # (B,T,hs)
        wei = q @ k.transpose(-2, -1) * (hs ** -0.5)      # ① QKᵀ ② ÷√d
        wei = wei.masked_fill(torch.tril(torch.ones(T, T)) == 0, float('-inf'))  # ③ 因果
        wei = F.softmax(wei, dim=-1)                      # ④ softmax
        return wei @ v                                     # ⑤ 加权聚合

class MultiHead(nn.Module):
    """多头：nh 个头并行，各自在 hs 维子空间算注意力，再拼起来投影回 C"""
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head(hs) for _ in range(nh)])
        self.proj = nn.Linear(C, C, bias=False)   # 输出投影（model.py 的 c_proj）

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B,T,nh*hs)=(B,T,C)
        return self.proj(out)

class MLP(nn.Module):
    """FFN：放大 4 倍 → GELU 非线性 → 缩回（model.py 的 MLP）"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(C, 4 * C)     # c_fc：384→1536 的迷你版
        self.proj = nn.Linear(4 * C, C)   # c_proj

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))

class Block(nn.Module):
    """残差 + Pre-Norm（先 LayerNorm 再子层，最后 x + f(x)）"""
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = MultiHead()
        self.ln2 = nn.LayerNorm(C)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # 残差：x + Attention(Norm(x))
        x = x + self.mlp(self.ln2(x))    # 残差：x + MLP(Norm(x))
        return x

class GPT(nn.Module):
    """整体：tok_emb + pos_emb → n_layer 个 Block → ln_f → lm_head"""
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, C)          # 字符嵌入
        self.pos_emb = nn.Embedding(block_size, C) # 位置嵌入
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(C)                # 最后一层归一化
        self.lm_head = nn.Linear(C, V)             # 向量 → logits

    def forward(self, idx):
        T = idx.shape[1]
        pos = torch.arange(T)                       # 位置编号 0..T-1
        x = self.tok_emb(idx) + self.pos_emb(pos)   # (B,T,C)
        x = self.blocks(x)                          # 过 n_layer 个 Block
        x = self.ln_f(x)
        return self.lm_head(x)                      # (B,T,V)

model = GPT()
print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")

# ---------- 4. 训练 ----------
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

# ---------- 5. 生成一段文本（和 bigram.py 的生成对比） ----------
@torch.no_grad()
def generate(prompt, length=300):
    model.eval()
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
    for _ in range(length):
        idx_cond = idx[:, -block_size:]              # 只保留最后 block_size 个
        logits = model(idx_cond)[:, -1, :]           # 取最后一个位置的 logits
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1) # 按概率采样
        idx = torch.cat([idx, nxt], dim=1)
    return ''.join(itos[i] for i in idx[0].tolist())

print("\n--- 生成示例 ---")
print(generate("O Romeo", length=300))

print("\n对照：bigram ≈ 2.48；单头注意力 ≈ 2.46；GPT ≈ 1.72")
