# -*- coding: utf-8 -*-
r"""
bigram.py — Phase 2：从零手写 Bigram（二元）语言模型
只用 numpy，不用任何神经网络。核心思想：
  预测下一个字符时，只看"当前这个字符"，完全无视更早的历史。
  （这就是"一阶马尔可夫"：P(下一个 | 当前)）
运行： .\venv\Scripts\python.exe bigram.py
"""
import numpy as np

# ---------- 1. 读数据 + 建字符表 ----------
text = open('data/shakespeare_char/input.txt', encoding='utf-8').read()
chars = sorted(set(text))                        # 去重排序 → 65 个字符
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}       # 字符 → 编号
itos = {i: c for i, c in enumerate(chars)}       # 编号 → 字符
print(f"字符表大小 V = {V}")
data = [stoi[c] for c in text]                   # 全文 → 编号列表
print(f"总字符数 = {len(data)}")

# ---------- 2. 切 train/val（和 prepare.py 一致：前 90% / 后 10%） ----------
n = int(0.9 * len(data))
train, val = data[:n], data[n:]

# ---------- 3. 统计二元组：N[i][j] = 字符 i 后面跟字符 j 的次数 ----------
N = np.zeros((V, V), dtype=np.int64)
for i in range(len(train) - 1):
    N[train[i], train[i + 1]] += 1
N = N + 1            # 拉普拉斯平滑 +1：防止某对从没出现过 → log(0) 爆炸

# ---------- 4. 归一化成概率：P[i][j] = P(下一个=j | 当前=i)，每行和=1 ----------
P = N / N.sum(axis=1, keepdims=True)

# ---------- 5. 看看模型真实学到的规律 ----------
for ch in [' ', 'e', 'q', '.']:
    i = stoi[ch]
    top = P[i].argsort()[::-1][:5]
    print(f"  '{ch}' 之后最可能跟: " + ", ".join(f"'{itos[t]}'({P[i, t]:.2f})" for t in top))

# ---------- 6. 算损失：NLL = -log P(正确字符) 的平均 ----------
def nll(seq):
    seq = np.asarray(seq)
    return -np.log(P[seq[:-1], seq[1:]]).mean()

print(f"\ntrain NLL = {nll(train):.4f},  val NLL = {nll(val):.4f}")
print(f"  对比：均匀瞎猜 = {np.log(V):.4f},  nanoGPT(GPT) val ≈ 1.72")

# ---------- 7. 生成 200 个字符 ----------
rng = np.random.default_rng(1337)
out = [rng.integers(V)]
for _ in range(200):
    out.append(rng.choice(V, p=P[out[-1]]))
print(''.join(itos[i] for i in out))
