"""
Week 5 Practical: Transformer Architecture
===========================================
Objectives:
  - Implement a complete decoder-only Transformer in PyTorch
  - Train it on a character-level text dataset (Shakespeare / custom)
  - Plot training/validation loss curves
  - Generate text using greedy and temperature sampling
  - Inspect model parameter counts by component
"""

from __future__ import annotations

import math
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 0. Hyper-parameters — keep small so this runs on CPU
# ─────────────────────────────────────────────────────────────────────────────
BLOCK_SIZE   = 128     # context window (tokens)
BATCH_SIZE   = 32
N_LAYERS     = 4
N_HEADS      = 4
D_MODEL      = 128
D_FF         = 512
DROPOUT      = 0.1
LEARNING_RATE = 3e-4
MAX_ITERS    = 3000
EVAL_INTERVAL = 300
EVAL_ITERS   = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data — download Shakespeare if not present
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = Path("shakespeare.txt")

def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("Downloading Shakespeare dataset…")
    urllib.request.urlretrieve(url, DATA_PATH)
    print(f"Saved to {DATA_PATH}")

if not DATA_PATH.exists():
    try:
        download_shakespeare()
    except Exception as e:
        print(f"Download failed ({e}). Creating a tiny synthetic dataset.")
        DATA_PATH.write_text(
            "To be or not to be that is the question. " * 500 +
            "All the world is a stage and all the men and women merely players. " * 500
        )

text = DATA_PATH.read_text(encoding="utf-8")
print(f"Dataset: {len(text):,} characters")

# Character-level tokeniser
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
print(f"Vocabulary size: {vocab_size}")
print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data loader
# ─────────────────────────────────────────────────────────────────────────────
def get_batch(split_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    d = train_data if split_name == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i: i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1: i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model components
# ─────────────────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.c_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.attn_drop = nn.Dropout(DROPOUT)
        self.resid_drop = nn.Dropout(DROPOUT)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        Q, K, V = self.c_attn(x).split(D_MODEL, dim=2)
        d_k = C // N_HEADS
        Q = Q.view(B, T, N_HEADS, d_k).transpose(1, 2)
        K = K.view(B, T, N_HEADS, d_k).transpose(1, 2)
        V = V.view(B, T, N_HEADS, d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.GELU(),
            nn.Linear(D_FF, D_MODEL),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ffn  = FeedForward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb   = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.drop      = nn.Dropout(DROPOUT)
        self.blocks    = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYERS)])
        self.ln_f      = nn.LayerNorm(D_MODEL)
        self.head      = nn.Linear(D_MODEL, vocab_size, bias=False)
        # Weight tying
        self.token_emb.weight = self.head.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None) -> tuple:
        B, T = idx.shape
        tok  = self.token_emb(idx)
        pos  = self.pos_emb(torch.arange(T, device=DEVICE))
        x    = self.drop(tok + pos)
        x    = self.blocks(x)
        x    = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ─────────────────────────────────────────────────────────────────────────────
# 4. Parameter count
# ─────────────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameter counts:")
    print(f"  Total      : {total:,}")
    print(f"  Trainable  : {trainable:,}")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:15s}: {n:10,}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Training loop
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(model: nn.Module) -> dict:
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        ls = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            ls.append(loss.item())
        losses[split] = sum(ls) / len(ls)
    model.train()
    return losses


def train(model: nn.Module) -> tuple[list, list]:
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    train_losses, val_losses = [], []
    t0 = time.time()

    for it in range(1, MAX_ITERS + 1):
        if it % EVAL_INTERVAL == 0 or it == 1:
            losses = estimate_loss(model)
            elapsed = time.time() - t0
            print(f"  iter {it:5d} | train={losses['train']:.4f} | "
                  f"val={losses['val']:.4f} | {elapsed:.1f}s")
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

    return train_losses, val_losses


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    model = MiniGPT().to(DEVICE)
    count_parameters(model)

    print(f"\nTraining MiniGPT for {MAX_ITERS} iterations…")
    train_losses, val_losses = train(model)

    # ── Loss curve ────────────────────────────────────────────────────────────
    iters_logged = list(range(EVAL_INTERVAL, MAX_ITERS + 1, EVAL_INTERVAL))
    if len(iters_logged) > len(train_losses):
        iters_logged = iters_logged[:len(train_losses)]

    plt.figure(figsize=(8, 4))
    plt.plot(iters_logged, train_losses, label="Train loss")
    plt.plot(iters_logged, val_losses, label="Val loss")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy loss")
    plt.title("MiniGPT Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("week05_training_curve.png", dpi=150)
    plt.show()
    print("Saved: week05_training_curve.png")

    # ── Text generation ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Text generation")
    print("=" * 60)
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

    for temp in [0.5, 1.0, 1.5]:
        generated = model.generate(context.clone(), max_new_tokens=200, temperature=temp)
        text_out = decode(generated[0].tolist())
        print(f"\nTemperature={temp}:\n{text_out[:300]}")

    print("\nAll tasks complete.")
