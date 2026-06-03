"""
Week 6 Practical: Pre-training and Scaling Laws
================================================
Objectives:
  - Train a family of MiniGPT models at different sizes
  - Plot loss vs parameter count (scaling law experiment)
  - Estimate FLOPs for each training run
  - Compute and compare perplexity on a held-out set
  - Verify the Chinchilla token-per-parameter relationship
"""

import math
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data (reuse Shakespeare from Week 5 or create inline)
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = Path("shakespeare.txt")
if not DATA_PATH.exists():
    try:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            DATA_PATH
        )
    except Exception:
        DATA_PATH.write_text(("All the world is a stage. " * 1000 +
                               "To be or not to be. " * 1000) * 5)

text = DATA_PATH.read_text(encoding="utf-8")
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data   = data[split_idx:]
print(f"Vocab: {vocab_size}  |  Train: {len(train_data):,}  |  Val: {len(val_data):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Configurable MiniGPT
# ─────────────────────────────────────────────────────────────────────────────
class GPTConfig:
    def __init__(self, n_layers=4, n_heads=4, d_model=128, block_size=128,
                 dropout=0.0, vocab_size=65):
        self.n_layers   = n_layers
        self.n_heads    = n_heads
        self.d_model    = d_model
        self.d_ff       = 4 * d_model
        self.block_size = block_size
        self.dropout    = dropout
        self.vocab_size = vocab_size


class SelfAttn(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.register_buffer("mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            .view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        d_k = C // self.cfg.n_heads
        h = self.cfg.n_heads
        Q, K, V = self.c_attn(x).split(C, dim=2)
        Q = Q.view(B, T, h, d_k).transpose(1, 2)
        K = K.view(B, T, h, d_k).transpose(1, 2)
        V = V.view(B, T, h, d_k).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        out = (w @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = SelfAttn(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ConfigurableGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks  = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.head    = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight
        self.apply(lambda m: nn.init.normal_(m.weight, 0, 0.02)
                   if isinstance(m, (nn.Linear, nn.Embedding)) else None)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size),
                               targets.view(-1)) if targets is not None else None
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# 3. FLOPs estimation
# ─────────────────────────────────────────────────────────────────────────────
def estimate_flops_per_token(cfg: GPTConfig) -> int:
    """Approximate FLOPs for one forward pass per token (6*N rule)."""
    N = cfg.n_layers * (
        # attention: QKV + output projection
        4 * cfg.d_model * cfg.d_model
        # attention scores: 2 * seq * d_model (approx, seq-independent estimate)
        # FFN
        + 2 * cfg.d_model * cfg.d_ff
    )
    return 6 * N   # factor 6 for forward + backward


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training helper
# ─────────────────────────────────────────────────────────────────────────────
def quick_train(cfg: GPTConfig, n_tokens_to_train: int,
                batch_size: int = 32, lr: float = 3e-4) -> float:
    """Train for approximately n_tokens_to_train and return final val loss."""
    model = ConfigurableGPT(cfg).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = max(1, n_tokens_to_train // (batch_size * cfg.block_size))

    model.train()
    for step in range(steps):
        ix = torch.randint(len(train_data) - cfg.block_size, (batch_size,))
        x  = torch.stack([train_data[i: i + cfg.block_size] for i in ix]).to(DEVICE)
        y  = torch.stack([train_data[i + 1: i + cfg.block_size + 1] for i in ix]).to(DEVICE)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Evaluate on val
    model.eval()
    val_losses = []
    for _ in range(20):
        ix = torch.randint(len(val_data) - cfg.block_size, (batch_size,))
        x  = torch.stack([val_data[i: i + cfg.block_size] for i in ix]).to(DEVICE)
        y  = torch.stack([val_data[i + 1: i + cfg.block_size + 1] for i in ix]).to(DEVICE)
        with torch.no_grad():
            _, loss = model(x, y)
        val_losses.append(loss.item())
    return sum(val_losses) / len(val_losses)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Scaling experiment
# ─────────────────────────────────────────────────────────────────────────────
SCALING_CONFIGS = [
    GPTConfig(n_layers=2, n_heads=2, d_model=64,  block_size=64,  vocab_size=vocab_size),
    GPTConfig(n_layers=2, n_heads=4, d_model=128, block_size=64,  vocab_size=vocab_size),
    GPTConfig(n_layers=4, n_heads=4, d_model=128, block_size=128, vocab_size=vocab_size),
    GPTConfig(n_layers=4, n_heads=4, d_model=256, block_size=128, vocab_size=vocab_size),
    GPTConfig(n_layers=6, n_heads=4, d_model=256, block_size=128, vocab_size=vocab_size),
]

TOKENS_PER_RUN = 500_000   # ~30 seconds per config on CPU


# ─────────────────────────────────────────────────────────────────────────────
# 6. Chinchilla token/param analysis
# ─────────────────────────────────────────────────────────────────────────────
def chinchilla_optimal_tokens(n_params: int) -> int:
    """Chinchilla: optimal tokens ≈ 20 × N."""
    return 20 * n_params


def chinchilla_optimal_params(n_tokens: int) -> int:
    """Chinchilla: optimal params ≈ N_tokens / 20."""
    return n_tokens // 20


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Task 1: Parameter count and FLOPs per config ──────────────────────────
    print("=" * 70)
    print("TASK 1: Parameter counts and FLOPs per model configuration")
    print("=" * 70)
    print(f"{'Layers':>6} {'Heads':>5} {'d_model':>7} {'Params':>12} "
          f"{'FLOPs/tok':>12} {'Chinch. tokens':>15}")
    for cfg in SCALING_CONFIGS:
        n   = ConfigurableGPT(cfg).n_params()
        flops = estimate_flops_per_token(cfg)
        chin  = chinchilla_optimal_tokens(n)
        print(f"{cfg.n_layers:>6} {cfg.n_heads:>5} {cfg.d_model:>7} "
              f"{n:>12,} {flops:>12,} {chin:>15,}")

    # ── Task 2: Scaling experiment ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"TASK 2: Scaling experiment ({TOKENS_PER_RUN:,} tokens per model)")
    print("=" * 70)
    results = []
    for i, cfg in enumerate(SCALING_CONFIGS):
        n_params = ConfigurableGPT(cfg).n_params()
        print(f"[{i+1}/{len(SCALING_CONFIGS)}] Training {n_params:,}-param model…", flush=True)
        val_loss = quick_train(cfg, TOKENS_PER_RUN)
        ppl = math.exp(val_loss)
        flops_total = estimate_flops_per_token(cfg) * TOKENS_PER_RUN
        results.append((n_params, val_loss, ppl, flops_total))
        print(f"   → val_loss={val_loss:.4f}  PPL={ppl:.2f}")

    # ── Task 3: Plot scaling law ───────────────────────────────────────────────
    params_list = [r[0] for r in results]
    losses_list = [r[1] for r in results]
    ppls_list   = [r[2] for r in results]
    flops_list  = [r[3] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(params_list, losses_list, "o-", color="steelblue", linewidth=2, markersize=8)
    for n, l in zip(params_list, losses_list):
        axes[0].annotate(f"{n//1000}K", (n, l), textcoords="offset points",
                         xytext=(5, 5), fontsize=8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Parameters (log scale)")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("Scaling Law: Loss vs Parameters")
    axes[0].grid(alpha=0.3)

    axes[1].plot(flops_list, losses_list, "o-", color="darkorange", linewidth=2, markersize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Total FLOPs (log scale)")
    axes[1].set_ylabel("Validation loss")
    axes[1].set_title("Scaling Law: Loss vs Compute")
    axes[1].grid(alpha=0.3)

    plt.suptitle("MiniGPT Scaling Experiment on Shakespeare", fontsize=12)
    plt.tight_layout()
    plt.savefig("week06_scaling_laws.png", dpi=150)
    plt.show()
    print("Saved: week06_scaling_laws.png")

    # ── Task 4: Chinchilla analysis ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TASK 4: Chinchilla analysis")
    print("=" * 70)
    print("Real-world examples vs Chinchilla optimum:")
    examples = [
        ("GPT-3",       175_000_000_000, 300_000_000_000),
        ("Chinchilla",   70_000_000_000, 1_400_000_000_000),
        ("LLaMA-3 8B",   8_000_000_000, 15_000_000_000_000),
    ]
    print(f"{'Model':15} {'Params':>15} {'Tokens':>18} {'Tok/Param':>10} {'Chinch opt tok':>16}")
    for name, params, tokens in examples:
        ratio = tokens / params
        opt   = chinchilla_optimal_tokens(params)
        print(f"{name:15} {params:>15,} {tokens:>18,} {ratio:>10.1f} {opt:>16,}")

    print("\nAll tasks complete.")
