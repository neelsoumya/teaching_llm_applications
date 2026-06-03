"""
Week 4 Practical: The Attention Mechanism
==========================================
Objectives:
  - Implement scaled dot-product attention from scratch in NumPy
  - Implement causal (masked) self-attention in PyTorch
  - Implement multi-head attention in PyTorch
  - Extract and visualise real attention weights from a pretrained model
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("transformers not installed – skipping HuggingFace attention section")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Scaled dot-product attention in NumPy (no gradients – for understanding)
# ─────────────────────────────────────────────────────────────────────────────
def attention_numpy(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                    mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.
    Q, K: (seq_len, d_k)
    V:    (seq_len, d_v)
    Returns: (output, attention_weights)  both (seq_len, d_v) and (seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d_k)            # (seq_len, seq_len)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)     # mask out future positions

    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))  # stable softmax
    weights = weights / weights.sum(axis=-1, keepdims=True)

    output = weights @ V                           # (seq_len, d_v)
    return output, weights


def make_causal_mask(seq_len: int) -> np.ndarray:
    """Return a boolean upper-triangular mask (True = masked)."""
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Self-attention in PyTorch (single head)
# ─────────────────────────────────────────────────────────────────────────────
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x: torch.Tensor,
                causal: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, d_model)
        Returns: (output, attention_weights)
        """
        Q = self.W_Q(x)   # (batch, seq_len, d_k)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if causal:
            seq_len = x.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        output = weights @ V
        return output, weights


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-head attention in PyTorch
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, d_model)
        Returns: (output, attention_weights)
           output:  (batch, seq_len, d_model)
           weights: (batch, n_heads, seq_len, seq_len)
        """
        B, T, D = x.shape
        h = self.n_heads

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # (B, T, D) → (B, h, T, d_k)
            return t.view(B, T, h, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_Q(x))
        K = split_heads(self.W_K(x))
        V = split_heads(self.W_V(x))

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)         # (B, h, T, T)
        attended = weights @ V                       # (B, h, T, d_k)

        # Concatenate heads and project
        out = attended.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_O(out)
        return out, weights


# ─────────────────────────────────────────────────────────────────────────────
# 4. Visualise attention weights
# ─────────────────────────────────────────────────────────────────────────────
def plot_attention_weights(weights: np.ndarray, tokens: list[str],
                           title: str = "Attention Weights",
                           filename: str = "week04_attention.png"):
    """
    weights: (seq_len, seq_len) numpy array
    tokens:  list of token strings
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(weights, cmap="Blues", vmin=0, vmax=weights.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(tokens, fontsize=10)
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending from)")
    ax.set_title(title)

    # Annotate cells
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if weights[i, j] < 0.5 * weights.max() else "white")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Extract real attention weights from a pretrained model
# ─────────────────────────────────────────────────────────────────────────────
def extract_and_plot_bert_attention(text: str):
    if not HAS_TRANSFORMERS:
        return
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Plot all 12 heads from layer 0
    attn_layer0 = outputs.attentions[0][0].numpy()   # (heads, seq, seq)
    n_heads = attn_layer0.shape[0]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(attn_layer0[h], cmap="Blues")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        ax.set_title(f"Head {h + 1}", fontsize=9)
    plt.suptitle(f"BERT Layer 0 — All 12 Attention Heads\n'{text}'", fontsize=11)
    plt.tight_layout()
    plt.savefig("week04_bert_all_heads.png", dpi=120)
    plt.show()
    print("Saved: week04_bert_all_heads.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Task 1: NumPy attention on toy example ────────────────────────────────
    print("=" * 60)
    print("TASK 1: Scaled dot-product attention (NumPy, bidirectional)")
    print("=" * 60)
    np.random.seed(42)
    seq_len, d_k = 6, 8
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
    out, weights = attention_numpy(Q, K, V)
    print(f"Output shape: {out.shape}")
    print(f"Attention weights (each row sums to 1):\n{weights.round(3)}")
    print(f"Row sums: {weights.sum(axis=-1).round(6)}")

    # ── Task 2: Causal masked attention ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 2: Causal masked attention (NumPy)")
    print("=" * 60)
    mask = make_causal_mask(seq_len)
    out_causal, weights_causal = attention_numpy(Q, K, V, mask=mask)
    print("Causal attention weights (upper triangle is 0):")
    print(weights_causal.round(3))
    # Verify upper triangle is 0
    assert np.allclose(np.triu(weights_causal, k=1), 0, atol=1e-6), "Causal mask failed!"
    print("Causal mask verified: upper triangle is zero. ✓")

    tokens_demo = ["The", "cat", "sat", "on", "the", "mat"]
    plot_attention_weights(weights, tokens_demo,
                           title="Bidirectional Attention Weights",
                           filename="week04_attn_bidirectional.png")
    plot_attention_weights(weights_causal, tokens_demo,
                           title="Causal Masked Attention Weights",
                           filename="week04_attn_causal.png")

    # ── Task 3: PyTorch single-head self-attention ────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 3: Single-head self-attention (PyTorch)")
    print("=" * 60)
    d_model, d_k_pt = 32, 16
    attn1 = SingleHeadSelfAttention(d_model, d_k_pt)
    x = torch.randn(1, 6, d_model)
    out_pt, w_pt = attn1(x, causal=True)
    print(f"Input shape:   {x.shape}")
    print(f"Output shape:  {out_pt.shape}")
    print(f"Weights shape: {w_pt.shape}")
    print(f"Row sums: {w_pt[0].sum(dim=-1).detach().numpy().round(4)}")

    # ── Task 4: Multi-head attention ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 4: Multi-head attention (PyTorch)")
    print("=" * 60)
    d_model_mha, n_heads = 64, 4
    mha = MultiHeadAttention(d_model_mha, n_heads, causal=True)
    x_mha = torch.randn(2, 8, d_model_mha)   # batch=2
    out_mha, w_mha = mha(x_mha)
    print(f"Input:   {x_mha.shape}")
    print(f"Output:  {out_mha.shape}")
    print(f"Weights: {w_mha.shape}   (batch, heads, seq, seq)")

    # Visualise head 0, batch item 0
    w_head0 = w_mha[0, 0].detach().numpy()
    tokens8 = [f"tok{i}" for i in range(8)]
    plot_attention_weights(w_head0, tokens8,
                           title="Multi-Head Attention — Head 0 (causal)",
                           filename="week04_mha_head0.png")

    # ── Task 5: Real BERT attention weights ──────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 5: BERT real attention weights")
    print("=" * 60)
    if HAS_TRANSFORMERS:
        extract_and_plot_bert_attention("The doctor gave the patient a prescription.")
    else:
        print("transformers not available – skipping")

    print("\nAll tasks complete.")
