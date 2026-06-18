"""
Week 17 Practical: Efficiency — Attention Variants and Alternative Architectures
==================================================================================
Objectives:
  - Measure time and memory scaling of naive attention vs tiled attention
  - Implement a 1D SSM (S4-style): verify recurrent == convolutional form
  - Implement the Mamba selective SSM mechanism (input-dependent B, C, Δ)
  - Compare KV-cache memory: MHA vs GQA vs SSM fixed state
  - Implement sliding window attention and compare to full attention
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — Attention scaling: naive vs tiled vs sparse
# ─────────────────────────────────────────────────────────────────────────────
def naive_attention(Q, K, V):
    """Standard O(n²) attention — materialises the full n×n matrix."""
    d_k   = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return weights @ V


def tiled_attention(Q, K, V, tile_size=64):
    """
    Memory-efficient tiled attention (simplified FlashAttention idea).
    Processes Q in tiles; for each Q tile iterates over all K,V tiles.
    Uses online softmax to accumulate without the full n×n matrix.
    Same mathematical result as naive_attention.
    """
    B, n, d = Q.shape
    output  = torch.zeros_like(Q)

    for q_start in range(0, n, tile_size):
        q_end = min(q_start + tile_size, n)
        Q_i   = Q[:, q_start:q_end, :]          # (B, tile, d)

        m_i = torch.full((B, q_end - q_start), float('-inf'), device=Q.device)
        d_i = torch.zeros(B, q_end - q_start, device=Q.device)
        o_i = torch.zeros(B, q_end - q_start, d, device=Q.device)

        for k_start in range(0, n, tile_size):
            k_end   = min(k_start + tile_size, n)
            K_j     = K[:, k_start:k_end, :]
            V_j     = V[:, k_start:k_end, :]

            s_ij    = (Q_i @ K_j.transpose(-2, -1)) / math.sqrt(d)  # (B, tq, tk)
            m_ij    = s_ij.max(dim=-1).values                         # (B, tq)
            m_new   = torch.maximum(m_i, m_ij)

            exp_s   = torch.exp(s_ij - m_new.unsqueeze(-1))           # (B, tq, tk)
            exp_old = torch.exp(m_i - m_new)                          # (B, tq)

            d_i = exp_old * d_i + exp_s.sum(dim=-1)
            o_i = (exp_old.unsqueeze(-1) * o_i + exp_s @ V_j)
            m_i = m_new

        output[:, q_start:q_end, :] = o_i / d_i.unsqueeze(-1)

    return output


def sliding_window_attention(Q, K, V, window=32):
    """Each position attends only to a local window of size 2w+1."""
    B, n, d = Q.shape
    output  = torch.zeros_like(Q)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        q  = Q[:, i:i+1, :]          # (B, 1, d)
        k  = K[:, lo:hi, :]          # (B, w, d)
        v  = V[:, lo:hi, :]
        s  = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        w  = F.softmax(s, dim=-1)
        output[:, i:i+1, :] = w @ v
    return output


def measure_time_memory(fn, *args, n_warmup=2, n_repeat=5):
    """Measure mean wall-clock time (ms) and peak GPU memory (MB)."""
    for _ in range(n_warmup):
        _ = fn(*args)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _  = fn(*args)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    mem_mb = torch.cuda.max_memory_allocated() / 1e6 if DEVICE == "cuda" else 0.0
    return float(np.mean(times)), mem_mb


def task1_attention_scaling():
    print("=" * 65)
    print("TASK 1: Attention time and memory scaling")
    print("=" * 65)

    seq_lens   = [128, 256, 512, 1024, 2048]
    d_model    = 64
    batch_size = 1

    naive_times, tiled_times, sparse_times = [], [], []
    naive_mems,  tiled_mems               = [], []

    print(f"\n{'n':>6}  {'Naive (ms)':>12}  {'Tiled (ms)':>12}  {'Window (ms)':>12}")
    print("-" * 50)
    for n in seq_lens:
        Q = torch.randn(batch_size, n, d_model, device=DEVICE)
        K = torch.randn(batch_size, n, d_model, device=DEVICE)
        V = torch.randn(batch_size, n, d_model, device=DEVICE)

        t_naive,  m_naive  = measure_time_memory(naive_attention, Q, K, V)
        t_tiled,  m_tiled  = measure_time_memory(tiled_attention, Q, K, V)
        t_sparse, _        = measure_time_memory(sliding_window_attention, Q, K, V, 32)

        naive_times.append(t_naive)
        tiled_times.append(t_tiled)
        sparse_times.append(t_sparse)
        naive_mems.append(m_naive)
        tiled_mems.append(m_tiled)
        print(f"{n:>6}  {t_naive:>12.2f}  {t_tiled:>12.2f}  {t_sparse:>12.2f}")

    # Verify tiled == naive
    Q = torch.randn(1, 128, d_model)
    K = torch.randn(1, 128, d_model)
    V = torch.randn(1, 128, d_model)
    diff = (naive_attention(Q, K, V) - tiled_attention(Q, K, V)).abs().max().item()
    print(f"\n  Max absolute diff (naive vs tiled): {diff:.2e}  "
          f"({'✓ identical' if diff < 1e-4 else '⚠ different'})")

    # Plot scaling
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(seq_lens, naive_times,  "o-", label="Full attention",    color="red")
    axes[0].plot(seq_lens, tiled_times,  "s-", label="Tiled (FA-style)",  color="steelblue")
    axes[0].plot(seq_lens, sparse_times, "^-", label="Sliding window",    color="green")
    axes[0].set_xlabel("Sequence length n")
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_title("Attention Time Scaling")
    axes[0].legend()

    n2 = [n**2 / seq_lens[0]**2 * naive_times[0] for n in seq_lens]
    axes[0].plot(seq_lens, n2, "k--", alpha=0.3, label="O(n²) reference")
    axes[0].legend()

    if DEVICE == "cuda":
        axes[1].plot(seq_lens, naive_mems, "o-", color="red",      label="Full attention")
        axes[1].plot(seq_lens, tiled_mems, "s-", color="steelblue", label="Tiled (FA-style)")
        axes[1].set_xlabel("Sequence length n")
        axes[1].set_ylabel("Peak GPU memory (MB)")
        axes[1].set_title("Attention Memory Scaling")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "GPU memory profiling\nrequires CUDA",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title("GPU Memory (CUDA only)")

    plt.tight_layout()
    plt.savefig("week17_attention_scaling.png", dpi=150)
    plt.show()
    print("Saved: week17_attention_scaling.png")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — S4-style SSM: recurrent vs convolutional forms
# ─────────────────────────────────────────────────────────────────────────────
class SimpleSSM(nn.Module):
    """
    A minimal S4-inspired SSM with learnable A, B, C, Δ.
    State dim N, input/output dim d=1 for clarity.

    Recurrent form:  h_t = Ā h_{t-1} + B̄ u_t
                     y_t = C h_t
    Convolutional form: y = K * u  where K_k = C Ā^k B̄
    """
    def __init__(self, N=16, d=1):
        super().__init__()
        self.N = N
        self.d = d
        # Diagonal SSM for simplicity (A is diagonal, parameterised by log)
        self.log_A  = nn.Parameter(torch.randn(N) * 0.1 - 1.0)  # stable: A < 0
        self.B      = nn.Parameter(torch.randn(N, d) * 0.1)
        self.C      = nn.Parameter(torch.randn(d, N) * 0.1)
        self.log_dt = nn.Parameter(torch.zeros(1))                # log step size

    @property
    def A(self):
        return -torch.exp(self.log_A)    # negative real diagonal → stable

    @property
    def dt(self):
        return torch.exp(self.log_dt)

    def discretise(self):
        """ZOH discretisation: Ā = exp(Δ A), B̄ ≈ Δ B (simplified for diag A)."""
        A_bar = torch.exp(self.dt * self.A)          # (N,) diagonal
        B_bar = (self.dt * self.B)                    # (N, d) — simplified
        return A_bar, B_bar

    def recurrent_forward(self, u):
        """u: (T, d) → y: (T, d)  via step-by-step recurrence."""
        A_bar, B_bar = self.discretise()
        T   = u.shape[0]
        h   = torch.zeros(self.N, device=u.device)
        ys  = []
        for t in range(T):
            h  = A_bar * h + (B_bar @ u[t])        # (N,)
            y  = self.C @ h                          # (d,)
            ys.append(y)
        return torch.stack(ys, dim=0)               # (T, d)

    def convolutional_forward(self, u):
        """
        u: (T, d) → y: (T, d)  via convolution with the SSM kernel K.
        K_k = C Ā^k B̄   for k = 0, 1, ..., T-1
        Uses FFT-based convolution: O(T log T) instead of O(T²).
        """
        A_bar, B_bar = self.discretise()
        T   = u.shape[0]

        # Build kernel K of length T
        powers = torch.pow(A_bar.unsqueeze(0), torch.arange(T, device=u.device).unsqueeze(1))
        # powers: (T, N)  where powers[k, i] = A_bar[i]^k
        K = (powers @ B_bar.squeeze(-1).unsqueeze(-1) * self.C.T).sum(dim=-1)
        # Actually compute: K[k] = C @ diag(Ā^k) @ B̄ = (C * Ā^k) @ B̄
        K_vals = []
        for k in range(T):
            A_k   = A_bar ** k           # (N,)
            K_k   = (self.C * A_k) @ B_bar   # (d, d)
            K_vals.append(K_k[0, 0])    # scalar for d=1
        K_tensor = torch.stack(K_vals)  # (T,) — the SSM impulse response

        # FFT convolution
        K_f = torch.fft.rfft(K_tensor, n=2 * T)
        u_f = torch.fft.rfft(u[:, 0],  n=2 * T)
        y_f = K_f * u_f
        y   = torch.fft.irfft(y_f, n=2 * T)[:T]
        return y.unsqueeze(-1)          # (T, d)


def task2_ssm_equivalence():
    print("\n" + "=" * 65)
    print("TASK 2: S4-style SSM — recurrent vs convolutional equivalence")
    print("=" * 65)

    T   = 64
    ssm = SimpleSSM(N=16, d=1)
    u   = torch.randn(T, 1)

    y_rec  = ssm.recurrent_forward(u)
    y_conv = ssm.convolutional_forward(u)

    max_diff = (y_rec - y_conv).abs().max().item()
    print(f"\n  Sequence length T = {T}")
    print(f"  State dimension N = {ssm.N}")
    print(f"  Max diff recurrent vs convolutional: {max_diff:.2e}")
    print(f"  {'✓ Forms are equivalent' if max_diff < 1e-4 else '⚠ Discrepancy detected'}")

    # Plot the SSM impulse response (the kernel K)
    with torch.no_grad():
        impulse = torch.zeros(T, 1)
        impulse[0, 0] = 1.0
        K_response = ssm.recurrent_forward(impulse)

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.plot(K_response[:, 0].detach().numpy())
    plt.xlabel("Time step k")
    plt.ylabel("K[k]")
    plt.title("SSM Impulse Response (Kernel)")

    plt.subplot(1, 2, 2)
    plt.plot(y_rec[:, 0].detach().numpy(),  label="Recurrent",     alpha=0.7)
    plt.plot(y_conv[:, 0].detach().numpy(), label="Convolutional", alpha=0.7, linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("Output")
    plt.title("SSM Output: Recurrent vs Convolutional")
    plt.legend()
    plt.tight_layout()
    plt.savefig("week17_ssm_equivalence.png", dpi=150)
    plt.show()
    print("Saved: week17_ssm_equivalence.png")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — Mamba-style selective SSM
# ─────────────────────────────────────────────────────────────────────────────
class SelectiveSSM(nn.Module):
    """
    Simplified Mamba selective SSM.
    Key difference from S4: B, C, Δ are *functions of the input* u_t.
    This makes the SSM content-aware.
    """
    def __init__(self, d_model=32, N=16):
        super().__init__()
        self.d_model = d_model
        self.N       = N

        # Fixed A (diagonal, stable)
        A = torch.arange(1, N + 1, dtype=torch.float).log()
        self.register_buffer("A_log", A)

        # Input-dependent projections (the "selective" part)
        self.x_proj = nn.Linear(d_model, N + N + 1, bias=False)  # → B, C, log_Δ
        self.dt_proj = nn.Linear(1, d_model, bias=True)           # Δ → full dim

        self.D = nn.Parameter(torch.ones(d_model))   # skip connection

    def forward(self, u):
        """
        u: (B, T, d_model)
        Returns: (B, T, d_model)
        """
        B_batch, T, d = u.shape
        N = self.N

        # Project input to get input-dependent B, C, Δ
        BCdelta = self.x_proj(u)                     # (B, T, 2N+1)
        B_in    = BCdelta[:, :, :N]                  # (B, T, N)
        C_in    = BCdelta[:, :, N:2*N]               # (B, T, N)
        log_dt  = BCdelta[:, :, 2*N:]                # (B, T, 1)
        dt      = F.softplus(log_dt)                 # (B, T, 1) — positive step size

        # Discretise: Ā_t = exp(dt_t * A), B̄_t = dt_t * B_t
        A       = -torch.exp(self.A_log)             # (N,) — negative diagonal
        A_bar   = torch.exp(dt * A.unsqueeze(0).unsqueeze(0))   # (B, T, N)
        B_bar   = dt * B_in                                       # (B, T, N)

        # Recurrent scan (sequential for clarity; production uses parallel scan)
        h     = torch.zeros(B_batch, N, device=u.device)
        ys    = []
        for t in range(T):
            # h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ u_t (simplified: project u_t to N dims)
            u_proj = u[:, t, :d].mean(dim=-1, keepdim=True).expand(-1, N)  # (B, N)
            h = A_bar[:, t, :] * h + B_bar[:, t, :] * u_proj               # (B, N)
            y = (C_in[:, t, :] * h).sum(dim=-1, keepdim=True)              # (B, 1)
            # Expand to d_model (simplified)
            ys.append(y.expand(-1, d))

        ssm_out = torch.stack(ys, dim=1)              # (B, T, d)
        return ssm_out + self.D * u                   # skip connection


def task3_mamba_selective():
    print("\n" + "=" * 65)
    print("TASK 3: Mamba-style selective SSM")
    print("=" * 65)

    d_model = 32
    N       = 16
    T       = 64
    B_batch = 2
    model   = SelectiveSSM(d_model=d_model, N=N)

    u = torch.randn(B_batch, T, d_model)
    y = model(u)
    print(f"\n  Input shape:  {u.shape}")
    print(f"  Output shape: {y.shape}")

    # Key demonstration: selective = different state per token
    # Show that A_bar varies across time (input-dependent)
    with torch.no_grad():
        BCdelta = model.x_proj(u[0])         # (T, 2N+1)
        log_dt  = BCdelta[:, 2*N:]
        dt      = F.softplus(log_dt)         # (T, 1)

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.plot(dt[:, 0].numpy())
    plt.xlabel("Token position t")
    plt.ylabel("Δ_t (step size)")
    plt.title("Input-dependent step size Δ_t\n(varies per token — 'selective')")
    plt.axhline(dt.mean().item(), color="red", linestyle="--", label="mean Δ")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y[0, :, :8].detach().numpy())
    plt.xlabel("Token position t")
    plt.ylabel("Output (first 8 dims)")
    plt.title("Selective SSM Output")
    plt.tight_layout()
    plt.savefig("week17_mamba_selective.png", dpi=150)
    plt.show()
    print("Saved: week17_mamba_selective.png")
    print(f"\n  Δ_t range: [{dt.min().item():.3f}, {dt.max().item():.3f}]")
    print("  Δ_t varies per token → SSM selectively attends to different inputs.")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 — KV cache memory: MHA vs GQA vs SSM
# ─────────────────────────────────────────────────────────────────────────────
def kv_cache_memory_mb(n_tokens, n_layers, d_model, n_heads, n_kv_heads,
                        dtype_bytes=2):
    """
    Estimate KV cache memory in MB.
    MHA: n_kv_heads = n_heads
    GQA: n_kv_heads < n_heads
    """
    d_head   = d_model // n_heads
    kv_bytes = n_tokens * n_layers * 2 * n_kv_heads * d_head * dtype_bytes
    return kv_bytes / 1e6


def ssm_state_memory_mb(n_layers, d_model, N, dtype_bytes=2):
    """SSM fixed state: n_layers * d_model * N values (constant in n_tokens)."""
    return n_layers * d_model * N * dtype_bytes / 1e6


def task4_kv_cache_comparison():
    print("\n" + "=" * 65)
    print("TASK 4: KV cache memory: MHA vs GQA vs SSM")
    print("=" * 65)

    # LLaMA-3 8B-style config
    n_layers = 32
    d_model  = 4096
    n_heads  = 32
    n_kv_heads_gqa = 8   # GQA
    N_ssm    = 16        # SSM state dimension per channel

    seq_lens = [1024, 4096, 16384, 65536, 131072, 262144]

    mha_mems  = [kv_cache_memory_mb(n, n_layers, d_model, n_heads, n_heads)   for n in seq_lens]
    gqa_mems  = [kv_cache_memory_mb(n, n_layers, d_model, n_heads, n_kv_heads_gqa) for n in seq_lens]
    ssm_mem   = ssm_state_memory_mb(n_layers, d_model, N_ssm)

    print(f"\n  Config: {n_layers} layers, d_model={d_model}, "
          f"{n_heads} Q heads, {n_kv_heads_gqa} KV heads (GQA)")
    print(f"  SSM state dim N={N_ssm} per channel\n")
    print(f"  {'n_tokens':>10}  {'MHA (MB)':>10}  {'GQA (MB)':>10}  {'SSM (MB)':>10}")
    print("  " + "-" * 46)
    for n, m_mha, m_gqa in zip(seq_lens, mha_mems, gqa_mems):
        print(f"  {n:>10,}  {m_mha:>10.1f}  {m_gqa:>10.1f}  {ssm_mem:>10.1f}")

    plt.figure(figsize=(9, 4))
    plt.plot([n/1000 for n in seq_lens], mha_mems, "o-",
             label=f"MHA ({n_heads} KV heads)", color="red", linewidth=2)
    plt.plot([n/1000 for n in seq_lens], gqa_mems, "s-",
             label=f"GQA ({n_kv_heads_gqa} KV heads)", color="steelblue", linewidth=2)
    plt.axhline(ssm_mem, color="green", linestyle="--", linewidth=2,
                label=f"SSM fixed state ({ssm_mem:.1f} MB)")
    plt.xlabel("Sequence length (thousands of tokens)")
    plt.ylabel("Memory (MB)")
    plt.title("KV Cache Memory: MHA vs GQA vs SSM\n(LLaMA-3 8B-style config, fp16)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("week17_kv_cache_memory.png", dpi=150)
    plt.show()
    print("Saved: week17_kv_cache_memory.png")
    print(f"\n  SSM memory is CONSTANT regardless of sequence length.")
    print(f"  At 131k tokens: MHA uses {mha_mems[-2]:.0f} MB, GQA uses "
          f"{gqa_mems[-2]:.0f} MB, SSM uses {ssm_mem:.1f} MB.")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 — Sliding window vs full attention: output quality
# ─────────────────────────────────────────────────────────────────────────────
def task5_sliding_window_quality():
    print("\n" + "=" * 65)
    print("TASK 5: Sliding window vs full attention — output comparison")
    print("=" * 65)

    n, d    = 64, 32
    windows = [4, 8, 16, 32, n]   # last = full attention

    torch.manual_seed(0)
    Q = torch.randn(1, n, d)
    K = torch.randn(1, n, d)
    V = torch.randn(1, n, d)

    full_out = naive_attention(Q, K, V)

    print(f"\n  Sequence length: {n}   d_model: {d}")
    print(f"\n  {'Window size':>12}  {'Max diff':>10}  {'Cosine sim':>12}  {'Coverage':>10}")
    print("  " + "-" * 50)

    diffs, cosims, labels = [], [], []
    for w in windows:
        if w == n:
            out   = full_out
            label = "Full"
        else:
            out   = sliding_window_attention(Q, K, V, window=w)
            label = f"w={w}"

        diff  = (out - full_out).abs().mean().item()
        cos   = F.cosine_similarity(out.view(-1), full_out.view(-1), dim=0).item()
        cover = min(2*w + 1, n) / n
        diffs.append(diff)
        cosims.append(cos)
        labels.append(label)
        print(f"  {label:>12}  {diff:>10.4f}  {cos:>12.4f}  {cover:>9.1%}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(labels, diffs, color=["steelblue"]*4 + ["green"])
    plt.xlabel("Attention type")
    plt.ylabel("Mean absolute diff vs full attention")
    plt.title("Output Quality: Sliding Window vs Full Attention")

    plt.subplot(1, 2, 2)
    plt.bar(labels, cosims, color=["steelblue"]*4 + ["green"])
    plt.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
    plt.ylim(0.8, 1.01)
    plt.xlabel("Attention type")
    plt.ylabel("Cosine similarity to full attention")
    plt.title("Output Cosine Similarity")
    plt.tight_layout()
    plt.savefig("week17_sliding_window_quality.png", dpi=150)
    plt.show()
    print("Saved: week17_sliding_window_quality.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    task1_attention_scaling()
    task2_ssm_equivalence()
    task3_mamba_selective()
    task4_kv_cache_comparison()
    task5_sliding_window_quality()

    print("\n" + "=" * 65)
    print("All Week 17 tasks complete.")
    print("=" * 65)
