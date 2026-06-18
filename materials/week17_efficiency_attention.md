# Week 17 — Efficiency: Attention Variants and Alternative Architectures

## Lecture Overview

The standard transformer with full self-attention scales quadratically in both time and memory with sequence length: O(n²·d). For a 128k-token context window this becomes prohibitive. This week we examine the engineering and mathematical innovations that make long-context and efficient inference practical: sparse attention, FlashAttention, linear attention approximations, and the state space model (SSM) family — Mamba and S4 — which offer a genuinely different computational paradigm to the transformer.

---

## 1. The Quadratic Cost of Full Attention

### 1.1 Where the Quadratic Comes From

Recall scaled dot-product attention:

```
Attention(Q, K, V) = softmax(Q K^T / √d_k) V
```

Step by step:

| Operation | Shape | Cost |
|-----------|-------|------|
| Q K^T (similarity matrix) | (n, d_k) × (d_k, n) → (n, n) | O(n² d_k) |
| softmax over rows | (n, n) | O(n²) |
| Weights × V | (n, n) × (n, d_v) → (n, d_v) | O(n² d_v) |
| **Total** | | **O(n² d)** |

Memory cost: storing the (n, n) attention matrix requires O(n²) space. For n = 128,000 tokens, this is ~65 billion entries at fp32 — around 260 GB just for the attention matrix.

### 1.2 Why It Matters

| Context length | Attention matrix (fp16) | Practical? |
|---------------|-------------------------|------------|
| 2,048 | ~16 MB | Yes |
| 32,768 | ~4 GB | Barely |
| 128,000 | ~65 GB | No (without tricks) |
| 1,000,000 | ~4 TB | No |

Long-context models (Gemini 1.5 with 1M tokens, Claude with 200k, GPT-4 Turbo with 128k) all rely on engineering solutions described in this lecture.

---

## 2. Sparse Attention Patterns

The key insight: for most tokens and most heads, the full n×n attention matrix is nearly sparse — most attention weights are very small. We can approximate the full attention by computing only the entries that matter.

### 2.1 Local (Window) Attention

Each token attends only to its w nearest neighbours:

```
token i attends to tokens max(0, i-w) ... min(n-1, i+w)
```

Cost: O(n·w·d) — linear in n for fixed window size w.

Used in: Longformer (Beltagy et al., 2020), BigBird (Zaheer et al., 2020).

Limitation: information cannot flow between distant tokens in a single layer. Depth compensates partially — after L layers, information can travel L·w positions — but very long-range dependencies require many layers.

### 2.2 Strided / Dilated Attention

Attend to every k-th token rather than contiguous neighbours. Covers a larger receptive field without increasing cost.

```
token i attends to tokens i, i-k, i-2k, i-3k, ...
```

### 2.3 Global + Local Attention

Some tokens (e.g. `[CLS]`, sentence separators, or task-specific tokens) attend to and from all positions globally, while most tokens use local attention.

BigBird: global tokens + local window + random attention → O(n) total complexity, provably Turing-complete.

### 2.4 Axial Attention

For 2D inputs (images, video): instead of attending over all n² positions jointly, attend along rows then columns separately.

```
n² → 2n attention operations, each of length n_row or n_col
```

Cost: O(n^{3/2}) for square inputs.

### 2.5 Sliding Window Attention (SWA) — Mistral

Mistral 7B uses sliding window attention with w = 4096 inside each layer. Combined with 32 transformer layers, the effective receptive field is 32 × 4096 = 131,072 tokens despite each individual attention operation being O(n·w).

```python
# Conceptual: restrict attention mask to window
mask = torch.zeros(n, n)
for i in range(n):
    mask[i, max(0, i - window_size): i + 1] = 1
scores = scores.masked_fill(mask == 0, float('-inf'))
```

---

## 3. FlashAttention

Dao et al. (2022, 2023) — FlashAttention is the most important attention engineering advance of the past five years. It achieves **the same mathematical result** as standard attention with:

- **O(n) memory** instead of O(n²)
- **2–4× faster** wall-clock time on A100/H100 GPUs
- **No approximation** — exact attention, not an estimate

### 3.1 The Memory Bottleneck: HBM vs SRAM

Modern GPU memory has a hierarchy:
- **HBM (High Bandwidth Memory)**: large (40–80GB), slow (~2 TB/s bandwidth)
- **SRAM (on-chip)**: tiny (~20MB on A100), very fast (~20 TB/s bandwidth)

Standard attention is **memory-bandwidth bound**: most time is spent reading and writing the n×n matrix to/from HBM, not doing arithmetic.

### 3.2 The FlashAttention Algorithm

FlashAttention tiles the computation into blocks that fit in SRAM:

```
Split Q into tiles Q_1, ..., Q_T  (T = n / block_size)
Split K, V into tiles K_1, ..., T_s, V_1, ..., V_s

For each Q tile:
    Load Q_i into SRAM
    For each K, V tile:
        Load K_j, V_j into SRAM
        Compute local attention scores: S_ij = Q_i K_j^T / √d_k
        Update running softmax denominator and output (online softmax)
    Write output tile O_i back to HBM
```

The key trick is **online softmax** (Milakov & Gimelshein, 2018): computing the softmax incrementally without materialising the full (n, n) matrix, using the log-sum-exp identity to maintain a running maximum and normalisation constant.

```python
# Online softmax pseudocode
m = -inf          # running max
d = 0.0           # running denominator
o = zeros(d_v)    # running output

for each block j:
    s = Q_i @ K_j.T / sqrt(d_k)    # (block_size, block_size)
    m_new = max(m, s.max())
    d_new = exp(m - m_new) * d + exp(s - m_new).sum()
    o = (exp(m - m_new) * d * o + exp(s - m_new) @ V_j) / d_new
    m, d = m_new, d_new
```

### 3.3 FlashAttention-2 and FlashAttention-3

**FlashAttention-2** (Dao, 2023): better work partitioning across GPU thread blocks; fewer non-matmul FLOPs; better handling of causal masking. ~2× faster than FA1.

**FlashAttention-3** (Shah et al., 2024): exploits Hopper (H100) architecture features — overlaps matmul and softmax using warp specialisation; achieves ~75% of H100 theoretical FLOPs throughput.

### 3.4 Impact

FlashAttention is now the default attention implementation in:
- PyTorch (`F.scaled_dot_product_attention` with FA backend)
- HuggingFace Transformers
- vLLM, llama.cpp, all major LLM serving frameworks

```python
# PyTorch 2.0+ uses FlashAttention automatically
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
```

---

## 4. Linear Attention Approximations

Can we reduce the O(n²) complexity of attention to O(n) by approximating the softmax?

### 4.1 The Kernel Trick for Attention

Standard attention:

```
Attention(Q, K, V)_i  =  Σ_j softmax(q_i · k_j / √d) v_j
```

If we replace `softmax(q · k)` with a kernel function `κ(q, k) = φ(q)^T φ(k)` where φ is a feature map:

```
Attention(Q, K, V)_i  =  Σ_j φ(q_i)^T φ(k_j) v_j
                       =  φ(q_i)^T (Σ_j φ(k_j) v_j^T)
```

The sum `Σ_j φ(k_j) v_j^T` can be computed *once* (O(nd) cost), then each query can be answered in O(d²) — overall O(n) complexity.

### 4.2 Performer (Choromanski et al., 2021)

Uses random Fourier features to approximate softmax:

```
exp(q · k) ≈ E[φ(q)^T φ(k)]    where φ(x) = exp(x·ω + b)
```

with ω drawn from a Gaussian distribution. Unbiased estimator; O(n·r) complexity for r random features.

Limitation: approximation quality degrades for long sequences or when attention is very peaked.

### 4.3 Linear Transformer / RWKV

RWKV (Peng et al., 2023): reformulates attention as a recurrent computation with a linear kernel. During inference, processes tokens one at a time with O(1) per-step cost — like an RNN. During training, uses the parallel form for efficiency.

```
RWKV uses:  κ(q, k) = exp(q) · exp(k)   (element-wise, no dot product)
```

This makes the kernel decomposable and allows a recurrent form:

```
h_t = α h_{t-1} + k_t ⊗ v_t    (numerator state)
z_t = β z_{t-1} + k_t            (denominator state)
y_t = q_t ⊗ h_t / (q_t ⊗ z_t)  (output)
```

### 4.4 Limitations of Linear Attention

- Approximation error: linear attention cannot represent all patterns that softmax attention can.
- Performance gap: on most benchmarks, linear attention underperforms full attention, sometimes substantially.
- The "quadratic wall" is real but linear attention is not a free lunch — quality matters.

---

## 5. State Space Models: S4 and Mamba

An entirely different approach: replace attention with a **structured state space model** (SSM).

### 5.1 Continuous-Time State Space Models

A linear SSM maps an input signal u(t) to an output signal y(t) through a hidden state h(t):

```
h'(t) = A h(t) + B u(t)      (state equation)
y(t)  = C h(t) + D u(t)      (output equation)
```

where:
- A ∈ R^{N×N}: state transition matrix
- B ∈ R^{N×1}: input projection
- C ∈ R^{1×N}: output projection
- N: state dimension (hidden size)

This is a classical control theory formulation. The question is: can this replace attention?

### 5.2 Discretisation

For sequence modelling we work with discrete tokens. Discretise using zero-order hold with step size Δ:

```
Ā = exp(Δ A)
B̄ = (ΔA)^{-1}(exp(ΔA) − I) ΔB
```

The discrete recurrence:

```
h_t = Ā h_{t-1} + B̄ u_t
y_t = C h_t
```

### 5.3 S4 — Structured State Spaces for Sequences

Gu et al. (2022): The key insight is that with a specific parameterisation of A (HiPPO matrix — designed to optimally memorise history), the SSM can capture very long-range dependencies, and can be computed efficiently as a **convolution** during training:

```
y = K * u    where K = (C B̄, C Ā B̄, C Ā² B̄, ...) is the SSM kernel
```

Convolutions are O(n log n) via FFT. So S4 is:
- **Recurrent** at inference: O(N) per step, O(n) total
- **Convolutional** during training: O(n log n)
- **Long-range dependencies**: the HiPPO matrix ensures the state captures all past inputs in an optimal polynomial basis

### 5.4 Mamba — Selective State Space Models

Gu and Dao (2023): S4 has a critical limitation — A, B, C, Δ are the same for every input token. The model cannot selectively focus on some inputs more than others (no "content-based" routing).

Mamba introduces **input-dependent (selective) SSM parameters**:

```
B_t, C_t, Δ_t = linear(u_t)    ← depend on the current token
```

Now the state transition varies per token, allowing Mamba to:
- Selectively remember or forget information based on content
- Filter out irrelevant tokens from the state

This is the key mechanism that enables Mamba to match or exceed transformer performance on language modelling benchmarks while maintaining linear time and constant-memory inference.

### 5.5 The Mamba Block

```
Input x
  │
  ├── Linear → SSM (selective) → activation → multiply ─┐
  │                                                       │
  └── Linear → activation ──────────────────────────────┤
                                                         │
                                                       output
```

Mamba replaces the attention + FFN transformer block with a single SSM block containing two parallel paths (SSM path and gate path), similar in spirit to gated recurrent units.

### 5.6 Mamba vs Transformer

| Property | Transformer | Mamba |
|----------|------------|-------|
| Training complexity | O(n² d) | O(n d N) |
| Inference per token | O(n d) (KV cache) | O(d N) — constant! |
| Memory (inference) | O(n d) (KV cache grows) | O(d N) — fixed! |
| Long-range modelling | Excellent (full attention) | Good (selective SSM) |
| In-context learning | Excellent | Weaker |
| Hardware efficiency | Good (with FlashAttention) | Good (parallel scan) |
| Max tested scale | ~1T parameters | ~3B parameters (as of 2024) |

### 5.7 Hybrid Models

In practice, the community has converged on **hybrid architectures** combining attention layers and SSM layers:

- **Jamba** (AI21 Labs, 2024): alternating Mamba and attention layers. 52B total params, MoE. Handles 256k context.
- **Zamba** (Zyphra, 2024): mostly Mamba layers with a few shared attention layers.
- **Falcon Mamba** (TII, 2024): pure Mamba at 7B scale.
- **Griffin** (DeepMind, 2024): gated linear recurrences + local attention.

The motivation: pure attention excels at in-context learning and associative recall; SSMs excel at compression and constant-memory streaming. Combining them gets the best of both.

### 5.8 "Maybe Not?" — Limitations of SSMs

The slide note "Maybe not?" is a fair caveat. Current evidence:

- On language modelling perplexity at scale (>7B), transformers with FlashAttention still lead.
- In-context learning (few-shot) is notably weaker in pure SSMs.
- Associative recall tasks (where the answer depends on matching specific tokens seen much earlier) are hard for fixed-size states.
- Most frontier models (GPT-4, Claude, Gemini, LLaMA) remain transformer-based.

SSMs are a promising and theoretically elegant alternative, but the jury is still out at scale.

---

## 6. Multi-Query and Grouped-Query Attention

Two engineering variants that reduce the KV cache memory without approximating attention.

### 6.1 Multi-Query Attention (MQA)

Shazeer (2019): share a single key and value head across all query heads.

```
Standard MHA:  n_heads Q heads, n_heads K heads, n_heads V heads
MQA:           n_heads Q heads, 1 K head, 1 V head
```

KV cache memory: reduced by n_heads×. Slight quality degradation.

### 6.2 Grouped-Query Attention (GQA)

Ainslie et al. (2023): a middle ground — G groups of query heads share K and V heads.

```
GQA: n_heads Q heads, G K heads, G V heads   (G < n_heads)
```

LLaMA-2 70B uses GQA with G = 8 (64 query heads, 8 key/value heads). LLaMA-3 uses GQA throughout.

KV cache reduction: n_heads/G ×. Quality nearly identical to MHA.

---

## 7. Putting It Together: Efficient LLM Architecture

Modern efficient LLMs combine:

```
Pre-filling (training / prompt processing):
  FlashAttention-2/3 + GQA + sliding window (for very long context)

Decoding (autoregressive generation):
  KV cache + GQA (reduces cache size) + speculative decoding

Very long context (>100k tokens):
  RoPE with long-context extension (YaRN, LongRoPE) +
  sliding window or local attention in some layers +
  FlashAttention

Alternative inference budget:
  Consider hybrid Mamba/attention or pure Mamba
  for streaming/edge deployment
```

---

## 8. Practical This Week

See `practicals/week17_practical.py`:
- Measure the wall-clock time and memory cost of naive attention vs a tiled approximation as a function of sequence length n; plot the scaling.
- Implement a simple 1D SSM (S4-style) from scratch and verify the equivalence between the recurrent and convolutional forms.
- Implement the Mamba selective SSM mechanism: input-dependent B, C, Δ; run the parallel scan.
- Compare inference-time memory usage: full KV cache (MHA) vs GQA vs SSM fixed state, for varying sequence lengths.
- Implement sliding window attention and compare output quality vs full attention on a short language modelling task.

---

## 9. Further Reading

- Dao et al. (2022) — "FlashAttention" — https://arxiv.org/abs/2205.14135
- Dao (2023) — "FlashAttention-2" — https://arxiv.org/abs/2307.08691
- Shah et al. (2024) — "FlashAttention-3" — https://arxiv.org/abs/2407.08608
- Beltagy et al. (2020) — "Longformer" — https://arxiv.org/abs/2004.05150
- Zaheer et al. (2020) — "BigBird" — https://arxiv.org/abs/2007.14062
- Choromanski et al. (2021) — "Rethinking Attention with Performers" — https://arxiv.org/abs/2009.14794
- Peng et al. (2023) — "RWKV: Reinventing RNNs for the Transformer Era" — https://arxiv.org/abs/2305.13048
- Gu et al. (2022) — "Efficiently Modelling Long Sequences with Structured State Spaces" (S4) — https://arxiv.org/abs/2111.00396
- Gu and Dao (2023) — "Mamba: Linear-Time Sequence Modelling with Selective State Spaces" — https://arxiv.org/abs/2312.00752
- Ainslie et al. (2023) — "GQA: Training Generalised Multi-Query Transformer Models" — https://arxiv.org/abs/2305.13245
- Team AI21 (2024) — "Jamba: A Hybrid Transformer-Mamba Language Model" — https://arxiv.org/abs/2403.19887
- De et al. (2024) — "Griffin: Mixing Gated Linear Recurrences and Local Attention" — https://arxiv.org/abs/2402.19427

---

## Discussion Questions

1. Explain why standard self-attention is O(n²) in both time and memory. Sketch the FlashAttention tiling strategy and explain how it reduces memory to O(n) without changing the mathematical result.
2. Compare sparse attention (e.g. sliding window) and linear attention (e.g. Performer) as approaches to reducing the quadratic cost. What does each sacrifice compared to full attention?
3. Describe the selective SSM mechanism in Mamba. What is the key difference from S4, and why does it matter for language modelling?
4. A production system needs to handle 500k-token contexts for document analysis. The model is a standard transformer. List three techniques from this week that could make this feasible, and describe the trade-off each introduces.
5. "Mamba might not replace transformers." Summarise the current evidence for and against SSMs as a viable alternative to attention-based architectures at frontier scale.
