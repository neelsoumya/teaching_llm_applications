# Week 6 — Pre-training and Scaling Laws

## Lecture Overview

This week we examine *how* large language models are trained: the data pipelines, objectives, compute budgets, and the empirical scaling laws that govern the relationship between model size, data, compute, and performance.

---

## 1. The Pre-training Paradigm

Modern LLMs are trained in two stages:

1. **Pre-training**: self-supervised learning on massive, diverse text corpora. No human labels required. The objective is next-token prediction (or masked language modelling). This is where the vast majority of compute is spent.

2. **Adaptation**: fine-tuning, instruction tuning, or RLHF to align the model to user intent (covered Week 7).

Pre-training is what gives the model its general knowledge, reasoning ability, and language competence.

---

## 2. Pre-training Data

### 2.1 Common Data Sources

| Source | Examples |
|--------|---------|
| Web crawls | Common Crawl, C4, RefinedWeb |
| Books | Books1, Books2, Project Gutenberg |
| Code | GitHub, The Stack |
| Wikipedia | All languages |
| Scientific papers | arXiv, PubMed, Semantic Scholar |
| Conversations | Reddit (PushShift), StackExchange |

GPT-3 was trained on ~300B tokens. LLaMA-3 was trained on over 15 trillion tokens.

### 2.2 Data Quality and Filtering

Raw Common Crawl is noisy. Standard filtering steps:
- **Language identification**: keep only target language(s).
- **Quality heuristics**: remove pages with too many special characters, very short pages, duplicates.
- **Deduplication**: exact-match and near-duplicate removal (MinHash). Critical — deduplication dramatically improves performance per token.
- **Toxicity filtering**: remove harmful content.
- **Data mixing**: carefully chosen proportions of different sources.

### 2.3 Data Contamination

If evaluation benchmarks (e.g. MMLU, HumanEval) appear in training data, reported performance is inflated. Most labs now deduplicate against known benchmarks, but contamination remains a live concern.

---

## 3. The Training Objective

For decoder-only LLMs the pre-training objective is **cross-entropy loss** over next-token prediction:

```
L = -1/T * sum_{t=1}^{T} log P(w_t | w_1, ..., w_{t-1})
```

This is equivalent to maximising the log-likelihood of the training corpus.

**Perplexity** is the exponentiated average negative log-likelihood:

```
PPL = exp(L)
```

Lower perplexity = better model. As a sanity check: a model that assigns uniform probability over a 50k vocabulary has PPL = 50,000.

---

## 4. Training Infrastructure

Training a 70B-parameter model requires:
- **Hardware**: thousands of A100 or H100 GPUs.
- **Parallelism strategies**:
  - **Data parallelism**: each GPU holds the full model; different batches.
  - **Tensor parallelism**: split individual weight matrices across GPUs.
  - **Pipeline parallelism**: different layers on different GPUs.
  - **Sequence parallelism**: split the sequence dimension.
- **Mixed precision**: bfloat16 for activations; fp32 for master weights and optimiser state.
- **Gradient checkpointing**: recompute activations during backward pass to save memory at the cost of extra compute.
- **Optimizer**: AdamW with learning rate warmup + cosine decay schedule.

---

## 5. Scaling Laws

Kaplan et al. (2020) and Hoffmann et al. (2022) characterise how LLM performance scales with:
- **N**: number of model parameters
- **D**: number of training tokens
- **C**: total compute (measured in FLOPs)

The approximate relationship is:

```
C ≈ 6 * N * D
```

(Each forward pass through N parameters over D tokens, times 6 for forward + backward.)

### 5.1 Kaplan Scaling Laws (OpenAI, 2020)

Loss follows a power law:

```
L(N) ∝ N^(-α_N)
L(D) ∝ D^(-α_D)
L(C) ∝ C^(-α_C)
```

Key finding: given a fixed compute budget, it is better to train a **larger model on fewer tokens** than a smaller model on more tokens.

### 5.2 Chinchilla Scaling Laws (DeepMind, 2022)

Hoffmann et al. showed Kaplan's recommendation was wrong — models were under-trained.

**Chinchilla optimal**: for a compute-optimal model, train with roughly **20 tokens per parameter**.

```
N_opt ∝ C^0.5
D_opt ∝ C^0.5
```

| Model | Parameters | Tokens | Tokens / Param |
|-------|-----------|--------|---------------|
| GPT-3 | 175B | 300B | ~1.7 (under-trained) |
| Chinchilla | 70B | 1.4T | ~20 (optimal) |
| LLaMA-3 | 8B | 15T | ~1875 (over-trained for inference) |

LLaMA-3 deliberately over-trains small models: a smaller model trained on more tokens can be **cheaper to deploy** while matching a larger model trained to Chinchilla optimum.

---

## 6. Emergent Abilities

Wei et al. (2022) document abilities that appear *abruptly* at a certain model scale:

- Multi-step arithmetic
- Chain-of-thought reasoning
- Instruction following
- Multi-lingual translation (for low-resource languages)

These are not predictable from small-scale experiments. This makes forecasting LLM capabilities difficult.

**Debate**: some researchers argue emergence is an artefact of evaluation metrics — with smoother metrics, emergence looks more gradual.

---

## 7. Training Instabilities

Large-scale training is prone to:
- **Loss spikes**: sudden large increases in loss, often requiring rollback to a checkpoint.
- **Gradient explosions**: clipped with `max_norm` parameter.
- **Divergence from bad data batches**: mitigated by careful data filtering.

Monitoring training requires tracking loss, gradient norm, learning rate, and hardware utilisation in real time.

---

## 8. Practical This Week

See `practicals/week06_practical.py`:
- Train a small GPT-like model on a text corpus (e.g. Shakespeare or a custom dataset).
- Plot training loss and compute perplexity on a held-out set.
- Empirically verify scaling: train models of different sizes and plot loss vs parameter count.
- Estimate compute (FLOPs) for each run and plot against loss.

---

## 9. Further Reading

- Kaplan et al. (2020) — "Scaling Laws for Neural Language Models" — https://arxiv.org/abs/2001.08361
- Hoffmann et al. (2022) — "Training Compute-Optimal Large Language Models" (Chinchilla) — https://arxiv.org/abs/2203.15556
- Wei et al. (2022) — "Emergent Abilities of Large Language Models" — https://arxiv.org/abs/2206.07682
- Touvron et al. (2023) — "LLaMA 2" — https://arxiv.org/abs/2307.09288

---

## Discussion Questions

1. Why is deduplication of training data so important for model quality?
2. Explain the Chinchilla finding in plain language. Why were earlier models under-trained?
3. You have a compute budget of 10²³ FLOPs. Using the Chinchilla relationship, estimate the optimal model size and number of training tokens.
4. Why might a company prefer to deploy a smaller, over-trained model over a larger Chinchilla-optimal model?
