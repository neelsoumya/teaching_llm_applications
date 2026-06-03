# Week 4 — The Attention Mechanism

## Lecture Overview

Attention is the single most important innovation in modern LLMs. This week we derive it from first principles, implement it in PyTorch, and develop intuitions for what attention heads learn to do.

---

## 1. Motivation: The Problem With Fixed-Size Representations

In early sequence-to-sequence models (encoder-decoder RNNs for machine translation), the entire source sentence was compressed into a single fixed-size vector. This bottleneck caused quality to degrade for long sentences.

**Attention** (Bahdanau et al., 2015) was introduced as a solution: instead of compressing to a single vector, let the decoder look back at *all* encoder states and weight them dynamically based on relevance.

---

## 2. Scaled Dot-Product Attention

The modern formulation (Vaswani et al., 2017) is clean and fully parallelisable.

### Inputs

Given a sequence of n tokens, we compute three matrices from the input embeddings X ∈ R^{n × d}:

```
Q = X W_Q    (Queries,  n × d_k)
K = X W_K    (Keys,     n × d_k)
V = X W_V    (Values,   n × d_v)
```

W_Q, W_K, W_V are learned weight matrices.

### Formula

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

### Step-by-Step

1. **Similarity scores**: Q K^T gives an n×n matrix of dot products. Entry (i, j) measures how much token i should attend to token j.
2. **Scaling**: divide by sqrt(d_k) to prevent large dot products from pushing softmax into saturation.
3. **Softmax**: convert scores to a probability distribution (rows sum to 1).
4. **Weighted sum**: multiply by V to produce output vectors that are weighted combinations of value vectors.

### Intuition

- **Query**: what am I looking for?
- **Key**: what do I contain / advertise?
- **Value**: what information do I contribute if selected?

For token i, the query vector asks a question; the key vectors of all tokens answer that question; the resulting attention weights determine how much of each value to mix in.

---

## 3. Causal (Masked) Self-Attention

In a decoder-only LLM (GPT family), a token must not attend to *future* tokens (that would be cheating during training and generation).

We apply a **causal mask**: set Q K^T entries for j > i to -∞ before the softmax.

```python
import torch
import torch.nn.functional as F
import math

def causal_self_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    
    n = scores.size(-1)
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights
```

---

## 4. Multi-Head Attention

A single attention head can only ask one "question" at a time. **Multi-head attention** runs h parallel attention heads, each with its own projection matrices.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O

head_i = Attention(Q W_Q_i, K W_K_i, V W_V_i)
```

The output is projected back to d_model with W_O.

**Why multiple heads?**
Different heads learn to capture different types of relationships:
- Syntactic dependencies (subject-verb agreement)
- Coreference (this → the dog)
- Positional patterns (previous token, next token)
- Semantic similarity

---

## 5. Cross-Attention

In encoder-decoder transformers (T5, original Transformer), the decoder uses **cross-attention** to attend to the encoder's output.

The queries come from the decoder; the keys and values come from the encoder. This is how the model conditions generation on the source sequence.

---

## 6. Computational Complexity

Standard self-attention is O(n²·d) in time and O(n²) in memory, where n is the sequence length. This is why long contexts are expensive.

For context windows of 128k tokens, this would require ~16GB of memory just for attention weights. Efficient variants (FlashAttention, sparse attention, linear attention) address this.

### FlashAttention

Computes attention in tiles, never materialising the full n×n matrix. Provides:
- Same mathematical result as standard attention.
- O(n) memory usage.
- 2–4× faster in practice.

Used by default in modern LLM implementations.

---

## 7. Visualising Attention

Attention weights are interpretable (to some degree). Tools like BertViz allow you to interactively visualise which tokens attend to which.

```python
# Conceptual example of extracting attention weights
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)

inputs = tokenizer("The dog chased the cat", return_tensors="pt")
outputs = model(**inputs)

# outputs.attentions: tuple of (batch, heads, seq_len, seq_len) per layer
attn = outputs.attentions[0][0]  # layer 0, batch 0: shape (heads, seq_len, seq_len)
```

---

## 8. Limitations and Caveats

- Attention weights are not always meaningful interpretations. High attention weight ≠ high causal influence.
- Different heads specialise unpredictably; not all heads are interpretable.
- Attention is necessary but not sufficient for explaining model behaviour.

---

## 9. Practical This Week

See `practicals/week04_practical.py`:
- Implement scaled dot-product attention from scratch in NumPy and then PyTorch.
- Implement causal masked attention.
- Implement multi-head attention.
- Visualise attention weights on a short sentence and interpret what different heads attend to.

---

## 10. Further Reading

- Bahdanau et al. (2015) — "Neural Machine Translation by Jointly Learning to Align and Translate" — https://arxiv.org/abs/1409.0473
- Vaswani et al. (2017) — "Attention is All You Need" — https://arxiv.org/abs/1706.03762
- Dao et al. (2022) — "FlashAttention" — https://arxiv.org/abs/2205.14135
- [Illustrated Transformer (Jalammar)](https://jalammar.github.io/illustrated-transformer/)
- [Intro to attention video](https://www.youtube.com/watch?v=XN7sevVxyUM)
- Existing notebook: `Lesson_3-selfattention.ipynb`

---

## Discussion Questions

1. Explain in your own words the role of Q, K, and V in the attention mechanism.
2. Why do we scale by 1/sqrt(d_k)? What happens if we don't?
3. Why can't a decoder-only model attend to future tokens during training?
4. A model has 12 attention heads each with d_k = 64 and d_model = 768. What is the dimension of W_O?
