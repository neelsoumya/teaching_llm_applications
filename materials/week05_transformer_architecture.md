# Week 5 — Transformer Architecture

## Lecture Overview

This week we assemble all the components — embedding, positional encoding, multi-head attention, feed-forward networks, residual connections, layer normalisation — into a complete Transformer block, and then into full encoder-only, encoder-decoder, and decoder-only models.

---

## 1. The Transformer Block

A single Transformer block consists of:

```
x → LayerNorm → Multi-Head Self-Attention → + (residual) → LayerNorm → FFN → + (residual)
```

In PyTorch pseudocode:

```python
def transformer_block(x, attn, ffn):
    # Self-attention sub-layer with residual
    x = x + attn(layer_norm(x))
    # Feed-forward sub-layer with residual
    x = x + ffn(layer_norm(x))
    return x
```

Note: this is "Pre-LN" (layer norm before the sub-layer), which is more stable than the original "Post-LN" variant.

---

## 2. Components in Detail

### 2.1 Multi-Head Self-Attention

Covered in Week 4. Allows each token to gather information from all other tokens.

### 2.2 Feed-Forward Network (FFN)

A position-wise (applied independently to each token) two-layer MLP:

```
FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2
```

- W_1 : d_model → d_ff (typically d_ff = 4 * d_model)
- W_2 : d_ff → d_model
- GELU activation (smoother than ReLU; used in GPT-2 onwards)

The FFN has been described as a "key-value memory" — it stores factual associations learned during pre-training.

### 2.3 Residual Connections

```
output = sub_layer(x) + x
```

Residual connections (He et al., 2016, from ResNets) allow gradients to flow directly to earlier layers, enabling training of very deep networks.

### 2.4 Layer Normalisation

```
LayerNorm(x) = gamma * (x - mean(x)) / std(x) + beta
```

Normalises each token's representation independently across the embedding dimension. Stabilises training.

---

## 3. Encoder-Only Models (BERT family)

- **Purpose**: produce rich contextual representations; good for classification, NER, QA.
- **Attention type**: bidirectional — every token can attend to every other token.
- **Training objective**: Masked Language Modelling (MLM) — randomly mask 15% of tokens; predict the masked tokens.
- **Examples**: BERT, RoBERTa, DeBERTa, DistilBERT.

```
Input: "The [MASK] sat on the mat"
Target: "cat"
```

---

## 4. Encoder-Decoder Models (T5, BART family)

- **Purpose**: sequence-to-sequence tasks — translation, summarisation, question answering.
- **Encoder**: bidirectional attention; processes the full input.
- **Decoder**: causal attention + cross-attention to encoder output.
- **Training objective**: span corruption (T5) or denoising (BART).
- **Examples**: T5, FLAN-T5, BART, mT5.

---

## 5. Decoder-Only Models (GPT family)

- **Purpose**: text generation; the dominant architecture for large LLMs.
- **Attention type**: causal (left-to-right only).
- **Training objective**: next-token prediction (standard language modelling).
- **Examples**: GPT-2, GPT-3, GPT-4, LLaMA, Mistral, Falcon, Gemma.

```
Input:  "The cat sat on the"
Target: "mat"
```

This is the architecture we focus on for the rest of the course.

---

## 6. The Full Decoder-Only Model

```
Input token IDs
        │
        ▼
Token Embedding (V × d_model)
        +
Positional Encoding
        │
        ▼
[Transformer Block 1]
[Transformer Block 2]
    ...
[Transformer Block N]
        │
        ▼
Layer Norm
        │
        ▼
Linear (d_model → V)    ← weight-tied with token embedding
        │
        ▼
Softmax → probability distribution over vocabulary
```

For GPT-2 small: N=12 blocks, d_model=768, d_ff=3072, 12 heads, vocab 50,257.
For LLaMA-3 70B: N=80 blocks, d_model=8192, grouped query attention, vocab 128,256.

---

## 7. Weight Tying

The token embedding matrix (V × d_model) is *shared* with the final linear layer that projects back to vocabulary size. This reduces parameters and also ensures the representation space is consistent between input and output.

---

## 8. KV Cache

At inference time, generating token t+1 requires the keys and values from all previous positions 1..t. Computing these from scratch each step would be O(n²) total. The **KV cache** stores previously computed K and V matrices and reuses them.

- Memory cost: 2 × N_layers × n × d_model × 2 bytes (fp16) per batch.
- For a 70B model with 8k context: ~70GB — hence the importance of efficient serving infrastructure.

---

## 9. Scaling Model Size

| Model | Layers | d_model | Heads | Parameters |
|-------|--------|---------|-------|-----------|
| GPT-2 small | 12 | 768 | 12 | 117M |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B |
| LLaMA-2 7B | 32 | 4096 | 32 | 7B |
| LLaMA-2 70B | 80 | 8192 | 64 | 70B |
| GPT-3 | 96 | 12288 | 96 | 175B |

Modern large models also use:
- **Grouped Query Attention (GQA)**: fewer key/value heads than query heads — reduces KV cache memory.
- **SwiGLU activation**: gated linear unit in the FFN (LLaMA style).
- **RMSNorm**: simpler than LayerNorm; removes the mean subtraction.

---

## 10. Practical This Week

See `practicals/week05_practical.py`:
- Implement a complete decoder-only Transformer from scratch in PyTorch.
- Train it on a small character-level or word-level text dataset.
- Plot training and validation loss.
- Generate text from the trained model using greedy decoding and temperature sampling.

---

## 11. Further Reading

- Vaswani et al. (2017) — "Attention is All You Need" — https://arxiv.org/abs/1706.03762
- Devlin et al. (2019) — "BERT" — https://arxiv.org/abs/1810.04805
- Brown et al. (2020) — GPT-3 — https://arxiv.org/abs/2005.14165
- [Karpathy: Build GPT-2 from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [3Blue1Brown transformer video](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- Existing notebook: `[1_1]_Transformer_from_Scratch_(exercises).ipynb`

---

## Discussion Questions

1. Why are residual connections essential for training very deep transformers?
2. Compare encoder-only, encoder-decoder, and decoder-only architectures. For each, give a task it is best suited for.
3. A GPT-2-small model has 12 layers, d_model=768, and vocab 50,257. Approximately how many parameters are in the embedding matrix alone? What fraction is this of the total 117M?
4. Explain why weight tying between the input embedding and the output linear layer makes mathematical sense.
