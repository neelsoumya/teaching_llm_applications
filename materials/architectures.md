# Architectures (Stanford CS 336)

- [🎥 Video lecture from Stanford CS336 course](https://youtu.be/lVynu4bo1rY?si=nXDSMP5oy4CJt4Mh)

## Intro

- Positional embedding (sines and cosines)

- FFN: ReLU

- LayerNorm, post-norm

## 🎮 Practical

- SWIGLU in FFN
- Rotary Positional Embeddings
- Linear layers (have no bias terms)
- LayerNorm in front of the block

- `Llama`
- why choose these?

## Zoo of models

- why so many models/architectures?
- 🤔 stabilty tricks (during training)
- how many vocab elements?
- train efficiently on GPUs
- `QK-norm`
- Hybrid attention
- lots of experimentation

- _Concept_ 🧩 🚀 💡 architecture modifications that make training more stable
- when `Llama 2` came out, everyone started making minor modifications to it

## Pre vs. Post norm

- Original transformer paper residual layer (post norm)
- GPT 2, 3, Llama 2 are pre-norm

### Why Pre-norm?

- helps with training stability
- smoother gradient flow

- ⚠️ All modern LLMs push the layer norm outside the residual stream (but before the compuations/multi-head attention)

- Residual layer has the norm

![image](../images/pre_vs_post_norm.png)
