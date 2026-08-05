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

## Residual stream

- The residual stream operates as a "parallel highway" for information within a Transformer block.

- The Main Path (blue arrows): As data moves through the Self-Attention and Feed-Forward layers, it is refined and transformed to learn complex patterns and relationships.

The Residual Stream (yellow arrows)::

- This path allows the original, unprocessed input from the previous layer to skip directly to the block's end, where it is added back to the processed signal.

- This mechanism is fundamental for Large Language Models because it ensures that the model preserves previous context and original information as it gets deeper. 

 - Without this direct pathway, essential context could be lost during the heavy processing required at each layer, making it impossible to train deep networks effectively.

![image](../images/residual_stream.png) 

![image](../images/residualstream.png) 

- 💡 without residual connections, the information flow would be bottlenecked at each layer, making it difficult for the model to learn long-range dependencies or retain information across deep stacks of layers. 

## Pre vs. Post norm

- Original transformer paper residual layer (post norm)
- GPT 2, 3, Llama 2 are pre-norm

### Why Pre-norm?

- helps with training stability
- smoother gradient flow

- ⚠️ All modern LLMs push the layer norm outside the residual stream (but before the compuations/multi-head attention)

- Residual layer has the norm

![image](../images/pre_vs_post_norm.png)

- keep your residual stream clean
- allow gradients to flow backwards more easily

- [🎥 gradients are stable during training](https://youtu.be/lVynu4bo1rY?si=XvpNpyE-wAOdeZzg&t=715)



## Double norm

- LayerNorm at the beginning and/or end of Multi-Head Attention (MHA) and Feed Forward Network (FFN) _but_ still outside the residual stream

- 



