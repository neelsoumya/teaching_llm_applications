# Week 3 — Embeddings and Representations

## Lecture Overview

This week we go deep into *embeddings* — how discrete token IDs become dense, meaningful vectors, how those vectors encode semantic relationships, and how to work with them in practice.

## Thoughts

- [Embeddings and why we need them](https://cambiotraining.github.io/ml-unsupervised/materials/normalization.html#hands-on-coding) see section on categorical data and encoding

- words in similar context have similar meanings

- we need embeddings which take care of context

- embeddings optimized for specific task. See why and see notes for UG class [here](https://github.com/neelsoumya/teaching_unsupervised_machine_learning_private) and [here](https://github.com/neelsoumya/public_teaching_unsupervised_learning/) and [notes](https://github.com/neelsoumya/public_teaching_unsupervised_learning/blob/main/pca_notes_advanced_2.pdf) and [notes](https://github.com/neelsoumya/teaching_unsupervised_machine_learning_private/tree/main/supervisions)

-  Lecture Prep: Fundamentals of Embeddings and Large Language Models

- 1. Core Concepts: Why Do We Need Embeddings?
Computers, neural networks, and mathematical algorithms inherently process numbers—they cannot natively comprehend raw text, symbols, or semantic nuance. To bridge this gap, textual data must be converted into a numerical format. 

However, simple methods like assigning a unique integer to each word or using one-hot encoding fall short because they fail to capture meaning. 

### The Evolution of Representations
* **Discrete Symbols (The Problem):** If "apple" is assigned the index `1` and "orange" is assigned `2`, a mathematical model treats them as completely independent units. The numerical distance between 1 and 2 holds no intrinsic semantic relationship, meaning the model cannot infer that both are types of fruit.
* **Vector Spaces (The Solution):** Embeddings map discrete symbols into a continuous, high-dimensional vector space. In this space, geometric proximity correlates directly with semantic similarity. Words or concepts with similar meanings are located closer together, allowing models to leverage mathematical tools (like cosine similarity) to capture human-like intuition.



---

-  End-to-End Training and Task Specificity

A crucial point for students to grasp is that embeddings are **not static or absolute representations**; they are tied explicitly to a specific objective and are trained end-to-end.

* **Context & Task Definition:** When a representation is optimized for a particular task, the dimensions of the embedding learn to reflect the properties most useful for solving that task. The objective dictates the underlying geometry of the vector space.

* **Example: Word Prediction:** If the target task is next-word prediction (the foundational objective of LLMs), the trained embeddings will manifest as highly specific numbers optimized exclusively to facilitate that prediction loop. 

* **End-to-End Optimization:** Rather than manually assigning features, initial embedding weights are usually randomized. Through the process of backpropagation, the gradients flow all the way back to the input embedding layer. The network iteratively tunes these vector coordinates until the spatial layout minimizes the error rate of the target task.

---

-  Recommended Educational Resources (3Blue1Brown)

For highly visual and intuitive breakdowns of these exact concepts, the following video lessons from the 3Blue1Brown channel are exceptional tools for both lecture planning and student reference:

* **[Large Language Models Explained Briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs):** A visually compelling introduction covering how neural networks, chatbots, and transformers process and generate human language.
* **[But What Is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk):** This foundational lesson breaks down the math underlying deep learning layers, weights, and biases, which is vital for understanding how embedding vectors are manipulated mathematically.

---

## 1. From Tokens to Vectors

After tokenisation we have a sequence of integers, e.g. `[1045, 2293, 4671]`. A neural network cannot directly compute on integers. We need to map each integer to a continuous vector.

An **embedding layer** is simply a lookup table:

```
E : V → R^d
```

where V is the vocabulary size (e.g. 50,257) and d is the embedding dimension (e.g. 768 or 4096). Each token ID is mapped to a vector of d floats.

In PyTorch:

```python
import torch
import torch.nn as nn

vocab_size = 50257
d_model = 768

embedding = nn.Embedding(vocab_size, d_model)
token_ids = torch.tensor([1045, 2293, 4671])
vectors = embedding(token_ids)   # shape: (3, 768)
```

These vectors are *learned* during training — they start random and gradually encode useful structure.

---

## 2. The Distributional Hypothesis

*"You shall know a word by the company it keeps."* — J. R. Firth (1957)

Words that appear in similar contexts tend to have similar meanings. By training on next-token prediction, the model is forced to give similar representations to tokens that behave similarly in text.

---

## 3. Word2Vec: The Classic Embedding Model

Mikolov et al. (2013) introduced Word2Vec — a simple two-layer neural network trained with two objectives:
- **Skip-gram**: predict context words from a centre word.
- **CBOW**: predict the centre word from context words.

The result: embeddings that capture analogy relationships:

```
vec("king") - vec("man") + vec("woman") ≈ vec("queen")
```

### Why This Works

The embedding encodes dimensions corresponding loosely to:
- Royalty / power
- Gender
- Plurality
- Tense
- ...

These are *not* explicitly programmed — they emerge from co-occurrence statistics.

---

## 4. Contextual Embeddings

Word2Vec gives a *static* embedding: "bank" has one vector regardless of context (river bank vs financial bank). This is a major limitation.

Modern LLMs produce **contextual embeddings**: the representation of a token depends on all surrounding tokens. The same word gets a different vector in different sentences.

This is a direct consequence of the attention mechanism (covered next week).

---

## 5. Sentence and Document Embeddings

For tasks like semantic search or clustering, we need a single vector representing an entire sentence or document.

### Mean Pooling

Average the token embeddings:

```python
sentence_embedding = token_embeddings.mean(dim=0)
```

### CLS Token (BERT-style)

Use the embedding of the special `[CLS]` token, which is designed to summarise the full input.

### Dedicated Sentence Embedding Models

- `sentence-transformers` library: fine-tuned models optimised for semantic similarity (e.g. `all-MiniLM-L6-v2`, `all-mpnet-base-v2`).
- Produce embeddings optimised for cosine similarity.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = ["The cat sat on the mat.", "A feline rested on a rug."]
embeddings = model.encode(sentences)   # shape: (2, 384)
```

---

## 6. Measuring Similarity

Given two embedding vectors u and v, the most common similarity metric is **cosine similarity**:

```
cos_sim(u, v) = (u · v) / (||u|| * ||v||)
```

- Range: -1 to 1
- 1 = identical direction (most similar)
- 0 = orthogonal (unrelated)
- -1 = opposite

```python
import torch
import torch.nn.functional as F

sim = F.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0))
```

---

## 7. Visualising Embeddings

High-dimensional embeddings (d = 768, 4096) cannot be visualised directly. We use dimensionality reduction:

### PCA

Projects to 2D by finding the axes of maximum variance. Fast, linear, interpretable.

### t-SNE

Non-linear projection. Preserves *local* structure — similar vectors cluster together. Good for visualising cluster structure.

### UMAP

Faster than t-SNE, better at preserving global structure. Widely used in modern practice.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(embeddings)

plt.scatter(reduced[:, 0], reduced[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, reduced[i])
plt.show()
```

---

## 8. Positional Encodings

The transformer has no built-in notion of order — the same set of tokens in different orders would produce the same result without position information. We add **positional encodings** to the embeddings.

### Sinusoidal Positional Encoding (original Transformer)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- Each position gets a unique pattern of sines and cosines.
- The model can generalise to sequence lengths not seen in training.

### Learned Positional Embeddings (GPT-2 style)

Simply learn a separate embedding for each position. Works well in practice but does not generalise to longer sequences.

### Rotary Positional Encoding (RoPE) — LLaMA, Mistral

Encodes position as a rotation of the query and key vectors. Better generalisation to longer contexts. This is the modern standard.

---

## 9. Practical This Week

See `practicals/week03_practical.py`:
- Load a sentence transformer model and compute embeddings for a set of sentences.
- Visualise the embedding space using t-SNE and UMAP.
- Compute cosine similarity between sentence pairs and rank by similarity.
- Implement sinusoidal positional encoding from scratch and visualise the patterns.

---

## 10. Further Reading

- Mikolov et al. (2013) — "Efficient Estimation of Word Representations in Vector Space" — https://arxiv.org/abs/1301.3781
- Devlin et al. (2019) — "BERT" — https://arxiv.org/abs/1810.04805
- Su et al. (2021) — "RoFormer: Enhanced Transformer with Rotary Position Embedding" — https://arxiv.org/abs/2104.09864
- [Sentence Transformers library](https://www.sbert.net/)
- [3Blue1Brown embedding video](https://www.youtube.com/watch?v=wjZofJX0v4M)

---

## Discussion Questions

1. Explain why Word2Vec embeddings satisfy the analogy relation `king - man + woman ≈ queen`.
2. Why are contextual embeddings more powerful than static embeddings for NLP tasks?
3. You have a dataset of 10,000 scientific paper abstracts. Describe a pipeline using sentence embeddings to find the 5 most similar papers to a new abstract.
