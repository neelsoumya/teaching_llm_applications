# Week 1 — Introduction and History of Large Language Models

## Lecture Overview

This first lecture sets the scene. We ask: *what is a language model, why has it become so powerful, and how did we get here?*

---

## 1. What Is a Language Model?

A **language model** assigns a probability to a sequence of tokens (words, subwords, characters):

```
P(w_1, w_2, ..., w_n)
```

Equivalently, using the chain rule:

```
P(w_1, ..., w_n) = P(w_1) * P(w_2 | w_1) * P(w_3 | w_1, w_2) * ...
```

This is the **autoregressive** view: each token is predicted given all previous tokens.

A **large** language model is simply one trained at scale — billions of parameters on trillions of tokens.

---

## 2. A Brief History

### 2.1 Statistical Language Models (1990s–2000s)

- **n-gram models**: estimate P(w_t | w_{t-n+1}, ..., w_{t-1}) using counts.
- Advantages: simple, interpretable, fast.
- Disadvantages: poor generalisation; the curse of dimensionality; no notion of meaning.

### 2.2 Neural Language Models (2003–2012)

- Bengio et al. (2003): first neural language model using feed-forward networks and learned word embeddings.
- Word2Vec (Mikolov et al., 2013): embeddings capture semantic relationships (king - man + woman ≈ queen).
- Recurrent Neural Networks (RNNs) and LSTMs: process sequences step by step, maintaining hidden state.
- **Key limitation of RNNs**: vanishing gradients; sequential computation cannot be parallelised.

### 2.3 The Attention Revolution (2014–2017)

- Bahdanau et al. (2015): *attention mechanism* for neural machine translation.
- Instead of compressing the entire input into a single vector, attend to all positions simultaneously.
- Vaswani et al. (2017): **"Attention is All You Need"** — the Transformer.
  - Replaces recurrence entirely with multi-head self-attention.
  - Enables massively parallel training.

### 2.4 The Pre-training Era (2018–present)

| Year | Model | Organisation | Parameters |
|------|-------|--------------|------------|
| 2018 | GPT-1 | OpenAI | 117M |
| 2018 | BERT | Google | 340M |
| 2019 | GPT-2 | OpenAI | 1.5B |
| 2020 | GPT-3 | OpenAI | 175B |
| 2022 | ChatGPT / InstructGPT | OpenAI | ~175B |
| 2023 | LLaMA-2 | Meta | 7B–70B |
| 2024 | Llama-3, Mistral, Gemini, Claude | Meta/Mistral/Google/Anthropic | 7B–1T+ |

The core recipe: **pre-train on massive text corpora** (self-supervised), then **fine-tune or prompt** for downstream tasks.

---

## 3. Why Do LLMs Work?

### 3.1 The Distributional Hypothesis

*Words that appear in similar contexts tend to have similar meanings.* (Harris, 1954; Firth, 1957)

Training on next-token prediction forces the model to learn:
- Syntax and grammar
- World knowledge
- Reasoning patterns (implicitly)
- Style, tone, and register

### 3.2 Emergent Abilities

At sufficient scale, LLMs develop **emergent abilities** not present in smaller models:
- Multi-step arithmetic
- Chain-of-thought reasoning
- In-context learning (few-shot)
- Code generation

Wei et al. (2022) document many such emergent abilities as a function of model size.

### 3.3 In-Context Learning

One of the most surprising properties: LLMs can perform new tasks **just from examples in the prompt**, without updating weights. This is called *in-context learning* (ICL).

---

## 4. Common Applications

| Domain | Example Application |
|--------|---------------------|
| Text generation | Creative writing, summarisation |
| Code | GitHub Copilot, code review |
| Q&A / search | RAG-based assistants |
| Healthcare | Clinical note summarisation, diagnostic support |
| Science | Hypothesis generation, literature review |
| Education | Tutoring, automated feedback |
| Agents | Web browsing, task automation |

---

## 5. Key Concepts to Remember

- **Token**: the basic unit processed by an LLM (not always a word).
- **Context window**: the maximum number of tokens the model can process at once.
- **Parameters**: the weights of the neural network, learned during training.
- **Pre-training**: unsupervised training on large text corpora.
- **Fine-tuning**: adapting a pre-trained model on a smaller, task-specific dataset.
- **Prompt**: the input text given to the model at inference time.

---

## 6. The Transformer at a Glance

(We cover this in depth in Weeks 4–5; here is a high-level sketch.)

```
Input tokens
    │
    ▼
[Token Embedding + Positional Encoding]
    │
    ▼
[Transformer Block × N]
  ┌──────────────────────────────────┐
  │  Multi-Head Self-Attention       │
  │  + Add & Norm                    │
  │  Feed-Forward Network            │
  │  + Add & Norm                    │
  └──────────────────────────────────┘
    │
    ▼
[Linear + Softmax → Probability over vocabulary]
```

---

## 7. Practical This Week

See `practicals/week01_practical.py`:
- Query an open-source LLM via the Hugging Face API.
- Inspect the raw token probabilities for a short prompt.
- Visualise which token the model predicts next.

---

## 8. Further Reading

- Vaswani et al. (2017) — "Attention is All You Need" — https://arxiv.org/abs/1706.3762
- Brown et al. (2020) — "Language Models are Few-Shot Learners" (GPT-3) — https://arxiv.org/abs/2005.14165
- Wei et al. (2022) — "Emergent Abilities of Large Language Models" — https://arxiv.org/abs/2206.07682
- [FT Generative AI visual explainer](https://ig.ft.com/generative-ai/)
- [Cambridge AI notes: Introduction to LLMs](https://docs.science.ai.cam.ac.uk/large-language-models/Introduction/Introduction/)

---

## Discussion Questions

1. What are the limitations of n-gram models that motivated neural language models?
2. What does "emergent" mean in the context of LLMs, and why is it surprising?
3. Give two examples of tasks you might want to apply an LLM to in your own research domain.
