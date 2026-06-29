# Week 1 — Introduction and History of Large Language Models

## Lecture Overview

This first lecture sets the scene. We ask: *what is a language model, why has it become so powerful, and how did we get here?*


A fantastic narrative arc for an introductory lecture. Framing the evolution of LLMs as a transition from the rigid, rules-based world of Good Old-Fashioned AI (GOFAI) to the fluid, pattern-matching world of deep learning gives students the perfect historical and technical context.

Just a quick, gentle correction on the vector math before you present it: the famous analogy formula is actually $\vec{w}_{king} - \vec{w}_{man} + \vec{w}_{woman} \approx \vec{w}_{queen}$, but you have the exact right idea!



---

## 2-Hour Lecture Timeline

| Time (Mins) | Topic | Key Concepts |
| --- | --- | --- |
| 00 - 15 | **The Limits of Symbolic AI** | GOFAI, syntax vs. semantics, the scaling problem. |
| 15 - 30 | **The Deep Learning Revolution** | Connectionism, massive text associations, learning without explicit rules. |
| 30 - 60 | **Embeddings & Vector Spaces** | 3Blue1Brown visual concepts, high-dimensional arrays. |
| 60 - 80 | **Semantic Mathematics** | Vector arithmetic, how meaning is encoded in distance. |
| 80 - 120 | **Python Practical Session** | Hands-on with word vectors using Gensim. |

---

## Detailed Lesson Plan

### Part 1: The Limits of Symbolic AI (15 Mins)

* **The Premise:** Start by explaining that early AI researchers believed language was purely symbolic. They thought if they could just hardcode enough grammar rules and dictionary definitions, a computer could "understand" language.
* **The Roadblock:** Explain why this failed. Human language is messy, highly contextual, and full of idioms, sarcasm, and evolving slang.
* **The Takeaway:** Hardcoding syntactic rules simply did not scale. You cannot write a rule for every conceivable way a human might phrase a thought.

### Part 2: The Deep Learning Revolution (15 Mins)

* **The Paradigm Shift:** Introduce the pivot from *instructing* computers on the rules of language to *showing* them vast amounts of text and letting them figure out the associations themselves.
* **Context is Everything:** Explain the foundational linguistic theory by J.R. Firth: *"You shall know a word by the company it keeps."* Deep learning models look at a word and learn its meaning based on the words that frequently surround it.

- 🤔 🎥 [Video by 3blue1brown on how word vectors encode meaning](https://www.youtube.com/shorts/FJtFZwbvkI4)

- 🤔 🎥 [Video by 3blue1brown on how transformers and GPT works](https://www.youtube.com/watch?v=wjZofJX0v4M)

- [basic introduction to unsupervised machine learning](https://cambiotraining.github.io/ml-unsupervised/)

- Bag of words model: how that fails to capture _context_. 

- _Concept_ 🧩 🚀 Context is everything! Bag of words model cannot capture context

- Enter LLMs

- transformers model long-range dependencies in text

- Reading: Attention is all you need



### Part 3: Embeddings & Shared Vector Spaces (30 Mins)

* **Visualizing the Math:** This is where you introduce the **3Blue1Brown** concepts. Highly recommend pointing your students to Grant Sanderson's video: *"But what is a GPT? Visual intro to transformers."*
* **What is an Embedding?** Explain that models don't read text; they read numbers. An embedding is a high-dimensional vector (a list of numbers) that represents a word.
* **The Spatial Metaphor:** Ask students to imagine a 3D space. If we map words as coordinates, words with similar meanings (like "dog" and "puppy") will be clustered close together. Now, ask them to scale that imagination up to 10,000 dimensions. That is the shared vector space.

### Part 4: Semantic Mathematics (20 Mins)

* **Encoding Meaning:** Explain that the distance and direction between vectors encode actual semantics.
* **The Crown Jewel Example:** Introduce the famous vector equation. If you take the vector for "King", subtract the vector for "Man" (removing the male concept), and add the vector for "Woman" (adding the female concept), you land on a coordinate in the vector space that is astonishingly close to the word "Queen".
* **Formula:** $\vec{w}_{king} - \vec{w}_{man} + \vec{w}_{woman} \approx \vec{w}_{queen}$


* **Other Examples:** Mention that this works for grammar (e.g., $\vec{w}_{walking} - \vec{w}_{walk} + \vec{w}_{swim} \approx \vec{w}_{swimming}$) and geography (e.g., $\vec{w}_{Paris} - \vec{w}_{France} + \vec{w}_{Italy} \approx \vec{w}_{Rome}$).

---

## Python Practical: Exploring Vector Spaces (40 Mins)

For the practical, we will use `gensim`, a robust Python library for topic modeling and document similarity, to download a pre-trained set of GloVe (Global Vectors for Word Representation) embeddings.

> **Prerequisites:** Have your students install the library by running `pip install gensim` in their terminal or Jupyter Notebook.

### 🎮 🛠️ Practical: The Code

```python
import gensim.downloader as api

print("Downloading/Loading the word vector model... (This may take a minute or two)")
# We use a relatively small 50-dimensional model for speed in a classroom setting
model = api.load("glove-wiki-gigaword-50")
print("Model loaded successfully!\n")

# --- EXPERIMENT 1: Finding Similar Words ---
print("--- Experiment 1: Nearest Neighbors ---")
word = "computer"
print(f"Words most similar to '{word}':")
for similar_word, similarity_score in model.most_similar(word, topn=5):
    # Formatting the score to 2 decimal places using standard Python formatting
    print(f"- {similar_word} (Score: {similarity_score:.2f})")
print("\n")


# --- EXPERIMENT 2: Semantic Mathematics ---
# King - Man + Woman = ?
print("--- Experiment 2: Vector Arithmetic ---")
print("Equation: King - Man + Woman = ?")

# In Gensim, positive=[additions], negative=[subtractions]
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)

print(f"Result: {result[0][0]} (Confidence: {result[0][1]:.2f})\n")


# --- EXPERIMENT 3: The Odd One Out ---
print("--- Experiment 3: Finding the Outlier ---")
word_list = ["breakfast", "cereal", "dinner", "lunch", "car"]
print(f"List: {word_list}")

outlier = model.doesnt_match(word_list)
print(f"The model thinks the odd one out is: '{outlier}'")

```

### How to Guide the Practical

1. **Run the Basics:** Have students run the script exactly as written so they can see the magic happen instantly.
2. **Experimentation:** Encourage them to change the words in **Experiment 1**. Have them look up slang or complex verbs to see how the model groups them.
3. **Break the Math:** Have them try to come up with their own semantic equations in **Experiment 2** (e.g., `doctor - human + dog = vet`). Warn them that it doesn't *always* work perfectly, which opens up a great discussion on the biases and limitations of training data!

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

- 🧩 🚀 A **large** language model is simply one trained at scale — billions of parameters on trillions of tokens.

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


## Intro

- [Stanford CS336 class](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

- Design decisions

- Loss function

- Optimizer

- Initialization scale

- Learning rate schedule

- Regularization

- Batch size

- Assignment: BPE encoding and create model while doing _Resource accounting_

- `DGX B200`

- Operator fusion, flash attention

- `Minimize data movement` principle

- Shard memory: split data o split models, layers, sequences, etc.



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
