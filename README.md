# MSc Course: Large Language Models — Internals and Applications

**University of York — Department of Computer Science**

> A 12-week masters-level course on the internals of large language models and their real-world applications. The course combines rigorous theory with hands-on Python practicals, culminating in a group project.

---

## Course Overview

This course takes students from first principles through to deploying LLM-powered applications. The first half focuses on *internals* (tokenisation, embeddings, attention, transformers, pre-training, fine-tuning). The second half focuses on *applications* (RAG, agents, tool use, evaluation, safety, healthcare AI, multimodal models).

Each week has:
- A **lecture note** (detailed Markdown in `materials/`)
- A **practical script** (Python in `practicals/`)

---

## Directory Structure

```
teaching_llm_applications/
├── README.md                        ← this file
├── requirements.txt                 ← Python dependencies
├── materials/
│   ├── week01_intro_and_history.md
│   ├── week02_tokenisation.md
│   ├── week03_embeddings.md
│   ├── week04_attention.md
│   ├── week05_transformer_architecture.md
│   ├── week06_pretraining_and_scaling.md
│   ├── week07_finetuning_and_rlhf.md
│   ├── week08_prompting_and_context_engineering.md
│   ├── week09_rag_and_retrieval.md
│   ├── week10_agents_and_tool_use.md
│   ├── week11_evaluation_safety_ethics.md
│   └── week12_applications_and_project.md
└── practicals/
    ├── week01_practical.py
    ├── week02_practical.py
    ├── week03_practical.py
    ├── week04_practical.py
    ├── week05_practical.py
    ├── week06_practical.py
    ├── week07_practical.py
    ├── week08_practical.py
    ├── week09_practical.py
    ├── week10_practical.py
    ├── week11_practical.py
    └── week12_practical.py
```

---

## Week-by-Week Plan

| Week | Topic | Lecture Notes | Practical |
|------|-------|---------------|-----------|
| 1 | Introduction and History of LLMs | [materials/week01_intro_and_history.md](materials/week01_intro_and_history.md) | [practicals/week01_practical.py](practicals/week01_practical.py) |
| 2 | Tokenisation | [materials/week02_tokenisation.md](materials/week02_tokenisation.md) | [practicals/week02_practical.py](practicals/week02_practical.py) |
| 3 | Embeddings and Representations | [materials/week03_embeddings.md](materials/week03_embeddings.md) | [practicals/week03_practical.py](practicals/week03_practical.py) |
| 4 | The Attention Mechanism | [materials/week04_attention.md](materials/week04_attention.md) | [practicals/week04_practical.py](practicals/week04_practical.py) |
| 5 | Transformer Architecture | [materials/week05_transformer_architecture.md](materials/week05_transformer_architecture.md) | [practicals/week05_practical.py](practicals/week05_practical.py) |
| 6 | Pre-training and Scaling Laws | [materials/week06_pretraining_and_scaling.md](materials/week06_pretraining_and_scaling.md) | [practicals/week06_practical.py](practicals/week06_practical.py) |
| 7 | Fine-tuning and RLHF | [materials/week07_finetuning_and_rlhf.md](materials/week07_finetuning_and_rlhf.md) | [practicals/week07_practical.py](practicals/week07_practical.py) |
| 8 | Prompting and Context Engineering | [materials/week08_prompting_and_context_engineering.md](materials/week08_prompting_and_context_engineering.md) | [practicals/week08_practical.py](practicals/week08_practical.py) |
| 9 | Retrieval-Augmented Generation (RAG) | [materials/week09_rag_and_retrieval.md](materials/week09_rag_and_retrieval.md) | [practicals/week09_practical.py](practicals/week09_practical.py) |
| 10 | Agents and Tool Use | [materials/week10_agents_and_tool_use.md](materials/week10_agents_and_tool_use.md) | [practicals/week10_practical.py](practicals/week10_practical.py) |
| 11 | Evaluation, Safety, and Ethics | [materials/week11_evaluation_safety_ethics.md](materials/week11_evaluation_safety_ethics.md) | [practicals/week11_practical.py](practicals/week11_practical.py) |
| 12 | Applications and Final Project | [materials/week12_applications_and_project.md](materials/week12_applications_and_project.md) | [practicals/week12_practical.py](practicals/week12_practical.py) |

---

## Lecture Notes

### Part 1 — Internals

- **Week 1** — [Introduction and History](materials/week01_intro_and_history.md)
  What are LLMs? A brief history from n-grams through Word2Vec to the Transformer era. Emergent abilities and in-context learning.

- **Week 2** — [Tokenisation](materials/week02_tokenisation.md)
  Byte-pair encoding (BPE), WordPiece, SentencePiece, tiktoken. Special tokens, vocabulary size trade-offs, tokenisation artefacts.

- **Week 3** — [Embeddings and Representations](materials/week03_embeddings.md)
  Token embeddings, contextual vs static embeddings, sentence embeddings, cosine similarity, PCA / t-SNE / UMAP, sinusoidal and RoPE positional encodings.

- **Week 4** — [The Attention Mechanism](materials/week04_attention.md)
  Scaled dot-product attention (Q, K, V), causal masking, multi-head attention, cross-attention, FlashAttention, visualising attention weights.

- **Week 5** — [Transformer Architecture](materials/week05_transformer_architecture.md)
  Full decoder-only architecture. LayerNorm, residual connections, feed-forward networks, weight tying, KV cache, encoder-only and encoder-decoder variants. Scaling model size.

- **Week 6** — [Pre-training and Scaling Laws](materials/week06_pretraining_and_scaling.md)
  Pre-training data pipelines, next-token prediction objective, training infrastructure. Kaplan and Chinchilla scaling laws. Emergent abilities. Training instabilities.

- **Week 7** — [Fine-tuning and RLHF](materials/week07_finetuning_and_rlhf.md)
  Supervised fine-tuning (SFT), instruction tuning, LoRA, QLoRA, RLHF (reward model, PPO), Constitutional AI, Direct Preference Optimisation (DPO).

### Part 2 — Applications

- **Week 8** — [Prompting and Context Engineering](materials/week08_prompting_and_context_engineering.md)
  Zero-shot, few-shot, chain-of-thought, self-consistency. Prompt patterns (role, format, decomposition, step-back). Structured output. Prompt injection and security.

- **Week 9** — [Retrieval-Augmented Generation (RAG)](materials/week09_rag_and_retrieval.md)
  RAG architecture (indexing, retrieval, generation). Chunking strategies. Embedding models for retrieval. Vector stores (FAISS, ChromaDB). Hybrid retrieval and re-ranking. Advanced patterns: HyDE, multi-query, Self-RAG. Evaluation with RAGAS.

- **Week 10** — [Agents and Tool Use](materials/week10_agents_and_tool_use.md)
  LLM agents, the ReAct framework, tool/function calling, agent memory and planning. Multi-agent systems (LangGraph, AutoGen, CrewAI, smolagents). Agent evaluation and safety.

- **Week 11** — [Evaluation, Safety, and Ethics](materials/week11_evaluation_safety_ethics.md)
  Automatic metrics (BLEU, ROUGE, BERTScore). Benchmarks (MMLU, HumanEval, GSM8K, TruthfulQA). LLM-as-judge. Hallucination types and mitigations. Bias and fairness. AI safety and alignment (RLHF, Constitutional AI). Responsible deployment checklist.

- **Week 12** — [Applications and Final Project](materials/week12_applications_and_project.md)
  LLMs in healthcare, science, education, and software engineering. Multimodal models. Production architecture. Final project guidelines, assessment rubric, and project ideas.

---

## Practicals

All practicals are self-contained Python scripts with inline comments. Run with:

```bash
python practicals/weekNN_practical.py
```

| Practical | What you build |
|-----------|----------------|
| [Week 1](practicals/week01_practical.py) | Query GPT-2, inspect next-token probabilities, visualise top-k predictions, compare temperatures, compute perplexity |
| [Week 2](practicals/week02_practical.py) | Compare tokenisers (tiktoken, BERT, GPT-2 HF), train BPE from scratch, visualise token boundaries, estimate API cost |
| [Week 3](practicals/week03_practical.py) | Sentence embeddings, cosine similarity matrix, semantic search, PCA / t-SNE / UMAP visualisation, sinusoidal positional encoding |
| [Week 4](practicals/week04_practical.py) | Scaled dot-product attention (NumPy), causal masked attention, multi-head attention (PyTorch), extract real BERT attention weights |
| [Week 5](practicals/week05_practical.py) | Complete decoder-only Transformer from scratch, train on Shakespeare, plot loss curves, generate text at multiple temperatures |
| [Week 6](practicals/week06_practical.py) | Scaling law experiment: train models of 5 different sizes, plot loss vs parameters and vs FLOPs, Chinchilla analysis |
| [Week 7](practicals/week07_practical.py) | LoRA from scratch, fine-tune a toy LM, Bradley-Terry reward model, DPO loss demonstration |
| [Week 8](practicals/week08_practical.py) | Zero-shot vs few-shot vs CoT comparison, self-consistency decoding, structured extraction, prompt injection demo |
| [Week 9](practicals/week09_practical.py) | Full RAG pipeline (chunk → embed → FAISS index → retrieve → generate), BM25 hybrid retrieval, faithfulness evaluation |
| [Week 10](practicals/week10_practical.py) | ReAct agent from scratch, tool calling loop, two-agent orchestrator-specialist system, success rate evaluation |
| [Week 11](practicals/week11_practical.py) | ROUGE / BERTScore / LLM-as-judge evaluation, hallucination rate measurement, gender bias probing, jailbreak attempts |
| [Week 12](practicals/week12_practical.py) | End-to-end LLM application (RAG + agent + Streamlit UI), systematic evaluation on 20 queries |

---

## Learning Outcomes

By the end of this course students will be able to:

1. Explain the architecture and training objectives of modern LLMs from first principles.
2. Implement core components (tokenisers, attention, transformer blocks) in PyTorch.
3. Fine-tune and adapt pre-trained models using PEFT/LoRA.
4. Design and evaluate prompting strategies including chain-of-thought and few-shot learning.
5. Build a retrieval-augmented generation (RAG) pipeline.
6. Construct simple LLM-powered agents with tool use.
7. Evaluate LLM outputs for accuracy, bias, and safety.
8. Deploy LLM applications in healthcare, science, and other domains.

---

## Assessment

| Component | Weight | Details |
|-----------|--------|---------|
| Weekly practicals (submitted as scripts/notebooks) | 30% | 12 short submissions |
| Mid-term written assignment (Week 6) | 20% | 1500-word essay on scaling laws or fine-tuning |
| Final group project | 50% | Working LLM application + 10-page report + demo |

---

## Prerequisites

- Python (intermediate level)
- Basic probability and linear algebra
- Some familiarity with machine learning (e.g. logistic regression, neural networks)

No prior NLP experience required.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=<your-openai-key>
ANTHROPIC_API_KEY=<your-anthropic-key>
HF_TOKEN=<your-huggingface-token>
```

Run any practical:

```bash
python practicals/week01_practical.py
```

---

## External Resources

- [FT Generative AI visual explainer](https://ig.ft.com/generative-ai/)
- [Cambridge AI for Science LLM notes](https://docs.science.ai.cam.ac.uk/large-language-models/Introduction/Introduction/)
- [Andrej Karpathy — Build GPT-2 from scratch (video)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [3Blue1Brown deep learning playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Illustrated Transformer (Jalammar)](https://jalammar.github.io/illustrated-transformer/)
- [Hugging Face documentation](https://huggingface.co/docs)
- [LangChain documentation](https://python.langchain.com/)
- [Stanford CME-295 cheatsheet](https://github.com/afshinea/stanford-cme-295-transformers-large-language-models/blob/main/en/cheatsheet-transformers-large-language-models.pdf)
- [RAGAS evaluation framework](https://github.com/explodinggradients/ragas)
- [AI Safety teaching resources](https://github.com/neelsoumya/AI_safety_teaching_resources)

### Key Papers

| Paper | Link |
|-------|------|
| Vaswani et al. (2017) — Attention is All You Need | https://arxiv.org/abs/1706.03762 |
| Brown et al. (2020) — GPT-3 | https://arxiv.org/abs/2005.14165 |
| Kaplan et al. (2020) — Scaling Laws | https://arxiv.org/abs/2001.08361 |
| Hoffmann et al. (2022) — Chinchilla | https://arxiv.org/abs/2203.15556 |
| Wei et al. (2022) — Emergent Abilities | https://arxiv.org/abs/2206.07682 |
| Wei et al. (2022) — Chain-of-Thought | https://arxiv.org/abs/2201.11903 |
| Hu et al. (2022) — LoRA | https://arxiv.org/abs/2106.09685 |
| Ouyang et al. (2022) — InstructGPT / RLHF | https://arxiv.org/abs/2203.02155 |
| Rafailov et al. (2023) — DPO | https://arxiv.org/abs/2305.18290 |
| Lewis et al. (2020) — RAG | https://arxiv.org/abs/2005.11401 |
| Yao et al. (2022) — ReAct | https://arxiv.org/abs/2210.03629 |
| Dettmers et al. (2023) — QLoRA | https://arxiv.org/abs/2305.14314 |

---

## Contact

**Soumya Banerjee**
- York: soumya.banerjee@york.ac.uk
- Cambridge: sb2333@cam.ac.uk
