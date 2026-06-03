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

## Structure

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

| Week | Topic | Theme |
|------|-------|-------|
| 1 | Introduction and History | What are LLMs and why do they work? |
| 2 | Tokenisation | How text becomes numbers |
| 3 | Embeddings and Representations | Meaning in vector space |
| 4 | The Attention Mechanism | Scaled dot-product and multi-head attention |
| 5 | Transformer Architecture | Full encoder-decoder and decoder-only models |
| 6 | Pre-training and Scaling Laws | How LLMs are trained; emergent abilities |
| 7 | Fine-tuning and RLHF | Adapting LLMs; PEFT; LoRA; instruction tuning |
| 8 | Prompting and Context Engineering | Prompt patterns; few-shot; chain-of-thought |
| 9 | Retrieval-Augmented Generation (RAG) | Grounding LLMs in external knowledge |
| 10 | Agents and Tool Use | ReAct; function calling; multi-agent systems |
| 11 | Evaluation, Safety, and Ethics | Benchmarks; hallucinations; bias; responsible AI |
| 12 | Applications and Project | Healthcare AI; science; project presentations |

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
| Weekly practicals (submitted as notebooks/scripts) | 30% | 12 short submissions |
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
# Clone or download the repo, then:
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=<your-key>
HF_TOKEN=<your-huggingface-token>
```

Run any practical:

```bash
python practicals/week01_practical.py
```

---

## Resources

- [FT Generative AI visual explainer](https://ig.ft.com/generative-ai/)
- [Cambridge AI for Science LLM notes](https://docs.science.ai.cam.ac.uk/large-language-models/Introduction/Introduction/)
- [Andrej Karpathy — Build GPT-2 from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [3Blue1Brown deep learning playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Illustrated Transformer (Jalammar)](https://jalammar.github.io/illustrated-transformer/)
- [Hugging Face documentation](https://huggingface.co/docs)
- [LangChain documentation](https://python.langchain.com/)
- [Stanford CME-295 cheatsheet](https://github.com/afshinea/stanford-cme-295-transformers-large-language-models/blob/main/en/cheatsheet-transformers-large-language-models.pdf)

---

## Contact

**Soumya Banerjee**
- York: soumya.banerjee@york.ac.uk
- Cambridge: sb2333@cam.ac.uk
