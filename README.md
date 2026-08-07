# MSc Course: Large Language Models — Internals and Applications


> An 18-week masters-level course on the internals of large language models and their real-world applications. The course combines rigorous theory with hands-on Python practicals, culminating in a group project.

---

## Course Overview

This course takes students from first principles through to deploying LLM-powered applications. The first half focuses on *internals* (tokenisation, embeddings, attention, transformers, pre-training, fine-tuning). The second half focuses on *applications* (RAG, agents, tool use, evaluation, safety, healthcare AI, multimodal models). Week 13 provides a dedicated deep dive into Reinforcement Learning from Human Feedback (RLHF), covering the full mathematical derivation, PPO, DPO, reward hacking, Constitutional AI, and process reward models. Week 14 covers mechanistic interpretability (superposition, sparse autoencoders, circuits, the logit lens, grokking, knowledge editing). Week 15 is a dedicated deep dive into evaluation (evals): the full evaluation stack from automatic metrics through human evaluation, LLM-as-judge, benchmark design, red-teaming, safety evals, and evaluation for deployed LLM applications. Week 16 provides a deep dive into Direct Preference Optimisation (DPO): the full mathematical derivation from the RLHF objective, the gradient interpretation, practical implementation, failure modes (likelihood displacement, over-optimisation, distribution shift), and a survey of variants including IPO, KTO, SimPO, and ORPO. Week 17 covers efficiency in attention and alternative architectures: the quadratic cost of full attention, sparse attention patterns (sliding window, BigBird), FlashAttention (tiling and online softmax), linear attention approximations (Performer, RWKV), state space models (S4, Mamba), grouped-query attention, and hybrid Mamba/transformer architectures. Week 18 is a short, practical primer on reinforcement learning — just enough RL intuition (policies, rewards, REINFORCE, baselines) to understand why PPO, DPO, and GRPO are shaped the way they are.

Each week has:
- A **lecture note** (detailed Markdown in `materials/`)
- A **practical script** (Python in `practicals/`)

---

## Material

- [Introduction](materials/week01_intro_and_history.md)

- [Transformers fundamentals](materials/transformers.md)

- [Week 5 Transformers Architecture](materials/week05_transformer_architecture.md)

- [Masked language models (BERT vs. GPT)](materials/masked_language_modelling_vs_generative.md)

- [Stanford CS336 my own notes](materials/CS336_Stanford.md)

- [Practicals](materials/practicals.md)

- [LLMs from scratch](materials/LLMs_from_scratch.md)

- [BERT SLMs](materials/BERT_SLMs.md)

- [Finetuning](materials/week07_finetuning_and_rlhf.md)

- [Costs](materials/costs.md)

- [Scaling](materials/scaling.md)

- [LoRA](materials/LORA.md)

- [Energy](materials/energy.md)

- [Explainable AI in the context of LLMs and mechanistic interpretability + reading](materials/explainableAI.md)

- [Guest lectures](materials/guest_lectures.md)

- [Things to cover](materials/things_to_cover.md)

- [K-V Cache](materials/kvcache.md)

- [Paged attention and virtual LLM (vLLM)](materials/paged_attention.md)

- [Prefill](materials/prefill.md)

- [Flash attention](materials/flash_attention.md)


- [Practicals using `nanoGPT`](materials/practicals.md)

- [Practicals using Baby steps paper by Frank](materials/practicals_babysteps.md)

- [Quantization](materials/quantization.md)

- [Resource accounting](materials/resource_accounting.md)

- [Architectures](materials/architectures.md)

- [Softmax](materials/softmax.md)

- More advanced topics

- [Reasoning models and GRPO](materials/reasoning_models_GRPO.md)

- [Different kinds of attention](materials/different_kinds_attention.md)

- [📝 Sarvam](materials/sarvam.md)

## Installation

```bash
pip install -r requirements.txt
```

## Declaration

>Parts of this course material are generated using generative AI. I take full responsibility for all content and have verified it. Lots of the text and material are also heavily inspired by other sources which are all detailed in this document. I take and make no claims to originality. These are just to be used as teaching notes.

## Acknowledgements and courses for inspiration

- [Jay Alamar Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

- [Stanford CS336 Language Modelling from Scratch](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) and [course webpage](https://cs336.stanford.edu/)

- [deeplearning.ai course on transformers by Jay Alammar and Maarten Grootendorst](https://learn.deeplearning.ai/courses/how-transformer-llms-work)

- [Lecture notes by Francois Chollet](https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer/)

- [🤗 Huggingface resources on transformers](https://huggingface.co/learn/llm-course/chapter1/4)

- [Stanford CME295](https://cme295.stanford.edu/) and [on youtube](https://www.youtube.com/watch?v=k5Fh-UgTuCo)

- Ilya Sutskever and Andrej Karpathy video on how to code transformers from scratch

- `NanoGPT`

- Vizuara videos coding transformers from scratch

- DAMTP and CAISH courses math department cambridge 2025

- Bluedot impact courses


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
│   ├── week12_applications_and_project.md
│   ├── week13_rlhf_deep_dive.md
│   ├── week14_mechanistic_interpretability.md
│   ├── week15_evals.md
│   ├── week16_dpo.md
│   ├── week17_efficiency_attention.md
│   └── week18_intro_to_rl.md
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
    ├── week12_practical.py
    ├── week13_practical.py
    ├── week14_practical.py
    ├── week15_practical.py
    ├── week16_practical.py
    ├── week17_practical.py
    └── week18_practical.py
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
| 7 | Fine-tuning and RLHF (overview) | [materials/week07_finetuning_and_rlhf.md](materials/week07_finetuning_and_rlhf.md) | [practicals/week07_practical.py](practicals/week07_practical.py) |
| 8 | Prompting and Context Engineering | [materials/week08_prompting_and_context_engineering.md](materials/week08_prompting_and_context_engineering.md) | [practicals/week08_practical.py](practicals/week08_practical.py) |
| 9 | Retrieval-Augmented Generation (RAG) | [materials/week09_rag_and_retrieval.md](materials/week09_rag_and_retrieval.md) | [practicals/week09_practical.py](practicals/week09_practical.py) |
| 10 | Agents and Tool Use | [materials/week10_agents_and_tool_use.md](materials/week10_agents_and_tool_use.md) | [practicals/week10_practical.py](practicals/week10_practical.py) |
| 11 | Evaluation, Safety, and Ethics | [materials/week11_evaluation_safety_ethics.md](materials/week11_evaluation_safety_ethics.md) | [practicals/week11_practical.py](practicals/week11_practical.py) |
| 12 | Applications and Final Project | [materials/week12_applications_and_project.md](materials/week12_applications_and_project.md) | [practicals/week12_practical.py](practicals/week12_practical.py) |
| 13 | **RLHF: Deep Dive** | [materials/week13_rlhf_deep_dive.md](materials/week13_rlhf_deep_dive.md) | [practicals/week13_practical.py](practicals/week13_practical.py) |
| 14 | Mechanistic Interpretability | [materials/week14_mechanistic_interpretability.md](materials/week14_mechanistic_interpretability.md) | [practicals/week14_practical.py](practicals/week14_practical.py) |
| 15 | **Evals: Evaluating LLMs** | [materials/week15_evals.md](materials/week15_evals.md) | [practicals/week15_practical.py](practicals/week15_practical.py) |
| 16 | **Direct Preference Optimisation (DPO)** | [materials/week16_dpo.md](materials/week16_dpo.md) | [practicals/week16_practical.py](practicals/week16_practical.py) |
| 17 | **Efficiency: Attention Variants & Architectures** | [materials/week17_efficiency_attention.md](materials/week17_efficiency_attention.md) | [practicals/week17_practical.py](practicals/week17_practical.py) |
| 18 | **Introduction to Reinforcement Learning** | [materials/week18_intro_to_rl.md](materials/week18_intro_to_rl.md) | [practicals/week18_practical.py](practicals/week18_practical.py) |

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

- **Week 7** — [Fine-tuning and RLHF (overview)](materials/week07_finetuning_and_rlhf.md)
  Supervised fine-tuning (SFT), instruction tuning, LoRA, QLoRA, RLHF (reward model, PPO), Constitutional AI, Direct Preference Optimisation (DPO). See Week 13 for the full mathematical treatment of RLHF.

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

### Part 3 — Advanced Topics

- **Week 13** — [RLHF: Deep Dive](materials/week13_rlhf_deep_dive.md)
  Full mathematical derivation of RLHF from first principles. Bradley-Terry preference model. Reward model architecture and training. PPO actor-critic loop applied to language models: clipped surrogate objective, advantage estimation, KL controller, value clipping. Reward hacking: definition, examples, measurement, and mitigations. DPO derivation from the KL-constrained RLHF objective. Variants and alternatives: RLOO, GRPO, iterative DPO, rejection sampling fine-tuning. Constitutional AI and RLAIF. Process reward models (PRMs) for multi-step reasoning. Annotator agreement and reward model quality. Evaluation: win rate, MT-Bench, AlpacaEval.

- **Week 14** — [Mechanistic Interpretability](materials/week14_mechanistic_interpretability.md)
  The superposition hypothesis. Sparse autoencoders (SAEs) for feature disentanglement. Circuits: induction heads, the IOI circuit, the greater-than circuit. Activation patching and causal scrubbing. The residual stream as a communication bus. The logit lens. MLP layers as key-value memories. Knowledge editing (ROME, MEMIT). Linear probing and its causal limitations. Grokking and the modular arithmetic circuit. Automated interpretability (ACDC, Bills et al.). Universality. Open problems and the TransformerLens toolkit.

- **Week 15** — [Evals: Evaluating LLMs](materials/week15_evals.md)
  The evaluation stack from automatic metrics to safety evals. BLEU, ROUGE, BERTScore, MAUVE, pass@k. Benchmark design, contamination, and saturation (MMLU, GSM8K, HumanEval, SWE-bench, GPQA). Human evaluation: pairwise preference, Chatbot Arena, ELO ratings, annotator agreement. LLM-as-judge: setup, position bias, verbosity bias, self-preference, MT-Bench, AlpacaEval. Behavioural evals and red-teaming: jailbreak taxonomy, automated red-teaming, HarmBench. Safety and alignment evals: sycophancy, deceptive alignment, TruthfulQA, WMDP, dangerous capability evaluations. Eval design: the eval checklist, statistical significance (McNemar), eval frameworks (LM Eval Harness, HELM, Inspect AI). Evaluation for deployed LLM applications: component vs end-to-end, offline vs online, implicit feedback.

- **Week 16** — [Direct Preference Optimisation (DPO)](materials/week16_dpo.md)
  Full mathematical derivation of DPO from the KL-constrained RLHF objective: the analytical optimal policy, rearranging for the implicit reward, cancellation of the partition function, the Bradley-Terry substitution, and the final DPO loss. Gradient interpretation and implicit curriculum. PyTorch implementation from scratch. Key hyperparameters (β, reference policy, data quality). Training diagnostics: chosen/rejected rewards, reward margin, preference accuracy, likelihood displacement. Failure modes: likelihood displacement mechanism, over-optimisation, distribution shift, preference noise. DPO variants: IPO (bounded margin), KTO (unpaired data), SimPO (no reference model, length normalisation), ORPO (one-stage SFT + alignment). Iterative and online DPO. DPO vs PPO decision guide. Practical tips with TRL’s DPOTrainer.

- **Week 17** — [Efficiency: Attention Variants and Alternative Architectures](materials/week17_efficiency_attention.md)
  The quadratic O(n²) cost of full self-attention: time, memory, and practical limits. Sparse attention patterns: local/window attention, strided/dilated attention, global+local (Longformer, BigBird), sliding window attention in Mistral. FlashAttention: HBM vs SRAM bottleneck, tiling, online softmax algorithm, FlashAttention-2 and -3, PyTorch integration. Linear attention approximations: the kernel trick for attention, Performer (random Fourier features), RWKV (linear recurrent form), limitations. State space models (SSMs): continuous-time formulation, ZOH discretisation, S4 (HiPPO matrix, convolutional form, O(n log n) training), Mamba (selective/input-dependent B, C, Δ), Mamba vs transformer comparison, hybrid models (Jamba, Griffin). Grouped-query attention (GQA) and multi-query attention (MQA): KV cache reduction. Efficient inference architecture: combining FlashAttention + GQA + sliding window + speculative decoding.

- **Week 18** — [Introduction to Reinforcement Learning](materials/week18_intro_to_rl.md)
  A short, practical RL primer: the agent-environment loop, policies, rewards, and returns. The core REINFORCE update rule and the intuition behind it (push up actions that worked, push down ones that didn't). Variance reduction via baselines and a one-paragraph sketch of actor-critic. A one-line summary of the conceptual path from REINFORCE to PPO and GRPO. Why single-turn RLHF behaves like a bandit problem rather than full multi-step RL.

---

## Guest Lectures

Practitioner and research perspectives complement the core weekly material. See [materials/guest_lectures.md](materials/guest_lectures.md) for full details on each speaker, talk overviews, suggested background reading, and guidance on how to prepare questions.

| Speaker | Topic | Relevant weeks |
|---------|-------|---------------|
| Cole Robertson | Speech, language, and LLMs in industry — real-time spoken dialogue, ASR + LLM pipelines, startup deployment | 1, 8, 12 |
| Glasgow Startup *(TBC)* | LLMs for real-world applications — prototyping, fine-tuning vs RAG vs prompting, evaluation in the wild | 7, 8, 9, 11, 12 |

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
| [**Week 13**](practicals/week13_practical.py) | **Full RLHF pipeline from scratch**: synthetic preference dataset, Bradley-Terry reward model, PPO actor-critic update loop, DPO training, reward hacking detection; RM score / KL divergence / win rate plots |
| [**Week 14**](practicals/week14_practical.py) | **Mechanistic interpretability**: logit lens across GPT-2 layers; activation patching heatmap (IOI heads); linear probe for verb detection per layer; sparse autoencoder trained on MLP activations with feature inspection; grokking on modular arithmetic |
| [**Week 15**](practicals/week15_practical.py) | **Evals**: BLEU/ROUGE/BERTScore/MAUVE comparison; LLM-as-judge pipeline with position-bias mitigation (swap A/B); position bias measurement; sycophancy probing; mini red-team across three jailbreak categories; factual QA eval with McNemar significance testing |
| [**Week 16**](practicals/week16_practical.py) | **DPO from scratch**: implement DPO, IPO, and SimPO losses; train a toy LM on synthetic preferences; monitor chosen/rejected rewards, margin, accuracy, and KL; demonstrate likelihood displacement; two-iteration online DPO; comparison plots across all three methods |
| [**Week 17**](practicals/week17_practical.py) | **Efficiency**: time/memory scaling of naive vs tiled (FlashAttention-style) vs sliding-window attention; S4-style SSM from scratch — verify recurrent == convolutional form via FFT; Mamba selective SSM with input-dependent Δ; KV-cache memory comparison (MHA vs GQA vs SSM) across sequence lengths up to 262k tokens; output quality analysis of sliding window at varying window sizes |
| [**Week 18**](practicals/week18_practical.py) | **REINFORCE basics**: a minimal policy-gradient agent on a toy bandit, trained with and without a baseline, showing the variance reduction directly in a plot |

---

## Learning Objectives

The objectives are organised by theme, expressed using measurable action verbs at masters level, and grounded in Bloom's revised taxonomy. They map directly to the weekly topics and assessments.

### 1. Foundations and History *(Weeks 1–2)*

By the end of Week 2, students will be able to:

- **Explain** the historical progression from n-gram language models and RNNs to the Transformer architecture, identifying the key limitations each generation addressed.
- **Define** core terminology — token, context window, parameter, pre-training, fine-tuning, prompt, perplexity — accurately and consistently.
- **Describe** the self-supervised next-token prediction objective and explain why training on it at scale gives rise to broad language competence and emergent abilities.
- **Identify** the key differences between encoder-only, encoder-decoder, and decoder-only architectures and match each to appropriate downstream tasks.
- **Implement** a byte-pair encoding (BPE) tokeniser from scratch and explain the trade-offs between vocabulary size, sequence length, and cross-lingual coverage.
- **Analyse** tokenisation artefacts — number representation, non-English text, code indentation, whitespace sensitivity — and predict their downstream effects on model behaviour.

### 2. Representations and Embeddings *(Week 3)*

By the end of Week 3, students will be able to:

- **Explain** the distributional hypothesis and how it motivates learning word and sentence representations from co-occurrence statistics in large text corpora.
- **Distinguish** between static embeddings (Word2Vec) and contextual embeddings produced by transformer models, and justify why contextual representations are preferred for most modern tasks.
- **Apply** sentence embedding models to compute semantic similarity and build a nearest-neighbour retrieval system over a document corpus.
- **Visualise** high-dimensional embedding spaces using PCA, t-SNE, and UMAP, and interpret the resulting cluster structure in terms of semantic relationships.
- **Implement** sinusoidal positional encodings from scratch and explain the motivation for rotary positional encodings (RoPE) used in modern LLMs such as LLaMA and Mistral.

### 3. The Attention Mechanism *(Week 4)*

By the end of Week 4, students will be able to:

- **Derive** the scaled dot-product attention formula from the query-key-value abstraction, explaining the role of each component in information routing.
- **Justify** the scaling factor 1/√d_k and explain both theoretically and empirically what happens when it is omitted.
- **Implement** causal masked self-attention and multi-head attention in PyTorch from first principles, without using `nn.MultiheadAttention`.
- **Explain** the O(n²) computational complexity of self-attention and describe how FlashAttention reduces memory requirements without changing the mathematical result.
- **Interpret** real attention weight patterns extracted from a pretrained BERT model, while acknowledging the established limitations of attention as a mechanistic explanation tool.

### 4. Transformer Architecture *(Week 5)*

By the end of Week 5, students will be able to:

- **Assemble** a complete decoder-only transformer from scratch in PyTorch — including token embeddings, positional encodings, multi-head attention, feed-forward layers, residual connections, and layer normalisation — and train it to generate coherent text.
- **Explain** the roles of residual connections and pre-layer normalisation in enabling stable training of networks with tens or hundreds of layers.
- **Describe** the KV cache mechanism and calculate its memory footprint for a given model configuration and context length.
- **Compare** encoder-only, encoder-decoder, and decoder-only model families across architecture, training objective, and suitability for different tasks.
- **Count** and break down the parameters of a transformer model by component (embedding, attention projections, feed-forward, layer norms) and verify against published totals.

### 5. Pre-training and Scaling Laws *(Week 6)*

By the end of Week 6, students will be able to:

- **Describe** the full pre-training data pipeline — sourcing, deduplication, quality filtering, toxicity filtering, and domain mixing — and explain why each step affects downstream model quality.
- **Articulate** both the Kaplan (2020) and Chinchilla (2022) scaling laws and explain the key difference in their prescriptions for compute-optimal training.
- **Apply** the Chinchilla relationship (N_opt ∝ C^0.5, D_opt ∝ C^0.5) to estimate optimal model size and token count for a given FLOPs budget.
- **Explain** why practitioners often deliberately over-train small models beyond the Chinchilla optimum, and analyse the deployment cost implications.
- **Empirically verify** power-law scaling by training models of multiple sizes and plotting loss against parameter count and total compute.
- **Identify** at least three causes of training instability (loss spikes, gradient explosions, bad data batches) and the standard mitigations for each.

### 6. Fine-tuning and Alignment *(Week 7)*

By the end of Week 7, students will be able to:

- **Explain** supervised fine-tuning (SFT) and instruction tuning, describe the risk of catastrophic forgetting, and state at least two mitigations.
- **Implement** LoRA (Low-Rank Adaptation) from scratch, calculate the trainable parameter reduction ratio for given dimensions and rank, and apply it to a frozen base model.
- **Compare** full fine-tuning, LoRA, QLoRA, prefix tuning, and prompt tuning across the dimensions of parameter efficiency, GPU memory requirement, and typical task performance.
- **Describe** the three stages of RLHF — SFT, reward model training, PPO optimisation — at an overview level (mathematical depth covered in Week 13).
- **Implement** the Bradley-Terry preference loss for reward model training and the DPO objective as a reward-model-free alternative to PPO.
- **Critically compare** RLHF and DPO as alignment strategies, identifying the implementation complexity, data requirements, and known failure modes of each.

### 7. Prompting and Context Engineering *(Week 8)*

By the end of Week 8, students will be able to:

- **Apply** zero-shot, few-shot, and chain-of-thought prompting strategies to a given reasoning or generation task and quantify the effect on output quality across a representative evaluation set.
- **Implement** self-consistency decoding — sample multiple reasoning chains, take the majority-vote answer — and explain why it improves accuracy on multi-step reasoning benchmarks.
- **Design** system prompts, output format instructions, and few-shot demonstrations that reliably elicit structured (JSON / XML) output from a language model.
- **Identify and apply** established prompt patterns — role/persona, output format, step-back, decomposition — to appropriate problem types.
- **Explain** the "lost in the middle" phenomenon and derive practical placement guidelines for critical information in long-context prompts.
- **Describe** prompt injection attacks, explain the mechanism by which user-supplied content can override system instructions, and implement at least two mitigations.

### 8. Retrieval-Augmented Generation *(Week 9)*

By the end of Week 9, students will be able to:

- **Design and implement** a complete RAG pipeline — document loading, chunking, embedding, vector indexing, similarity retrieval, prompt construction, and generation — for a real document corpus.
- **Select** an appropriate chunking strategy (fixed-size, sentence-level, recursive character, semantic, structure-aware) for a given document type and justify the choice in terms of retrieval precision and recall.
- **Compare** dense retrieval, sparse retrieval (BM25), and hybrid retrieval approaches, explaining the trade-offs in terms of semantic generalisation, exact-match performance, and computational cost.
- **Implement** re-ranking using a cross-encoder and explain the bi-encoder / cross-encoder precision-latency trade-off.
- **Evaluate** a RAG system using RAGAS or equivalent across: retrieval recall, context precision, faithfulness, and answer relevance.
- **Describe** advanced RAG patterns — HyDE, multi-query retrieval, parent-child chunking, Self-RAG, Corrective RAG — and identify the specific failure mode each addresses.

### 9. Agents and Tool Use *(Week 10)*

By the end of Week 10, students will be able to:

- **Implement** a ReAct-style agent loop from scratch using a real LLM API, including tool dispatch, result injection, and multi-turn context management.
- **Define** tool schemas compatible with the OpenAI or Anthropic function-calling specification and integrate at least two tools (e.g. web search, code execution) into a working agent.
- **Distinguish** between in-context memory, external vector memory, episodic memory, and procedural memory in agent architectures, and select the appropriate type for a given task requirement.
- **Design** a multi-agent system with at least two specialised subagents and an orchestrator, specifying the communication protocol, task decomposition strategy, and error recovery mechanism.
- **Evaluate** an agent on a structured task suite, reporting success rate, mean steps to completion, and failure mode analysis.
- **Identify** safety risks specific to agentic systems — irreversible real-world actions, prompt injection via tool results, scope creep — and apply appropriate mitigations including sandboxing, human-in-the-loop checkpoints, and audit logging.

### 10. Evaluation, Safety, and Ethics *(Week 11)*

By the end of Week 11, students will be able to:

- **Apply** reference-based metrics (ROUGE, BERTScore, BLEU) and reference-free metrics (perplexity, MAUVE) to evaluate generated text, and articulate the specific failure modes of each.
- **Design** an LLM-as-judge evaluation rubric for an open-ended generation task and identify its known failure modes including position bias, verbosity bias, and self-preference.
- **Measure** hallucination rate on a factual QA benchmark, categorise hallucinations by type (intrinsic, extrinsic, factual), and propose at least two concrete mitigations.
- **Probe** a language model for gender or demographic bias using templated test sets (WinoBias-style) and interpret the results in terms of representation and allocation harms.
- **Explain** Goodhart's Law in the context of LLM evaluation and provide a concrete example from the RLHF or benchmark optimisation literature.
- **Apply** a responsible deployment checklist — encompassing red-teaming, content filtering, scope definition, human escalation paths, production monitoring, and user communication of limitations — to a proposed LLM application.
- **Critically assess** the ethical implications of deploying LLMs in high-stakes domains such as healthcare, legal advice, and education, with reference to real-world case studies.

### 11. Applications and Deployment *(Week 12)*

By the end of Week 12, students will be able to:

- **Identify** the specific challenges of applying LLMs in healthcare — hallucination risk, regulatory compliance (GDPR, MHRA), clinical data sensitivity, PPIE requirements — and propose a deployment architecture that addresses them.
- **Select** an appropriate model and deployment strategy (local vs API, full fine-tuning vs RAG vs prompting) for a given application domain, latency budget, privacy constraint, and cost envelope, and justify the decision.
- **Build** a working end-to-end LLM application combining at least two of: RAG, fine-tuning, agents, and structured prompting, packaged as a runnable Streamlit or FastAPI service.
- **Evaluate** the application systematically on a representative test set of at least 20 queries and produce a written analysis of success cases, failure modes, and proposed improvements.
- **Communicate** technical design decisions, evaluation methodology, and results clearly to both technical and non-technical audiences in a written report and live demonstration.

### 12. RLHF: Deep Dive *(Week 13)*

By the end of Week 13, students will be able to:

- **Derive** the Bradley-Terry paired comparison model from first principles and explain why pairwise preferences are preferred over absolute quality scores for collecting human feedback.
- **Explain** the full three-stage RLHF pipeline — SFT, reward model training, PPO optimisation — at a mathematical level, including the role and form of the KL divergence penalty in the RL objective.
- **Implement** the Bradley-Terry reward model loss, train a reward model on a synthetic preference dataset, and measure its held-out preference prediction accuracy.
- **Describe** the PPO actor-critic update in the context of language model fine-tuning, including the clipped surrogate objective, advantage estimation with GAE, value function training, and practical stabilisation techniques (reward normalisation, KL controller, entropy bonus).
- **Implement** a simplified PPO training loop for a toy language model policy and track RM score, KL divergence from the reference policy, and true reward throughout training.
- **Derive** the DPO objective from the KL-constrained RLHF optimisation problem, explaining every algebraic step, and implement it as a standard supervised training loss.
- **Define** reward hacking precisely, give at least three concrete behavioural examples from deployed systems, and propose monitoring and mitigation strategies including RM ensembles, iterative data collection, and human evaluation checkpoints.
- **Compare** PPO, DPO, RLOO, GRPO, iterative DPO, and rejection sampling fine-tuning across data requirements, computational cost, training stability, and suitability for different task types.
- **Explain** Constitutional AI and RLAIF as approaches that reduce dependence on human preference annotation, and identify their limitations.
- **Describe** process reward models (PRMs), explain why step-level reward signals improve performance on multi-step reasoning tasks, and outline the annotation challenges they introduce.
- **Detect** reward hacking empirically by monitoring the divergence between RM score and true human preference across training iterations and interpreting the resulting plots.

### 13. Direct Preference Optimisation *(Week 16)*

By the end of Week 16, students will be able to:

- **Derive** the DPO loss from the KL-constrained RLHF objective in full algebraic detail: writing the analytical optimal policy, expressing the reward in terms of the policy, substituting into the Bradley-Terry model, and explaining why the partition function Z(x) cancels.
- **Interpret** the DPO gradient, explaining the role of the residual factor (1 − σ(ĥ)) as an implicit difficulty-weighting mechanism that down-weights pairs the model already ranks correctly.
- **Implement** the DPO loss from scratch in PyTorch, including correct computation of per-sequence log-probabilities for both the policy and a frozen reference model.
- **Monitor** DPO training using the full diagnostic suite: chosen reward, rejected reward, reward margin, preference accuracy, absolute log-probabilities, and KL divergence from the reference policy.
- **Explain** likelihood displacement — the failure mode where DPO decreases the absolute probability of the preferred completion even as the margin increases — and propose at least two mitigations.
- **Compare** DPO, IPO, KTO, SimPO, and ORPO: for each variant, identify the specific limitation of standard DPO it addresses, the mathematical change it makes, and the trade-off it introduces.
- **Implement** IPO and SimPO losses from scratch and evaluate all three methods on the same synthetic preference dataset.
- **Describe** iterative (online) DPO, explain why it addresses distribution shift, and implement a two-iteration online DPO loop showing improved reward accuracy on successive iterations.
- **Select** between DPO and PPO for a given alignment task, justifying the choice across the dimensions of engineering complexity, data requirements, stability, and task type.
- **Apply** TRL’s DPOTrainer to fine-tune a real model with LoRA, configuring the key hyperparameters (β, learning rate, max sequence length) and interpreting the logged diagnostics.

### 14. Introduction to Reinforcement Learning *(Week 18)*

By the end of Week 18, students will be able to:

- **Explain** the agent-environment loop and how RL differs from supervised learning (no fixed correct-answer labels; feedback via reward instead).
- **State** the REINFORCE update rule and explain its intuition: increasing the probability of actions that led to good outcomes, decreasing it for actions that led to poor ones.
- **Implement** REINFORCE from scratch on a toy bandit task.
- **Explain** why subtracting a baseline reduces variance without introducing bias, and demonstrate this empirically.
- **Describe**, at a high level, how actor-critic, TRPO, PPO, and GRPO each build on REINFORCE.
- **Explain** why single-turn RLHF for LLMs can be treated as a bandit problem rather than requiring full multi-step RL machinery.

---

### Mapping to Bloom's Revised Taxonomy

The objectives above span all six cognitive levels:

| Level | Verbs used | Examples from this course |
|-------|-----------|--------------------------|
| **Remember** | Define, identify, list, name | Define token, perplexity, LoRA rank, KV cache, Bradley-Terry model |
| **Understand** | Explain, describe, distinguish, summarise | Explain causal masking, Chinchilla scaling, reward hacking, KL penalty |
| **Apply** | Implement, apply, calculate, build | Implement attention, LoRA, RAG pipeline, PPO loop, DPO loss |
| **Analyse** | Analyse, compare, interpret, diagnose | Interpret attention weights, diagnose reward hacking, compare PPO vs DPO |
| **Evaluate** | Evaluate, assess, justify, critique | Assess alignment strategies, critique deployment proposals, score with LLM-as-judge |
| **Create** | Design, construct, assemble, produce | Design multi-agent system, build evaluated LLM application, implement full RLHF pipeline |

---


## Prerequisites

- Python (intermediate level)
- Basic probability and linear algebra
- Some familiarity with machine learning (e.g. logistic regression, neural networks)

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
- [TRL library (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl/index)
- [OpenRLHF — scalable RLHF framework](https://github.com/OpenRLHF/OpenRLHF)

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
| Christiano et al. (2017) — Deep RL from Human Preferences | https://arxiv.org/abs/1706.03741 |
| Ouyang et al. (2022) — InstructGPT / RLHF | https://arxiv.org/abs/2203.02155 |
| Schulman et al. (2017) — Proximal Policy Optimisation | https://arxiv.org/abs/1707.06347 |
| Bai et al. (2022) — Constitutional AI | https://arxiv.org/abs/2212.08073 |
| Rafailov et al. (2023) — DPO | https://arxiv.org/abs/2305.18290 |
| Lightman et al. (2023) — Let's Verify Step by Step (PRMs) | https://arxiv.org/abs/2305.20050 |
| Skalse et al. (2022) — Defining and Characterizing Reward Hacking | https://arxiv.org/abs/2209.13085 |
| Lewis et al. (2020) — RAG | https://arxiv.org/abs/2005.11401 |
| Yao et al. (2022) — ReAct | https://arxiv.org/abs/2210.03629 |
| Dettmers et al. (2023) — QLoRA | https://arxiv.org/abs/2305.14314 |
| DeepSeek-R1 (2025) — Incentivising Reasoning via RL (GRPO) | https://arxiv.org/abs/2501.12948 |
| Azar et al. (2023) — IPO | https://arxiv.org/abs/2310.12036 |
| Ethayarajh et al. (2024) — KTO | https://arxiv.org/abs/2402.01306 |
| Meng et al. (2024) — SimPO | https://arxiv.org/abs/2405.14734 |
| Hong et al. (2024) — ORPO | https://arxiv.org/abs/2403.07691 |
| Rafailov et al. (2024) — Scaling Laws for DPO Overoptimisation | https://arxiv.org/abs/2406.02900 |
| Dao et al. (2022) — FlashAttention | https://arxiv.org/abs/2205.14135 |
| Dao (2023) — FlashAttention-2 | https://arxiv.org/abs/2307.08691 |
| Beltagy et al. (2020) — Longformer | https://arxiv.org/abs/2004.05150 |
| Zaheer et al. (2020) — BigBird | https://arxiv.org/abs/2007.14062 |
| Choromanski et al. (2021) — Performers | https://arxiv.org/abs/2009.14794 |
| Peng et al. (2023) — RWKV | https://arxiv.org/abs/2305.13048 |
| Gu et al. (2022) — S4 | https://arxiv.org/abs/2111.00396 |
| Gu and Dao (2023) — Mamba | https://arxiv.org/abs/2312.00752 |
| Ainslie et al. (2023) — GQA | https://arxiv.org/abs/2305.13245 |
| Team AI21 (2024) — Jamba | https://arxiv.org/abs/2403.19887 |
| Williams (1992) — REINFORCE | https://link.springer.com/article/10.1007/BF00992696 |

---

## Inspiration

1. **Stanford CS324: Understanding and Developing Large Language Models** — this is probably the best single Stanford reference for your purpose because it explicitly combines **modeling, theory, ethics, and systems**, and it is designed to give hands-on experience with massive language models. ([Stanford CRFM][1])

2. **Stanford CS224N: Natural Language Processing with Deep Learning** — a very strong backbone course. The current offering explicitly covers **deep learning for NLP and LLMs**, and its assessment structure is especially useful for inspiration: one assignment each on **word vectors**, **neural-network foundations**, **self-attention and Transformers**, and **LLM benchmarking/evaluation**. ([Stanford University][2])

3. **Stanford CS25: Transformers United V6** — this is less of a core methods course and more of a **transformer seminar**, but it is valuable if you want to see how Stanford frames cutting-edge transformer topics and guest talks from major researchers. It is especially useful for a reading-seminar or guest-lecture component. ([Stanford University][3])

4. **UC Berkeley CS 194/294-267: Understanding Large Language Models: Foundations and Safety** — very useful if you want to balance internals with **interpretability, scaling laws, robustness, alignment, privacy, watermarking, agency, reasoning, and evaluation**. It is a strong model for a course that treats LLMs as both a technical and safety-relevant system. ([rdi.berkeley.edu][4])

5. **UC Berkeley INFO 290: Applied Generative AI and Large Language Models** — this is a good reference for the **applied** side of the syllabus: transformer architectures, prompt engineering, API integration, RAG, open-source models, fine-tuning, graph enhancements, and agentic technologies. ([UC Berkeley School of Information][5])

6. **MIT OCW 6.7960 Deep Learning, Lecture 8: Architectures: Transformers** — a clean, compact lecture resource for explaining the core internals: **tokens, attention, and positional codes**, with a nice framing that connects transformers to other architectures. ([MIT OpenCourseWare][6])

7. **MIT OCW 15.773, Lecture 10: Adapting LLMs with Parameter-Efficient Fine-Tuning** — useful for the post-pretraining part of the course, especially **instruction tuning** and adapting base models. ([MIT OpenCourseWare][7])

-  [deeplearning.ai](https://learn.deeplearning.ai/) search for transformers.

- [Francois Chollet's online book](https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer/)


A sensible design pattern would be to use **CS224N** for the course spine, **CS324** for the “LLM systems + theory” lens, **MIT’s transformer lecture** for the internals exposition, and **Berkeley’s courses** for safety, interpretability, and application modules. ([Stanford University][2])

[1]: https://crfm.stanford.edu/courses.html "Stanford CRFM"
[2]: https://web.stanford.edu/class/cs224n/ "Stanford CS 224N | Natural Language Processing with Deep Learning"
[3]: https://web.stanford.edu/class/cs25/ "CS25: Transformers United V6 | CS25"
[4]: https://rdi.berkeley.edu/understanding_llms/s24 "CS 194/294-267 Understanding Large Language Models: Foundations and Safety | Spring 2024"
[5]: https://www.ischool.berkeley.edu/courses/info/290/genai "Info 290. Applied Generative AI and Large Language Models | UC Berkeley School of Information"
[6]: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/resources/mit6_7960f24_lec08_mp4/ "Lec 08. Architectures: Transformers | Deep Learning | Electrical Engineering and Computer Science | MIT OpenCourseWare"
[7]: https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/resources/15773-sp24-lecture-10-version-4_mp4/ "10: Generative AI – Adapting LLMs with Parameter-Efficient Fine-Tuning | Hands-On Deep Learning | Sloan School of Management | MIT OpenCourseWare"



## Contact

**Soumya Banerjee**
- York: soumya.banerjee@york.ac.uk
- Cambridge: sb2333@cam.ac.uk
