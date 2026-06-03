# Week 12 — Applications and Final Project

## Lecture Overview

In this final week we survey the state of the art in LLM applications across several important domains — healthcare, science, education, and software engineering — and set expectations for the final project. Students present their preliminary project demos.

---

## 1. LLMs in Healthcare

Healthcare is one of the most high-stakes and consequential application domains.

### 1.1 Clinical NLP Tasks

| Task | Example models / approaches |
|------|-----------------------------|
| Clinical note summarisation | BART, FLAN-T5, GPT-4 with RAG |
| ICD coding | Fine-tuned BERT variants |
| De-identification | NER models + LLM post-processing |
| Radiology report generation | Multimodal LLMs (CheXagent, GPT-4V) |
| Drug-drug interaction | Fine-tuned biomedical LLMs |
| Patient-facing chatbots | RAG over clinical guidelines |

### 1.2 Challenges

- **Hallucination is life-threatening**: a model that invents a drug dosage is dangerous.
- **Regulatory compliance**: GDPR, HIPAA, MDR — strict constraints on data and outputs.
- **Clinician trust**: models must be explainable and auditable.
- **Data scarcity**: labelled clinical data is expensive and rare.
- **Rare diseases**: LLMs trained on general web data underperform on rare conditions.

### 1.3 Notable Models

- **Med-PaLM 2** (Google): expert-level performance on USMLE.
- **BioMedLM** (Stanford): smaller model for biomedical NLP.
- **Clinical-Longformer**: long-context model for clinical notes.

### 1.4 Patient and Public Involvement (PPIE)

Deploying healthcare AI without involving patients is unethical and often counterproductive. PPIE ensures:
- Systems address real patient needs.
- Outputs are accessible and understandable.
- Privacy and consent are properly designed in.

---

## 2. LLMs for Science

### 2.1 Literature Review and Synthesis

RAG-based systems over corpora of scientific papers:
- Semantic Scholar API, PubMed, arXiv.
- Multi-hop reasoning across papers.
- Automated hypothesis generation.

### 2.2 Mathematical Reasoning

- LLMs can solve some competition maths problems (GSM8K, MATH).
- Formal verification with Lean 4: LLMs can write proof sketches; a proof assistant verifies correctness.
- **AlphaProof** (DeepMind, 2024): solves IMO problems using a combination of LLMs and formal verification.

### 2.3 Code for Science

- GitHub Copilot, Claude Code, GPT-4o: accelerate data analysis pipelines.
- Generating simulation code, data cleaning scripts, statistical analyses.

### 2.4 Hypothesis Generation

Systems like BACON and BioDiscoveryAgent use LLMs to:
- Identify gaps in the literature.
- Propose novel experiments.
- Design research protocols.

---

## 3. LLMs for Education

- **Personalised tutoring**: Socratic dialogue, adaptive hints.
- **Automated feedback**: on essays, code, mathematical proofs.
- **Content generation**: quiz questions, worked examples, summaries.
- **Accessibility**: simplifying complex text for different reading levels.

### Risks

- Students offloading thinking to the model.
- Undetectable academic misconduct.
- Reinforcing misconceptions if the model is wrong.

---

## 4. LLMs for Software Engineering

- **Code generation**: Copilot, Cursor, Claude Code.
- **Code review**: automated PR reviews.
- **Bug localisation**: find and explain bugs.
- **Documentation generation**: docstrings, READMEs.
- **Test generation**: unit test synthesis.

SWE-bench shows current models can resolve ~12–50% of real GitHub issues autonomously (depending on model and scaffolding).

---

## 5. Multimodal LLMs

Modern frontier models process multiple modalities:

| Modality | Example tasks |
|----------|--------------|
| Text + Image | VQA, document parsing, chart understanding |
| Text + Audio | Transcription, speech-to-text, voice assistants |
| Text + Video | Video summarisation, action recognition |
| Text + Code | Code execution, data analysis |

Models: GPT-4o, Claude 3.5, Gemini 1.5, LLaVA (open source).

---

## 6. Building Production LLM Applications

### 6.1 Architecture

```
User → Frontend (Streamlit / React)
           │
           ▼
       Orchestration layer (LangChain / custom)
           │
    ┌──────┴──────┐
    │             │
  LLM API     Vector Store
  (OpenAI /   (FAISS / Chroma)
   Anthropic /
   local)
           │
           ▼
       Output + Logging
```

### 6.2 Key Engineering Decisions

- **Model choice**: capability vs cost vs latency vs privacy.
- **Caching**: cache identical or near-identical queries (semantic caching with embeddings).
- **Rate limiting**: respect API quotas; implement exponential backoff.
- **Streaming**: stream tokens to the UI for perceived responsiveness.
- **Fallback**: graceful degradation if the primary model is unavailable.
- **Logging and monitoring**: log all inputs/outputs for debugging and safety review.

### 6.3 Local vs API Models

| Factor | Local (Ollama, llama.cpp) | API (OpenAI, Anthropic) |
|--------|--------------------------|------------------------|
| Privacy | High | Lower |
| Cost | Hardware upfront | Per token |
| Capability | Smaller models | Frontier models |
| Latency | Variable | Consistent |
| Setup | Complex | Simple |

---

## 7. Final Project Guidelines

### 7.1 Scope

Build a working LLM-powered application. It should:
- Use at least one of: RAG, fine-tuning, agents, or structured prompting.
- Solve a real problem in a domain of your choice.
- Include evaluation: quantitative metrics + qualitative analysis.

### 7.2 Deliverables

| Item | Details |
|------|---------|
| Code | Clean, documented Python; hosted on GitHub |
| Report | 8–10 pages: problem, approach, evaluation, limitations |
| Demo | 10-minute live demonstration |
| Poster | A3 poster for the showcase session |

### 7.3 Project Ideas

- **Healthcare chatbot**: RAG over clinical guidelines; evaluate against clinician gold standard.
- **Research assistant**: multi-agent deep research over arXiv papers.
- **Code review agent**: automated PR review with tool calling.
- **Explainable AI assistant**: use LLM to narrate SHAP/LIME explanations for a black-box model.
- **Educational tutor**: Socratic dialogue agent for a specific subject.
- **Multilingual QA**: evaluate fairness across languages.
- **Scientific hypothesis generator**: RAG + agent over a domain literature corpus.

### 7.4 Assessment Rubric

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Technical depth | 30% | Quality of LLM integration, prompt design, architecture |
| Evaluation | 25% | Rigorous, appropriate metrics; honest analysis of failure modes |
| Novelty | 15% | Original idea or non-trivial combination of techniques |
| Code quality | 15% | Readable, documented, reproducible |
| Presentation | 15% | Clear demo; well-written report |

---

## 8. Practical This Week

See `practicals/week12_practical.py`:
- Build a minimal end-to-end LLM application combining RAG + an agent tool (your choice of domain).
- Package it as a Streamlit app.
- Run a systematic evaluation on 20 test queries.
- Write a two-page reflection: what worked, what failed, and what you would change.

---

## 9. Course Summary

Over 12 weeks we have covered:

1. History and intuition of LLMs
2. Tokenisation (BPE, WordPiece, SentencePiece)
3. Embeddings, similarity, positional encodings
4. Attention mechanism (Q/K/V, multi-head, causal)
5. Transformer architecture (encoder-only, decoder-only, encoder-decoder)
6. Pre-training and scaling laws (Chinchilla, emergent abilities)
7. Fine-tuning and RLHF (SFT, LoRA, QLoRA, DPO)
8. Prompting and context engineering (CoT, few-shot, self-consistency)
9. Retrieval-Augmented Generation (chunking, vector stores, evaluation)
10. Agents and tool use (ReAct, function calling, multi-agent)
11. Evaluation, safety, and ethics (benchmarks, hallucination, bias)
12. Applications and project (healthcare, science, production systems)

---

## 10. Further Reading

- Moor et al. (2023) — "Med-PaLM 2" — https://arxiv.org/abs/2305.09617
- Thirunavukarasu et al. (2023) — "Large language models in medicine" — Nature Medicine
- AlphaProof (DeepMind, 2024) — https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/
- [Streamlit documentation](https://docs.streamlit.io/)
- Existing notebooks: `explainability_using_LLM.ipynb`, `deep_research_agent_ag2.ipynb`

---

## Discussion Questions

1. You are building an LLM application for clinical triage in an NHS hospital. List five ethical and practical requirements it must satisfy before deployment.
2. Compare local model deployment (e.g. Ollama + LLaMA-3) with API-based deployment (e.g. GPT-4o) for a healthcare application. What are the key trade-offs?
3. Reflect on the course: which technique do you think will have the most impact in your own research domain, and why?
