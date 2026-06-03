# Week 11 — Evaluation, Safety, and Ethics

## Lecture Overview

Building capable LLM systems is only half the challenge. This week we ask: how do we know if our system is actually good? How do we measure, improve, and ensure safety? And what are the broader ethical responsibilities of LLM practitioners?

---

## 1. Why Evaluation Is Hard

LLM outputs are:
- **Open-ended**: there is rarely a single correct answer.
- **Multi-dimensional**: quality, factual accuracy, helpfulness, safety, fluency are all distinct.
- **Context-dependent**: the same output may be good in one context and bad in another.
- **Adversarially exploitable**: optimising for a metric often degrades the underlying quality (Goodhart's Law).

---

## 2. Automatic Evaluation Metrics

### 2.1 Reference-Based Metrics

Compare model output to a gold-standard reference.

| Metric | What it measures | Limitations |
|--------|-----------------|-------------|
| BLEU | N-gram overlap (translation) | Misses meaning; brittle |
| ROUGE | N-gram recall (summarisation) | Same issues |
| METEOR | Alignment + synonyms | Better than BLEU |
| BERTScore | Contextual embedding similarity | More robust; still imperfect |
| BLEURT | Fine-tuned regression model | Expensive |

```python
from evaluate import load

rouge = load("rouge")
results = rouge.compute(
    predictions=["The cat sat on the mat."],
    references=["A cat is sitting on a mat."]
)
print(results)   # {'rouge1': 0.73, 'rouge2': 0.40, 'rougeL': 0.73, ...}
```

### 2.2 Reference-Free Metrics

- **Perplexity**: how well does the model predict its own output? Low PPL indicates fluency.
- **Self-BLEU**: diversity of generated outputs.
- **MAUVE**: distribution overlap between model and human text.

---

## 3. Benchmarks

### 3.1 General Knowledge and Reasoning

| Benchmark | Task | Notes |
|-----------|------|-------|
| MMLU | Multiple-choice, 57 subjects | Knowledge breadth |
| HellaSwag | Sentence completion | Common sense |
| ARC (Easy/Challenge) | Science QA | Reasoning |
| WinoGrande | Coreference resolution | Language understanding |
| TruthfulQA | Truthfulness | Hallucination |

### 3.2 Code

| Benchmark | Task |
|-----------|------|
| HumanEval | Python function synthesis |
| MBPP | Simple Python programming |
| SWE-bench | Real GitHub issue resolution |

### 3.3 Reasoning

| Benchmark | Task |
|-----------|------|
| GSM8K | Grade school mathematics |
| MATH | Harder mathematics |
| BBH (BIG-Bench Hard) | Diverse hard tasks |
| ARC-AGI | Abstract visual reasoning |

### 3.4 Safety

| Benchmark | Task |
|-----------|------|
| TruthfulQA | Avoids known false beliefs |
| HarmBench | Resistance to jailbreaks |
| BBQ | Social bias in QA |

### Benchmark Contamination

If a model's training data contains benchmark questions, reported scores are inflated. This is a pervasive problem — be sceptical of leaderboard numbers.

---

## 4. LLM-as-Judge

Use a capable LLM (e.g. GPT-4, Claude) to evaluate outputs. Provide a rubric:

```
You are an evaluation assistant. Rate the following answer on a scale of 1–5 for:
- Factual accuracy
- Helpfulness
- Clarity

Question: {question}
Answer: {answer}

Respond ONLY with JSON: {"factual_accuracy": int, "helpfulness": int, "clarity": int, "rationale": string}
```

- Cheap, scalable, relatively consistent.
- Biased towards responses that resemble the judge model's own style.
- Susceptible to position bias (favours the first answer in pairwise comparisons).

---

## 5. Hallucination

**Hallucination**: the model generates plausible-sounding but factually incorrect information with apparent confidence.

### Types

| Type | Example |
|------|---------|
| Factual hallucination | Inventing a citation, fake statistics |
| Intrinsic hallucination | Answer contradicts the provided context |
| Extrinsic hallucination | Answer goes beyond the context with unsupported claims |

### Causes

- Training on noisy web data containing errors.
- Next-token prediction does not penalise plausible-sounding lies.
- Insufficient knowledge about a topic.
- Over-confidence from RLHF (rewarding confident-sounding answers).

### Mitigations

- RAG: ground answers in retrieved documents.
- Cite sources and verify claims post-generation.
- Train models to say "I don't know."
- Use smaller, specialised models for high-stakes domains.
- Factual consistency models (e.g. FactScore).

---

## 6. Bias and Fairness

LLMs trained on web text absorb societal biases:
- **Representation bias**: certain demographic groups are described more stereotypically.
- **Allocation bias**: models may recommend worse opportunities to certain groups.
- **Toxicity**: generating harmful content about protected groups.

### Measurement

- **WinoBias**: gender bias in coreference resolution.
- **BBQ**: bias benchmarks for question answering.
- **BOLD**: fairness in biographical generation.

### Mitigations

- Diverse and balanced training data.
- RLHF / Constitutional AI to reduce harmful outputs.
- Post-hoc filters and classifiers.
- Ongoing red-teaming.

---

## 7. AI Safety and Alignment

### 7.1 Short-Term Safety Concerns

- **Jailbreaks**: adversarial prompts that bypass safety filters.
- **Prompt injection**: malicious instructions embedded in user or retrieved content.
- **Misuse**: generating disinformation, phishing emails, CBRN instructions.

### 7.2 Longer-Term Alignment

- **Goal misalignment**: a model optimised for a proxy reward may pursue it in unintended ways.
- **Deceptive alignment**: a model that behaves well in training but pursues different goals at deployment.
- **Situational awareness**: the LLM "knows" it is in training vs deployment and behaves differently.

### 7.3 Frameworks

- **HHH (Helpful, Harmless, Honest)**: Anthropic's guiding principles.
- **Constitutional AI**: use a written constitution to guide self-improvement.
- **Responsible Scaling Policies**: commit to safety evaluations before scaling.

---

## 8. Privacy and Data

- LLMs may memorise and reproduce training data including personal information.
- Extraction attacks can recover training data fragments.
- Deployed models may leak conversation content in multi-tenant systems.

Best practices:
- Do not include PII in training data without consent.
- Audit model outputs for memorised content.
- Apply differential privacy during fine-tuning for sensitive data.

---

## 9. Responsible Deployment Checklist

Before deploying an LLM application:

- [ ] Evaluate on a diverse, representative test set.
- [ ] Red-team for harmful outputs and jailbreaks.
- [ ] Define clear scope (what the system should and should not do).
- [ ] Implement content filtering / moderation.
- [ ] Provide human escalation paths.
- [ ] Monitor outputs in production.
- [ ] Document limitations and communicate them to users.
- [ ] Establish a process for user feedback and model updates.

---

## 10. Practical This Week

See `practicals/week11_practical.py`:
- Evaluate a model's outputs on a small QA dataset using ROUGE, BERTScore, and LLM-as-judge.
- Measure hallucination rate: compare model answers to ground truth on a factual QA set.
- Probe for gender bias using a WinoBias-style template set.
- Attempt two jailbreak attacks and document whether they succeed.

---

## 11. Further Reading

- Maynez et al. (2020) — "On Faithfulness and Factuality in Abstractive Summarization" — https://arxiv.org/abs/2005.00661
- Ji et al. (2023) — "Survey of Hallucination in NLG" — https://arxiv.org/abs/2202.03629
- Bender et al. (2021) — "On the Dangers of Stochastic Parrots" — https://dl.acm.org/doi/10.1145/3442188.3445922
- Anthropic (2022) — "Constitutional AI" — https://arxiv.org/abs/2212.08073
- [AI Safety teaching resources](https://github.com/neelsoumya/AI_safety_teaching_resources)

---

## Discussion Questions

1. Why is BLEU a poor metric for open-ended generation? What would you use instead?
2. Describe three types of hallucination and a mitigation for each.
3. A hospital wants to deploy an LLM for clinical triage. List five evaluation criteria and how you would measure each.
4. What is Goodhart's Law and why is it relevant to LLM evaluation?
