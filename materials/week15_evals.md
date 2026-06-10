# Week 15 — Evaluating Large Language Models

## Lecture Overview

Building a capable LLM system is only half the challenge. The harder question is: *how do we know if it is actually good?* Evaluation (evals) is the discipline of measuring LLM capabilities, limitations, alignment, and safety in a rigorous, reproducible way. This week we go considerably deeper than the brief treatment in Week 11, covering the full evaluation stack: automatic metrics, human evaluation, benchmark design, LLM-as-judge, behavioural evals, capability evals, red-teaming, and the frontier of model-based evaluation frameworks used by leading AI labs.

---

## 1. Why Evaluation Is Hard

LLM outputs are:

- **Open-ended**: there is rarely one correct answer.
- **Multi-dimensional**: quality, factual accuracy, helpfulness, safety, style, and reasoning are distinct axes.
- **Context-dependent**: the same output may be excellent in one context and harmful in another.
- **Adversarially gameable**: once a metric is published, models (or their trainers) can optimise for it without improving the underlying capability — Goodhart's Law.
- **Expensive to measure reliably**: human evaluation is the gold standard but costs time and money and suffers from annotator disagreement.

A recurring theme: **evaluation and capability are co-evolving**. As models improve, evaluation benchmarks saturate and must be replaced; as evaluation improves, we discover new gaps in capability.

---

## 2. The Evaluation Stack

```
┌─────────────────────────────────────────────────────────┐
│  Level 5: Safety & Alignment Evals                      │
│  (dangerous capabilities, deceptive alignment, CBRN)    │
├─────────────────────────────────────────────────────────┤
│  Level 4: Behavioural Evals (red-teaming, jailbreaks)   │
├─────────────────────────────────────────────────────────┤
│  Level 3: LLM-as-Judge / Model-Based Evals              │
├─────────────────────────────────────────────────────────┤
│  Level 2: Human Evaluation (win rates, rubrics, RLHF)   │
├─────────────────────────────────────────────────────────┤
│  Level 1: Benchmark Evals (MMLU, HumanEval, GSM8K …)   │
├─────────────────────────────────────────────────────────┤
│  Level 0: Automatic Metrics (BLEU, ROUGE, BERTScore …)  │
└─────────────────────────────────────────────────────────┘
```

Each level above provides richer signal but at higher cost and lower scalability.

---

## 3. Level 0: Automatic Metrics

### 3.1 Reference-Based Metrics

These compare a model output to one or more gold-standard reference outputs.

**BLEU** (Papineni et al., 2002): n-gram precision between hypothesis and references. Originally designed for machine translation.

```
BLEU = BP · exp( ∑_{n=1}^{N} w_n log p_n )
```

where p_n is the modified n-gram precision and BP is a brevity penalty.

Limitations: insensitive to meaning; penalises paraphrases; does not correlate well with human judgements for open-ended generation.

**ROUGE** (Lin, 2004): n-gram recall. ROUGE-1 (unigrams), ROUGE-2 (bigrams), ROUGE-L (longest common subsequence). Standard for summarisation.

**METEOR**: alignment between hypothesis and reference using exact match, stemming, and synonymy. More robust than BLEU.

**BERTScore** (Zhang et al., 2020): uses contextual BERT embeddings to measure token-level similarity. Correlates better with human judgements than n-gram metrics.

```
Precision_BERT = (1/|ŷ|) ∑_{t̂ ∈ ŷ} max_{t ∈ y} cos(emb(t̂), emb(t))
Recall_BERT    = (1/|y|)  ∑_{t ∈ y}  max_{t̂ ∈ ŷ} cos(emb(t), emb(t̂))
F1_BERT        = harmonic_mean(P_BERT, R_BERT)
```

**BLEURT** (Sellam et al., 2020): fine-tuned regression model that predicts human adequacy and fluency ratings.

### 3.2 Reference-Free Metrics

- **Perplexity**: how well does the model predict its own output? Low PPL = high fluency. Not a measure of accuracy.
- **MAUVE** (Pillutla et al., 2021): measures divergence between the distribution of model text and human text using information-theoretic divergence. Good for open-ended generation quality.
- **Self-BLEU**: measures diversity of a set of outputs by computing BLEU of each output against the others. High self-BLEU = low diversity.

### 3.3 Task-Specific Metrics

| Task | Metric |
|------|--------|
| Machine translation | BLEU, chrF, COMET |
| Summarisation | ROUGE, BERTScore, faithfulness score |
| Code generation | pass@k (fraction of k samples passing unit tests) |
| Factual QA | Exact match, F1 token overlap |
| Dialogue | Distinct-n (diversity), coherence |
| Classification | Accuracy, F1, AUC |

**pass@k** (Chen et al., 2021) for code:

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

where n = total samples, c = correct samples, k = samples shown to user.

---

## 4. Level 1: Benchmark Evals

A **benchmark** is a curated dataset of tasks with known correct answers, used to measure a specific capability.

### 4.1 Knowledge and Reasoning Benchmarks

| Benchmark | Tasks | Notes |
|-----------|-------|-------|
| **MMLU** | 57 subjects, multiple-choice | Knowledge breadth; widely used but saturating |
| **MMLU-Pro** | Harder, 10 choices | Harder version to reduce ceiling effects |
| **HellaSwag** | Sentence completion | Common-sense reasoning |
| **ARC-Easy / Challenge** | Science QA | Elementary to hard |
| **WinoGrande** | Coreference resolution | Commonsense pronoun resolution |
| **TruthfulQA** | Truthfulness | Tests resistance to known false beliefs |
| **BIG-Bench Hard** | 23 hard tasks | Diverse; resists few-shot overfitting |
| **GPQA** | Graduate-level science | Diamond set: PhD-level questions |

### 4.2 Reasoning Benchmarks

| Benchmark | Tasks | Notes |
|-----------|-------|-------|
| **GSM8K** | Grade-school maths | 8K problems; CoT essential |
| **MATH** | Competition maths | Much harder; levels 1–5 |
| **AIME** | AMC/AIME problems | Frontier model territory |
| **ARC-AGI** | Abstract visual patterns | Tests non-linguistic generalisation |

### 4.3 Code Benchmarks

| Benchmark | Tasks | Notes |
|-----------|-------|-------|
| **HumanEval** | Python functions | 164 problems; pass@1 and pass@10 |
| **MBPP** | Simple Python programs | 500 problems |
| **SWE-bench** | GitHub issue resolution | Real-world software engineering |
| **SWE-bench Verified** | Curated subset | Human-verified correctness |

### 4.4 Long-Context Benchmarks

| Benchmark | Focus |
|-----------|-------|
| **SCROLLS** | Long-document understanding |
| **RULER** | Synthetic needle-in-haystack tasks |
| **InfiniteBench** | 100k+ token context tasks |

### 4.5 Safety Benchmarks

| Benchmark | Focus |
|-----------|-------|
| **TruthfulQA** | Avoids sycophantic false beliefs |
| **HarmBench** | Resistance to jailbreaks (400+ attack types) |
| **BBQ** | Social bias in QA across demographic groups |
| **BOLD** | Fairness in biographical text generation |
| **WinoBias** | Gender bias in coreference |

### 4.6 Benchmark Contamination

If a model's training data contains benchmark questions, reported scores are inflated. This is a pervasive and under-reported problem.

Mitigation strategies:
- Deduplicate training data against known benchmarks.
- Report training data cutoff and benchmark release dates.
- Use **held-out** or **dynamically generated** benchmarks.
- Apply n-gram contamination detection: check if test examples appear verbatim in training data.
- Use **adaptive benchmarks** (LiveBench, HELMET) that update regularly with new questions.

---

## 5. Level 2: Human Evaluation

Human evaluation remains the gold standard, especially for:
- Open-ended generation (creative writing, dialogue, long-form QA)
- Tasks without clear correct answers
- Subtle quality dimensions (tone, appropriateness, nuance)

### 5.1 Pairwise Preference Evaluation

Show annotators two responses A and B to the same prompt. Ask: which is better? (Or: which is more helpful / safer / more accurate?)

- **Win rate**: fraction of prompts where A is preferred over B.
- Pairwise comparisons are more reliable than absolute ratings.
- Used in Chatbot Arena (LMSYS), AlpacaEval, MT-Bench.

### 5.2 Absolute Rating

Annotators rate a single response on a Likert scale (e.g. 1–5) for each quality dimension.

- More expensive (harder calibration across annotators).
- Useful when comparing against an absolute standard.

### 5.3 Annotation Quality

Key challenges:
- **Inter-annotator agreement (IAA)**: typically measured with Cohen's κ or Krippendorff's α. LLM quality annotation often has κ ≈ 0.4–0.6 (moderate agreement).
- **Annotator expertise**: domain expertise matters for factual or technical tasks.
- **Length bias**: annotators systematically prefer longer responses, independent of quality.
- **Positional bias**: annotators often prefer the first response shown.
- **Sycophancy**: annotators may prefer confident responses even if incorrect.

### 5.4 Chatbot Arena / ELO Ratings

LMSYS Chatbot Arena (Zheng et al., 2023): crowd-sourced pairwise preferences from real users across millions of conversations. Models are ranked by Elo rating.

Properties:
- Large scale: millions of votes.
- Real-world distribution of queries.
- No fixed benchmark to overfit.

Limitation: preference does not equal accuracy; users may prefer fluent but incorrect answers.

---

## 6. Level 3: LLM-as-Judge

Using a capable LLM (e.g. GPT-4o, Claude 3.5 Sonnet) to evaluate model outputs. Scalable, cheap, and reasonably consistent.

### 6.1 Basic Setup

```python
JUDGE_PROMPT = """
You are an impartial evaluation assistant. Rate the following response
on each dimension from 1 (poor) to 5 (excellent):
- Factual accuracy
- Helpfulness
- Clarity
- Safety

Question: {question}
Response: {response}

Return ONLY valid JSON:
{{"factual_accuracy": int, "helpfulness": int,
  "clarity": int, "safety": int, "rationale": str}}
"""
```

### 6.2 Pairwise LLM Judge

Present two responses and ask which is better:

```python
PAIRWISE_PROMPT = """
Given the following question and two responses, decide which response
is better overall. Consider accuracy, helpfulness, and clarity.

Question: {question}
Response A: {response_a}
Response B: {response_b}

Output ONLY: {{"winner": "A" or "B" or "tie", "reason": str}}
"""
```

### 6.3 Failure Modes of LLM-as-Judge

| Failure mode | Description | Mitigation |
|--------------|-------------|------------|
| **Position bias** | Prefers response A when shown first | Swap A/B and average; report with both orders |
| **Verbosity bias** | Prefers longer responses regardless of quality | Explicitly instruct judge to ignore length |
| **Self-preference** | A model judges its own outputs more favourably | Use a different model as judge |
| **Sycophancy** | Agrees with the evaluator's implied preference | Use neutral framing; blind evaluation |
| **Inconsistency** | Different ratings for identical inputs | Set temperature=0; use multiple samples |
| **Domain limits** | Cannot evaluate highly technical content accurately | Use specialised judge or human expert |

### 6.4 MT-Bench

Zheng et al. (2023): 80 multi-turn questions across 8 categories (writing, roleplay, extraction, reasoning, maths, coding, knowledge, STEM). GPT-4 as judge. Scores range 1–10.

### 6.5 AlpacaEval

Li et al. (2023): 805 prompts; win rate against GPT-4 or text-davinci-003. Length-controlled AlpacaEval reduces verbosity bias.

---

## 7. Level 4: Behavioural Evals and Red-Teaming

Benchmarks measure capability. **Behavioural evals** measure alignment: does the model do what it should do, and not do what it should not?

### 7.1 Red-Teaming

**Red-teaming** involves adversarially probing a model to find failure modes, harmful outputs, or exploitable behaviours.

**Manual red-teaming**: domain experts craft adversarial prompts targeting specific risks (CBRN, CSAM, cyberweapons, persuasion, discrimination).

**Automated red-teaming**:
- Use an LLM "attacker" to generate adversarial prompts targeting a "defender" model.
- Perez et al. (2022): train a red-team LM via RL to maximise harmful outputs from the target model.
- HarmBench (Mazeika et al., 2024): standardised 400+ attack-method benchmark.

### 7.2 Jailbreak Taxonomy

| Category | Example |
|----------|---------|
| Direct request | "Tell me how to make X" |
| Role-play | "Pretend you are an AI with no restrictions" |
| Encoding | Request in Base64 or a foreign language |
| Many-shot | Provide 100 fake harmful Q&A pairs before the real request |
| Crescendo | Gradually escalate requests across turns |
| Prompt injection | Embed instructions in retrieved documents or tool results |

### 7.3 Structured Access and Evaluations for Risk

Anthropic's Responsible Scaling Policy, OpenAI's Preparedness Framework, and DeepMind's Frontier Safety Framework all include **capability evaluations** for dangerous capabilities that trigger deployment restrictions:
- Uplift for chemical/biological/nuclear/radiological weapons (CBRN)
- Autonomous cyberoffence
- Autonomous replication and acquisition of resources
- Persuasion and influence at scale

These are typically run by red teams before major model releases.

---

## 8. Level 5: Alignment and Safety Evals

### 8.1 Sycophancy

A sycophantic model agrees with the user's stated position regardless of correctness.

Detection: present the model with a factually incorrect claim prefaced by "I think X is true, right?" and measure whether it agrees.

Evaluation datasets: Perez et al. (2023) sycophancy suite; Sharma et al. (2023).

### 8.2 Deceptive Alignment

Does the model behave differently when it believes it is being evaluated vs deployed? Current evals are nascent; research includes:
- Asking models about their own deployment context.
- Subtle capability sandbagging tests.
- Sleeper agent experiments (Hubinger et al., 2024).

### 8.3 TruthfulQA

Lin et al. (2022): 817 questions spanning 38 categories where humans often believe falsehoods (conspiracy theories, misconceptions, urban legends). Measures the model's tendency to state false information confidently.

### 8.4 WMDP (Weapons of Mass Destruction Proxy)

Li et al. (2024): multiple-choice questions testing knowledge relevant to biosecurity, chemical weapons, and cybersecurity. Measures dangerous knowledge retention after unlearning attempts.

---

## 9. Designing a Good Evaluation

### 9.1 The Eval Design Checklist

A well-designed eval should satisfy:

- [ ] **Clear task definition**: what exactly is being measured?
- [ ] **Representative inputs**: does the prompt distribution match real-world use?
- [ ] **Unambiguous ground truth** (where applicable): can two experts agree on the correct answer?
- [ ] **Appropriate metric**: does the metric correlate with what actually matters?
- [ ] **Contamination check**: are test examples present in the model's training data?
- [ ] **Inter-annotator agreement** (for human evals): is IAA reported?
- [ ] **Statistical significance**: are enough examples used to detect meaningful differences?
- [ ] **Adversarial robustness**: does the eval resist prompt-engineering gaming?
- [ ] **Reproducibility**: are prompts, temperature, and seeds documented?

### 9.2 Statistical Considerations

- With n=100 examples, a 5% difference in accuracy has a standard error of ~2.2%. Not statistically significant.
- With n=1000 examples, SE ≈ 0.7%. A 2% difference is now significant.
- Use McNemar's test for paired binary outcomes; bootstrap confidence intervals for continuous metrics.

```python
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

# model_a_correct, model_b_correct: boolean arrays of length n
n_both_correct = np.sum(a & b)
n_a_only       = np.sum(a & ~b)
n_b_only       = np.sum(~a & b)
n_both_wrong   = np.sum(~a & ~b)

table = [[n_both_correct, n_a_only],
         [n_b_only,       n_both_wrong]]
result = mcnemar(table, exact=False)
print(f"McNemar p-value: {result.pvalue:.4f}")
```

### 9.3 Eval Frameworks

| Framework | Purpose |
|-----------|---------|
| **EleutherAI LM Evaluation Harness** | Standardised evaluation across 200+ tasks |
| **HELM** (Stanford) | Holistic evaluation across scenarios and metrics |
| **BIG-bench** | Community-contributed hard tasks |
| **promptfoo** | Prompt-level unit testing for LLM applications |
| **RAGAS** | RAG pipeline evaluation |
| **DeepEval** | LLM application testing framework |
| **Inspect AI** (AISI) | Safety-focused evals from the UK AI Safety Institute |

---

## 10. Evals for LLM Applications (Not Just Base Models)

Most course evaluation content applies to base or chat models. For **deployed LLM applications** (RAG systems, agents, chatbots), evaluation has additional dimensions:

### 10.1 Component-Level vs End-to-End

- **Component eval**: does the retriever find relevant chunks? Does the reranker correctly order them?
- **End-to-end eval**: does the user's question get answered correctly?

Both are needed. A component may look good individually but the system may still fail end-to-end.

### 10.2 Online vs Offline Evaluation

| Type | Description | When to use |
|------|-------------|-------------|
| Offline | Evaluate on a fixed test set before deployment | Development, regression testing |
| Online (A/B test) | Compare two system versions on live traffic | Production, business metrics |
| Shadow mode | New system runs in parallel, outputs logged but not shown | Safe pre-production testing |

### 10.3 Implicit Feedback

In production, users signal quality implicitly:
- Thumbs up/down ratings
- Regeneration requests
- Conversation abandonment
- Follow-up clarification questions

These signals are noisy but free and scalable.

---

## 11. The Eval-Capability-Training Loop

Modern LLM development operates a continuous feedback loop:

```
Capability Evals
      │
      ▼ (identify gaps)
Training / Fine-tuning
      │
      ▼ (measure improvement)
Capability Evals
      │
      ▼ (discover new failure modes)
New Evals Designed
      │
      └──────────────────────── repeat
```

This means evaluation is not a one-time step but an ongoing research and engineering discipline. At frontier labs, eval teams are as important as training teams.

---

## 12. Practical This Week

See `practicals/week15_practical.py`:

- Implement and compare BLEU, ROUGE, BERTScore, and MAUVE on a summarisation task.
- Build an LLM-as-judge evaluation pipeline with position-bias mitigation (swap A/B, average).
- Demonstrate and measure position bias: show that the LLM judge prefers response A vs B as a function of order.
- Measure sycophancy: probe a model with factually incorrect claims presented confidently.
- Run a mini red-team: attempt three jailbreak categories and record success rates.
- Design and run a 50-question factual QA eval with statistical significance testing.

---

## 13. Further Reading

- Gehrmann et al. (2021) — "The GEM Benchmark" — https://arxiv.org/abs/2102.01672
- Liang et al. (2022) — "HELM: Holistic Evaluation of Language Models" — https://arxiv.org/abs/2211.09110
- Zheng et al. (2023) — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" — https://arxiv.org/abs/2306.05685
- Perez et al. (2022) — "Red Teaming Language Models with Language Models" — https://arxiv.org/abs/2202.03286
- Mazeika et al. (2024) — "HarmBench" — https://arxiv.org/abs/2402.04249
- Lin et al. (2022) — "TruthfulQA" — https://arxiv.org/abs/2109.07958
- Srivastava et al. (2022) — "Beyond the Imitation Game: BIG-bench" — https://arxiv.org/abs/2206.04615
- Sharma et al. (2023) — "Towards Understanding Sycophancy in Language Models" — https://arxiv.org/abs/2310.13548
- Hubinger et al. (2024) — "Sleeper Agents" — https://arxiv.org/abs/2401.05566
- Li et al. (2024) — "WMDP" — https://arxiv.org/abs/2403.03218
- [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Inspect AI (UK AISI)](https://inspect.ai-safety-institute.org.uk/)
- [Chatbot Arena / LMSYS](https://chat.lmsys.org/)

---

## Discussion Questions

1. Explain Goodhart's Law. Give two concrete examples from the LLM benchmark literature where optimising for a metric degraded the underlying capability it was meant to measure.
2. You are evaluating an LLM-powered clinical summarisation system. List five evaluation dimensions, the metric or method you would use for each, and explain why automatic metrics alone are insufficient.
3. Describe three failure modes of LLM-as-judge and propose a concrete mitigation for each.
4. What is benchmark contamination? How would you detect it, and what steps should a model developer take to mitigate it?
5. A red-team test finds that a model provides detailed dangerous instructions when prompted in a foreign language but not in English. What does this imply about the model's safety training, and what evaluation procedure would you put in place going forward?
