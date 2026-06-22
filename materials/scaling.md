# Scaling

- maybe this means that intelligence is task-specific and different species have different performance constraints. We can compare intelligence across species but our confidence in the quest for a single benchmark for intelligence for multiple species is misplaced.

- same conclusions for ConceptARC paper. There are different performance constraints on machines and humans (machines are faster, etc.). Machines may `reason` in ways different to humans. They may also use different `concepts`. Ultimately our hope that machines may reason in the same way that we do is also misplaced (cite Rich Sutton essay `Bitter Lessons ...`). We made the same mistake with chess: we thought that we will imbue machines with the same strategies that we use while playing chess. _The winning techniques were approaches that used brute-force search with engineering-ey approaches_.

- this is especially releavnt since current approaches to ARC use a lot of _effort_ and _energy_ and _computational resources_.  Thousands of dollars per task. We need to come up with a theory of intelligence and energetics that spans biological intelligence and artificial engineered inteligence (such as LLMs). it will also allow us to create a theory of how energetics and evolutionary constraints shape intelligence.

- Some equations for this theory given below: optimal foraging theory of infomation (trainng cost, data acquisition, Inference cost, number of parameters, computational budget per task, number of tokens per task) . 

- treat intelligence less as a single score and more as a family of efficiency frontiers under task and resource constraints

- `this ties into scaling lecture in class`


- same conclusions for ConceptARC paper. There are different performance constraints on machines and humans (machines are faster, etc.). Machines may `reason` in ways different to humans. They may also use different `concepts`. Ultimately our hope that machines may reason in the same way that we do is also misplaced (cite Rich Sutton essay `Bitter Lessons ...`). We made the same mistake with chess: we thought that we will imbue machines with the same strategies that we use while playing chess. _The winning techniques were approaches that used brute-force search with engineering-ey approaches_.

- this is especially releavnt since current approaches to ARC use a lot of _effort_ and _energy_ and _computational resources_.  Thousands of dollars per task. We need to come up with a theory of intelligence and energetics that spans biological intelligence and artificial engineered inteligence (such as LLMs). it will also allow us to create a theory of how energetics and evolutionary constraints shape intelligence.

- Some equations for this theory given below: optimal foraging theory of infomation (trainng cost, data acquisition, Inference cost, number of parameters, computational budget per task, number of tokens per task) . 

- `treat intelligence less as a single score and more as a family of efficiency frontiers under task and resource constraints`

- Energetic Niche Theory of Intelligence

- For biology, those costs are metabolic, attentional, developmental, and social. For AI, they are FLOPs, latency, tokens, memory bandwidth, training data, and dollar cost.

- performance–budget frontier for substrate `s`.
Then “more intelligent” means not “higher on one benchmark,” but “better frontier shape” across the relevant task distribution.


- practical and theory on all of the `costs` of LLMs

- data, training, inference, curation, alignment, RLHF, guardrails, human costs

- scaling papers with notebooks CAISH

- scaling curves papers

## Resources

# Large Language Model Scaling: Papers, Interactive Resources, and Further Reading

> [!NOTE]
> **Learning Goals**
>
> By the end of this topic, you should be able to:
>
> * Explain what scaling laws are and why they matter.
> * Understand the relationship between model size, data, and compute.
> * Describe the transition from the Kaplan scaling laws to the Chinchilla scaling laws.
> * Discuss practical bottlenecks to continued scaling.
> * Critically evaluate whether scaling alone is sufficient for achieving more general intelligence.

---

# Why Study Scaling?

The modern era of AI has been shaped by a simple observation: **larger models trained on more data with more compute tend to perform better in predictable ways**. Understanding these relationships has become essential for understanding the development of Large Language Models (LLMs).

Scaling laws help us answer questions such as:

* How much performance improvement can we expect from a larger model?
* Is it better to increase parameters or training data?
* What are the limits of current approaches?
* Can scaling alone lead to more general forms of intelligence?

---

# Essential Reading

## 1. The Philosophical Foundation

### Rich Sutton — *The Bitter Lesson*

**Why read it?**

This short essay provides the intellectual backdrop for modern AI. Sutton argues that throughout AI history, methods that leverage increasing computation repeatedly outperform approaches built around handcrafted human knowledge.

🔗 [https://www.incompleteideas.net/IncIdeas/BitterLesson.html](https://www.incompleteideas.net/IncIdeas/BitterLesson.html)

> [!TIP]
> **Discussion Question**
>
> Is the success of modern LLMs evidence that Sutton was right? What examples from AI history support or challenge his argument?

---

## 2. The Original Scaling Laws

### Kaplan et al. (2020) — *Scaling Laws for Neural Language Models*

**Key idea:** Language model performance follows remarkably smooth power-law relationships with:

* Model size
* Dataset size
* Training compute

This paper established scaling laws as a central framework for understanding AI progress.

🔗 [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)

> [!IMPORTANT]
> One of the most surprising findings is that performance improvements remain highly predictable across many orders of magnitude.

---

## 3. Scaling Produces New Behaviours

### Brown et al. (2020) — *Language Models are Few-Shot Learners*

The GPT-3 paper demonstrated that simply scaling a language model can lead to qualitatively new capabilities such as:

* In-context learning
* Few-shot learning
* Zero-shot task performance

🔗 [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

> [!NOTE]
> This paper sparked renewed interest in the possibility that new capabilities may emerge from scale alone.

---

## 4. The Chinchilla Revision

### Hoffmann et al. (2022) — *Training Compute-Optimal Large Language Models*

This paper challenged prevailing assumptions about model scaling.

The central argument:

* Many large models were **undertrained**
* More training data was needed
* Compute should be distributed more efficiently between parameters and tokens

🔗 [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)

> [!WARNING]
> Bigger models are not necessarily better models. Training strategy matters.

---

## 5. Revisiting Chinchilla

### Epoch AI — *Chinchilla Scaling: A Replication Attempt*

An important follow-up study that revisits and tests the Chinchilla conclusions using a much larger collection of models.

🔗 [https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt)

> [!TIP]
> Ask yourself:
>
> * Are scaling laws universal?
> * How robust are these results?
> * What assumptions underlie these analyses?

---

# Interactive Resources

> [!NOTE]
> These resources are ideal for tutorials, lab sessions, and self-study.

---

## AI Safety Course — Chapter 1: Capabilities

A highly accessible introduction to scaling and capability development.

Features:

* Explanations of parameters, compute, and data
* Exercises
* Review questions
* AI safety context

🔗 [https://ai-safety-course.github.io/chapters/chapter-1/](https://ai-safety-course.github.io/chapters/chapter-1/)

---

## Epoch AI Distributed Training Simulator

An interactive simulator that allows you to explore:

* GPU counts
* Training time
* Parallelisation strategies
* Compute budgets

🔗 [https://epoch.ai/latest/introducing-the-distributed-training-interactive-simulator](https://epoch.ai/latest/introducing-the-distributed-training-interactive-simulator)

> [!TIP]
> Classroom Activity:
>
> Have students estimate the infrastructure required to train a GPT-4-scale model and compare assumptions.

---

## Data Bottleneck Explorer

### Epoch AI — *Will We Run Out of Data?*

Investigates a major question for the future of scaling:

**What happens when we run out of high-quality human-generated text?**

🔗 [https://epoch.ai/publications/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data](https://epoch.ai/publications/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data)

> [!IMPORTANT]
> Future AI progress may be constrained by data availability rather than compute.

---

# Hands-On Notebooks

## Empirical Scaling Harness

Students can:

* Train small transformer models
* Generate scaling curves
* Fit power laws
* Test extrapolation predictions

🔗 [https://github.com/mmcmanus1/empirical-scaling-harness](https://github.com/mmcmanus1/empirical-scaling-harness)

> [!TIP]
> Recommended mini-project:
>
> Replicate a scaling-law curve using a subset of experiments and compare your fitted exponent with the published result.

---

## Reasoning Scaling Law

A research-oriented notebook exploring scaling behaviour in reasoning tasks.

Students can:

* Generate synthetic tasks
* Train models
* Evaluate reasoning performance
* Investigate scaling trends

🔗 [https://github.com/WANGXinyiLinda/reasoning-scaling-law](https://github.com/WANGXinyiLinda/reasoning-scaling-law)

---

## Chinchilla's Wild Implications

An accessible explanation of the practical consequences of compute-optimal scaling.

Includes:

* Visualisations
* Worked examples
* Colab notebook

🔗 [https://www.alignmentforum.org/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications](https://www.alignmentforum.org/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications)

---

# Further Reading

## AI Safety, Ethics, and Society Textbook

### Section 2.4 — Scaling Laws

A concise pedagogical introduction with exercises and review questions.

🔗 [https://www.aisafetybook.com/textbook/scaling-laws](https://www.aisafetybook.com/textbook/scaling-laws)

---

## JAX Scaling Book

### How To Scale Your Model

A practical guide to large-scale training systems.

Topics include:

* Distributed training
* TPU architectures
* GPU clusters
* Communication bottlenecks

🔗 [https://jax-ml.github.io/scaling-book/](https://jax-ml.github.io/scaling-book/)

---

# Suggested Lecture Flow

## Part 1 — Why Scaling Matters

1. The Bitter Lesson
2. Historical trends in AI

---

## Part 2 — Empirical Scaling Laws

3. Kaplan et al.
4. Power-law behaviour
5. Predictability of performance

---

## Part 3 — Emergent Capabilities

6. GPT-3
7. Few-shot learning
8. In-context learning

---

## Part 4 — Compute-Optimal Training

9. Chinchilla
10. Parameters vs. tokens
11. Replication studies

---

## Part 5 — Limits of Scaling

12. Compute bottlenecks
13. Energy constraints
14. Data limitations

---

## Part 6 — Hands-On Exploration

15. Scaling-law notebook
16. Interactive simulator
17. Student experiments

---

# Reflection Questions

> [!QUESTION]
>
> 1. Why do power laws appear so frequently in modern AI?
> 2. Is scaling discovering intelligence or merely exploiting statistics more efficiently?
> 3. What might eventually limit further scaling?
> 4. Does scaling alone lead to reasoning and abstraction?
> 5. How do scaling laws relate to François Chollet's arguments in ARC and the measurement of intelligence?
> 6. Are there domains where scaling may fail entirely?

---

> [!SUMMARY]
>
> The central story of modern AI can be viewed as:
>
> **The Bitter Lesson → Scaling Laws → GPT-3 → Chinchilla → Data and Compute Limits**
>
> Understanding this progression provides a foundation for understanding both the successes and the future challenges of large language models.
