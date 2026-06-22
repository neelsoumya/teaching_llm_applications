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

# Resources for Teaching LLM Scaling

This page collects core papers, interactive resources, and supporting reading for a class on scaling large language models. The emphasis is on resources that are useful for teaching: clear, canonical, and easy to explore in class.

## Core papers to anchor the topic

### 1) Rich Sutton — *The Bitter Lesson*

A short but foundational essay arguing that general methods that leverage computation tend to outperform hand-engineered knowledge over time. It is an excellent opening piece for a lesson on why scaling became so central in modern AI.

[Read the essay](https://www.incompleteideas.net/IncIdeas/BitterLesson.html)

### 2) Kaplan et al. (2020) — *Scaling Laws for Neural Language Models*

The classic empirical paper on scaling laws. It shows that language-model loss follows power laws in model size, dataset size, and compute across a very wide range of scales.

[Read on arXiv](https://arxiv.org/abs/2001.08361)

### 3) Brown et al. (2020) — *Language Models are Few-Shot Learners*

The GPT-3 paper. Very useful for showing that scaling changes not only performance but also the style of learning, especially in-context and few-shot behavior.

[Read on arXiv](https://arxiv.org/abs/2005.14165)

### 4) Hoffmann et al. (2022) — *Training Compute-Optimal Large Language Models*

The Chinchilla paper. This is the key update to the scaling story: for a fixed compute budget, model size and training tokens should be scaled together more carefully than many earlier recipes suggested.

[Read on arXiv](https://arxiv.org/abs/2203.15556)

### 5) Epoch AI — *Chinchilla scaling: A replication attempt*

A useful follow-up for class discussion. It revisits the Chinchilla result at larger scale and includes an interactive visualization.

[Read on Epoch AI](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt)

---

## Best interactive / playable resources

### 1) AI Safety Course — Chapter 1: *Capabilities*

A classroom-friendly course page with a dedicated scaling section, explanations of compute/data/parameter count, review questions, and a scaling-law exercise.

[Open the chapter](https://ai-safety-course.github.io/chapters/chapter-1/)

### 2) Epoch AI — Distributed training interactive simulator

A very good in-class demo for systems bottlenecks. Students can explore how hardware, parallelism, and scaling assumptions affect training feasibility.

[Open the simulator](https://epoch.ai/latest/introducing-the-distributed-training-interactive-simulator)

### 3) Epoch AI — Data-limit / data-runout projection page

Useful for discussing what happens when scaling runs into data constraints. The page includes interactive visualizations about how long human-generated text data may last under different scenarios.

[Open the page](https://epoch.ai/publications/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data)

### 4) Empirical Scaling Harness

A hands-on notebook-style resource that trains small transformers at multiple scales, fits power laws, and tests holdout predictions. Good for a classroom demo or a student project.

[View the repository](https://github.com/mmcmanus1/empirical-scaling-harness)

### 5) Reasoning scaling law

An experimental notebook resource for exploring reasoning-related scaling patterns. It generates synthetic graphs, trains and evaluates language models, and is designed to run in Colab.

[View the repository](https://github.com/WANGXinyiLinda/reasoning-scaling-law)

### 6) *Chinchilla’s wild implications*

A less formal but very useful explanatory post with an accompanying Colab notebook. Good for intuition-building and discussion.

[Read on the Alignment Forum](https://www.alignmentforum.org/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications)

---

## Supporting reading

### 1) AI Safety, Ethics, and Society Textbook — Section 2.4: *Scaling Laws*

A clean pedagogical explanation with review questions and answers. Very suitable for self-study or as a tutorial handout.

[Read the section](https://www.aisafetybook.com/textbook/scaling-laws)

### 2) JAX-ML — *How To Scale Your Model*

More systems-oriented than theory-oriented, but useful for understanding scaling in real hardware terms: TPUs/GPUs, communication, and parallelization.

[Read the guide](https://jax-ml.github.io/scaling-book/)

### 3) Epoch AI — *Will we run out of data to train large language models?*

A strong extension beyond the classic scaling-law story, focused on data bottlenecks, overtraining, and compute-optimal training.

[Read the report](https://epoch.ai/publications/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data)

---

## Suggested lecture arc

A clean sequence for teaching is:

1. *The Bitter Lesson*
2. Kaplan et al. on scaling laws
3. GPT-3 and few-shot learning
4. Chinchilla and compute-optimal training
5. Interactive simulator and data bottlenecks
6. Hands-on notebook exercise

This progression moves from principle, to empirical law, to current practical constraints, and finally to experimentation.

---

## Optional classroom activities

* Ask students to compare the Kaplan and Chinchilla views of scaling.
* Use the Epoch AI simulator to discuss why “more parameters” is not the same as “better training.”
* Have students inspect a notebook that fits a power law and identify what assumptions are being made.
* End with a discussion of where scaling may hit limits: compute, data, evaluation, or architecture.
