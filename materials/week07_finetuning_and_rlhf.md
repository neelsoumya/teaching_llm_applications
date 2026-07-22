# Week 7 — Fine-tuning and RLHF

## Lecture Overview

- [Code from Jay Alammar book](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter11)

- Pre-trained LLMs are powerful but raw: they continue text rather than follow instructions, and they may produce harmful or unhelpful outputs. This week we cover the techniques used to adapt pre-trained models: supervised fine-tuning (SFT), parameter-efficient fine-tuning (PEFT), LoRA, and reinforcement learning from human feedback (RLHF).

---

## 1. Why Fine-tuning?

A pre-trained model trained on next-token prediction will, given the prompt `"Translate this to French: The cat sat on the mat"`, continue the text in a way that looks like training data — perhaps generating more English examples — rather than actually translating.

Fine-tuning teaches the model to:
- Follow instructions
- Answer questions helpfully
- Refuse harmful requests
- Adopt a particular style or domain

---

## 2. Supervised Fine-tuning (SFT)

**SFT** continues training on a curated dataset of (prompt, completion) pairs:

```
Prompt:     "Summarise the following article in two sentences: ..."
Completion: "The article discusses ... The key finding is ..."
```

The training objective is the same (cross-entropy on the completion tokens). The model learns the mapping from instructions to helpful responses.

### SFT Datasets

- **InstructGPT / Alpaca / FLAN**: instruction-following datasets.
- **ShareGPT**: conversations with ChatGPT shared by users.
- **OpenAssistant**: human-annotated conversations.
- **Domain-specific**: clinical notes, code, legal text.

### Catastrophic Forgetting

Fine-tuning on a small dataset can cause the model to "forget" capabilities learned during pre-training. Mitigated by:
- Low learning rates
- Few epochs
- Mixing pre-training data with SFT data
- PEFT methods (see below)

---

## 3. Parameter-Efficient Fine-tuning (PEFT)

Fine-tuning all parameters of a 7B or 70B model requires enormous GPU memory and is expensive. **PEFT** methods update only a small fraction of parameters.

### 3.1 LoRA (Low-Rank Adaptation)

Hu et al. (2022) propose representing weight updates as low-rank matrices.

For a weight matrix W ∈ R^{d × k}, instead of learning ΔW ∈ R^{d × k} (dk parameters), learn:

```
ΔW = B A

where B ∈ R^{d × r}, A ∈ R^{r × k}, r << min(d, k)
```

The number of trainable parameters is `r(d + k)` instead of `dk`.

With r=8, a 4096×4096 matrix goes from 16.7M to 65k trainable parameters — a 256× reduction.

The adapted forward pass:

```python
h = W x + (B @ A) x * scaling_factor
```

The original weights W are frozen; only A and B are updated.

### 3.2 QLoRA

Dettmers et al. (2023): quantise the frozen base model to 4-bit (NF4 quantisation), then apply LoRA on top.

This allows fine-tuning a 65B model on a single 48GB GPU — previously impossible.

### 3.3 Other PEFT Methods

| Method | Idea |
|--------|------|
| Adapters | Insert small bottleneck layers between transformer layers |
| Prefix tuning | Prepend learned "soft tokens" to the input |
| Prompt tuning | Only tune the embedding of a short learned prefix |
| IA³ | Scale activations with learned vectors |

---

## 4. Instruction Tuning

**Instruction tuning** is SFT on a diverse mixture of tasks formatted as instructions. The key insight (Wei et al., 2022; FLAN): fine-tuning on many instruction types improves generalisation to *unseen* instructions.

FLAN-T5 and FLAN-PaLM were trained on 1,800+ tasks. They significantly outperform the base model on zero-shot benchmarks.

---

## 5. Reinforcement Learning from Human Feedback (RLHF)

SFT alone is not sufficient to produce consistently helpful, harmless, and honest outputs. RLHF (Christiano et al., 2017; Ouyang et al., 2022) adds a human preference signal.

### 5.1 Three Stages

**Stage 1 — SFT**: Fine-tune the base model on a curated dataset of demonstrations.

**Stage 2 — Reward Model Training**:
- Sample multiple completions for the same prompt.
- Human annotators rank completions by preference.
- Train a **reward model** (RM) to predict human preference scores.

```
RM(prompt, completion) → scalar reward
```

**Stage 3 — RL Optimisation (PPO)**:
- Use Proximal Policy Optimisation (PPO) to update the SFT model to maximise the reward model's score.
- Add a KL divergence penalty against the SFT model to prevent reward hacking:

```
reward = RM(prompt, completion) - β * KL(π_θ || π_SFT)
```

### 5.2 Reward Hacking

The RM is an imperfect proxy for human preferences. The LLM may learn to produce outputs that score highly on the RM without being genuinely helpful — for example, very long or very confident responses.

### 5.3 Constitutional AI (Anthropic)

An alternative to RLHF: use the LLM itself to critique and revise its responses based on a set of principles (a "constitution"), reducing reliance on human labels.

### 5.4 Direct Preference Optimisation (DPO)

Rafailov et al. (2023): sidestep explicit RL by directly optimising the policy on preference pairs with a simple classification loss. Simpler to implement and often performs as well as PPO.

```
L_DPO = -log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))
```

where y_w is the preferred completion and y_l is the dispreferred completion.

---

## 6. Full Fine-tuning Workflow (with LoRA)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                          lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # e.g. "trainable params: 4,194,304 || all params: 6,742,609,920"

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset,
                     dataset_text_field="text", max_seq_length=512)
trainer.train()
```

---

## 7. Practical This Week

See `practicals/week07_practical.py`:
- Fine-tune a small open-source model (e.g. `facebook/opt-125m` or `gpt2`) on a custom instruction dataset using LoRA.
- Compare outputs before and after fine-tuning.
- Plot training loss.
- Implement a simple preference dataset and train a reward model from scratch on a toy problem.

---

## 8. Further Reading

- Hu et al. (2022) — "LoRA: Low-Rank Adaptation of Large Language Models" — https://arxiv.org/abs/2106.09685
- Dettmers et al. (2023) — "QLoRA" — https://arxiv.org/abs/2305.14314
- Ouyang et al. (2022) — "Training language models to follow instructions with human feedback" (InstructGPT) — https://arxiv.org/abs/2203.02155
- Rafailov et al. (2023) — "Direct Preference Optimisation" — https://arxiv.org/abs/2305.18290
- Existing notebook: `fine_tune_llm.ipynb`, `PEFT.md`, `LLaMA_lora_int8.ipynb`

---

## Discussion Questions

1. Explain the three stages of RLHF. Why is the KL penalty against the SFT model important?
2. A weight matrix has dimensions 4096×4096. With LoRA rank r=16, how many trainable parameters does the update have? What is the compression ratio?
3. What is "reward hacking" and why is it a problem?
4. Compare SFT, RLHF, and DPO as methods for alignment. What are the trade-offs?
