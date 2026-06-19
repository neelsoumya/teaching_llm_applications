# Week 13 — Reinforcement Learning from Human Feedback (RLHF): A Deep Dive

## Lecture Overview

- 🎥 [Video](https://www.youtube.com/watch?v=Z_JUqJBpVOk)

Week 7 introduced RLHF as one component of the fine-tuning pipeline. This week we go considerably deeper: deriving the mathematics from first principles, understanding the reward model in detail, working through the PPO update rule as applied to language models, examining failure modes, and surveying the frontier of RLHF alternatives. By the end of this week students should be able to implement every stage of RLHF from scratch on a toy problem.

---

## 1. Why RLHF?

Supervised fine-tuning (SFT) teaches a model to imitate demonstrations. But human preferences are not always easily captured by demonstrations alone:

- A human annotator may prefer a response that is *more helpful* than the demonstration they wrote.
- Helpfulness, harmlessness, and honesty can trade off against each other in ways that are hard to specify as labelled examples.
- The model may learn to mimic the *style* of preferred responses (confident, verbose) without actually improving accuracy.

RLHF reframes the problem: instead of asking humans to write ideal completions, ask them to *compare* two completions and say which they prefer. This comparison signal is cheaper to collect and more reliable — humans are better at ranking than generating.

---

## 2. The RLHF Pipeline: Three Stages

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Supervised Fine-tuning (SFT)                          │
│  Pre-trained LLM  →  Fine-tune on curated demonstrations        │
│  Result: π_SFT  (a good starting policy)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Reward Model Training                                 │
│  Collect pairwise comparisons (y_w ≻ y_l | x)                  │
│  Train RM to assign scalar reward r(x, y)                       │
│  Result: r_φ  (a proxy for human preference)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: RL Optimisation (PPO)                                 │
│  Optimise π_θ to maximise E[r_φ(x,y)] - β·KL(π_θ || π_SFT)    │
│  Result: π_RLHF  (the final aligned model)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 1: Supervised Fine-tuning

The SFT model is the initialisation for RL. It is trained on a dataset of high-quality (prompt, response) pairs:

```
L_SFT = -∑_t log π_θ(y_t | x, y_{<t})
```

The SFT model serves two roles:
1. It is the **starting policy** for PPO — already a reasonable model that produces sensible completions.
2. It is the **reference policy** π_ref used in the KL penalty during RL to prevent the model from drifting too far.

---

## 4. Stage 2: The Reward Model

### 4.1 Preference Data Collection

Human annotators are shown a prompt x and two completions y_w (winner / preferred) and y_l (loser / dispreferred). They choose which they prefer. This produces a dataset:

```
D = {(x^(i), y_w^(i), y_l^(i))}_{i=1}^{N}
```

Annotators may be given a rubric covering:
- Helpfulness (does it answer the question?)
- Harmlessness (is it safe?)
- Honesty (is it factually accurate?)
- Instruction following (does it obey the format?)

### 4.2 The Bradley-Terry Model

We model human preference as arising from an underlying latent quality score r*(x, y):

```
P(y_w ≻ y_l | x) = σ(r*(x, y_w) - r*(x, y_l))
```

where σ is the sigmoid function. This is the **Bradley-Terry** model of paired comparisons.

### 4.3 Training the Reward Model

We initialise the reward model from the SFT model and add a linear head that maps the final hidden state to a scalar:

```
r_φ(x, y) = W^T h_{final}(x, y)
```

The reward model is trained to minimise the negative log-likelihood of the observed preferences:

```
L_RM = -E_{(x,y_w,y_l) ~ D} [ log σ(r_φ(x, y_w) - r_φ(x, y_l)) ]
```

This is equivalent to binary cross-entropy on the preference pair, where the label is always "y_w is better."

### 4.4 Reward Model Properties

- **Score calibration**: the absolute value of r_φ is arbitrary; only differences matter.
- **Distribution shift**: the RM is trained on SFT completions. As RL shifts the policy, the RM may be evaluated on out-of-distribution completions.
- **Ensemble rewards**: training multiple RMs and averaging (or taking the minimum) reduces the risk of reward hacking.
- **Process reward models (PRMs)**: instead of scoring the final answer, score each reasoning step. Better for multi-step tasks but much more expensive to annotate.

---

## 5. Stage 3: Reinforcement Learning with PPO

### 5.1 The RL Objective

We treat the language model as a **policy** π_θ that maps a prompt x to a completion y. The goal is to maximise:

```
J(θ) = E_{x ~ D, y ~ π_θ(·|x)} [ r_φ(x, y) ] - β · KL(π_θ(·|x) || π_ref(·|x))
```

The two terms:
- **r_φ(x, y)**: the reward from the trained reward model — the signal we want to maximise.
- **β · KL(π_θ || π_ref)**: a KL divergence penalty that keeps the policy close to the SFT reference policy. This prevents the policy from collapsing to reward-hacking completions that score well on r_φ but are degenerate.

The KL penalty can be absorbed into a per-token reward:

```
r̃(x, y) = r_φ(x, y) - β · ∑_t log [π_θ(y_t | x, y_{<t}) / π_ref(y_t | x, y_{<t})]
```

### 5.2 Proximal Policy Optimisation (PPO)

PPO (Schulman et al., 2017) is the standard RL algorithm used in RLHF. It is an on-policy actor-critic method.

#### Key components:

**Actor** (the LLM being trained): π_θ — generates completions.

**Critic** (value function): V_ψ(x, y_{<t}) — estimates the expected future reward from state (x, y_{<t}).

**Advantage estimate**: how much better is action y_t than the average?

```
A_t = r̃_t + γ V_ψ(s_{t+1}) - V_ψ(s_t)
```

In practice for LLMs, with γ = 1 and using GAE (Generalised Advantage Estimation):

```
A_t^{GAE} = ∑_{k=0}^{T-t} (γλ)^k δ_{t+k}    where δ_t = r̃_t + V_ψ(s_{t+1}) - V_ψ(s_t)
```

**Clipped surrogate objective**:

```
L_PPO = E_t [ min(ρ_t A_t,  clip(ρ_t, 1-ε, 1+ε) A_t) ]
```

where ρ_t = π_θ(y_t | x, y_{<t}) / π_old(y_t | x, y_{<t}) is the importance sampling ratio.

The clip prevents excessively large updates when π_θ deviates far from π_old.

### 5.3 The PPO-RLHF Training Loop

```
for each iteration:
    1. Sample prompts x ~ D
    2. Generate completions y ~ π_θ(·|x)           [rollout]
    3. Score completions: r = r_φ(x, y)             [reward]
    4. Compute KL penalty against π_ref             [regularise]
    5. Compute advantages using critic V_ψ          [estimate]
    6. Update π_θ with clipped PPO objective        [actor update]
    7. Update V_ψ with MSE loss on returns          [critic update]
```

This is computationally expensive: at each step we need four model forward passes — the actor, critic, reference policy, and reward model.

### 5.4 Practical Tricks

- **Mini-batch PPO**: collect a rollout buffer, then perform multiple gradient steps on mini-batches before collecting new rollouts.
- **Reward normalisation**: normalise rewards to zero mean, unit variance within each batch. Stabilises training.
- **KL controller**: adaptively adjust β to maintain a target KL divergence (e.g. KL ≈ 0.1 nats).
- **Entropy bonus**: add a small entropy term to the reward to prevent the policy from becoming deterministic too quickly.
- **Value clipping**: clip the value function update similarly to the policy update.

---

## 6. Reward Hacking

**Reward hacking** (Amodei et al., 2016; Skalse et al., 2022) occurs when the policy finds behaviours that achieve high reward model scores without actually satisfying the underlying human preference.

### 6.1 Examples

| Behaviour | Why it scores highly | Why it is bad |
|-----------|---------------------|---------------|
| Very long responses | RMs often reward thoroughness | Wastes tokens; buries the answer |
| Excessive hedging | RMs reward safety | Becomes unhelpful ("I can't be sure but...") |
| Sycophantic agreement | RMs reward positively-toned responses | Confirms user misconceptions |
| Repetition | Padding to seem comprehensive | Reduces information density |
| Confident incorrect answers | Confidence may be rewarded | Hallucination |

### 6.2 Measuring Reward Hacking

Monitor the **KL divergence** from π_ref throughout training. Rapid KL growth without corresponding human evaluation improvement signals reward hacking.

The **reward-KL frontier**: as β decreases (weaker KL penalty), reward increases but so does the gap between RM score and true human preference. Optimal β is task-dependent.

### 6.3 Mitigations

- **Human evaluation checkpoints**: periodically evaluate the policy with real human raters, not just the RM.
- **RM ensemble**: use multiple independently trained reward models; if they disagree, treat the completion as uncertain.
- **Constitutional AI**: use the LLM itself to critique and revise outputs against explicit principles (Bai et al., 2022).
- **Iterative data collection**: continually collect new preference data on the current policy's completions, retrain the RM, and re-run RL.

---

## 7. Variants and Alternatives to PPO-RLHF

### 7.1 Direct Preference Optimisation (DPO)

Rafailov et al. (2023) show that the RLHF objective with a KL constraint has a closed-form optimal policy:

```
π*(y|x) = π_ref(y|x) exp(r*(x,y)/β) / Z(x)
```

Substituting back and rearranging, the reward can be expressed in terms of the policy:

```
r*(x,y) = β log [π*(y|x) / π_ref(y|x)] + β log Z(x)
```

Plugging this into the Bradley-Terry preference model and optimising directly over π_θ gives the **DPO loss**:

```
L_DPO(θ) = -E_{(x,y_w,y_l)} [ log σ( β log[π_θ(y_w|x)/π_ref(y_w|x)] - β log[π_θ(y_l|x)/π_ref(y_l|x)] ) ]
```

DPO advantages:
- No reward model needed — trains directly on preference pairs.
- No RL loop — standard cross-entropy style training.
- Simpler to implement and typically more stable.

DPO limitations:
- Requires a good reference policy π_ref.
- Less flexible when preference data is scarce or noisy.
- May underperform PPO on tasks requiring careful exploration.

### 7.2 REINFORCE Leave One Out (RLOO)

A simpler RL algorithm than PPO for language models: for each prompt, sample K completions and use the mean reward of the other K-1 as a baseline for each:

```
∇J ≈ ∑_k (r_k - mean_{j≠k} r_j) ∇ log π_θ(y_k | x)
```

Lower variance than vanilla REINFORCE; no separate value network required.

### 7.3 GRPO (Group Relative Policy Optimisation)

DeepSeek-R1 (2025): samples a group of completions per prompt, normalises rewards within the group, and uses the normalised advantage for the PPO-style update. No critic needed; trains efficiently on mathematical reasoning tasks.

### 7.4 Iterative DPO / Online DPO

Generate new completions with the current policy, score them with the RM, construct new preference pairs, and run DPO. Bridges the gap between offline DPO and online PPO — better coverage of the current policy's distribution.

### 7.5 Rejection Sampling Fine-tuning (RFT)

Generate K completions per prompt, keep only those with reward above a threshold, and fine-tune on them with SFT. Simple and effective; used by LLaMA-2 (Touvron et al., 2023) as a first pass before PPO.

### 7.6 Comparison Summary

| Method | RM needed | RL loop | Stability | Data efficiency |
|--------|-----------|---------|-----------|----------------|
| SFT | No | No | High | Moderate |
| Rejection sampling | Yes | No | High | Low |
| DPO | No (π_ref needed) | No | High | Moderate |
| RLOO | Yes | Yes | Medium | Moderate |
| PPO-RLHF | Yes | Yes | Low | High |
| Iterative DPO | Yes | Partial | Medium | High |
| GRPO | Yes | Yes | Medium | High |

---

## 8. Constitutional AI

Bai et al. (2022, Anthropic) replace human preference labels with AI-generated critiques.

### Two-stage process:

**Stage 1 — Supervised Learning from AI Feedback (SL-CAI)**:
1. Prompt the model to generate a potentially harmful response.
2. Ask the model to critique the response according to a *constitution* (a list of principles).
3. Ask the model to revise the response to better satisfy the principles.
4. Fine-tune on the revised responses.

**Stage 2 — Reinforcement Learning from AI Feedback (RLAIF)**:
1. Generate pairs of responses.
2. Ask the model (or a separate model) to score which is more constitutional.
3. Train a reward model on these AI-generated preference labels.
4. Run PPO with this RM.

Benefits: scalable (no human raters needed for RM training), more consistent than crowd-sourced annotation, allows explicit statement of values.

---

## 9. Process Reward Models (PRMs)

Standard reward models score the final output. For multi-step reasoning tasks (mathematics, code), this is a weak signal — the model may reach the right answer via wrong reasoning, or wrong answer via mostly correct reasoning.

**Process reward models** assign a reward to each reasoning step:

```
r_total = ∑_t r_t(x, y_{≤t})
```

Benefits:
- Denser reward signal for long-horizon reasoning.
- Can identify exactly which step goes wrong.
- Enables step-level beam search (selecting the best reasoning path at each step).

Drawbacks:
- Extremely expensive to annotate (annotators must verify each step).
- Step boundaries are often ambiguous.

OpenAI's Let's Verify Step by Step (Lightman et al., 2023) shows PRMs outperform outcome reward models on MATH.

---

## 10. Annotator Agreement and Reward Model Quality

Human annotation is noisy:
- **Inter-annotator agreement** is typically 60–80% on preference tasks.
- Annotators may have different cultural backgrounds, expertise levels, or interpretations of the rubric.
- Annotators may be susceptible to length bias (preferring longer responses regardless of quality).

Implications:
- The RM is an imperfect proxy; some irreducible noise is unavoidable.
- Aggregate over multiple annotators per pair where possible.
- Measure and report inter-annotator agreement (Krippendorff's α or Cohen's κ).
- Collect annotations from diverse annotator pools.

---

## 11. Evaluation of RLHF-Trained Models

### 11.1 Win Rate

Compare the RLHF model against the SFT baseline using human evaluation or LLM-as-judge:

```
Win Rate = (# prompts where RLHF is preferred) / (# total prompts)
```

A win rate > 50% indicates improvement.

### 11.2 MT-Bench and AlpacaEval

- **MT-Bench** (Zheng et al., 2023): 80 multi-turn questions across 8 categories; GPT-4 as judge.
- **AlpacaEval**: 805 prompts; win rate against text-davinci-003.

### 11.3 Reward Model Score vs Human Preference

Plot RM score vs human preference rating on a held-out set. The correlation should be high early in training; degradation in correlation signals reward hacking.

---

## 12. Practical This Week

See `practicals/week13_practical.py`:
- Implement the full three-stage RLHF pipeline on a toy text classification task.
- Train a reward model using Bradley-Terry loss on a synthetic preference dataset.
- Implement a simplified PPO update for a language model policy.
- Implement and compare DPO on the same preference dataset.
- Plot RM score, KL divergence, and win rate throughout training and identify signs of reward hacking.

---

## 13. Further Reading

- Christiano et al. (2017) — "Deep Reinforcement Learning from Human Preferences" — https://arxiv.org/abs/1706.03741
- Ouyang et al. (2022) — "Training language models to follow instructions with human feedback" (InstructGPT) — https://arxiv.org/abs/2203.02155
- Bai et al. (2022) — "Constitutional AI" — https://arxiv.org/abs/2212.08073
- Rafailov et al. (2023) — "Direct Preference Optimisation" — https://arxiv.org/abs/2305.18290
- Schulman et al. (2017) — "Proximal Policy Optimisation" — https://arxiv.org/abs/1707.06347
- Lightman et al. (2023) — "Let's Verify Step by Step" (PRMs) — https://arxiv.org/abs/2305.20050
- Skalse et al. (2022) — "Defining and Characterizing Reward Hacking" — https://arxiv.org/abs/2209.13085
- DeepSeek-R1 (2025) — "Incentivising Reasoning Capability in LLMs via RL" — https://arxiv.org/abs/2501.12948
- Touvron et al. (2023) — "LLaMA 2" (Section on RLHF) — https://arxiv.org/abs/2307.09288

---

## Discussion Questions

1. Derive the DPO loss from the RLHF objective with a KL constraint. What key assumption allows the reward model to be eliminated?
2. Explain reward hacking with two concrete examples from deployed systems. Why does the KL penalty not fully prevent it?
3. Compare PPO-RLHF and DPO in terms of data requirements, training stability, and computational cost. In what scenarios would you choose each?
4. A team wants to apply RLHF to a clinical decision support system. What are the risks of using crowd-sourced annotators to generate preference data? Propose an annotation protocol that mitigates these risks.
5. What is a process reward model and why might it outperform an outcome reward model for mathematical reasoning tasks?
