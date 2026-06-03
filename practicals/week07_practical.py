"""
Week 7 Practical: Fine-tuning and RLHF
========================================
Objectives:
  - Fine-tune a small LLM with LoRA on a custom instruction dataset
  - Compare base model vs fine-tuned model outputs
  - Implement a toy reward model from scratch
  - Simulate a DPO training step
"""

import os
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 1. A toy instruction dataset
# ─────────────────────────────────────────────────────────────────────────────
INSTRUCTION_DATA = [
    {"instruction": "What is the capital of France?",
     "response": "The capital of France is Paris."},
    {"instruction": "Explain what a neural network is in one sentence.",
     "response": "A neural network is a computational model inspired by biological neurons that learns patterns from data through interconnected layers of parameters."},
    {"instruction": "Write a Python function to add two numbers.",
     "response": "def add(a, b):\n    return a + b"},
    {"instruction": "What is the boiling point of water in Celsius?",
     "response": "Water boils at 100 degrees Celsius at standard atmospheric pressure."},
    {"instruction": "Summarise the transformer architecture in two sentences.",
     "response": "The transformer uses self-attention to model relationships between all tokens in a sequence simultaneously. It consists of stacked encoder and/or decoder blocks containing multi-head attention and feed-forward layers connected by residual connections and layer normalisation."},
    {"instruction": "What does RLHF stand for?",
     "response": "RLHF stands for Reinforcement Learning from Human Feedback."},
    {"instruction": "List three applications of large language models.",
     "response": "Three applications of LLMs are: (1) code generation and review, (2) document summarisation, and (3) question-answering assistants."},
    {"instruction": "What is gradient descent?",
     "response": "Gradient descent is an optimisation algorithm that iteratively adjusts model parameters in the direction of steepest descent of the loss function."},
] * 10   # repeat to get more training examples


def format_instruction(item: dict) -> str:
    return f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. LoRA implementation from scratch
# ─────────────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """
    A Linear layer augmented with a low-rank LoRA adapter.
    The original weight W is frozen; only A and B are updated.
    """
    def __init__(self, original_linear: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        d_out, d_in = original_linear.weight.shape
        self.r      = r
        self.alpha  = alpha
        self.scale  = alpha / r

        # Frozen original weight
        self.weight = original_linear.weight
        self.bias   = original_linear.bias
        self.weight.requires_grad_(False)

        # Trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(r, d_in) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base_out + lora_out

    def n_trainable_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()

    def n_total_params(self) -> int:
        return self.weight.numel() + (self.bias.numel() if self.bias is not None else 0) + \
               self.lora_A.numel() + self.lora_B.numel()


def inject_lora(model: nn.Module, target_names: list[str], r: int = 4) -> nn.Module:
    """Replace named Linear sub-modules with LoRALinear."""
    for name, module in list(model.named_modules()):
        for target in target_names:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent_name, attr = name.rsplit(".", 1)
                parent = model
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
                setattr(parent, attr, LoRALinear(module, r=r))
    return model


def count_trainable(model: nn.Module):
    total   = sum(p.numel() for p in model.parameters())
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trained:,}  ({100*trained/total:.2f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Toy fine-tuning with a character model + LoRA
# ─────────────────────────────────────────────────────────────────────────────
# Reuse the small GPT from week05 but add LoRA
import sys, importlib
from pathlib import Path

# We'll define an inline mini-model to avoid dependency on week05_practical.py
class ToyFFN(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class ToyLM(nn.Module):
    """A very small LM for demonstrating LoRA fine-tuning."""
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.emb   = nn.Embedding(vocab_size, d_model)
        self.ffns  = nn.ModuleList([ToyFFN(d_model) for _ in range(n_layers)])
        self.lns   = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head  = nn.Linear(d_model, vocab_size)
        self.emb.weight = self.head.weight

    def forward(self, idx: torch.Tensor, targets=None):
        x = self.emb(idx)
        for ln, ffn in zip(self.lns, self.ffns):
            x = x + ffn(ln(x))
        logits = self.head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1)) if targets is not None else None
        return logits, loss


def make_char_dataset(texts: list[str]) -> tuple[list[int], int, dict, dict]:
    all_text = "\n".join(texts)
    chars    = sorted(set(all_text))
    stoi     = {c: i for i, c in enumerate(chars)}
    itos     = {i: c for c, i in stoi.items()}
    ids      = [stoi[c] for c in all_text]
    return ids, len(chars), stoi, itos


# ─────────────────────────────────────────────────────────────────────────────
# 4. Toy reward model
# ─────────────────────────────────────────────────────────────────────────────
class ToyRewardModel(nn.Module):
    """
    A reward model that scores (prompt, completion) pairs.
    For demonstration: trained on labelled preference pairs.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def bradley_terry_loss(r_w: torch.Tensor, r_l: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry loss for preference learning.
    r_w: rewards for preferred completions
    r_l: rewards for dispreferred completions
    """
    return -F.logsigmoid(r_w - r_l).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 5. DPO loss
# ─────────────────────────────────────────────────────────────────────────────
def dpo_loss(pi_logps_w: torch.Tensor, pi_logps_l: torch.Tensor,
             ref_logps_w: torch.Tensor, ref_logps_l: torch.Tensor,
             beta: float = 0.1) -> torch.Tensor:
    """
    Direct Preference Optimisation loss (Rafailov et al., 2023).
    pi_logps_w:  log P_θ(y_w | x) — policy log probs for preferred completion
    pi_logps_l:  log P_θ(y_l | x) — policy log probs for dispreferred completion
    ref_logps_w: log P_ref(y_w | x)
    ref_logps_l: log P_ref(y_l | x)
    """
    log_ratio_w = pi_logps_w - ref_logps_w
    log_ratio_l = pi_logps_l - ref_logps_l
    return -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Task 1: Inspect LoRA parameter savings ────────────────────────────────
    print("=" * 60)
    print("TASK 1: LoRA parameter analysis")
    print("=" * 60)
    print(f"\n{'d_in':>6} {'d_out':>6} {'rank r':>7} | {'Full ΔW':>10} {'LoRA A+B':>10} {'Ratio':>8}")
    for d in [768, 2048, 4096]:
        for r in [4, 8, 16]:
            full  = d * d
            lora  = r * d + d * r
            ratio = full / lora
            print(f"{d:>6} {d:>6} {r:>7} | {full:>10,} {lora:>10,} {ratio:>8.1f}×")

    # ── Task 2: Fine-tune a toy model with LoRA ───────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 2: Fine-tuning a toy LM with LoRA")
    print("=" * 60)

    texts = [format_instruction(d) for d in INSTRUCTION_DATA]
    ids, vocab_sz, stoi, itos = make_char_dataset(texts)
    data = torch.tensor(ids, dtype=torch.long)
    block_size = 64

    base_model = ToyLM(vocab_sz, d_model=128, n_layers=3)
    n_base = sum(p.numel() for p in base_model.parameters())
    print(f"\nBase model parameters: {n_base:,}")

    # Freeze base model
    for p in base_model.parameters():
        p.requires_grad_(False)

    # Inject LoRA into FFN layers
    lora_model = inject_lora(base_model, target_names=["fc1", "fc2"], r=4)
    print("After LoRA injection:")
    count_trainable(lora_model)

    # Train with LoRA
    opt = torch.optim.AdamW(
        [p for p in lora_model.parameters() if p.requires_grad], lr=1e-3
    )
    losses = []
    for step in range(500):
        ix  = torch.randint(len(data) - block_size, (16,))
        x   = torch.stack([data[i: i + block_size] for i in ix])
        y   = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
        _, loss = lora_model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    plt.figure(figsize=(8, 4))
    window = 20
    smooth = [sum(losses[max(0, i-window):i+1]) / min(window, i+1) for i in range(len(losses))]
    plt.plot(losses, alpha=0.3, label="Raw loss")
    plt.plot(smooth, label="Smoothed loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("LoRA Fine-tuning Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("week07_lora_training.png", dpi=150)
    plt.show()
    print(f"Final loss: {losses[-1]:.4f}")

    # ── Task 3: Toy reward model ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 3: Toy reward model with Bradley-Terry loss")
    print("=" * 60)

    # Simulate preference dataset: preferred vs dispreferred response embeddings
    random.seed(42)
    torch.manual_seed(42)

    def make_fake_embedding(quality: float, d: int = 32) -> torch.Tensor:
        """Higher quality → embedding closer to a 'good' prototype."""
        good_proto = torch.ones(d)
        bad_proto  = -torch.ones(d)
        emb = quality * good_proto + (1 - quality) * bad_proto
        return emb + torch.randn(d) * 0.3

    N_PAIRS = 200
    qualities_w = torch.tensor([random.uniform(0.6, 1.0) for _ in range(N_PAIRS)])
    qualities_l = torch.tensor([random.uniform(0.0, 0.5) for _ in range(N_PAIRS)])
    X_w = torch.stack([make_fake_embedding(q.item()) for q in qualities_w])
    X_l = torch.stack([make_fake_embedding(q.item()) for q in qualities_l])

    reward_model = ToyRewardModel(input_dim=32)
    rm_opt = torch.optim.Adam(reward_model.parameters(), lr=1e-3)

    rm_losses = []
    for epoch in range(200):
        r_w = reward_model(X_w)
        r_l = reward_model(X_l)
        loss = bradley_terry_loss(r_w, r_l)
        rm_opt.zero_grad()
        loss.backward()
        rm_opt.step()
        rm_losses.append(loss.item())

    plt.figure(figsize=(8, 4))
    plt.plot(rm_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Bradley-Terry Loss")
    plt.title("Reward Model Training")
    plt.tight_layout()
    plt.savefig("week07_reward_model.png", dpi=150)
    plt.show()
    print(f"Final RM loss: {rm_losses[-1]:.4f}")

    with torch.no_grad():
        r_preferred   = reward_model(X_w).mean().item()
        r_dispreferred = reward_model(X_l).mean().item()
    print(f"Mean reward (preferred)   : {r_preferred:.4f}")
    print(f"Mean reward (dispreferred): {r_dispreferred:.4f}")
    assert r_preferred > r_dispreferred, "Reward model should prefer good completions!"
    print("Reward model correctly ranks preferred > dispreferred. ✓")

    # ── Task 4: DPO loss demonstration ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 4: DPO loss demonstration")
    print("=" * 60)

    # Simulate log probabilities
    torch.manual_seed(0)
    batch = 8
    # Policy assigns higher log-prob to preferred (correct training signal)
    pi_logps_w   = torch.randn(batch) - 1.0   # higher (less negative)
    pi_logps_l   = torch.randn(batch) - 3.0   # lower
    ref_logps_w  = torch.randn(batch) - 2.0
    ref_logps_l  = torch.randn(batch) - 2.0

    loss_val = dpo_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, beta=0.1)
    print(f"DPO loss (policy prefers correct answer): {loss_val.item():.4f}")

    # Swap to simulate wrong policy
    loss_wrong = dpo_loss(pi_logps_l, pi_logps_w, ref_logps_w, ref_logps_l, beta=0.1)
    print(f"DPO loss (policy prefers wrong answer):   {loss_wrong.item():.4f}")
    print("Higher loss when policy is wrong — DPO gradient pushes the right direction. ✓")

    print("\nAll tasks complete.")
