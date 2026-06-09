"""
Week 13 Practical: Reinforcement Learning from Human Feedback (RLHF)
======================================================================
Objectives:
  - Implement the three-stage RLHF pipeline from scratch on a toy problem
  - Train a reward model using Bradley-Terry loss on synthetic preferences
  - Implement a simplified PPO update loop for a language model policy
  - Implement DPO on the same preference dataset
  - Visualise RM score, KL divergence, and win rate across training
  - Observe and diagnose reward hacking

Problem setup
-------------
We use a tiny character-level LM (the "policy") and a synthetic reward
function: responses are preferred when they contain more unique words and
are factually grounded (simulated). This lets us run the full pipeline
on CPU in a few minutes without needing API keys.
"""

import math
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 0. Toy vocabulary and policy LM
# ─────────────────────────────────────────────────────────────────────────────
VOCAB = [
    "<pad>", "<eos>",
    "the", "a", "is", "are", "was", "be", "not",
    "good", "bad", "great", "poor", "excellent", "terrible",
    "answer", "question", "response", "helpful", "harmful",
    "yes", "no", "maybe", "always", "never",
    "I", "you", "we", "they", "it",
    "can", "should", "will", "must", "may",
    "safe", "unsafe", "honest", "dishonest",
    "because", "therefore", "however", "but", "and",
    ".", ",", "?", "!"
]
V = len(VOCAB)
stoi = {w: i for i, w in enumerate(VOCAB)}
itos = {i: w for w, i in stoi.items()}
PAD_ID = stoi["<pad>"]
EOS_ID = stoi["<eos>"]
MAX_LEN = 12


class TinyLM(nn.Module):
    """
    A tiny transformer-like LM for the RLHF toy experiment.
    Input: token ids (batch, seq_len)
    Output: logits over vocabulary (batch, seq_len, V)
    """
    def __init__(self, vocab_size: int = V, d: int = 64, n_layers: int = 2):
        super().__init__()
        self.emb   = nn.Embedding(vocab_size, d, padding_idx=PAD_ID)
        self.pos   = nn.Embedding(MAX_LEN + 2, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=128,
                                       dropout=0.0, batch_first=True)
            for _ in range(n_layers)
        ])
        self.head  = nn.Linear(d, vocab_size)
        self.d = d

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos_ids = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.emb(idx) + self.pos(pos_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)   # (B, T, V)

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], max_new: int = MAX_LEN,
                 temperature: float = 1.0) -> list[int]:
        ids = list(prompt_ids)
        for _ in range(max_new):
            inp = torch.tensor([ids], device=DEVICE)
            logits = self(inp)[0, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)
            if next_id == EOS_ID:
                break
        return ids

    def log_prob_sequence(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the sum of log probabilities for a sequence.
        ids: (B, T)  — includes both prompt and response tokens
        Returns: (B,)
        """
        logits = self(ids[:, :-1])              # (B, T-1, V)
        targets = ids[:, 1:]                    # (B, T-1)
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp  = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        # Mask padding
        mask = (targets != PAD_ID).float()
        return (token_lp * mask).sum(dim=-1)    # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic reward function (ground truth, never seen by the RM)
# ─────────────────────────────────────────────────────────────────────────────
GOOD_WORDS  = {"good", "great", "excellent", "helpful", "safe", "honest",
               "yes", "can", "should", "therefore"}
BAD_WORDS   = {"bad", "poor", "terrible", "harmful", "unsafe", "dishonest",
               "never", "not", "no"}

def true_reward(token_ids: list[int]) -> float:
    """
    A ground-truth reward based on:
      - diversity (unique tokens / total tokens)
      - positive word count
      - length (moderate length preferred)
    """
    tokens = [itos[i] for i in token_ids if i not in (PAD_ID, EOS_ID)]
    if not tokens:
        return -1.0
    unique_ratio  = len(set(tokens)) / max(len(tokens), 1)
    positive_score = sum(1 for t in tokens if t in GOOD_WORDS) / max(len(tokens), 1)
    negative_score = sum(1 for t in tokens if t in BAD_WORDS) / max(len(tokens), 1)
    length_bonus   = 1.0 if 5 <= len(tokens) <= 10 else 0.5
    return float(unique_ratio + 2 * positive_score - 2 * negative_score + length_bonus)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Generate synthetic preference dataset
# ─────────────────────────────────────────────────────────────────────────────
def make_preference_dataset(policy: TinyLM, n_prompts: int = 200,
                             n_pairs_per_prompt: int = 2) -> list[dict]:
    """
    For each prompt, sample two completions. The one with higher true_reward
    is labelled as preferred (y_w). Add 15% label noise to simulate annotators.
    """
    dataset = []
    prompts = [[stoi["the"], stoi["answer"], stoi["is"]],
               [stoi["I"], stoi["can"], stoi["be"]],
               [stoi["it"], stoi["is"], stoi["a"]],
               [stoi["you"], stoi["should"], stoi["be"]]]

    for _ in range(n_prompts):
        prompt = random.choice(prompts)
        for _ in range(n_pairs_per_prompt):
            y1_ids = policy.generate(prompt, temperature=1.2)
            y2_ids = policy.generate(prompt, temperature=1.2)
            r1 = true_reward(y1_ids)
            r2 = true_reward(y2_ids)

            # Introduce 15% annotator noise
            if random.random() < 0.15:
                r1, r2 = r2, r1

            if r1 >= r2:
                y_w, y_l = y1_ids, y2_ids
            else:
                y_w, y_l = y2_ids, y1_ids

            dataset.append({"prompt": prompt, "y_w": y_w, "y_l": y_l,
                             "r_w": max(r1, r2), "r_l": min(r1, r2)})
    return dataset


def pad_sequence(ids: list[int], max_len: int) -> torch.Tensor:
    padded = ids[:max_len] + [PAD_ID] * max(0, max_len - len(ids))
    return torch.tensor(padded, dtype=torch.long)


def batch_to_tensors(pairs: list[dict], key: str) -> torch.Tensor:
    return torch.stack([pad_sequence(p[key], MAX_LEN + 2) for p in pairs]).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Stage 2: Reward model
# ─────────────────────────────────────────────────────────────────────────────
class RewardModel(nn.Module):
    """
    Wraps TinyLM and adds a scalar head.
    r(x, y) = linear(mean_pool(LM_hidden(x, y)))
    """
    def __init__(self, lm: TinyLM):
        super().__init__()
        self.lm   = copy.deepcopy(lm)
        self.head = nn.Linear(lm.d, 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T)  →  scalar reward (B,)"""
        B, T = ids.shape
        pos_ids = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.lm.emb(ids) + self.lm.pos(pos_ids)
        for layer in self.lm.layers:
            x = layer(x)
        # Mean-pool over non-padding tokens
        mask = (ids != PAD_ID).float().unsqueeze(-1)   # (B, T, 1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)           # (B,)


def train_reward_model(rm: RewardModel, dataset: list[dict],
                       n_epochs: int = 30, batch_size: int = 32,
                       lr: float = 1e-3) -> list[float]:
    opt = torch.optim.Adam(rm.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        random.shuffle(dataset)
        epoch_losses = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i: i + batch_size]
            yw = batch_to_tensors(batch, "y_w")
            yl = batch_to_tensors(batch, "y_l")
            r_w = rm(yw)
            r_l = rm(yl)
            # Bradley-Terry loss
            loss = -F.logsigmoid(r_w - r_l).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        avg = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg)
        if (epoch + 1) % 5 == 0:
            # Accuracy: does RM rank y_w > y_l?
            with torch.no_grad():
                all_yw = batch_to_tensors(dataset, "y_w")
                all_yl = batch_to_tensors(dataset, "y_l")
                acc = ((rm(all_yw) > rm(all_yl)).float().mean().item())
            print(f"  RM epoch {epoch+1:3d} | loss={avg:.4f} | acc={acc:.3f}")
    return losses


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stage 3a: PPO (simplified)
# ─────────────────────────────────────────────────────────────────────────────
class ValueHead(nn.Module):
    """Critic: estimates V(state) = expected future reward."""
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)


def ppo_step(policy: TinyLM, ref_policy: TinyLM, value_head: ValueHead,
             rm: RewardModel, prompts: list[list[int]],
             beta: float = 0.05, clip_eps: float = 0.2,
             lr_actor: float = 1e-4, lr_critic: float = 1e-3,
             n_updates: int = 4) -> dict:
    """One PPO iteration: rollout → score → update."""

    actor_opt  = torch.optim.Adam(policy.parameters(), lr=lr_actor)
    critic_opt = torch.optim.Adam(value_head.parameters(), lr=lr_critic)

    # ── Rollout ──
    rollouts = []
    for prompt in prompts:
        with torch.no_grad():
            y_ids = policy.generate(prompt, temperature=0.9)
        rollouts.append({"prompt": prompt, "y": y_ids})

    y_tensors = torch.stack([pad_sequence(r["y"], MAX_LEN + 2) for r in rollouts]).to(DEVICE)

    # ── Score with RM ──
    with torch.no_grad():
        rm_scores = rm(y_tensors).cpu().numpy()

    # ── KL penalty ──
    with torch.no_grad():
        pi_lp  = policy.log_prob_sequence(y_tensors)
        ref_lp = ref_policy.log_prob_sequence(y_tensors)
        kl = (pi_lp - ref_lp).cpu().numpy()

    rewards = rm_scores - beta * kl   # shape (N,)

    # ── Value estimate ──
    with torch.no_grad():
        pos_ids = torch.arange(y_tensors.size(1), device=DEVICE).unsqueeze(0)
        hidden  = policy.emb(y_tensors) + policy.pos(pos_ids)
        for layer in policy.layers:
            hidden = layer(hidden)
        mean_hidden = hidden.mean(dim=1)   # (N, d)
        values = value_head(mean_hidden).cpu().numpy()

    advantages = rewards - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns    = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)

    # ── Old log probs ──
    with torch.no_grad():
        old_lp = policy.log_prob_sequence(y_tensors)

    # ── PPO update ──
    for _ in range(n_updates):
        # Actor
        new_lp = policy.log_prob_sequence(y_tensors)
        ratio  = torch.exp(new_lp - old_lp.detach())
        surr1  = ratio * advantages_t
        surr2  = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_t
        actor_loss = -torch.min(surr1, surr2).mean()

        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        actor_opt.step()

        # Critic
        pos_ids = torch.arange(y_tensors.size(1), device=DEVICE).unsqueeze(0)
        h = policy.emb(y_tensors) + policy.pos(pos_ids)
        for layer in policy.layers:
            h = layer(h)
        v_pred = value_head(h.mean(dim=1))
        critic_loss = F.mse_loss(v_pred, returns)

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

    mean_kl = float(np.mean(np.abs(kl)))
    mean_rm  = float(np.mean(rm_scores))
    return {"mean_rm_score": mean_rm, "mean_kl": mean_kl,
            "mean_reward": float(np.mean(rewards)),
            "actor_loss": actor_loss.item()}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Stage 3b: DPO
# ─────────────────────────────────────────────────────────────────────────────
def dpo_train(policy: TinyLM, ref_policy: TinyLM, dataset: list[dict],
              beta: float = 0.1, n_epochs: int = 20,
              batch_size: int = 16, lr: float = 1e-4) -> list[float]:
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        random.shuffle(dataset)
        epoch_losses = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i: i + batch_size]
            yw = batch_to_tensors(batch, "y_w")
            yl = batch_to_tensors(batch, "y_l")

            pi_lp_w  = policy.log_prob_sequence(yw)
            pi_lp_l  = policy.log_prob_sequence(yl)
            with torch.no_grad():
                ref_lp_w = ref_policy.log_prob_sequence(yw)
                ref_lp_l = ref_policy.log_prob_sequence(yl)

            log_ratio_w = pi_lp_w - ref_lp_w
            log_ratio_l = pi_lp_l - ref_lp_l
            loss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            epoch_losses.append(loss.item())

        avg = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg)
        if (epoch + 1) % 5 == 0:
            print(f"  DPO epoch {epoch+1:3d} | loss={avg:.4f}")
    return losses


# ─────────────────────────────────────────────────────────────────────────────
# 6. Win rate evaluation (vs reference policy)
# ─────────────────────────────────────────────────────────────────────────────
EVAL_PROMPTS = [
    [stoi["the"], stoi["answer"], stoi["is"]],
    [stoi["I"], stoi["can"], stoi["be"]],
    [stoi["you"], stoi["should"], stoi["be"]],
    [stoi["it"], stoi["is"], stoi["a"]],
]

def evaluate_win_rate(policy: TinyLM, ref_policy: TinyLM,
                      n_samples: int = 50) -> float:
    """Compare policy vs ref_policy using true_reward as judge."""
    wins = 0
    for _ in range(n_samples):
        prompt = random.choice(EVAL_PROMPTS)
        y_policy = policy.generate(prompt, temperature=0.8)
        y_ref    = ref_policy.generate(prompt, temperature=0.8)
        if true_reward(y_policy) > true_reward(y_ref):
            wins += 1
    return wins / n_samples


def mean_true_reward(policy: TinyLM, n_samples: int = 100) -> float:
    rewards = []
    for _ in range(n_samples):
        prompt = random.choice(EVAL_PROMPTS)
        y = policy.generate(prompt, temperature=0.8)
        rewards.append(true_reward(y))
    return float(np.mean(rewards))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Stage 1: SFT model (pre-trained stand-in) ─────────────────────────────
    print("=" * 60)
    print("STAGE 1: SFT policy (random initialisation as stand-in)")
    print("=" * 60)
    sft_policy = TinyLM().to(DEVICE)
    ref_policy = copy.deepcopy(sft_policy)   # frozen reference
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    baseline_reward = mean_true_reward(sft_policy)
    print(f"SFT baseline mean true reward: {baseline_reward:.4f}")

    # ── Stage 2: Reward model ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: Collecting preferences and training reward model")
    print("=" * 60)
    print("Generating preference dataset…")
    pref_data = make_preference_dataset(sft_policy, n_prompts=300)
    print(f"  {len(pref_data)} preference pairs collected")
    # Split
    split_idx = int(0.9 * len(pref_data))
    train_pref = pref_data[:split_idx]
    eval_pref  = pref_data[split_idx:]

    rm = RewardModel(sft_policy).to(DEVICE)
    print("\nTraining reward model…")
    rm_losses = train_reward_model(rm, train_pref, n_epochs=30)

    # Evaluate RM accuracy on held-out pairs
    with torch.no_grad():
        yw_eval = batch_to_tensors(eval_pref, "y_w")
        yl_eval = batch_to_tensors(eval_pref, "y_l")
        rm_acc  = (rm(yw_eval) > rm(yl_eval)).float().mean().item()
    print(f"\nReward model held-out accuracy: {rm_acc:.3f}")

    # ── Stage 3a: PPO ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 3a: PPO-RLHF training")
    print("=" * 60)
    ppo_policy = copy.deepcopy(sft_policy).to(DEVICE)
    value_head = ValueHead(sft_policy.d).to(DEVICE)

    ppo_rm_scores, ppo_kls, ppo_true_rewards, ppo_win_rates = [], [], [], []
    N_PPO_ITERS  = 40
    PROMPTS_PER_ITER = 20

    for it in range(1, N_PPO_ITERS + 1):
        batch_prompts = [random.choice(EVAL_PROMPTS) for _ in range(PROMPTS_PER_ITER)]
        stats = ppo_step(ppo_policy, ref_policy, value_head, rm,
                         batch_prompts, beta=0.05, n_updates=3)
        true_r   = mean_true_reward(ppo_policy, n_samples=40)
        win_rate = evaluate_win_rate(ppo_policy, ref_policy, n_samples=40)

        ppo_rm_scores.append(stats["mean_rm_score"])
        ppo_kls.append(stats["mean_kl"])
        ppo_true_rewards.append(true_r)
        ppo_win_rates.append(win_rate)

        if it % 10 == 0:
            print(f"  PPO iter {it:3d} | RM={stats['mean_rm_score']:.3f} | "
                  f"KL={stats['mean_kl']:.3f} | true_r={true_r:.3f} | win={win_rate:.2f}")

    # ── Stage 3b: DPO ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 3b: DPO training")
    print("=" * 60)
    dpo_policy = copy.deepcopy(sft_policy).to(DEVICE)
    dpo_losses = dpo_train(dpo_policy, ref_policy, train_pref,
                           beta=0.1, n_epochs=40, batch_size=16)

    dpo_true_reward = mean_true_reward(dpo_policy, n_samples=100)
    dpo_win_rate    = evaluate_win_rate(dpo_policy, ref_policy, n_samples=100)
    print(f"\nDPO final mean true reward: {dpo_true_reward:.4f}")
    print(f"DPO win rate vs reference:  {dpo_win_rate:.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generating plots…")
    print("=" * 60)
    iters = list(range(1, N_PPO_ITERS + 1))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 1. RM training loss
    axes[0, 0].plot(rm_losses)
    axes[0, 0].set_title("Reward Model Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Bradley-Terry loss")

    # 2. PPO RM score over training
    axes[0, 1].plot(iters, ppo_rm_scores, color="steelblue", label="RM score")
    axes[0, 1].set_title("PPO: Reward Model Score")
    axes[0, 1].set_xlabel("PPO iteration")
    axes[0, 1].set_ylabel("Mean RM score")
    axes[0, 1].axhline(y=rm(batch_to_tensors(pref_data[:50], "y_l")).mean().item(),
                        color="red", linestyle="--", alpha=0.5, label="SFT baseline (approx)")
    axes[0, 1].legend()

    # 3. PPO KL divergence (reward hacking signal)
    axes[0, 2].plot(iters, ppo_kls, color="darkorange")
    axes[0, 2].set_title("PPO: KL Divergence from Reference")
    axes[0, 2].set_xlabel("PPO iteration")
    axes[0, 2].set_ylabel("|KL(π_θ || π_ref)|")
    axes[0, 2].axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Target KL")
    axes[0, 2].legend()

    # 4. PPO true reward (ground truth, not used in training)
    axes[1, 0].plot(iters, ppo_true_rewards, color="green")
    axes[1, 0].axhline(y=baseline_reward, color="gray", linestyle="--", label="SFT baseline")
    axes[1, 0].set_title("PPO: True Reward (Ground Truth)")
    axes[1, 0].set_xlabel("PPO iteration")
    axes[1, 0].set_ylabel("Mean true reward")
    axes[1, 0].legend()

    # 5. PPO win rate vs reference
    axes[1, 1].plot(iters, ppo_win_rates, color="purple")
    axes[1, 1].axhline(y=0.5, color="gray", linestyle="--", label="50% baseline")
    axes[1, 1].set_title("PPO: Win Rate vs Reference Policy")
    axes[1, 1].set_xlabel("PPO iteration")
    axes[1, 1].set_ylabel("Win rate")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()

    # 6. DPO training loss
    axes[1, 2].plot(dpo_losses, color="teal")
    axes[1, 2].set_title("DPO Training Loss")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("DPO loss")

    plt.suptitle("RLHF Experiment: PPO vs DPO on Toy Language Model", fontsize=13)
    plt.tight_layout()
    plt.savefig("week13_rlhf_results.png", dpi=150)
    plt.show()
    print("Saved: week13_rlhf_results.png")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY: SFT vs PPO vs DPO")
    print("=" * 60)
    print(f"{'Method':10} {'True reward':>14} {'Win rate vs SFT':>18}")
    print("-" * 44)
    print(f"{'SFT':10} {baseline_reward:>14.4f} {'0.500 (baseline)':>18}")
    print(f"{'PPO':10} {ppo_true_rewards[-1]:>14.4f} {ppo_win_rates[-1]:>18.3f}")
    print(f"{'DPO':10} {dpo_true_reward:>14.4f} {dpo_win_rate:>18.3f}")

    # ── Reward hacking analysis ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("REWARD HACKING ANALYSIS")
    print("=" * 60)
    print("Plotting RM score vs true reward to detect reward hacking…")
    # If RM score increases while true reward plateaus/decreases → reward hacking

    if len(ppo_rm_scores) > 5:
        rm_trend    = ppo_rm_scores[-5:] - np.array(ppo_rm_scores[:5])
        true_trend  = np.array(ppo_true_rewards[-5:]) - np.array(ppo_true_rewards[:5])
        print(f"  RM score change (last 5 vs first 5 iters): {float(np.mean(rm_trend)):+.3f}")
        print(f"  True reward change:                        {float(np.mean(true_trend)):+.3f}")
        if float(np.mean(rm_trend)) > 0.1 and float(np.mean(true_trend)) < 0:
            print("  ⚠ Possible reward hacking detected: RM score rising but true reward falling.")
        else:
            print("  ✓ No strong evidence of reward hacking in this run.")

    print("\nAll tasks complete.")
