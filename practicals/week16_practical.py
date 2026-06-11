"""
Week 16 Practical: Direct Preference Optimisation (DPO)
=========================================================
Objectives:
  - Implement DPO loss from scratch and verify diagnostics
  - Build a synthetic preference dataset from a toy LM
  - Train with DPO and monitor chosen/rejected rewards, margin, accuracy, KL
  - Implement IPO and SimPO and compare all three losses
  - Demonstrate likelihood displacement
  - Implement two-iteration online DPO and show improvement

No API key required — all experiments run on a tiny toy vocabulary and LM.
"""

import math
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Tiny vocabulary and LM  (reused from Week 13, self-contained here)
# ─────────────────────────────────────────────────────────────────────────────
VOCAB = [
    "<pad>", "<eos>",
    "the", "a", "is", "are", "was",
    "good", "bad", "great", "poor", "excellent", "terrible",
    "answer", "response", "helpful", "harmful",
    "yes", "no", "maybe",
    "I", "you", "we",
    "can", "should", "will",
    "safe", "unsafe", "honest",
    "because", "therefore", "however",
    ".", ",", "!",
]
V       = len(VOCAB)
stoi    = {w: i for i, w in enumerate(VOCAB)}
itos    = {i: w for w, i in stoi.items()}
PAD_ID  = stoi["<pad>"]
EOS_ID  = stoi["<eos>"]
MAX_LEN = 10

GOOD_WORDS = {"good", "great", "excellent", "helpful", "safe", "honest",
              "yes", "can", "should", "therefore"}
BAD_WORDS  = {"bad", "poor", "terrible", "harmful", "unsafe",
              "no", "however"}


class TinyLM(nn.Module):
    def __init__(self, vocab_size=V, d=64, n_layers=2):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, d, padding_idx=PAD_ID)
        self.pos  = nn.Embedding(MAX_LEN + 4, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=128,
                                       dropout=0.0, batch_first=True)
            for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab_size)
        self.d    = d

    def forward(self, idx):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        for layer in self.layers:
            x = layer(x)
        return self.head(x)          # (B, T, V)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new=MAX_LEN, temperature=1.0):
        ids = list(prompt_ids)
        for _ in range(max_new):
            inp    = torch.tensor([ids], device=DEVICE)
            logits = self(inp)[0, -1, :] / max(temperature, 1e-6)
            probs  = F.softmax(logits, dim=-1)
            nxt    = torch.multinomial(probs, 1).item()
            ids.append(nxt)
            if nxt == EOS_ID:
                break
        return ids


def pad_to(ids, length):
    return (ids + [PAD_ID] * length)[:length]


def sequence_log_prob(model, prompt_ids, completion_ids):
    """
    Sum of log P(completion_t | prompt + completion_{<t}) for each t.
    prompt_ids, completion_ids: (B, T) tensors
    Returns: (B,)
    """
    full      = torch.cat([prompt_ids, completion_ids], dim=1)   # (B, Tp+Tc)
    logits    = model(full[:, :-1])                               # (B, Tp+Tc-1, V)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    Tp        = prompt_ids.shape[1]
    # Completion positions start at index Tp-1 in the shifted sequence
    comp_logits = log_probs[:, Tp - 1:, :]                        # (B, Tc, V)
    token_lp    = comp_logits.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
    mask        = (completion_ids != PAD_ID).float()
    return (token_lp * mask).sum(dim=-1)                          # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  True reward for generating synthetic preferences
# ─────────────────────────────────────────────────────────────────────────────
def true_reward(token_ids):
    toks          = [itos[i] for i in token_ids if i not in (PAD_ID, EOS_ID)]
    if not toks:
        return -1.0
    unique_ratio  = len(set(toks)) / max(len(toks), 1)
    pos           = sum(1 for t in toks if t in GOOD_WORDS) / max(len(toks), 1)
    neg           = sum(1 for t in toks if t in BAD_WORDS)  / max(len(toks), 1)
    length_bonus  = 1.0 if 4 <= len(toks) <= 8 else 0.4
    return unique_ratio + 2 * pos - 2 * neg + length_bonus


PROMPTS = [
    [stoi["the"], stoi["answer"], stoi["is"]],
    [stoi["I"],   stoi["can"],    stoi["be"]],
    [stoi["you"], stoi["should"], stoi["be"]],
]

MAX_COMP = 8   # max completion tokens to score


def make_preference_dataset(policy, n_pairs=300, noise=0.10):
    """Sample two completions per prompt, label the better one as chosen."""
    dataset = []
    for _ in range(n_pairs):
        prompt = random.choice(PROMPTS)
        y1     = policy.generate(prompt, temperature=1.1)
        y2     = policy.generate(prompt, temperature=1.1)
        r1, r2 = true_reward(y1), true_reward(y2)
        if random.random() < noise:          # annotator noise
            r1, r2 = r2, r1
        chosen, rejected = (y1, y2) if r1 >= r2 else (y2, y1)
        Tp = len(prompt)
        dataset.append({
            "prompt":   pad_to(prompt,   Tp),
            "chosen":   pad_to(chosen[Tp:Tp + MAX_COMP],  MAX_COMP),
            "rejected": pad_to(rejected[Tp:Tp + MAX_COMP], MAX_COMP),
        })
    return dataset


def batch_tensors(batch, key, device=DEVICE):
    return torch.tensor([b[key] for b in batch], dtype=torch.long, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Loss functions
# ─────────────────────────────────────────────────────────────────────────────
def compute_dpo_loss(policy, ref_policy, prompt_t, chosen_t, rejected_t,
                     beta=0.1):
    """Standard DPO loss plus diagnostics."""
    pi_chosen_lp   = sequence_log_prob(policy,     prompt_t, chosen_t)
    pi_rejected_lp = sequence_log_prob(policy,     prompt_t, rejected_t)
    with torch.no_grad():
        ref_chosen_lp  = sequence_log_prob(ref_policy, prompt_t, chosen_t)
        ref_rejected_lp= sequence_log_prob(ref_policy, prompt_t, rejected_t)

    log_ratio_w = pi_chosen_lp   - ref_chosen_lp
    log_ratio_l = pi_rejected_lp - ref_rejected_lp
    loss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()

    with torch.no_grad():
        diag = {
            "chosen_reward":   (beta * log_ratio_w).mean().item(),
            "rejected_reward": (beta * log_ratio_l).mean().item(),
            "margin":          (beta * (log_ratio_w - log_ratio_l)).mean().item(),
            "accuracy":        (log_ratio_w > log_ratio_l).float().mean().item(),
            "chosen_lp":       pi_chosen_lp.mean().item(),
            "rejected_lp":     pi_rejected_lp.mean().item(),
        }
    return loss, diag


def compute_ipo_loss(policy, ref_policy, prompt_t, chosen_t, rejected_t,
                     beta=0.1):
    """
    IPO loss (Azar et al., 2023).
    Target: log-ratio margin = 1/(2β)
    """
    pi_chosen_lp   = sequence_log_prob(policy,     prompt_t, chosen_t)
    pi_rejected_lp = sequence_log_prob(policy,     prompt_t, rejected_t)
    with torch.no_grad():
        ref_chosen_lp  = sequence_log_prob(ref_policy, prompt_t, chosen_t)
        ref_rejected_lp= sequence_log_prob(ref_policy, prompt_t, rejected_t)

    log_ratio_diff = (pi_chosen_lp - ref_chosen_lp) - (pi_rejected_lp - ref_rejected_lp)
    target         = 1.0 / (2 * beta)
    loss           = ((log_ratio_diff - target) ** 2).mean()
    with torch.no_grad():
        diag = {"margin": log_ratio_diff.mean().item(),
                "accuracy": (log_ratio_diff > 0).float().mean().item()}
    return loss, diag


def compute_simpo_loss(policy, prompt_t, chosen_t, rejected_t,
                       beta=2.0, gamma=0.5):
    """
    SimPO loss (Meng et al., 2024).
    No reference model. Length-normalised log prob.
    gamma: target reward margin.
    """
    def length_norm_lp(completion_t):
        # Mean log prob over non-padding tokens
        lp_sum = sequence_log_prob(policy, prompt_t, completion_t)
        lengths = (completion_t != PAD_ID).float().sum(dim=-1).clamp(min=1)
        return lp_sum / lengths

    chosen_nlp   = length_norm_lp(chosen_t)
    rejected_nlp = length_norm_lp(rejected_t)
    loss = -F.logsigmoid(beta * (chosen_nlp - rejected_nlp) - gamma).mean()
    with torch.no_grad():
        diag = {"margin":   (chosen_nlp - rejected_nlp).mean().item(),
                "accuracy": (chosen_nlp > rejected_nlp).float().mean().item()}
    return loss, diag


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Training loop (generic)
# ─────────────────────────────────────────────────────────────────────────────
def train_dpo_variant(loss_fn, policy, dataset, n_epochs=30, batch_size=32,
                      lr=1e-3, label="DPO"):
    opt      = torch.optim.Adam(policy.parameters(), lr=lr)
    history  = {k: [] for k in ["loss", "margin", "accuracy",
                                  "chosen_lp", "rejected_lp"]}
    N        = len(dataset)
    Tp       = len(dataset[0]["prompt"])

    for epoch in range(n_epochs):
        random.shuffle(dataset)
        epoch_stats = {k: [] for k in history}
        for i in range(0, N, batch_size):
            batch      = dataset[i: i + batch_size]
            prompt_t   = batch_tensors(batch, "prompt")
            chosen_t   = batch_tensors(batch, "chosen")
            rejected_t = batch_tensors(batch, "rejected")

            loss, diag = loss_fn(prompt_t, chosen_t, rejected_t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            epoch_stats["loss"].append(loss.item())
            for k, v in diag.items():
                if k in epoch_stats:
                    epoch_stats[k].append(v)

        for k in history:
            if epoch_stats[k]:
                history[k].append(np.mean(epoch_stats[k]))

        if (epoch + 1) % 10 == 0:
            print(f"  [{label}] epoch {epoch+1:3d} | "
                  f"loss={history['loss'][-1]:.4f} | "
                  f"margin={history['margin'][-1]:.4f} | "
                  f"acc={history['accuracy'][-1]:.3f}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def eval_true_reward(policy, n=100):
    rewards = []
    for _ in range(n):
        p = random.choice(PROMPTS)
        y = policy.generate(p, temperature=0.8)
        rewards.append(true_reward(y))
    return float(np.mean(rewards))


def eval_reward_accuracy(policy, ref_policy, dataset, beta=0.1):
    """Fraction of preference pairs correctly ranked by implicit DPO reward."""
    correct = 0
    for item in dataset:
        pt = torch.tensor([item["prompt"]],   device=DEVICE)
        ct = torch.tensor([item["chosen"]],   device=DEVICE)
        rt = torch.tensor([item["rejected"]], device=DEVICE)
        with torch.no_grad():
            pi_c   = sequence_log_prob(policy,     pt, ct)
            pi_r   = sequence_log_prob(policy,     pt, rt)
            ref_c  = sequence_log_prob(ref_policy, pt, ct)
            ref_r  = sequence_log_prob(ref_policy, pt, rt)
        if (pi_c - ref_c) > (pi_r - ref_r):
            correct += 1
    return correct / len(dataset)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Initialise SFT reference policy ──────────────────────────────────────
    print("=" * 62)
    print("Initialising SFT reference policy")
    print("=" * 62)
    sft_policy = TinyLM().to(DEVICE)
    ref_policy = copy.deepcopy(sft_policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    print(f"Baseline mean true reward: {eval_true_reward(sft_policy):.4f}")

    # ── Build preference dataset ──────────────────────────────────────────────
    print("\nBuilding preference dataset (300 pairs)…")
    pref_data = make_preference_dataset(sft_policy, n_pairs=300)
    split     = int(0.85 * len(pref_data))
    train_set, eval_set = pref_data[:split], pref_data[split:]
    print(f"  Train: {len(train_set)}  |  Eval: {len(eval_set)}")

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 1 — Train DPO
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("TASK 1: DPO training")
    print("=" * 62)
    dpo_policy = copy.deepcopy(sft_policy).to(DEVICE)
    beta       = 0.1

    def dpo_fn(pt, ct, rt):
        return compute_dpo_loss(dpo_policy, ref_policy, pt, ct, rt, beta=beta)

    dpo_history = train_dpo_variant(dpo_fn, dpo_policy, train_set,
                                    n_epochs=40, label="DPO")

    dpo_true_r   = eval_true_reward(dpo_policy)
    dpo_rm_acc   = eval_reward_accuracy(dpo_policy, ref_policy, eval_set, beta=beta)
    print(f"\nDPO final | true reward: {dpo_true_r:.4f} | RM accuracy: {dpo_rm_acc:.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 2 — IPO
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("TASK 2: IPO training")
    print("=" * 62)
    ipo_policy = copy.deepcopy(sft_policy).to(DEVICE)

    def ipo_fn(pt, ct, rt):
        return compute_ipo_loss(ipo_policy, ref_policy, pt, ct, rt, beta=beta)

    ipo_history = train_dpo_variant(ipo_fn, ipo_policy, train_set,
                                    n_epochs=40, label="IPO")
    ipo_true_r  = eval_true_reward(ipo_policy)
    print(f"\nIPO final | true reward: {ipo_true_r:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 3 — SimPO
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("TASK 3: SimPO training (no reference model)")
    print("=" * 62)
    simpo_policy = copy.deepcopy(sft_policy).to(DEVICE)

    def simpo_fn(pt, ct, rt):
        return compute_simpo_loss(simpo_policy, pt, ct, rt, beta=2.0, gamma=0.5)

    simpo_history = train_dpo_variant(simpo_fn, simpo_policy, train_set,
                                      n_epochs=40, label="SimPO")
    simpo_true_r  = eval_true_reward(simpo_policy)
    print(f"\nSimPO final | true reward: {simpo_true_r:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 4 — Likelihood displacement demonstration
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("TASK 4: Likelihood displacement")
    print("=" * 62)
    print("Tracking absolute chosen/rejected log-probs throughout DPO training.")
    print("Displacement = chosen log-prob DECREASES even as margin increases.\n")

    if "chosen_lp" in dpo_history and dpo_history["chosen_lp"]:
        epochs_logged = range(1, len(dpo_history["chosen_lp"]) + 1)
        init_chosen   = dpo_history["chosen_lp"][0]
        final_chosen  = dpo_history["chosen_lp"][-1]
        init_margin   = dpo_history["margin"][0]
        final_margin  = dpo_history["margin"][-1]
        print(f"  Chosen log-prob:  {init_chosen:.3f} → {final_chosen:.3f}  "
              f"(Δ = {final_chosen - init_chosen:+.3f})")
        print(f"  Reward margin:    {init_margin:.3f} → {final_margin:.3f}  "
              f"(Δ = {final_margin - init_margin:+.3f})")
        if final_chosen < init_chosen and final_margin > init_margin:
            print("  ⚠  Likelihood displacement detected: margin rose but "
                  "chosen log-prob fell.")
        else:
            print("  ✓  No clear likelihood displacement in this run.")

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 5 — Iterative DPO (2 rounds)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("TASK 5: Iterative (online) DPO — 2 iterations")
    print("=" * 62)

    iter_policy  = copy.deepcopy(sft_policy).to(DEVICE)
    iter_ref     = copy.deepcopy(sft_policy).to(DEVICE)
    iter_ref.eval()
    for p in iter_ref.parameters():
        p.requires_grad_(False)

    iter_results = []
    for iteration in range(1, 3):
        print(f"\n  --- Iteration {iteration} ---")
        # Generate new on-policy data
        new_data = make_preference_dataset(iter_policy, n_pairs=150, noise=0.10)
        iter_ref_local = iter_ref   # keep same reference (SFT)

        def iter_dpo_fn(pt, ct, rt):
            return compute_dpo_loss(iter_policy, iter_ref_local, pt, ct, rt,
                                    beta=beta)

        train_dpo_variant(iter_dpo_fn, iter_policy, new_data,
                          n_epochs=20, label=f"IterDPO-{iteration}")
        r = eval_true_reward(iter_policy)
        a = eval_reward_accuracy(iter_policy, iter_ref_local, eval_set)
        iter_results.append({"iter": iteration, "true_reward": r, "rm_acc": a})
        print(f"  Iter {iteration}: true_reward={r:.4f}  rm_acc={a:.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 1. Loss comparison
    for hist, label, colour in [(dpo_history, "DPO", "steelblue"),
                                  (ipo_history, "IPO", "darkorange"),
                                  (simpo_history, "SimPO", "green")]:
        if hist["loss"]:
            axes[0, 0].plot(hist["loss"], label=label, color=colour)
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # 2. Reward margin
    for hist, label, colour in [(dpo_history, "DPO", "steelblue"),
                                  (ipo_history, "IPO", "darkorange"),
                                  (simpo_history, "SimPO", "green")]:
        if hist["margin"]:
            axes[0, 1].plot(hist["margin"], label=label, color=colour)
    axes[0, 1].axhline(0, color="grey", linestyle="--", alpha=0.5)
    axes[0, 1].set_title("Reward Margin (chosen − rejected)")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].legend()

    # 3. Accuracy comparison
    for hist, label, colour in [(dpo_history, "DPO", "steelblue"),
                                  (ipo_history, "IPO", "darkorange"),
                                  (simpo_history, "SimPO", "green")]:
        if hist["accuracy"]:
            axes[0, 2].plot(hist["accuracy"], label=label, color=colour)
    axes[0, 2].axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="chance")
    axes[0, 2].set_title("Preference Accuracy")
    axes[0, 2].set_xlabel("Epoch"); axes[0, 2].set_ylim(0, 1); axes[0, 2].legend()

    # 4. Likelihood displacement (DPO)
    if dpo_history["chosen_lp"] and dpo_history["rejected_lp"]:
        ep = range(len(dpo_history["chosen_lp"]))
        axes[1, 0].plot(ep, dpo_history["chosen_lp"],   label="Chosen log-prob",   color="green")
        axes[1, 0].plot(ep, dpo_history["rejected_lp"], label="Rejected log-prob", color="red")
        axes[1, 0].set_title("DPO: Likelihood Displacement")
        axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Mean log-prob")
        axes[1, 0].legend()

    # 5. True reward comparison
    methods  = ["SFT", "DPO", "IPO", "SimPO"]
    rewards  = [eval_true_reward(sft_policy), dpo_true_r, ipo_true_r, simpo_true_r]
    colours  = ["grey", "steelblue", "darkorange", "green"]
    axes[1, 1].bar(methods, rewards, color=colours)
    axes[1, 1].axhline(rewards[0], color="grey", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("True Reward: SFT vs Aligned Models")
    axes[1, 1].set_ylabel("Mean true reward")

    # 6. Iterative DPO
    if iter_results:
        iters   = [0] + [r["iter"] for r in iter_results]
        t_rwds  = [eval_true_reward(sft_policy)] + [r["true_reward"] for r in iter_results]
        axes[1, 2].plot(iters, t_rwds, "o-", color="purple", linewidth=2, markersize=8)
        axes[1, 2].set_title("Iterative DPO: True Reward per Iteration")
        axes[1, 2].set_xlabel("Iteration (0 = SFT baseline)")
        axes[1, 2].set_ylabel("Mean true reward")
        axes[1, 2].set_xticks(iters)

    plt.suptitle("Week 16: DPO Variants Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig("week16_dpo_results.png", dpi=150)
    plt.show()
    print("Saved: week16_dpo_results.png")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    baseline = eval_true_reward(sft_policy)
    print(f"{'Method':12s}  {'True reward':>12}  {'vs SFT':>8}")
    print("-" * 36)
    print(f"{'SFT':12s}  {baseline:>12.4f}  {'baseline':>8}")
    for name, r in [("DPO", dpo_true_r), ("IPO", ipo_true_r), ("SimPO", simpo_true_r)]:
        print(f"{name:12s}  {r:>12.4f}  {r - baseline:>+8.4f}")
    if iter_results:
        r_iter = iter_results[-1]["true_reward"]
        print(f"{'IterDPO-2':12s}  {r_iter:>12.4f}  {r_iter - baseline:>+8.4f}")

    print("\nAll Week 16 tasks complete.")
