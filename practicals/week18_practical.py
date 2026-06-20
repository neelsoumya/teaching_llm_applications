"""
Week 18 Practical: Introduction to Reinforcement Learning (Short Primer)
==========================================================================
A minimal REINFORCE example on a toy bandit task, with and without a
baseline, showing the variance reduction directly.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# A simple bandit: 5 actions, each with a different true mean reward.
TRUE_MEANS = np.array([0.1, 0.3, 0.9, 0.4, 0.2])
N_ACTIONS  = len(TRUE_MEANS)


def get_reward(action):
    return float(np.random.normal(TRUE_MEANS[action], 0.5))


class Policy(nn.Module):
    """A simple softmax policy over actions."""
    def __init__(self, n_actions):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_actions))

    def sample(self):
        probs = F.softmax(self.logits, dim=-1)
        dist  = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


def train_reinforce(n_steps=300, lr=0.05, use_baseline=False):
    policy = Policy(N_ACTIONS)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    baseline = 0.0
    rewards = []

    for _ in range(n_steps):
        action, log_prob = policy.sample()
        reward = get_reward(action)
        rewards.append(reward)

        advantage = reward - baseline if use_baseline else reward
        loss = -log_prob * advantage

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if use_baseline:
            baseline = 0.9 * baseline + 0.1 * reward  # running average

    return rewards


def smooth(x, w=15):
    return np.convolve(x, np.ones(w) / w, mode="valid")


if __name__ == "__main__":
    print(f"True action means: {TRUE_MEANS}")
    print(f"Best action: {TRUE_MEANS.argmax()} (mean reward {TRUE_MEANS.max()})\n")

    n_runs = 10
    no_baseline_runs = [train_reinforce(use_baseline=False) for _ in range(n_runs)]
    baseline_runs    = [train_reinforce(use_baseline=True)  for _ in range(n_runs)]

    nb_mean = np.mean(no_baseline_runs, axis=0)
    b_mean  = np.mean(baseline_runs, axis=0)

    # Variance across runs of the final average reward — this is the headline result
    nb_var = np.var([np.mean(r[-50:]) for r in no_baseline_runs])
    b_var  = np.var([np.mean(r[-50:]) for r in baseline_runs])

    print(f"Variance of final reward across runs:")
    print(f"  No baseline:   {nb_var:.4f}")
    print(f"  With baseline: {b_var:.4f}")
    print(f"  → Baseline reduces run-to-run variance by "
          f"{100 * (1 - b_var / nb_var):.0f}%" if nb_var > 0 else "")

    plt.figure(figsize=(7, 4))
    plt.plot(smooth(nb_mean), label="REINFORCE (no baseline)", color="red")
    plt.plot(smooth(b_mean),  label="REINFORCE (with baseline)", color="green")
    plt.axhline(TRUE_MEANS.max(), color="grey", linestyle="--", label="Best possible reward")
    plt.xlabel("Step")
    plt.ylabel("Reward (smoothed, averaged over runs)")
    plt.title("REINFORCE: With vs Without a Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig("week18_reinforce_baseline.png", dpi=150)
    plt.show()
    print("\nSaved: week18_reinforce_baseline.png")
