"""
Week 18 Practical: Introduction to Reinforcement Learning
============================================================
Objectives:
  - Implement value iteration and policy iteration on a gridworld MDP
  - Implement TD(0) and Monte Carlo value estimation; compare variance
  - Implement Q-learning and SARSA; compare on-policy vs off-policy behaviour
  - Implement REINFORCE (with and without baseline) on a toy environment
  - Implement a simple actor-critic and compare to REINFORCE
  - Implement UCB and epsilon-greedy on a multi-armed bandit; plot regret

No external RL library required — everything is implemented from scratch
using NumPy / PyTorch so the underlying mechanics are fully visible.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# 0.  A small Gridworld MDP
# ─────────────────────────────────────────────────────────────────────────────
class GridWorld:
    """
    A simple deterministic-by-default gridworld.

    Layout (4x4), S = start, G = goal (+1 reward), X = pit (-1 reward, terminal):
        . . . G
        . X . .
        . . X .
        S . . .

    Actions: 0=up, 1=down, 2=left, 3=right.
    Moving off the grid keeps the agent in place.
    Step reward: -0.04 (encourages shorter paths), Goal: +1, Pit: -1 (terminal).
    """
    def __init__(self, slip_prob=0.0):
        self.size = 4
        self.goal = (0, 3)
        self.pits = {(1, 1), (2, 2)}
        self.start = (3, 0)
        self.slip_prob = slip_prob   # probability of a random action (stochastic transitions)
        self.actions = [0, 1, 2, 3]  # up, down, left, right
        self.states = [(r, c) for r in range(self.size) for c in range(self.size)]

    def is_terminal(self, s):
        return s == self.goal or s in self.pits

    def step(self, s, a):
        if self.is_terminal(s):
            return s, 0.0, True

        if random.random() < self.slip_prob:
            a = random.choice(self.actions)

        r, c = s
        if a == 0:   r = max(0, r - 1)
        elif a == 1: r = min(self.size - 1, r + 1)
        elif a == 2: c = max(0, c - 1)
        elif a == 3: c = min(self.size - 1, c + 1)
        s_next = (r, c)

        if s_next == self.goal:
            return s_next, 1.0, True
        if s_next in self.pits:
            return s_next, -1.0, True
        return s_next, -0.04, False

    def transition_model(self):
        """Build P[s][a][s'] and R[s][a][s'] for dynamic programming (deterministic case)."""
        P, R = {}, {}
        for s in self.states:
            P[s], R[s] = {}, {}
            for a in self.actions:
                if self.is_terminal(s):
                    P[s][a] = {s: 1.0}
                    R[s][a] = {s: 0.0}
                else:
                    s_next, r, _ = self.step(s, a)
                    # Deterministic-only model used for DP (slip_prob=0 assumed here)
                    P[s][a] = {s_next: 1.0}
                    R[s][a] = {s_next: r}
        return P, R


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — Value iteration and policy iteration
# ─────────────────────────────────────────────────────────────────────────────
def value_iteration(env, gamma=0.95, theta=1e-6, max_iters=1000):
    P, R = env.transition_model()
    V = {s: 0.0 for s in env.states}
    history = []

    for it in range(max_iters):
        delta = 0.0
        for s in env.states:
            if env.is_terminal(s):
                continue
            v_old = V[s]
            V[s] = max(
                sum(P[s][a][s_next] * (R[s][a][s_next] + gamma * V[s_next])
                    for s_next in P[s][a])
                for a in env.actions
            )
            delta = max(delta, abs(v_old - V[s]))
        history.append(delta)
        if delta < theta:
            break

    policy = {}
    for s in env.states:
        if env.is_terminal(s):
            policy[s] = None
            continue
        policy[s] = max(
            env.actions,
            key=lambda a: sum(P[s][a][s_next] * (R[s][a][s_next] + gamma * V[s_next])
                               for s_next in P[s][a])
        )
    return V, policy, history


def policy_iteration(env, gamma=0.95, theta=1e-6, max_iters=100):
    P, R = env.transition_model()
    policy = {s: random.choice(env.actions) for s in env.states}
    V = {s: 0.0 for s in env.states}
    n_policy_changes = []

    for it in range(max_iters):
        # Policy evaluation (iterative, to convergence)
        while True:
            delta = 0.0
            for s in env.states:
                if env.is_terminal(s):
                    continue
                v_old = V[s]
                a = policy[s]
                V[s] = sum(P[s][a][s_next] * (R[s][a][s_next] + gamma * V[s_next])
                          for s_next in P[s][a])
                delta = max(delta, abs(v_old - V[s]))
            if delta < theta:
                break

        # Policy improvement
        changed = 0
        for s in env.states:
            if env.is_terminal(s):
                continue
            old_a = policy[s]
            policy[s] = max(
                env.actions,
                key=lambda a: sum(P[s][a][s_next] * (R[s][a][s_next] + gamma * V[s_next])
                                  for s_next in P[s][a])
            )
            if policy[s] != old_a:
                changed += 1
        n_policy_changes.append(changed)
        if changed == 0:
            break

    return V, policy, n_policy_changes


def print_policy(policy, env, title=""):
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→", None: " "}
    print(f"\n  {title}")
    for r in range(env.size):
        row = []
        for c in range(env.size):
            s = (r, c)
            if s == env.goal:
                row.append(" G ")
            elif s in env.pits:
                row.append(" X ")
            else:
                row.append(f" {arrows[policy[s]]} ")
        print("  " + "".join(row))


def task1_dynamic_programming():
    print("=" * 65)
    print("TASK 1: Value Iteration and Policy Iteration")
    print("=" * 65)
    env = GridWorld(slip_prob=0.0)

    V_vi, policy_vi, hist_vi = value_iteration(env)
    V_pi, policy_pi, hist_pi = policy_iteration(env)

    print_policy(policy_vi, env, "Value Iteration — Optimal Policy")
    print_policy(policy_pi, env, "Policy Iteration — Optimal Policy")

    same = all(policy_vi[s] == policy_pi[s] for s in env.states if not env.is_terminal(s))
    print(f"\n  Policies match: {same}")
    print(f"  Value iteration converged in {len(hist_vi)} sweeps")
    print(f"  Policy iteration converged in {len(hist_pi)} policy-improvement steps")

    plt.figure(figsize=(7, 4))
    plt.plot(hist_vi, marker="o", label="Value iteration (max Δ per sweep)")
    plt.yscale("log")
    plt.xlabel("Sweep")
    plt.ylabel("Max value change (log scale)")
    plt.title("Value Iteration Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig("week18_value_iteration.png", dpi=150)
    plt.show()
    print("Saved: week18_value_iteration.png")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — Monte Carlo vs TD(0) value estimation
# ─────────────────────────────────────────────────────────────────────────────
def generate_episode(env, policy, max_steps=100):
    s = env.start
    trajectory = []
    for _ in range(max_steps):
        if env.is_terminal(s):
            break
        a = policy[s]
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory


def monte_carlo_value_estimation(env, policy, n_episodes=500, gamma=0.95):
    V = {s: 0.0 for s in env.states}
    counts = {s: 0 for s in env.states}
    errors = []

    # Reference: true value of start state from value iteration
    V_true, _, _ = value_iteration(env)

    for ep in range(n_episodes):
        traj = generate_episode(env, policy)
        G = 0.0
        visited_returns = {}
        for s, a, r in reversed(traj):
            G = r + gamma * G
            visited_returns[s] = G   # first-visit MC (overwritten = last visit in reverse = first visit)
        for s, G_s in visited_returns.items():
            counts[s] += 1
            V[s] += (G_s - V[s]) / counts[s]   # incremental mean

        errors.append(abs(V[env.start] - V_true[env.start]))
    return V, errors


def td0_value_estimation(env, policy, n_episodes=500, gamma=0.95, alpha=0.1):
    V = {s: 0.0 for s in env.states}
    errors = []
    V_true, _, _ = value_iteration(env)

    for ep in range(n_episodes):
        s = env.start
        for _ in range(100):
            if env.is_terminal(s):
                break
            a = policy[s]
            s_next, r, done = env.step(s, a)
            td_target = r + (0 if done else gamma * V[s_next])
            V[s] += alpha * (td_target - V[s])
            s = s_next
            if done:
                break
        errors.append(abs(V[env.start] - V_true[env.start]))
    return V, errors


def task2_mc_vs_td():
    print("\n" + "=" * 65)
    print("TASK 2: Monte Carlo vs TD(0) value estimation")
    print("=" * 65)
    env = GridWorld(slip_prob=0.0)
    _, optimal_policy, _ = value_iteration(env)

    # Run multiple seeds to show variance
    n_runs = 10
    mc_errors_all, td_errors_all = [], []
    for run in range(n_runs):
        random.seed(run)
        _, mc_err = monte_carlo_value_estimation(env, optimal_policy, n_episodes=300)
        _, td_err = td0_value_estimation(env, optimal_policy, n_episodes=300)
        mc_errors_all.append(mc_err)
        td_errors_all.append(td_err)

    mc_mean = np.mean(mc_errors_all, axis=0)
    mc_std  = np.std(mc_errors_all, axis=0)
    td_mean = np.mean(td_errors_all, axis=0)
    td_std  = np.std(td_errors_all, axis=0)

    print(f"\n  Mean abs error in V(start) after 300 episodes ({n_runs} runs):")
    print(f"    Monte Carlo: {mc_mean[-1]:.4f} ± {mc_std[-1]:.4f}")
    print(f"    TD(0):       {td_mean[-1]:.4f} ± {td_std[-1]:.4f}")

    plt.figure(figsize=(8, 4))
    episodes = np.arange(len(mc_mean))
    plt.plot(episodes, mc_mean, label="Monte Carlo", color="steelblue")
    plt.fill_between(episodes, mc_mean - mc_std, mc_mean + mc_std, alpha=0.2, color="steelblue")
    plt.plot(episodes, td_mean, label="TD(0)", color="darkorange")
    plt.fill_between(episodes, td_mean - td_std, td_mean + td_std, alpha=0.2, color="darkorange")
    plt.xlabel("Episode")
    plt.ylabel("Abs error in V(start)  (mean ± std over runs)")
    plt.title("Monte Carlo vs TD(0): Convergence and Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("week18_mc_vs_td.png", dpi=150)
    plt.show()
    print("Saved: week18_mc_vs_td.png")
    print("\n  Note the wider shaded region (variance across runs) for Monte Carlo.")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — Q-learning vs SARSA
# ─────────────────────────────────────────────────────────────────────────────
def epsilon_greedy_action(Q, s, actions, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)
    return max(actions, key=lambda a: Q[(s, a)])


def q_learning(env, n_episodes=500, gamma=0.95, alpha=0.1, epsilon=0.1):
    Q = {(s, a): 0.0 for s in env.states for a in env.actions}
    episode_returns = []
    for ep in range(n_episodes):
        s = env.start
        total_r = 0.0
        for _ in range(100):
            if env.is_terminal(s):
                break
            a = epsilon_greedy_action(Q, s, env.actions, epsilon)
            s_next, r, done = env.step(s, a)
            total_r += r
            max_next = 0.0 if done else max(Q[(s_next, a2)] for a2 in env.actions)
            Q[(s, a)] += alpha * (r + gamma * max_next - Q[(s, a)])
            s = s_next
            if done:
                break
        episode_returns.append(total_r)
    return Q, episode_returns


def sarsa(env, n_episodes=500, gamma=0.95, alpha=0.1, epsilon=0.1):
    Q = {(s, a): 0.0 for s in env.states for a in env.actions}
    episode_returns = []
    for ep in range(n_episodes):
        s = env.start
        a = epsilon_greedy_action(Q, s, env.actions, epsilon)
        total_r = 0.0
        for _ in range(100):
            if env.is_terminal(s):
                break
            s_next, r, done = env.step(s, a)
            total_r += r
            a_next = epsilon_greedy_action(Q, s_next, env.actions, epsilon)
            target = 0.0 if done else Q[(s_next, a_next)]
            Q[(s, a)] += alpha * (r + gamma * target - Q[(s, a)])
            s, a = s_next, a_next
            if done:
                break
        episode_returns.append(total_r)
    return Q, episode_returns


def task3_qlearning_vs_sarsa():
    print("\n" + "=" * 65)
    print("TASK 3: Q-learning (off-policy) vs SARSA (on-policy)")
    print("=" * 65)
    # Use a slippery gridworld to make the on/off-policy distinction matter
    env = GridWorld(slip_prob=0.1)

    Q_ql, returns_ql = q_learning(env, n_episodes=500, epsilon=0.2)
    Q_sarsa, returns_sarsa = sarsa(env, n_episodes=500, epsilon=0.2)

    def smooth(x, w=20):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    plt.figure(figsize=(8, 4))
    plt.plot(smooth(returns_ql), label="Q-learning (off-policy)", color="steelblue")
    plt.plot(smooth(returns_sarsa), label="SARSA (on-policy)", color="darkorange")
    plt.xlabel("Episode")
    plt.ylabel("Episode return (smoothed)")
    plt.title("Q-learning vs SARSA under ε-greedy exploration (slippery gridworld)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("week18_qlearning_vs_sarsa.png", dpi=150)
    plt.show()
    print("Saved: week18_qlearning_vs_sarsa.png")

    print(f"\n  Final smoothed return — Q-learning: {smooth(returns_ql)[-1]:.3f}")
    print(f"  Final smoothed return — SARSA:      {smooth(returns_sarsa)[-1]:.3f}")
    print("\n  SARSA accounts for the exploration policy's own risk (e.g. near pits);")
    print("  Q-learning learns the value of the greedy policy regardless of how it explores.")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 — REINFORCE (with and without baseline) on a toy bandit-style task
# ─────────────────────────────────────────────────────────────────────────────
class ToyContextualBandit:
    """
    A single-state contextual bandit: 5 actions, each with a different
    true mean reward (plus noise). Models the RLHF-as-bandit structure.
    """
    def __init__(self, n_actions=5):
        self.n_actions = n_actions
        self.true_means = np.array([0.1, 0.3, 0.9, 0.4, 0.2])

    def step(self, action):
        return float(np.random.normal(self.true_means[action], 0.5))


class SoftmaxPolicy(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_actions))

    def forward(self):
        return F.softmax(self.logits, dim=-1)

    def sample(self):
        probs = self.forward()
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


def reinforce_train(env, n_steps=500, lr=0.05, use_baseline=False):
    policy = SoftmaxPolicy(env.n_actions)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    baseline = 0.0
    rewards_history = []

    for step in range(n_steps):
        action, log_prob = policy.sample()
        reward = env.step(action)
        rewards_history.append(reward)

        advantage = reward - baseline if use_baseline else reward
        loss = -log_prob * advantage

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if use_baseline:
            baseline = 0.95 * baseline + 0.05 * reward   # running average baseline

    return rewards_history, policy.forward().detach().numpy()


def task4_reinforce():
    print("\n" + "=" * 65)
    print("TASK 4: REINFORCE with and without a baseline")
    print("=" * 65)
    env = ToyContextualBandit()
    print(f"\n  True action means: {env.true_means}")
    print(f"  Best action: {env.true_means.argmax()} (mean reward {env.true_means.max()})")

    n_runs = 8
    no_baseline_runs, baseline_runs = [], []
    for run in range(n_runs):
        torch.manual_seed(run); random.seed(run); np.random.seed(run)
        r_nb, probs_nb = reinforce_train(env, n_steps=400, use_baseline=False)
        torch.manual_seed(run); random.seed(run); np.random.seed(run)
        r_b,  probs_b  = reinforce_train(env, n_steps=400, use_baseline=True)
        no_baseline_runs.append(r_nb)
        baseline_runs.append(r_b)

    nb_mean = np.mean(no_baseline_runs, axis=0)
    b_mean  = np.mean(baseline_runs, axis=0)
    nb_std_across_runs = np.std([np.mean(r) for r in no_baseline_runs])
    b_std_across_runs  = np.std([np.mean(r) for r in baseline_runs])

    def smooth(x, w=20):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    plt.figure(figsize=(8, 4))
    plt.plot(smooth(nb_mean), label="REINFORCE (no baseline)", color="red")
    plt.plot(smooth(b_mean),  label="REINFORCE (with baseline)", color="green")
    plt.axhline(env.true_means.max(), color="grey", linestyle="--", label="Optimal mean reward")
    plt.xlabel("Step")
    plt.ylabel("Reward (smoothed, averaged over runs)")
    plt.title("REINFORCE: Effect of a Baseline on Learning Speed")
    plt.legend()
    plt.tight_layout()
    plt.savefig("week18_reinforce_baseline.png", dpi=150)
    plt.show()
    print("Saved: week18_reinforce_baseline.png")

    print(f"\n  Variance of mean reward across {n_runs} runs:")
    print(f"    No baseline:   {nb_std_across_runs:.4f}")
    print(f"    With baseline: {b_std_across_runs:.4f}")
    print(f"\n  Final learned action probabilities (with baseline): {probs_b.round(3)}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 — Simple actor-critic vs REINFORCE
# ─────────────────────────────────────────────────────────────────────────────
class CriticNet(nn.Module):
    """A learned baseline V(s) — trivial here since there's only one state,
    but written generally so it generalises to multi-state settings."""
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def forward(self):
        return self.value


def actor_critic_train(env, n_steps=400, lr_actor=0.05, lr_critic=0.1):
    policy = SoftmaxPolicy(env.n_actions)
    critic = CriticNet()
    actor_opt  = torch.optim.Adam(policy.parameters(), lr=lr_actor)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    rewards_history = []

    for step in range(n_steps):
        action, log_prob = policy.sample()
        reward = env.step(action)
        rewards_history.append(reward)

        v = critic()
        advantage = reward - v.item()

        actor_loss = -log_prob * advantage
        critic_loss = (reward - v) ** 2

        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

    return rewards_history


def task5_actor_critic():
    print("\n" + "=" * 65)
    print("TASK 5: Actor-Critic vs REINFORCE")
    print("=" * 65)
    env = ToyContextualBandit()

    n_runs = 8
    ac_runs, reinforce_runs = [], []
    for run in range(n_runs):
        torch.manual_seed(run); random.seed(run); np.random.seed(run)
        ac_runs.append(actor_critic_train(env, n_steps=400))
        torch.manual_seed(run); random.seed(run); np.random.seed(run)
        r_nb, _ = reinforce_train(env, n_steps=400, use_baseline=False)
        reinforce_runs.append(r_nb)

    def smooth(x, w=20):
        return np.convolve(np.mean(x, axis=0), np.ones(w) / w, mode="valid")

    plt.figure(figsize=(8, 4))
    plt.plot(smooth(ac_runs), label="Actor-Critic", color="purple")
    plt.plot(smooth(reinforce_runs), label="REINFORCE (no baseline)", color="red")
    plt.axhline(env.true_means.max(), color="grey", linestyle="--", label="Optimal mean reward")
    plt.xlabel("Step")
    plt.ylabel("Reward (smoothed, averaged over runs)")
    plt.title("Actor-Critic vs Plain REINFORCE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("week18_actor_critic.png", dpi=150)
    plt.show()
    print("Saved: week18_actor_critic.png")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 6 — Multi-armed bandit: epsilon-greedy vs UCB, regret comparison
# ─────────────────────────────────────────────────────────────────────────────
def run_bandit(env, strategy, n_steps=1000, epsilon=0.1, c=2.0):
    Q = np.zeros(env.n_actions)
    N = np.zeros(env.n_actions)
    regrets = []
    optimal_mean = env.true_means.max()

    for t in range(1, n_steps + 1):
        if strategy == "epsilon_greedy":
            if random.random() < epsilon:
                a = random.randrange(env.n_actions)
            else:
                a = int(np.argmax(Q))
        elif strategy == "ucb":
            if 0 in N:
                a = int(np.argmin(N))    # try every arm at least once
            else:
                ucb_values = Q + c * np.sqrt(np.log(t) / N)
                a = int(np.argmax(ucb_values))
        else:
            raise ValueError(strategy)

        r = env.step(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]

        regret = optimal_mean - env.true_means[a]
        regrets.append(regret)

    return np.cumsum(regrets)


def task6_bandit_regret():
    print("\n" + "=" * 65)
    print("TASK 6: Multi-armed bandit — epsilon-greedy vs UCB regret")
    print("=" * 65)
    env = ToyContextualBandit()

    n_runs = 20
    eps_regrets, ucb_regrets = [], []
    for run in range(n_runs):
        random.seed(run); np.random.seed(run)
        eps_regrets.append(run_bandit(env, "epsilon_greedy", n_steps=1000, epsilon=0.1))
        random.seed(run); np.random.seed(run)
        ucb_regrets.append(run_bandit(env, "ucb", n_steps=1000, c=2.0))

    eps_mean = np.mean(eps_regrets, axis=0)
    ucb_mean = np.mean(ucb_regrets, axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(eps_mean, label="ε-greedy (ε=0.1)", color="steelblue")
    plt.plot(ucb_mean, label="UCB (c=2.0)", color="darkorange")
    plt.xlabel("Step")
    plt.ylabel("Cumulative regret (averaged over runs)")
    plt.title("Multi-Armed Bandit: Cumulative Regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig("week18_bandit_regret.png", dpi=150)
    plt.show()
    print("Saved: week18_bandit_regret.png")

    print(f"\n  Final cumulative regret (mean over {n_runs} runs):")
    print(f"    ε-greedy: {eps_mean[-1]:.2f}")
    print(f"    UCB:      {ucb_mean[-1]:.2f}")
    print("\n  UCB's regret typically grows logarithmically rather than linearly,")
    print("  because it stops exploring sub-optimal arms once confident.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    task1_dynamic_programming()
    task2_mc_vs_td()
    task3_qlearning_vs_sarsa()
    task4_reinforce()
    task5_actor_critic()
    task6_bandit_regret()

    print("\n" + "=" * 65)
    print("All Week 18 tasks complete.")
    print("=" * 65)
