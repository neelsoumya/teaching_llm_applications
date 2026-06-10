"""
Week 14 Practical: Mechanistic Interpretability
=================================================
Objectives:
  - Load GPT-2 with TransformerLens and cache all intermediate activations
  - Implement the logit lens across all layers on factual completion prompts
  - Run activation patching to localise which heads carry subject information
    in an Indirect Object Identification (IOI) style task
  - Train linear probes on intermediate layer activations
  - Train a small sparse autoencoder (SAE) on MLP activations and
    visualise the top-activating tokens for learned features

Dependencies:
    pip install transformer_lens einops scikit-learn matplotlib

All tasks run on CPU with GPT-2 small (117M parameters).
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

try:
    import transformer_lens
    from transformer_lens import HookedTransformer, utils as tl_utils
    HAS_TL = True
except ImportError:
    HAS_TL = False
    print("transformer_lens not installed.")
    print("Install with:  pip install transformer_lens")
    print("Continuing with limited functionality.\n")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("scikit-learn not installed — probe section will be skipped.")

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

DEVICE = "cpu"   # GPT-2 small fits comfortably on CPU for this practical


# ─────────────────────────────────────────────────────────────────────────────
# 0. Load model
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    if not HAS_TL:
        return None
    print("Loading GPT-2 small via TransformerLens…")
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()
    print(f"  Layers : {model.cfg.n_layers}")
    print(f"  Heads  : {model.cfg.n_heads}")
    print(f"  d_model: {model.cfg.d_model}")
    print(f"  Vocab  : {model.cfg.d_vocab}\n")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 1. Logit Lens
# ─────────────────────────────────────────────────────────────────────────────
def logit_lens(model, prompt: str, top_k: int = 5) -> dict:
    """
    For each layer l, unembed the residual stream after that layer to see
    what the model 'would predict' at intermediate stages of computation.

    Returns a dict mapping layer_index → list of (token_str, prob) tuples.
    """
    tokens = model.to_tokens(prompt)                         # (1, T)
    token_strs = [model.to_string(t) for t in tokens[0]]

    _, cache = model.run_with_cache(tokens)

    results = {}
    for layer in range(model.cfg.n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"]     # (1, T, d)
        # Apply final layer norm then unembed
        resid_normed = model.ln_final(resid)
        logits = model.unembed(resid_normed)                 # (1, T, V)
        probs  = torch.softmax(logits[0, -1, :], dim=-1)     # last token pos
        top_probs, top_ids = torch.topk(probs, top_k)
        results[layer] = [(model.to_string(idx.item()), p.item())
                          for idx, p in zip(top_ids, top_probs)]
    return results, token_strs


def plot_logit_lens(results: dict, prompt: str, target_token: str,
                    filename: str = "week14_logit_lens.png"):
    """Plot the probability of the target token at each layer."""
    layers = sorted(results.keys())
    target_probs = []
    for l in layers:
        prob = next((p for t, p in results[l] if target_token.strip() in t), 0.0)
        target_probs.append(prob)

    plt.figure(figsize=(9, 4))
    plt.plot(layers, target_probs, "o-", color="steelblue", linewidth=2, markersize=6)
    plt.xlabel("Layer")
    plt.ylabel(f"P('{target_token.strip()}')")
    plt.title(f"Logit Lens: probability of '{target_token.strip()}' across layers\n"
              f"Prompt: \"{prompt}\"")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Direct Logit Attribution
# ─────────────────────────────────────────────────────────────────────────────
def direct_logit_attribution(model, prompt: str, target_token: str) -> dict:
    """
    Decompose the final logit for target_token into contributions from
    each attention layer and MLP layer via the residual stream.

    Uses the fact that:
        logit(t) = W_U @ (sum of all layer outputs) · e_target
    """
    tokens     = model.to_tokens(prompt)
    target_id  = model.to_single_token(target_token)
    _, cache   = model.run_with_cache(tokens)

    # Unembedding direction for target token
    W_U = model.W_U                                         # (d_model, V)
    unembed_dir = W_U[:, target_id]                         # (d_model,)

    contributions = {}
    for layer in range(model.cfg.n_layers):
        attn_out = cache[f"blocks.{layer}.hook_attn_out"][0, -1, :]   # (d,)
        mlp_out  = cache[f"blocks.{layer}.hook_mlp_out"][0, -1, :]    # (d,)
        contributions[f"attn_{layer}"] = (attn_out @ unembed_dir).item()
        contributions[f"mlp_{layer}"]  = (mlp_out  @ unembed_dir).item()

    return contributions


def plot_logit_attribution(contributions: dict, target_token: str,
                            filename: str = "week14_logit_attribution.png"):
    n_layers = len(contributions) // 2
    attn_contribs = [contributions[f"attn_{l}"] for l in range(n_layers)]
    mlp_contribs  = [contributions[f"mlp_{l}"]  for l in range(n_layers)]
    layers = list(range(n_layers))

    fig, ax = plt.subplots(figsize=(11, 4))
    width = 0.4
    ax.bar([l - width/2 for l in layers], attn_contribs, width,
           label="Attention", color="steelblue", alpha=0.8)
    ax.bar([l + width/2 for l in layers], mlp_contribs, width,
           label="MLP", color="darkorange", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Logit contribution → '{target_token.strip()}'")
    ax.set_title(f"Direct Logit Attribution for '{target_token.strip()}'")
    ax.legend()
    ax.set_xticks(layers)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Activation Patching (IOI-style)
# ─────────────────────────────────────────────────────────────────────────────
# IOI task: "When [A] and [B] went to the store, [B] gave a drink to"
# Correct answer: [A]. We patch residual stream activations from a
# corrupted prompt (names swapped) to find which layers carry [A]'s identity.

IOI_CLEAN     = "When Mary and John went to the store, John gave a drink to"
IOI_CORRUPTED = "When John and Mary went to the store, Mary gave a drink to"
IOI_TARGET    = " Mary"


def get_residual_stream(model, prompt: str) -> dict:
    """Cache residual stream after each layer for the last token position."""
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)
    resid = {}
    for layer in range(model.cfg.n_layers):
        resid[layer] = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].detach().clone()
    return resid


def patch_residual_and_score(model, clean_prompt: str, corrupted_prompt: str,
                              target_token: str) -> list[float]:
    """
    For each layer l:
      1. Run clean and corrupted prompts; cache residual streams.
      2. Re-run corrupted prompt, patching residual stream at layer l
         with the clean residual stream at that layer.
      3. Record the change in log-prob of target_token.
    Returns: list of patching effects per layer.
    """
    target_id = model.to_single_token(target_token)

    clean_resids     = get_residual_stream(model, clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    # Baseline (corrupted, no patching)
    with torch.no_grad():
        base_logits = model(corrupted_tokens)[0, -1, :]
    base_lp = F.log_softmax(base_logits, dim=-1)[target_id].item()

    effects = []
    for patch_layer in range(model.cfg.n_layers):
        clean_act = clean_resids[patch_layer]

        def make_hook(layer_idx, stored_act):
            def hook_fn(value, hook):
                if hook.layer() == layer_idx:
                    value[0, -1, :] = stored_act
                return value
            return hook_fn

        hook_name = f"blocks.{patch_layer}.hook_resid_post"
        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, make_hook(patch_layer, clean_act))]
            )[0, -1, :]
        patched_lp = F.log_softmax(patched_logits, dim=-1)[target_id].item()
        effects.append(patched_lp - base_lp)   # positive = patching helped

    return effects


def plot_patching_effects(effects: list[float], title: str,
                           filename: str = "week14_patching.png"):
    layers = list(range(len(effects)))
    colours = ["steelblue" if e > 0 else "salmon" for e in effects]
    plt.figure(figsize=(9, 4))
    plt.bar(layers, effects, color=colours, edgecolor="white", linewidth=0.5)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Layer (residual stream patched after this layer)")
    plt.ylabel("Δ log P(target) after patching")
    plt.title(title)
    plt.xticks(layers)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Linear Probes
# ─────────────────────────────────────────────────────────────────────────────
# Task: predict whether the *last* token of a sentence is a verb or a noun
# using a logistic regression probe trained on each layer's residual stream.

PROBE_SENTENCES = [
    # (sentence, label)   label: 1=ends-with-verb, 0=ends-with-noun
    ("The cat quickly ran",          1),
    ("She carefully wrote",          1),
    ("They loudly sang",             1),
    ("He slowly walked",             1),
    ("The bird happily flew",        1),
    ("Dogs always bark",             1),
    ("The children often play",      1),
    ("Scientists carefully observe", 1),
    ("The dog ate the bone",         0),
    ("She loves chocolate cake",     0),
    ("He drove the car",             0),
    ("They built the house",         0),
    ("The teacher graded the essay", 0),
    ("We visited the museum",        0),
    ("The chef prepared the meal",   0),
    ("Students took the exam",       0),
] * 6   # repeat to give more examples


def collect_probe_activations(model, sentences: list[tuple],
                               layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract residual stream activations at `layer` for the last token."""
    X, y = [], []
    for sent, label in sentences:
        tokens = model.to_tokens(sent)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        act = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].numpy()
        X.append(act)
        y.append(label)
    return np.array(X), np.array(y)


def train_probe_all_layers(model, sentences: list[tuple]) -> list[float]:
    """Train a logistic regression probe at each layer; return test accuracies."""
    if not HAS_SKLEARN:
        print("scikit-learn not available — skipping probes")
        return []
    accs = []
    for layer in range(model.cfg.n_layers):
        X, y = collect_probe_activations(model, sentences, layer)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                                    random_state=42, stratify=y)
        probe = LogisticRegression(max_iter=1000, C=1.0)
        probe.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, probe.predict(X_te))
        accs.append(acc)
        print(f"  Layer {layer:2d}: probe accuracy = {acc:.3f}")
    return accs


def plot_probe_accuracies(accs: list[float],
                           filename: str = "week14_probe_accuracy.png"):
    plt.figure(figsize=(9, 4))
    plt.plot(range(len(accs)), accs, "o-", color="green", linewidth=2, markersize=6)
    plt.axhline(0.5, color="gray", linestyle="--", label="Chance (50%)")
    plt.xlabel("Layer")
    plt.ylabel("Probe accuracy (verb vs noun)")
    plt.title("Linear Probe: Verb/Noun Classification Across Layers")
    plt.ylim(0.4, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sparse Autoencoder (SAE)
# ─────────────────────────────────────────────────────────────────────────────
class SparseAutoencoder(nn.Module):
    """
    A one-layer sparse autoencoder trained on model activations.
    Learns an overcomplete dictionary of monosemantic features.

    Architecture:
        Encoder: z = ReLU(W_enc (x - b_pre) + b_enc)
        Decoder: x̂ = W_dec z + b_pre
    Loss: ||x - x̂||² + λ||z||₁
    """
    def __init__(self, d_input: int, d_hidden: int, l1_coef: float = 1e-3):
        super().__init__()
        self.d_input  = d_input
        self.d_hidden = d_hidden
        self.l1_coef  = l1_coef

        self.W_enc = nn.Parameter(torch.randn(d_hidden, d_input) * 0.02)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.randn(d_input, d_hidden) * 0.02)
        self.b_pre = nn.Parameter(torch.zeros(d_input))

        # Normalise decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(F.linear(x - self.b_pre, self.W_enc, self.b_enc))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.W_dec) + self.b_pre

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z    = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = ((x - x_hat) ** 2).mean()
        sparsity   = z.abs().mean()
        loss = recon_loss + self.l1_coef * sparsity
        return loss, recon_loss, sparsity

    def normalise_decoder(self):
        """Keep decoder columns unit-norm after each gradient step."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)


def collect_mlp_activations(model, texts: list[str],
                              layer: int = 7) -> torch.Tensor:
    """
    Collect MLP output activations at `layer` for every token position
    across all texts. Returns a (N_tokens, d_model) tensor.
    """
    all_acts = []
    for text in texts:
        tokens = model.to_tokens(text)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        act = cache[f"blocks.{layer}.hook_mlp_out"][0]   # (T, d)
        all_acts.append(act)
    return torch.cat(all_acts, dim=0)   # (N, d)


def train_sae(sae: SparseAutoencoder, activations: torch.Tensor,
              n_epochs: int = 50, batch_size: int = 256,
              lr: float = 1e-3) -> list[float]:
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    losses = []
    N = activations.shape[0]
    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            x = activations[perm[i: i + batch_size]]
            loss, recon, sparsity = sae(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalise_decoder()
            epoch_loss += loss.item()
            n_batches  += 1
        avg = epoch_loss / n_batches
        losses.append(avg)
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                z_all = sae.encode(activations)
                frac_active = (z_all > 0).float().mean().item()
            print(f"  SAE epoch {epoch+1:3d} | loss={avg:.5f} | "
                  f"active_frac={frac_active:.4f}")
    return losses


def find_top_activating_tokens(sae: SparseAutoencoder,
                                activations: torch.Tensor,
                                all_tokens: list[str],
                                feature_idx: int,
                                top_k: int = 10) -> list[tuple[str, float]]:
    """Return the top-k tokens that most activate a given SAE feature."""
    with torch.no_grad():
        z = sae.encode(activations)
    feature_acts = z[:, feature_idx].numpy()
    top_indices  = np.argsort(feature_acts)[-top_k:][::-1]
    return [(all_tokens[i], float(feature_acts[i])) for i in top_indices]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    model = load_model()
    if model is None:
        print("TransformerLens not available. Install it to run this practical.")
        exit(0)

    # ── Task 1: Logit Lens ────────────────────────────────────────────────────
    print("=" * 60)
    print("TASK 1: Logit Lens")
    print("=" * 60)
    FACTUAL_PROMPTS = [
        ("The capital of France is",       " Paris"),
        ("The chemical symbol for gold is", " Au"),
        ("Shakespeare wrote",               " Hamlet"),
    ]
    for prompt, target in FACTUAL_PROMPTS:
        print(f"\nPrompt: '{prompt}'  →  target: '{target}'")
        results, token_strs = logit_lens(model, prompt, top_k=5)
        for layer in [0, 3, 6, 9, 11]:
            top5 = ", ".join(f"'{t}'={p:.3f}" for t, p in results[layer])
            print(f"  Layer {layer:2d}: {top5}")

    # Plot for first prompt
    results0, _ = logit_lens(model, FACTUAL_PROMPTS[0][0], top_k=10)
    plot_logit_lens(results0, FACTUAL_PROMPTS[0][0], FACTUAL_PROMPTS[0][1])

    # ── Task 2: Direct Logit Attribution ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 2: Direct Logit Attribution")
    print("=" * 60)
    for prompt, target in FACTUAL_PROMPTS:
        print(f"\nPrompt: '{prompt}'  target: '{target}'")
        contribs = direct_logit_attribution(model, prompt, target)
        top_pos = sorted(contribs.items(), key=lambda x: x[1], reverse=True)[:5]
        top_neg = sorted(contribs.items(), key=lambda x: x[1])[:3]
        print("  Top positive contributors:")
        for name, val in top_pos:
            print(f"    {name:12s}: {val:+.4f}")
        print("  Top negative contributors:")
        for name, val in top_neg:
            print(f"    {name:12s}: {val:+.4f}")

    contribs0 = direct_logit_attribution(model, FACTUAL_PROMPTS[0][0],
                                          FACTUAL_PROMPTS[0][1])
    plot_logit_attribution(contribs0, FACTUAL_PROMPTS[0][1])

    # ── Task 3: Activation Patching (IOI) ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 3: Activation Patching — IOI task")
    print("=" * 60)
    print(f"Clean     : '{IOI_CLEAN}'")
    print(f"Corrupted : '{IOI_CORRUPTED}'")
    print(f"Target    : '{IOI_TARGET}'")
    print("Patching residual stream layer by layer…")
    effects = patch_residual_and_score(model, IOI_CLEAN, IOI_CORRUPTED, IOI_TARGET)
    for layer, eff in enumerate(effects):
        bar = "█" * int(abs(eff) * 20) if eff > 0 else ""
        print(f"  Layer {layer:2d}: {eff:+.4f}  {bar}")

    plot_patching_effects(
        effects,
        title="Activation Patching: which layer carries subject identity?\n"
              f"Clean='{IOI_CLEAN[:30]}…'  Target='{IOI_TARGET}'"
    )

    # ── Task 4: Linear Probes ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 4: Linear Probes — Verb vs Noun at final token position")
    print("=" * 60)
    print("Training probes at each layer…")
    accs = train_probe_all_layers(model, PROBE_SENTENCES)
    if accs:
        best_layer = int(np.argmax(accs))
        print(f"\nBest layer: {best_layer}  (accuracy={accs[best_layer]:.3f})")
        plot_probe_accuracies(accs)

    # ── Task 5: Sparse Autoencoder ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 5: Sparse Autoencoder on MLP activations (layer 7)")
    print("=" * 60)

    SAE_CORPUS = [
        "The cat sat on the mat.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1)+fibonacci(n-2)",
        "The capital of France is Paris.",
        "She walked quickly to the store.",
        "import numpy as np\nX = np.random.randn(100, 10)",
        "The king and queen attended the royal banquet.",
        "Neural networks learn from data.",
        "The doctor prescribed antibiotics for the infection.",
        "print('Hello, world!')",
        "The sun rises in the east and sets in the west.",
        "He was never dishonest about his intentions.",
        "class Transformer(nn.Module):\n    def __init__(self):",
        "The patient's blood pressure was 120/80 mmHg.",
        "To be or not to be, that is the question.",
        "The gradient descent algorithm minimises the loss function.",
    ] * 20   # repeat for more data

    SAE_LAYER = 7
    print(f"Collecting MLP activations at layer {SAE_LAYER}…")
    all_tokens_flat = []
    for text in SAE_CORPUS:
        toks = model.to_tokens(text)[0]
        all_tokens_flat.extend([model.to_string(t.item()) for t in toks])

    activations = collect_mlp_activations(model, SAE_CORPUS, layer=SAE_LAYER)
    print(f"Activation matrix shape: {activations.shape}  "
          f"(tokens × d_model={activations.shape[1]})")

    # Normalise activations
    act_mean = activations.mean(0, keepdim=True)
    act_std  = activations.std(0, keepdim=True) + 1e-8
    activations_norm = (activations - act_mean) / act_std

    D_HIDDEN = activations.shape[1] * 4   # 4× overcomplete
    sae = SparseAutoencoder(d_input=activations.shape[1], d_hidden=D_HIDDEN,
                             l1_coef=5e-4)
    print(f"SAE: d_input={activations.shape[1]}, d_hidden={D_HIDDEN} "
          f"({D_HIDDEN // activations.shape[1]}× overcomplete)")
    print("Training SAE…")
    sae_losses = train_sae(sae, activations_norm, n_epochs=60, batch_size=256)

    # Plot SAE training loss
    plt.figure(figsize=(8, 4))
    plt.plot(sae_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (recon + L1 sparsity)")
    plt.title("Sparse Autoencoder Training Loss")
    plt.tight_layout()
    plt.savefig("week14_sae_loss.png", dpi=150)
    plt.show()
    print("Saved: week14_sae_loss.png")

    # Inspect top features
    print("\nTop-activating tokens for 10 random SAE features:")
    feature_indices = random.sample(range(D_HIDDEN), min(10, D_HIDDEN))
    for feat_idx in feature_indices:
        top_toks = find_top_activating_tokens(
            sae, activations_norm, all_tokens_flat[:len(activations_norm)],
            feat_idx, top_k=6
        )
        top_str = ", ".join(f"'{t}'({v:.2f})" for t, v in top_toks)
        print(f"  Feature {feat_idx:5d}: {top_str}")

    # Feature activation sparsity histogram
    with torch.no_grad():
        z_all = sae.encode(activations_norm)
    active_per_token = (z_all > 0).float().sum(dim=1).numpy()
    plt.figure(figsize=(8, 4))
    plt.hist(active_per_token, bins=40, color="steelblue", edgecolor="white")
    plt.xlabel("Number of active features per token")
    plt.ylabel("Count")
    plt.title("SAE Feature Sparsity: Active Features per Token")
    plt.tight_layout()
    plt.savefig("week14_sae_sparsity.png", dpi=150)
    plt.show()
    print("Saved: week14_sae_sparsity.png")

    # ── Task 6: Superposition demo ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 6: Superposition — neuron polysemanticity analysis")
    print("=" * 60)
    # Find the most polysemantic MLP neurons: highest entropy of top-token dist
    d_model = activations.shape[1]
    neuron_entropies = []
    for neuron_idx in range(min(d_model, 200)):
        acts = activations[:, neuron_idx].numpy()
        # Top-20 activating token activations
        top_idx = np.argsort(acts)[-20:]
        top_acts = acts[top_idx]
        top_acts = top_acts - top_acts.min() + 1e-8
        probs = top_acts / top_acts.sum()
        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
        neuron_entropies.append((neuron_idx, entropy))

    neuron_entropies.sort(key=lambda x: x[1], reverse=True)
    print("Most polysemantic neurons (highest entropy across top activating tokens):")
    for neuron_idx, ent in neuron_entropies[:5]:
        top_idx = np.argsort(activations[:, neuron_idx].numpy())[-8:][::-1]
        top_toks = [all_tokens_flat[i] for i in top_idx
                    if i < len(all_tokens_flat)]
        print(f"  Neuron {neuron_idx:4d} (entropy={ent:.3f}): "
              f"{[repr(t) for t in top_toks[:6]]}")

    print("\nLeast polysemantic neurons (lowest entropy — most monosemantic):")
    for neuron_idx, ent in neuron_entropies[-5:]:
        top_idx = np.argsort(activations[:, neuron_idx].numpy())[-8:][::-1]
        top_toks = [all_tokens_flat[i] for i in top_idx
                    if i < len(all_tokens_flat)]
        print(f"  Neuron {neuron_idx:4d} (entropy={ent:.3f}): "
              f"{[repr(t) for t in top_toks[:6]]}")

    print("\nAll tasks complete.")
