"""
Week 3 Practical: Embeddings and Representations
==================================================
Objectives:
  - Compute sentence embeddings with sentence-transformers
  - Visualise embedding space using PCA, t-SNE, and UMAP
  - Compute cosine similarity and build a simple semantic search engine
  - Implement sinusoidal positional encodings from scratch and visualise them
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("umap-learn not installed – UMAP section will be skipped")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("sentence-transformers not installed – using random embeddings for demo")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sentence embeddings
# ─────────────────────────────────────────────────────────────────────────────
SENTENCES = {
    "cats": [
        "The cat sat on the mat.",
        "A feline rested on a rug.",
        "My cat loves sleeping on soft surfaces.",
        "Cats often curl up on comfortable spots.",
    ],
    "science": [
        "Neural networks learn from data through gradient descent.",
        "Deep learning models require large amounts of labelled examples.",
        "Backpropagation computes gradients for training neural networks.",
        "Transformers use attention mechanisms to process sequences.",
    ],
    "weather": [
        "It is raining heavily outside today.",
        "The storm brought strong winds and flooding.",
        "Sunny skies are expected for the weekend.",
        "Heavy precipitation is forecast for Tuesday.",
    ],
    "food": [
        "The pasta was perfectly cooked al dente.",
        "She enjoyed a delicious bowl of ramen.",
        "The restaurant served excellent Italian cuisine.",
        "Freshly baked bread has a wonderful aroma.",
    ],
}

ALL_SENTENCES = [s for group in SENTENCES.values() for s in group]
LABELS = [label for label, group in SENTENCES.items() for _ in group]


def get_embeddings(sentences: list[str]) -> np.ndarray:
    """Return a (N, D) embedding matrix."""
    if HAS_SBERT:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(sentences, show_progress_bar=False)
    else:
        # Fallback: random embeddings with cluster structure for demo
        np.random.seed(42)
        n = len(sentences)
        d = 64
        embeddings = np.random.randn(n, d)
        # Add cluster offsets to simulate meaningful structure
        cluster_ids = {label: i for i, label in enumerate(SENTENCES.keys())}
        for i, label in enumerate(LABELS):
            embeddings[i] += cluster_ids[label] * 3
        return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cosine similarity
# ─────────────────────────────────────────────────────────────────────────────
def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalised = embeddings / (norms + 1e-8)
    return normalised @ normalised.T


def semantic_search(query: str, corpus: list[str], embeddings: np.ndarray,
                    top_k: int = 3) -> list[tuple[str, float]]:
    """Return top-k most similar sentences to the query."""
    if HAS_SBERT:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = model.encode([query])[0]
    else:
        np.random.seed(hash(query) % 1000)
        q_emb = np.random.randn(embeddings.shape[1])

    sims = [cosine_similarity(q_emb, emb) for emb in embeddings]
    ranked = sorted(zip(corpus, sims), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────
COLOURS = {"cats": "tab:blue", "science": "tab:orange",
           "weather": "tab:green", "food": "tab:red"}


def plot_embeddings(reduced: np.ndarray, labels: list[str], title: str, filename: str):
    plt.figure(figsize=(8, 6))
    for label, colour in COLOURS.items():
        mask = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced[mask, 0], reduced[mask, 1],
                    c=colour, label=label, s=80, alpha=0.8)
    for i, sent in enumerate(ALL_SENTENCES):
        plt.annotate(sent[:30] + "…" if len(sent) > 30 else sent,
                     (reduced[i, 0], reduced[i, 1]),
                     fontsize=6, alpha=0.6)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sinusoidal positional encoding (from scratch)
# ─────────────────────────────────────────────────────────────────────────────
def sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """
    Returns a (max_len, d_model) positional encoding matrix using
    the original Vaswani et al. (2017) sinusoidal formulation.
    """
    PE = np.zeros((max_len, d_model))
    positions = np.arange(max_len)[:, np.newaxis]          # (max_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) *
                      -(math.log(10000.0) / d_model))      # (d_model/2,)

    PE[:, 0::2] = np.sin(positions * div_term)
    PE[:, 1::2] = np.cos(positions * div_term)
    return PE


def plot_positional_encoding(PE: np.ndarray, filename: str = "week03_pos_enc.png"):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(PE, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel("Embedding dimension")
    plt.ylabel("Position")
    plt.title("Sinusoidal Positional Encoding")

    plt.subplot(1, 2, 2)
    for pos in [0, 5, 10, 20, 50]:
        plt.plot(PE[pos, :64], label=f"pos={pos}", alpha=0.8)
    plt.xlabel("Embedding dimension")
    plt.ylabel("Value")
    plt.title("PE vectors for selected positions (first 64 dims)")
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Task 1: Compute embeddings ────────────────────────────────────────────
    print("=" * 60)
    print("TASK 1: Computing sentence embeddings")
    print("=" * 60)
    embeddings = get_embeddings(ALL_SENTENCES)
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"  {len(ALL_SENTENCES)} sentences × {embeddings.shape[1]}-dim vectors")

    # ── Task 2: Cosine similarity ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 2: Cosine similarity between sentence pairs")
    print("=" * 60)
    pairs = [
        (0, 1),   # two cat sentences (should be high)
        (0, 4),   # cat vs neural network (should be low)
        (4, 5),   # two science sentences (should be high)
        (8, 9),   # two weather sentences (should be high)
    ]
    for i, j in pairs:
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"  sim={sim:.4f}  '{ALL_SENTENCES[i][:40]}' ↔ '{ALL_SENTENCES[j][:40]}'")

    # ── Task 3: Similarity matrix heatmap ─────────────────────────────────────
    sim_matrix = cosine_similarity_matrix(embeddings)
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    tick_labels = [s[:20] + "…" if len(s) > 20 else s for s in ALL_SENTENCES]
    plt.xticks(range(len(ALL_SENTENCES)), tick_labels, rotation=90, fontsize=6)
    plt.yticks(range(len(ALL_SENTENCES)), tick_labels, fontsize=6)
    plt.title("Sentence Embedding Cosine Similarity Matrix")
    plt.tight_layout()
    plt.savefig("week03_similarity_matrix.png", dpi=150)
    plt.show()
    print("Saved: week03_similarity_matrix.png")

    # ── Task 4: Semantic search ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 4: Semantic search")
    print("=" * 60)
    queries = [
        "What is a transformer model?",
        "My pet likes to nap.",
        "Will it be sunny tomorrow?",
    ]
    for q in queries:
        print(f"\nQuery: '{q}'")
        results = semantic_search(q, ALL_SENTENCES, embeddings, top_k=3)
        for sent, score in results:
            print(f"  {score:.4f}  {sent}")

    # ── Task 5: Dimensionality reduction and visualisation ────────────────────
    print("\n" + "=" * 60)
    print("TASK 5: Visualisation – PCA, t-SNE")
    print("=" * 60)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_reduced = pca.fit_transform(embeddings)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    plot_embeddings(pca_reduced, LABELS, "PCA (2D) of Sentence Embeddings",
                   "week03_pca.png")

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42, n_iter=1000)
    tsne_reduced = tsne.fit_transform(embeddings)
    plot_embeddings(tsne_reduced, LABELS, "t-SNE (2D) of Sentence Embeddings",
                   "week03_tsne.png")

    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_reduced = reducer.fit_transform(embeddings)
        plot_embeddings(umap_reduced, LABELS, "UMAP (2D) of Sentence Embeddings",
                       "week03_umap.png")

    # ── Task 6: Positional encoding ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 6: Sinusoidal positional encoding")
    print("=" * 60)
    PE = sinusoidal_positional_encoding(max_len=100, d_model=128)
    print(f"Positional encoding shape: {PE.shape}")

    # Check similarity between positions
    for p1, p2 in [(0, 1), (0, 5), (0, 50), (10, 11), (10, 20)]:
        sim = cosine_similarity(PE[p1], PE[p2])
        print(f"  cos_sim(pos={p1}, pos={p2}) = {sim:.4f}")

    plot_positional_encoding(PE)

    print("\nAll tasks complete.")
