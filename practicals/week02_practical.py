"""
Week 2 Practical: Tokenisation
================================
Objectives:
  - Inspect tokenisation with tiktoken (GPT-2 / GPT-4), BERT WordPiece, and
    LLaMA SentencePiece
  - Train a minimal BPE tokeniser from scratch on a small corpus
  - Visualise token boundaries in a sentence
  - Investigate tokenisation of numbers, code, and non-English text
"""

import re
import json
from collections import Counter, defaultdict

# Optional: comment out any import if the library is not installed
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("tiktoken not installed – skipping GPT tokeniser section")

try:
    from transformers import BertTokenizer, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("transformers not installed – skipping Hugging Face section")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Compare tokenisers on the same text
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Tokenisation is surprisingly important for LLM performance.",
    "The patient's haemoglobin level was 12.3 g/dL on 2024-03-15.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "Bonjour, comment allez-vous aujourd'hui?",
    "1 + 1 = 2, but 12345 + 67890 = 80235",
    "नमस्ते",   # Hindi
]


def tokenise_with_tiktoken(text: str, model: str = "gpt2") -> list[str]:
    if not HAS_TIKTOKEN:
        return []
    enc = tiktoken.encoding_for_model(model)
    ids = enc.encode(text)
    return [enc.decode([i]) for i in ids]


def tokenise_with_bert(text: str) -> list[str]:
    if not HAS_TRANSFORMERS:
        return []
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    return tok.tokenize(text)


def tokenise_with_gpt2_hf(text: str) -> list[str]:
    if not HAS_TRANSFORMERS:
        return []
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok.encode(text)
    return [tok.decode([i]) for i in ids]


def compare_tokenisers(text: str):
    print(f"\nText: {repr(text)}")
    print("-" * 60)

    if HAS_TIKTOKEN:
        gpt2_tokens = tokenise_with_tiktoken(text, "gpt2")
        print(f"GPT-2 tiktoken ({len(gpt2_tokens)} tokens): {gpt2_tokens}")

    if HAS_TRANSFORMERS:
        bert_tokens = tokenise_with_bert(text)
        print(f"BERT WordPiece ({len(bert_tokens)} tokens): {bert_tokens}")

        gpt2_hf = tokenise_with_gpt2_hf(text)
        print(f"GPT-2 HF        ({len(gpt2_hf)} tokens): {gpt2_hf}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Minimal BPE from scratch
# ─────────────────────────────────────────────────────────────────────────────
class MinimalBPE:
    """
    A minimal Byte-Pair Encoding tokeniser trained from scratch.
    Operates at the character level on whitespace-pre-tokenised text.
    """

    def __init__(self):
        self.vocab: dict[tuple, str] = {}   # pair → merged symbol
        self.merges: list[tuple] = []        # ordered list of merges

    def _get_pairs(self, vocab: dict[tuple, int]) -> Counter:
        """Count all adjacent symbol pairs across the vocabulary."""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair: tuple, vocab: dict[tuple, int]) -> dict[tuple, int]:
        """Merge all occurrences of pair in vocab."""
        new_vocab = {}
        bigram = pair[0] + pair[1]
        for word, freq in vocab.items():
            symbols = list(word)
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    new_symbols.append(bigram)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_vocab[tuple(new_symbols)] = freq
        return new_vocab

    def train(self, corpus: str, num_merges: int = 50):
        """Train BPE on a text corpus."""
        # Initialise: split into characters, add end-of-word marker
        words = corpus.lower().split()
        raw_vocab: dict[tuple, int] = Counter()
        for word in words:
            chars = tuple(list(word) + ["</w>"])
            raw_vocab[chars] += 1

        print(f"Initial vocabulary size (unique tokens): {len(set(t for w in raw_vocab for t in w))}")

        vocab = dict(raw_vocab)
        for merge_idx in range(num_merges):
            pairs = self._get_pairs(vocab)
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]
            vocab = self._merge_pair(best_pair, vocab)
            self.merges.append(best_pair)

            merged = best_pair[0] + best_pair[1]
            if merge_idx < 10 or merge_idx % 10 == 0:
                print(f"  Merge {merge_idx + 1:3d}: {best_pair} → '{merged}' "
                      f"(freq={pairs[best_pair]})")

        self.vocab = vocab
        all_symbols = set(t for w in vocab for t in w)
        print(f"\nFinal vocabulary size after {num_merges} merges: {len(all_symbols)}")
        return self

    def tokenise(self, text: str) -> list[str]:
        """Tokenise new text using learned merges."""
        words = text.lower().split()
        tokens = []
        for word in words:
            symbols = list(word) + ["</w>"]
            for pair in self.merges:
                i = 0
                new_symbols = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                        new_symbols.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols
            tokens.extend(symbols)
        return tokens


# ─────────────────────────────────────────────────────────────────────────────
# 3. Visualise token boundaries
# ─────────────────────────────────────────────────────────────────────────────
def visualise_token_boundaries(text: str, tokens: list[str]):
    """Print text with token boundaries marked using | separators."""
    # Reconstruct with colour-coded boundaries
    coloured = ""
    colours = ["\033[94m", "\033[92m"]  # blue, green
    for i, tok in enumerate(tokens):
        coloured += colours[i % 2] + tok.replace("Ġ", " ").replace("▁", " ") + "\033[0m"
    print(f"Original : {text}")
    print(f"Tokenised: {'|'.join(tokens)}")
    print(f"N tokens : {len(tokens)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Token cost estimation
# ─────────────────────────────────────────────────────────────────────────────
def estimate_api_cost(text: str, cost_per_1k_tokens: float = 0.002):
    """Estimate OpenAI API cost for a text string."""
    if not HAS_TIKTOKEN:
        print("tiktoken required for cost estimation")
        return
    enc = tiktoken.encoding_for_model("gpt-4o")
    n_tokens = len(enc.encode(text))
    cost = (n_tokens / 1000) * cost_per_1k_tokens
    print(f"Text length : {len(text)} chars")
    print(f"Token count : {n_tokens}")
    print(f"Est. cost   : ${cost:.6f} (at ${cost_per_1k_tokens}/1k tokens)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("TASK 1: Compare tokenisers across different models")
    print("=" * 60)
    for sent in SAMPLE_SENTENCES:
        compare_tokenisers(sent)

    print("\n" + "=" * 60)
    print("TASK 2: Train a minimal BPE tokeniser from scratch")
    print("=" * 60)
    # Use a simple corpus for demonstration
    corpus = (
        "low lower lowest new newer newest wide widen widest "
        "fast faster fastest slow slower slowest "
        "the cat sat on the mat the cat ate the rat "
        "low low low lower lower newest newest newest "
        "machine learning model training data neural network "
        "language model tokenisation subword vocabulary"
    )
    bpe = MinimalBPE()
    bpe.train(corpus, num_merges=30)

    print("\nTokenising new sentences:")
    test_sentences = [
        "the fastest machine learning model",
        "new lower levels of learning",
        "widest neural network training",
    ]
    for sent in test_sentences:
        toks = bpe.tokenise(sent)
        print(f"  '{sent}'")
        print(f"   → {toks} ({len(toks)} tokens)")

    print("\n" + "=" * 60)
    print("TASK 3: Visualise token boundaries (GPT-2 HF)")
    print("=" * 60)
    if HAS_TRANSFORMERS:
        vis_text = "Tokenisation splits text into subword units."
        vis_tokens = tokenise_with_gpt2_hf(vis_text)
        visualise_token_boundaries(vis_text, vis_tokens)

    print("\n" + "=" * 60)
    print("TASK 4: Token cost estimation")
    print("=" * 60)
    long_text = """
    Large language models have revolutionised natural language processing.
    They are trained on vast amounts of text data using self-supervised
    objectives such as next-token prediction. Applications include
    code generation, summarisation, translation, and question answering.
    """ * 5
    estimate_api_cost(long_text.strip())

    print("\n" + "=" * 60)
    print("TASK 5: Tokenisation quirks to investigate")
    print("=" * 60)
    quirky = [
        "1 + 1 = 2",
        "100000 + 200000 = 300000",
        "SoMeWeIrDcAsInG",
        "hello world",
        " hello world",   # note leading space
        "HELLO WORLD",
        "hello\nworld",
    ]
    if HAS_TIKTOKEN:
        enc = tiktoken.encoding_for_model("gpt2")
        for q in quirky:
            toks = [enc.decode([i]) for i in enc.encode(q)]
            print(f"  {repr(q):35s} → {toks}")
