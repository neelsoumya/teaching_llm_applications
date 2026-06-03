# Week 2 — Tokenisation

## Lecture Overview

Before a language model can process text, that text must be converted into a sequence of integer IDs. This week we explore *tokenisation*: how text is segmented into tokens, why the choice of tokeniser matters, and how to inspect and use tokenisers in practice.

---

## 1. Why Tokenisation?

Neural networks require numerical inputs. Text is a sequence of characters. The simplest approach would be character-level modelling — but:

- Sequences become very long, stressing the context window.
- Characters carry less information per token than words.

Word-level tokenisation is the other extreme:
- Short sequences, but huge vocabulary; unseen words (OOV) are a problem.
- No sharing of information between morphological variants (run / runs / running).

**Subword tokenisation** is the practical solution used by all modern LLMs:
- Vocabulary of 30,000–100,000 subword units.
- Frequent words are single tokens; rare words are split into parts.
- Handles any language and any word form.

---

## 2. Byte-Pair Encoding (BPE)

BPE is the most widely used subword algorithm (GPT-2, GPT-3, GPT-4, LLaMA).

### Algorithm

1. Start with a vocabulary of individual characters (plus a special end-of-word symbol).
2. Count all adjacent symbol pairs in the training corpus.
3. Merge the most frequent pair into a new symbol.
4. Repeat until the vocabulary reaches the target size.

### Example

Training text: `low low low lower lower newest newest`

Initial tokens: `l o w </w>  l o w e r </w>  n e w e s t </w>`

Most frequent pair: `l o` → merge to `lo`
Next: `lo w` → `low`
...and so on until `lowest` might become `low est </w>` etc.

### Properties

- Deterministic at inference time (greedy merge).
- Handles unseen character sequences gracefully.
- Language agnostic.

---

## 3. WordPiece and SentencePiece

**WordPiece** (BERT, DistilBERT):
- Similar to BPE but merges are chosen to maximise likelihood of the training data under a unigram language model.
- Uses `##` prefix to denote continuation subwords (e.g. `play ##ing`).

**SentencePiece** (LLaMA, T5, Mistral):
- Treats the input as a raw byte stream — no pre-tokenisation on whitespace.
- Can handle any script including CJK, Arabic, etc.
- Uses `▁` (U+2581) to denote word-initial subwords.

**Tiktoken** (OpenAI):
- BPE but applied to UTF-8 bytes rather than Unicode characters.
- Handles arbitrary bytes, so never produces unknown tokens.

---

## 4. Special Tokens

Every tokeniser defines special tokens with reserved IDs:

| Token | Purpose |
|-------|---------|
| `<|endoftext|>` | Separator between documents (GPT family) |
| `[CLS]` | Classification token (BERT) |
| `[SEP]` | Separator between sentences (BERT) |
| `[PAD]` | Padding to uniform length in batches |
| `<s>` / `</s>` | Start / end of sequence (LLaMA, Mistral) |
| `<unk>` | Unknown token (rare in modern tokenisers) |

---

## 5. Vocabulary Size and Its Effects

| Model | Tokeniser | Vocabulary Size |
|-------|-----------|----------------|
| GPT-2 | BPE | 50,257 |
| GPT-3 / 4 | Tiktoken (cl100k) | 100,277 |
| BERT | WordPiece | 30,522 |
| LLaMA-2 | SentencePiece | 32,000 |
| LLaMA-3 | Tiktoken | 128,256 |

Larger vocabulary → fewer tokens per sentence → shorter sequences → faster training and inference, but larger embedding matrix.

---

## 6. Tokenisation Artefacts and Gotchas

- **Numbers**: `12345` may become `['12', '345']` or `['1', '2', '3', '4', '5']` — arithmetic is hard.
- **Code**: whitespace and indentation are tokenised differently across models.
- **Non-English text**: languages with less training data are under-represented in the vocabulary; a single word may become many tokens, effectively increasing cost.
- **Case**: `Hello` and `hello` are often different tokens.
- **Leading spaces**: `" dog"` and `"dog"` are often different tokens in BPE.

---

## 7. Counting Tokens (Cost and Context)

API usage is billed per token. Context windows are limited in tokens. Use `tiktoken` to count tokens before sending to a model:

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("The quick brown fox jumps over the lazy dog.")
print(len(tokens))   # 9
```

---

## 8. Practical This Week

See `practicals/week02_practical.py`:
- Train a minimal BPE tokeniser from scratch on a small corpus using the `tokenizers` library.
- Compare tokenisations across GPT-2 (tiktoken), BERT (WordPiece), and LLaMA (SentencePiece).
- Visualise token boundaries in a sentence.

---

## 9. Further Reading

- Sennrich et al. (2016) — "Neural Machine Translation of Rare Words with Subword Units" (BPE) — https://arxiv.org/abs/1508.07909
- Kudo and Richardson (2018) — "SentencePiece" — https://arxiv.org/abs/1808.06226
- [Hugging Face tokeniser docs](https://huggingface.co/docs/tokenizers/index)
- [Tiktoken (OpenAI)](https://github.com/openai/tiktoken)
- Notebook from existing material: `tiktoken_demo.ipynb`

---

## Discussion Questions

1. Why is character-level modelling not used in practice for large-scale LLMs?
2. A user's message in Hindi becomes 3× more tokens than the same message in English. What are the consequences?
3. Sketch the BPE merge process on the toy string `aababab`. What are the first two merges?
