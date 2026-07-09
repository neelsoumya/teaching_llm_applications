## Practicals

- [Francois course](https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer/)

- [CS336 Stanford coursework](https://cs336.stanford.edu/#coursework)

- [`nanoGPT`](https://github.com/karpathy/nanogpt)

- Google Colab and restrict to only a few open-source models

- Huggingface spaces


## Installation

```python
python -m venv venv_llm

source venv_llm/bin/activate

```

```python
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

```bash
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT
```

## Quickstart (Week 1)

- Week 1

First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```python
python data/shakespeare_char/prepare.py
```

This encodes and decodes text using a very simple scheme (we just map characters to ints):

```python
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
```

> on a GPU

change in `train_shakespeare_char.py`

```python
# on macbook also add
device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
```

then run

```python
python train.py config/train_shakespeare_char.py
```

or

```python
cd /Users/soumyabanerjee/soumya_cam_mac/teaching/teaching_llm_applications/do_not_upload/nanoGPT-master
../../venv_llm/bin/python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

- NOTE:🧩 🚀  🎥 [`Flash Attention`](https://www.youtube.com/shorts/OSAelpOrqWs)

- See lecture on [Flash Attention](flash_attention.md)

> Flash Attention is a highly optimized algorithm that accelerates AI transformer models and reduces memory usage. It achieves this by efficiently managing data movement between a GPU's slow, high-bandwidth memory (HBM) and fast on-chip SRAM, making it a fundamental component for training and running large language models.


### Scaling laws using `nanoGPT`

- [Chinchilla paper](https://arxiv.org/pdf/2203.15556.pdf)

🤔
> Given a fixed FLOPs budget,1 how should one trade-off model size and the number of training tokens?

- 🎮 scaling laws practical VERY GOOD using [nanoGPT by Andrew Karpathy](https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb)


- pay attention to Approach 2. Andrew Karpathy says:

> Approach 2 is probably my favorite one because it fixes a flop budget and runs a number of model/dataset sizes, measures the loss, fits a parabolla, and gets the minimum. So it's a fairly direct measurement of what we're after. The best way to then calculate the compute-optimal number of tokens for any given model size, as an example, is via simple interpolation.

```python
# Approach 1 numbers
# # parameters, tokens
# raw = [
#     [400e6, 8e9],
#     [1e9, 20.2e9],
#     [10e9, 205.1e9],
#     [67e9, 1.5e12],
#     [175e9, 3.7e12],
#     [280e9, 5.9e12],
#     [520e9, 11e12],
#     [1e12, 21.2e12],
#     [10e12, 216.2e12],
# ]

# Approach 2 numbers
# parameters, tokens
raw = [
    [400e6, 7.7e9],
    [1e9, 20.0e9],
    [10e9, 219.5e9],
    [67e9, 1.7e12],
    [175e9, 4.3e12],
    [280e9, 7.1e12],
    [520e9, 13.4e12],
    [1e12, 26.5e12],
    [10e12, 292.0e12],
]

# fit a line by linear regression to the raw data
import numpy as np
x = np.array([np.log10(x[0]) for x in raw])
y = np.array([np.log10(x[1]) for x in raw])
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
print(f"y = {m}x + {c}")

plt.figure(figsize=(3, 3))
# plot the line
plt.plot([q[0] for q in raw], [10**(m*np.log10(q[0]) + c) for q in raw], label='linear regression', color='r')
# plot the raw data
plt.scatter([q[0] for q in raw], [q[1] for q in raw], label='raw data')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('parameters')
plt.ylabel('tokens')
plt.title('compute optimal models')
plt.grid()

# how many parameters required?
xquery = 124e6 # query model size here (e.g. GPT-2 small is 124M)
yquery = 10**(m*np.log10(xquery) + c)
print(f"predicted parameters for {xquery:e} tokens: {yquery:e}")
```


### Reproducing GPT-2 using `nanoGPT`

- [Reproducing GPT-2](https://github.com/karpathy/nanogpt#reproducing-gpt-2)

### Baselines using `nanoGPT`

- [Baselines using `nanoGPT`](https://github.com/karpathy/nanogpt#baselines)

### Finetuning using `nanoGPT`

- [Finetuning using `nanoGPT`](https://github.com/karpathy/nanogpt#finetuning)




# Practical: Understanding and Training Language Models with nanoGPT

**Course:** Large Language Models
**Duration:** ~3 hours (can be split into 2 sessions)
**Prerequisites:** Python, PyTorch basics, familiarity with transformer architecture (attention, embeddings, layer norm)

---

## 1. Learning Objectives

By the end of this practical, students should be able to:

1. Explain how a GPT-style decoder-only transformer is implemented end-to-end in ~300 lines of code.
2. Prepare a text dataset for character-level and token-level language modelling.
3. Train a small GPT model from scratch and monitor loss curves.
4. Sample text from a trained model and reason about temperature/top-k effects.
5. Modify architectural hyperparameters (depth, width, context length) and measure their effect on loss and generation quality.
6. Connect what they see in the code to concepts from lectures: causal self-attention, positional embeddings, weight tying, learning rate schedules.

---

## 2. Background

[nanoGPT](https://github.com/karpathy/nanoGPT) is Andrej Karpathy's minimal, readable reimplementation of GPT-2 style training. It is deliberately small (~300 lines for the model, ~300 for training) so that every operation is traceable. This makes it an ideal teaching tool: unlike production codebases (Megatron-LM, GPT-NeoX), there is no abstraction hiding the core mechanics.

The repository has two main files students should read closely:
- `model.py` — the GPT architecture (embeddings, transformer blocks, attention, MLP, final head).
- `train.py` — the training loop (data loading, optimizer, LR schedule, gradient accumulation, checkpointing).

We will use the **character-level Shakespeare** dataset for the first exercises (trains in minutes on CPU/GPU) and optionally move to **GPT-2 BPE tokenised** data for a second pass.

---

## 3. Environment Setup

### 3.1 Requirements

```bash
python -m venv nanogpt-env
source nanogpt-env/bin/activate   # on Windows: nanogpt-env\Scripts\activate

pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Notes for the class:
- `torch`: obviously, needs a CUDA build if you have a GPU. CPU works fine for the character-level demo.
- `tiktoken`: OpenAI's BPE tokenizer, used for GPT-2 style data prep.
- `wandb`: optional, only if you want live loss plots (can be disabled with `--wandb_log=False`).

### 3.2 Clone the repository

```bash
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT
```

### 3.3 Sanity check

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False`, everything still works — just set `--device=cpu` and use the smallest config below.

---

## 4. Part A — Character-Level GPT on Shakespeare (fast, ~10 min)

### 4.1 Prepare the data

```bash
cd data/shakespeare_char
python prepare.py
cd ../..
```

This downloads `tinyshakespeare` (~1MB of text), builds a character-level vocabulary, and writes `train.bin` / `val.bin` as `uint16` numpy arrays plus a `meta.pkl` with the vocab mapping.

**Discussion point for class:** ask students to open `prepare.py` and identify:
- How the vocabulary is built (`sorted(list(set(data)))`).
- Why data is stored as `.bin` files rather than re-tokenised on the fly (I/O efficiency, memory-mapping via `np.memmap`).

### 4.2 Inspect the model code

Open `model.py` and walk through, in this order:

1. **`CausalSelfAttention`** — note the causal mask via `torch.tril` and how `F.scaled_dot_product_attention` (or the manual softmax path) enforces that token *i* only attends to tokens ≤ *i*.
2. **`MLP`** — the position-wise feed-forward block (GELU, 4x expansion).
3. **`Block`** — pre-norm residual structure: `x = x + attn(ln1(x))`, `x = x + mlp(ln2(x))`.
4. **`GPT`** — token embedding + positional embedding, stack of blocks, final layer norm, and the `lm_head` linear layer. Point out **weight tying**: `self.transformer.wte.weight = self.lm_head.weight`.
5. **`generate`** method at the bottom — autoregressive sampling loop with temperature and top-k.

### 4.3 Train a tiny model

```bash
python train.py config/train_shakespeare_char.py \
  --device=cpu \
  --compile=False \
  --eval_iters=20 \
  --log_interval=1 \
  --block_size=64 \
  --batch_size=12 \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=128 \
  --max_iters=2000 \
  --lr_decay_iters=2000 \
  --dropout=0.0
```

On a laptop CPU this should take a few minutes and reach a validation loss around 1.5–1.7 (character-level, so this is expected to look "high" compared to word-level perplexity figures students may have seen in lectures — worth flagging explicitly).

If a GPU is available:
```bash
python train.py config/train_shakespeare_char.py --compile=False
```

### 4.4 Sample from the model

```bash
python sample.py --out_dir=out-shakespeare-char --num_samples=3 --max_new_tokens=200
```

Ask students to try:
```bash
python sample.py --out_dir=out-shakespeare-char --temperature=0.2 --num_samples=1
python sample.py --out_dir=out-shakespeare-char --temperature=1.2 --num_samples=1
```
and discuss the qualitative difference (more repetitive/conservative vs. more diverse/incoherent).

---

## 5. Part B — Reading the Training Loop

Open `train.py` and have students identify the following components (this maps directly to lecture material):

| Concept | Where in code |
|---|---|
| Learning rate warmup + cosine decay | `get_lr(it)` function |
| Gradient accumulation | `gradient_accumulation_steps`, the loop over `micro_step` |
| Mixed precision | `torch.amp.autocast`, `GradScaler` |
| AdamW optimizer with weight decay only on 2D params | `configure_optimizers` in `model.py` |
| Gradient clipping | `torch.nn.utils.clip_grad_norm_` |
| Checkpointing | `torch.save(checkpoint, ...)` inside the eval block |
| Distributed training (DDP) | `ddp` flag, `DistributedDataParallel` wrapper |

**Exercise:** ask students to add a print statement (or log to a list and plot with matplotlib) that records `lr` at every iteration, then plot it. They should recover the classic warmup-then-cosine-decay curve.

---

## 6. Part C — Minimal Reimplementation (conceptual exercise)

To make sure students actually understand attention (not just run someone else's code), have them fill in a stripped-down version of causal self-attention from scratch, then compare outputs against nanoGPT's implementation on a fixed random seed.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MinimalCausalSelfAttention(nn.Module):
    """
    A from-scratch causal self-attention module for students to complete.
    Compare against nanoGPT's model.CausalSelfAttention for correctness.
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # TODO(student): define q, k, v projections (can be one combined Linear or three)
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # causal mask, registered as a buffer so it moves with .to(device)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # TODO(student): project x to q, k, v and reshape into heads
        qkv = self.qkv_proj(x)                       # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)       # each (B, T, C)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # TODO(student): compute scaled dot-product attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, nh, T, T)

        # TODO(student): apply the causal mask BEFORE softmax
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v                                    # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-merge heads

        y = self.resid_dropout(self.out_proj(y))
        return y


if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, C, nh = 2, 8, 32, 4
    x = torch.randn(B, T, C)

    mine = MinimalCausalSelfAttention(n_embd=C, n_head=nh, block_size=T, dropout=0.0)
    out = mine(x)
    print("Output shape:", out.shape)  # expect (2, 8, 32)

    # Sanity check: causality. Perturbing a later token must not change
    # the output of an earlier position.
    x2 = x.clone()
    x2[:, -1, :] += 100.0  # large perturbation to the LAST token only
    out2 = mine(x2)
    diff_early = (out[:, :-1, :] - out2[:, :-1, :]).abs().max().item()
    print("Max difference in earlier positions after perturbing last token:", diff_early)
    assert diff_early < 1e-5, "Causality violated: earlier tokens are seeing future information!"
    print("Causality check passed.")
```

**Instructor note:** the causality check at the bottom is a good "aha" moment — students directly observe that perturbing token *T* has zero effect on outputs at positions `< T`, which is the empirical signature of the causal mask working correctly.

---

## 7. Part D — Hyperparameter Exploration (group exercise)

Split students into small groups. Each group trains the character-level model with **one** hyperparameter changed from the baseline (`config/train_shakespeare_char.py`), keeping `max_iters` fixed so runs are comparable in wall-clock time. Suggested axes:

| Group | Change | Question to answer |
|---|---|---|
| A | `n_layer`: 4 → 8 | Does depth reduce val loss at fixed iters? At what cost in wall-clock time? |
| B | `block_size`: 64 → 256 | Does longer context help character-level Shakespeare? Why might returns diminish? |
| C | `dropout`: 0.0 → 0.2 | Effect on train vs. val loss gap (overfitting diagnostic) |
| D | `learning_rate`: default → 10x higher/lower | Instability or slower convergence? |
| E | `n_head`: 4 → 1 vs 4 → 8 | Does number of heads (at fixed `n_embd`) matter much for this dataset size? |

Each group should record final train/val loss and produce a short (2-3 sentence) explanation grounded in what they saw in `model.py`. Reconvene and compare across groups — this generates a natural discussion of the **bias-variance tradeoff** and **compute-optimal scaling** (a segue into Chinchilla/scaling-law lecture material if relevant).

### 7.1 Helper script to log and plot results

```python
"""
compare_runs.py
Parses nanoGPT training logs (stdout redirected to a file) and plots
validation loss curves for multiple runs on one figure.

Usage:
    python train.py config/train_shakespeare_char.py --n_layer=8 > run_deep.log
    python train.py config/train_shakespeare_char.py --n_layer=4 > run_baseline.log
    python compare_runs.py run_baseline.log run_deep.log
"""
import re
import sys
import matplotlib.pyplot as plt

def parse_log(path):
    """
    nanoGPT prints lines like:
    iter 100: loss 1.4523, time 12.34ms, mfu 3.21%
    step 200: train loss 1.3021, val loss 1.4890
    We extract the 'val loss' lines.
    """
    iters, val_losses = [], []
    pattern = re.compile(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)")
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                iters.append(int(m.group(1)))
                val_losses.append(float(m.group(3)))
    return iters, val_losses

def main(log_paths):
    plt.figure(figsize=(7, 5))
    for path in log_paths:
        iters, val_losses = parse_log(path)
        if not iters:
            print(f"Warning: no val loss lines found in {path}")
            continue
        plt.plot(iters, val_losses, marker="o", label=path)

    plt.xlabel("Iteration")
    plt.ylabel("Validation loss")
    plt.title("nanoGPT validation loss comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("val_loss_comparison.png", dpi=150)
    print("Saved plot to val_loss_comparison.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_runs.py <log1> <log2> ...")
        sys.exit(1)
    main(sys.argv[1:])
```

---

## 8. Part E — Optional Extension: GPT-2 BPE Tokenisation and Fine-tuning

For advanced students or a second session, move from character-level to real subword tokenisation and (optionally) fine-tune from released GPT-2 weights.

### 8.1 Prepare OpenWebText-style data (BPE, word-level modelling)

```bash
cd data/shakespeare   # note: NOT shakespeare_char — this uses tiktoken BPE
python prepare.py
cd ../..
```

Have students diff this `prepare.py` against the char-level one:
```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode_ordinary(text)
```
vs. the character-level `stoi`/`itos` dictionaries. This is a good moment to discuss vocabulary size (50,257 for GPT-2 BPE vs. ~65 characters) and what that implies for embedding table size and the softmax over the vocabulary at the output.

### 8.2 Fine-tune from pretrained GPT-2 weights

```bash
python train.py config/finetune_shakespeare.py
```

This loads GPT-2 weights via HuggingFace `transformers` under the hood (`model.py`'s `from_pretrained` classmethod) and fine-tunes on the Shakespeare corpus. Compare generations before and after fine-tuning:

```bash
python sample.py --init_from=gpt2 --start="ROMEO:" --num_samples=1
python sample.py --out_dir=out-shakespeare --start="ROMEO:" --num_samples=1
```

Discuss: how quickly does style transfer occur? How many fine-tuning steps were needed vs. training from scratch in Part A?

---

## 9. Assessment / Deliverables

Ask each student (or pair) to submit:

1. A short report (max 2 pages) covering:
   - Their group's hyperparameter change (Section 7) and observed effect on loss.
   - The completed and passing causality check from Section 6.
   - One plot: validation loss vs. iteration for at least two configurations.
2. Answers to the following conceptual questions:
   - Why is the causal mask applied *before* the softmax rather than after?
   - Why does nanoGPT tie the input embedding and output projection weights? What does this save, and what assumption does it rely on?
   - Character-level vs. BPE tokenisation: what are the tradeoffs in sequence length, vocabulary size, and generalisation to unseen words?
   - What would need to change in `model.py` to support a non-causal (encoder-style, BERT-like) attention pattern?

---

## 10. Common Pitfalls / FAQ (for instructor reference)

- **`--compile=True` fails on some machines**: `torch.compile` requires a recent PyTorch + compatible GPU driver/toolchain. Default to `--compile=False` for classroom laptops to avoid losing time to environment debugging.
- **Out of memory on GPU**: reduce `batch_size` or `block_size`, or add `--dtype=bfloat16` / `float16`.
- **Val loss not printed**: check `eval_interval` and `eval_iters` — with a very short `max_iters`, evaluation may not trigger before training ends.
- **Confusing loss scale between char-level and BPE runs**: loss is per-token cross-entropy in nats; it is *not* directly comparable across tokenisations, since a "token" means something different in each case. Worth stating explicitly to avoid a common student misconception.
- **CPU training feels slow**: emphasise that the point of Part A is pedagogical traceability, not efficiency; Part E briefly touches why real GPT-2 training needs GPUs/TPUs at scale.

