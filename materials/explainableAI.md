# Explainable AI in the context of LLMs and mechanistic interpretability

- Explainability from the social sciences (Miller et al. paper)

- Class-contrastive counterfactual

> Your loan was denied because your salary was less than 40K. If your salary was greater than 50K, your loan would have been approved.

- Black box models

- 🤔 ❓ What kinds of issues can crop up?

- Apply this to LLMs

## Reading about J-space Anthropic paper

- [J-space Anthropic paper](https://kenhuangus.substack.com/p/claudes-hidden-workspace-why-j-space) and [here](https://transformer-circuits.pub/2026/workspace/index.html)


**REVIEW of exlainability**

- Post hoc attribution techniques, inherently interpretable architectures (e.g. decision trees or generalized additive models (GAMs)), and concept-based explanations do not open the black box

_NOTE_: 

- Read the following [review paper](https://dl.acm.org/doi/full/10.1145/3787104) and Neel Nanda's [blog](https://www.neelnanda.io/mechanistic-interpretability/glossary) that has this quote:

>specialized toolchains such as TransformerLens, CircuitsVis, and Neuroscope,researchers can map neurons, attention heads, and residual streams to interpretable computational motifs, paving the way toward transparent and auditable AI systems.

> Central challenges that confront Mechanistic Interpretability, including scalability to billion-parameter models, and entangled (polysemantic) features

>Mechanistic interpretability moves beyond post hoc analysis by reverse-engineering the internal computations of neural networks. Instead of correlating inputs and outputs, it identifies functional subgraphs–neurons, attention heads, and residual streams–that causally implement behaviors.

- `circuits microscope` for tracing activations. Read the [paper](https://distill.pub/2020/circuits/zoom-in/)

- image of sparse autoencoder

![image](https://dl.acm.org/cms/10.1145/3787104/asset/15c81570-d800-4501-a863-c4cc8620b220/assets/images/medium/csur-2025-0787-f07.jpg)

## Demo

- [demo](https://transformerlensorg.github.io/CircuitsVis/?path=/story/activations-textneuronactivations--multiple-samples)



## Mechanistic interpretability: short code in python and a small practical to demonstate this

- image summarizing the practical

![image](../images/mech_interp.png)

```python
"""
Mechanistic Interpretability: a minimal, teachable demo
--------------------------------------------------------
Two techniques, both < 15 lines of actual logic each:

1. LOGIT LENS  -> "what would the model predict if it stopped early?"
2. ATTENTION PATTERNS -> "which tokens is a given head looking at?"

Install:
    pip install transformer_lens

This uses GPT-2 small (117M params) - small enough to run on a laptop CPU.
"""

import torch
from transformer_lens import HookedTransformer

# ---- 1. Load the model ----------------------------------------------------
model = HookedTransformer.from_pretrained("gpt2-small")
model.eval()

prompt = "The Eiffel Tower is located in the city of"
tokens = model.to_tokens(prompt)

# run_with_cache stores EVERY intermediate activation, keyed by name.
# This is the core trick that makes TransformerLens good for teaching:
# you get full internal access "for free".
logits, cache = model.run_with_cache(tokens)

print(f"Prompt: {prompt!r}")
print(f"Model's final answer: {model.to_string(logits[0, -1].argmax())!r}\n")

# ---- 2. Logit Lens: decode the residual stream at every layer -------------
# The residual stream is the running "memory" that every layer reads from
# and writes to. If we apply the model's final steps (layer norm + unembed)
# to an EARLY layer's residual stream, we get a sneak peek at what the
# model "believed" before later layers refined it.
print("Logit Lens — prediction if we stopped after each layer:")
for layer in range(model.cfg.n_layers):
    resid = cache["resid_post", layer][0, -1]      # last token, this layer
    resid_normed = model.ln_final(resid)            # same final norm the real output uses
    layer_logits = model.unembed(resid_normed)       # project to vocabulary
    top_token = model.to_string(layer_logits.argmax())
    print(f"  Layer {layer:2d}: {top_token!r}")

# ---- 3. Attention patterns: what is one head looking at? ------------------
# Attention patterns are stored per layer, shape [batch, head, dest, src].
# This shows, for head 0 in layer 9 (an example — try others!), how much
# each token attends to every previous token when predicting the next one.
layer, head = 9, 0
pattern = cache["pattern", layer][0, head]   # [dest_pos, src_pos]
str_tokens = model.to_str_tokens(prompt)

print(f"\nAttention from last token, Layer {layer} Head {head}:")
last_token_attention = pattern[-1]  # attention paid by the final token
for tok, weight in zip(str_tokens, last_token_attention):
    print(f"  {tok!r:>12}: {weight.item():.3f}")
```


## Mechanistic Interpretability: A Practical Worksheet

**Goal:** by the end of this worksheet you should be able to (1) explain what the residual stream is, (2) use the logit lens to see a prediction "form" across layers, (3) find and characterize an attention head's behavior, and (4) run a simple causal intervention (activation patching) to test a hypothesis about what a component does.

**Setup**

```bash
pip install transformer_lens
```

Everything below uses GPT-2 small (117M params) so it runs comfortably on a laptop CPU. No GPU needed.

```python
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")
model.eval()
```

---

**Exercise 1 — The residual stream and the logit lens**

Background: every layer of a transformer reads from and writes to a shared "residual stream" — think of it as the model's running notepad. The final layer norm + unembedding matrix turn the *last* state of that notepad into next-token logits. The logit lens trick is to apply that same final step to *earlier* layers' notepad states, to see what the model "would have said" if it stopped early.

```python
prompt = "The Eiffel Tower is located in the city of"
tokens = model.to_tokens(prompt)
logits, cache = model.run_with_cache(tokens)

for layer in range(model.cfg.n_layers):
    resid = cache["resid_post", layer][0, -1]
    resid_normed = model.ln_final(resid)
    layer_logits = model.unembed(resid_normed)
    print(f"Layer {layer:2d}: {model.to_string(layer_logits.argmax())!r}")
```

*Try it yourself:*
- Run this on 3 prompts of your own choosing (mix easy factual ones and ambiguous ones).
- At what layer does the correct answer first appear? Does it stay stable, or does it flicker between candidates in later layers?
- Find a prompt where the *early*-layer guess is a reasonable-sounding wrong answer. What does that tell you about how the model builds up an answer?

---

**Exercise 2 — Reading an attention head**

Background: attention patterns tell you, for each token being predicted, how much `weight` it puts on every earlier token. Some heads have very interpretable, consistent jobs (e.g. always attend to the previous token; always attend to the subject of the sentence).

```python
layer, head = 9, 0
pattern = cache["pattern", layer][0, head]  # [dest_pos, src_pos]
str_tokens = model.to_str_tokens(prompt)

last_token_attention = pattern[-1]
for tok, weight in zip(str_tokens, last_token_attention):
    print(f"{tok!>12}: {weight.item():.3f}")
```

*Try it yourself:*
- Loop over all heads in layer 9 and print the top-attended source token for the last position in each. Do any heads look like they have a clear, describable job (e.g. "attends to the subject noun")?
- Pick one head that looks interesting and test it on 3 different prompts. Does its behavior generalize, or was your first observation a coincidence?
- Optional: install `circuitsvis` (`pip install circuitsvis`) for a visual attention pattern plot instead of printed numbers.


---

**Exercise 3 — Induction heads**

Background: induction heads are a well-studied circuit that implements a simple rule: "find the last time the current token appeared before, and predict whatever followed it." They're one of the few sub-circuits in transformers that's been fully reverse-engineered, which makes them a good target for a first real "find a mechanism" exercise.

```python
import torch

# A repeated random sequence is the classic way to elicit induction behavior:
# token at position i should predict whatever followed its previous occurrence.
seq_len = 25
rep_tokens = torch.randint(0, model.cfg.d_vocab, (1, seq_len))
rep_tokens = torch.cat([rep_tokens, rep_tokens], dim=1)  # repeat it once

rep_logits, rep_cache = model.run_with_cache(rep_tokens)

# Induction score: for each head, how much attention (on average, in the
# second half of the sequence) does it place on the token that followed
# the CURRENT token's previous occurrence?
def induction_score(cache, layer, head, seq_len):
    pattern = cache["pattern", layer][0, head]
    score = 0.0
    for dest in range(seq_len, 2 * seq_len):
        src = dest - seq_len + 1  # position right after the earlier occurrence
        score += pattern[dest, src].item()
    return score / seq_len

for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        s = induction_score(rep_cache, layer, head, seq_len)
        if s > 0.3:
            print(f"Layer {layer}, Head {head}: induction score {s:.2f}")
```

*Try it yourself:*
- Run this and note which layer/head combinations score highest.
- Take the single highest-scoring head and inspect its attention pattern directly (reuse Exercise 2's code) on the repeated sequence. Does it match the "look back one token after the last occurrence" story?
- Discuss: why would this be a useful thing for a language model to compute at all?

---

**Exercise 4 — Activation patching (a causal test)**

Background: so far you've only *observed* activations. Activation patching lets you intervene: take an activation from one run and splice it into another, then see if the output changes the way your hypothesis predicts. This is the closest thing mech interp has to a controlled experiment.

```python
clean_prompt = "The capital of France is"
corrupted_prompt = "The capital of Germany is"

clean_tokens = model.to_tokens(clean_prompt)
corrupted_tokens = model.to_tokens(corrupted_prompt)

clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits = model(corrupted_tokens)

paris_token = model.to_single_token(" Paris")

def patch_resid(activation, hook, layer_to_patch, clean_cache):
    activation[:, -1, :] = clean_cache["resid_post", layer_to_patch][:, -1, :]
    return activation

for layer in range(model.cfg.n_layers):
    patched_logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(f"blocks.{layer}.hook_resid_post",
                    lambda act, hook, l=layer: patch_resid(act, hook, l, clean_cache))]
    )
    prob = torch.softmax(patched_logits[0, -1], dim=-1)[paris_token].item()
    print(f"Patching layer {layer:2d} residual stream -> P('Paris') = {prob:.3f}")
```

*Try it yourself:*
- Run this and find the layer at which patching "flips" the model back toward predicting Paris. What does that tell you about where the fact "capital of France" is represented?
- Try patching only a specific attention head's output instead of the whole residual stream (hook name: `blocks.{layer}.attn.hook_z`). Can you narrow the effect down to one head?
- Design your own clean/corrupted pair for a different kind of fact (grammatical number agreement, a different factual association, etc.) and repeat the experiment.

---

**Wrap-up discussion questions**

- What's the difference between what Exercises 1–2 show you (observational) and what Exercise 4 shows you (causal)? Why does that difference matter for trusting a claim like "this head does X"?
- If you were auditing a model for a specific failure mode (e.g. a factual error or a bias), which of these four techniques would you reach for first, and why?
- What did NOT work the way you expected in any of the exercises above? That's usually the most interesting part to discuss.

---

**Further resources**

- TransformerLens docs and demo notebooks: https://github.com/TransformerLensOrg/TransformerLens
- Neel Nanda, "Concrete Steps to Get Started in Mech Interp": https://www.neelnanda.io/mechanistic-interpretability/getting-started
- ARENA course (Callum McDougall) — a full structured curriculum this worksheet borrows ideas from: https://arena-chapter1-transformer-interp.streamlit.app/
- Anthropic, "A Mathematical Framework for Transformer Circuits": https://transformer-circuits.pub/2021/framework/index.html
- Neel Nanda's YouTube channel (walkthroughs of exactly these techniques): https://www.youtube.com/@neelnanda2469