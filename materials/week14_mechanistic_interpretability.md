# Week 14 — Mechanistic Interpretability

## Lecture Overview

We can measure what LLMs *do* — benchmark scores, win rates, perplexity. But can we understand *why* they do it? **Mechanistic interpretability** (MI) is the programme of reverse-engineering the algorithms implemented by neural network weights: identifying the specific circuits, features, and computational motifs responsible for particular behaviours. This week we survey the theoretical foundations, the core toolkit, landmark findings, and the open problems that make MI one of the most active areas of AI safety and science research.

---

## 1. Motivation: Why Open the Black Box?

Standard evaluation tells us a model gets 90% on a benchmark. It does not tell us:

- Whether the model has learned the right reasoning process or a spurious shortcut.
- Which components to trust and which to distrust when deploying in high-stakes settings.
- How to predict model behaviour on genuinely out-of-distribution inputs.
- Whether safety fine-tuning has suppressed a capability or merely hidden it.

Mechanistic interpretability aims to answer these questions by working at the level of weights, activations, and circuits — not just inputs and outputs.

> "We want to be able to look at a neural network and say: *this* is what it is doing, and *this* is how it is doing it." — Chris Olah

---

## 2. The Superposition Hypothesis

### 2.1 Linear Representation of Features

A core hypothesis in MI (Elman, 1990; Olah et al., 2020; Elhage et al., 2022) is that neural networks represent *features* as directions in activation space:

```
activation(x) ≈ ∑_i f_i(x) · d_i
```

where f_i(x) is the presence/absence (or degree) of feature i in input x, and d_i is the corresponding direction vector in R^d.

When features are **orthogonal** — one feature per dimension — the model is said to use a *privileged basis*. This makes features easy to read off individual neurons.

### 2.2 Superposition

The number of meaningful features a model needs to represent (concepts, facts, syntactic roles) is far larger than the embedding dimension d. Elhage et al. (2022) show that models can represent more features than they have dimensions by storing them as **nearly-orthogonal directions** in a lower-dimensional space:

```
Number of representable features >> d_model
```

This is possible because random high-dimensional vectors are nearly orthogonal with high probability (Johnson-Lindenstrauss).

**Consequence**: individual neurons are *not* monosemantic — they respond to many unrelated features. A single neuron in GPT-2 may activate for `[banana, the colour yellow, curved objects, certain Unicode characters]` simultaneously.

### 2.3 Why Superposition Makes Interpretability Hard

- You cannot read off features by inspecting individual neurons.
- Ablating a neuron removes contributions from multiple unrelated features.
- The model may represent safety-relevant features in superposition with benign ones.

---

## 3. Sparse Autoencoders (SAEs)

If features are directions in superposition, we need a method to *disentangle* them. **Sparse autoencoders** (Cunningham et al., 2023; Bricken et al., 2023) are the current leading technique.

### 3.1 Architecture

An SAE learns an overcomplete dictionary of feature directions from model activations:

```
Encoder:  z = ReLU(W_enc (x - b_pre) + b_enc)    z ∈ R^M, M >> d
Decoder:  x̂ = W_dec z + b_pre
```

where x ∈ R^d is an activation vector (e.g. the residual stream at a particular layer and token position), z is a sparse feature activation vector, and M is the dictionary size (typically 4×–64× larger than d).

### 3.2 Training Objective

```
L = ||x - x̂||² + λ ||z||₁
```

The reconstruction loss ensures z faithfully encodes x. The L1 sparsity penalty encourages each activation to be explained by a small number of features.

### 3.3 What SAEs Find

After training on millions of activations from a real model, the learned feature directions tend to be **monosemantic** — each dictionary element fires on a coherent, human-interpretable concept:

- Feature 2318: fires on tokens following `def` in Python code
- Feature 7741: fires on words related to royalty (`king`, `queen`, `throne`, `crown`)
- Feature 1042: fires on negation words across many languages

Anthropic's work on Claude (Templeton et al., 2024) found millions of interpretable features in a frontier model, including a feature for the `Assistant` token that activates the "Assistant" role concept.

### 3.4 Limitations

- SAE features are not guaranteed to be causally relevant — they may be epiphenomenal.
- Training is expensive; a separate SAE is needed per layer and per model.
- Interpretation of features still requires manual inspection or automated labelling.
- Superposition may not be fully resolved: some features remain polysemantic.

---

## 4. Circuits

A **circuit** is a subgraph of the model's computational graph — a set of attention heads, MLP neurons, and the weights connecting them — that implements a specific algorithm.

### 4.1 The Indirect Object Identification (IOI) Circuit

Wang et al. (2022) study the sentence:

```
"When Mary and John went to the store, John gave a drink to ___"
```

The correct completion is `Mary`. They identify a ~26-head circuit in GPT-2 Medium that:

1. **Duplicate token heads** (layers 0–3): identify that `John` appears twice.
2. **S-inhibition heads** (layers 7–8): suppress the repeated subject (John) from being predicted.
3. **Name mover heads** (layers 9–10): copy the remaining name (Mary) to the output position.

The circuit was identified by:
- **Activation patching**: replace activations from a clean run with those from a corrupted run and measure the effect on the output.
- **Path patching**: isolate specific information paths between components.

### 4.2 The Induction Circuit

Olah et al. (2022) identify a two-head circuit in small transformers that implements **in-context copying**:

```
Prompt: [A][B] ... [A] → predict [B]
```

- **Previous token head** (layer 0): attends to the previous token; writes a "shifted key" signal.
- **Induction head** (layer 1): attends to tokens whose shifted key matches the current token; copies their successor.

Induction heads appear in virtually every transformer trained on natural language and are foundational for in-context learning.

### 4.3 Docstring Circuit, Greater-Than Circuit

Other well-studied circuits:
- **Docstring circuit** (Heimersheim & Janiak, 2023): how GPT-2 copies argument names from a Python function signature into its docstring.
- **Greater-than circuit** (Hanna et al., 2023): how a transformer compares two years and outputs the larger one, using specific attention heads as comparators.

---

## 5. Activation Patching and Causal Scrubbing

### 5.1 Activation Patching

To test whether component C is responsible for a behaviour:

1. Run the model on a *clean* input (correct behaviour) and a *corrupted* input (incorrect behaviour).
2. Replace the activation at C during the corrupted run with the activation from the clean run.
3. Measure the change in output logit difference.

If patching C restores the correct behaviour, C carries the relevant information for this task.

```python
# Conceptual PyTorch pseudocode
def activation_patch(model, clean_input, corrupted_input, hook_point):
    clean_acts   = {}
    corrupt_acts = {}

    # Collect clean activations
    def save_clean(act, hook):
        clean_acts[hook_point] = act.clone()
        return act

    # Run corrupted forward, replacing one activation
    def patch_activation(act, hook):
        act[:] = clean_acts[hook_point]
        return act

    model.run_with_hooks(clean_input,    fwd_hooks=[(hook_point, save_clean)])
    logits = model.run_with_hooks(corrupted_input, fwd_hooks=[(hook_point, patch_activation)])
    return logits
```

### 5.2 Causal Scrubbing

Chan et al. (2022) formalise circuit analysis as a *causal* question: does the hypothesised circuit account for the full causal structure of the model's behaviour on a task? Causal scrubbing replaces all activations *outside* the hypothesised circuit with random noise and checks whether the hypothesis is sufficient to explain the behaviour.

### 5.3 Attention Pattern Analysis

Attention weights show which tokens a head attends to, but not what information is moved. More informative: the **attention × value** product — which token's value vectors are actually written to the residual stream.

---

## 6. The Residual Stream as a Communication Bus

A key architectural insight (Elhage et al., 2021): the transformer's residual stream is a *shared communication channel*.

```
x^0       ← token embedding + positional encoding
x^1 = x^0 + Attn_1(x^0) + MLP_1(x^0)
x^2 = x^1 + Attn_2(x^1) + MLP_2(x^1)
...
x^L       → unembed → logits
```

Each layer *reads* from the residual stream (via QKV projections and MLP input) and *writes* back to it (via attention output and MLP output). Layers do not communicate through any other channel.

This means:
- **Composition**: head B in layer l+1 can only use information written to the residual stream by heads/MLPs in layers ≤ l.
- **Decomposition**: the final logits are the sum of contributions from every layer and every attention head. These contributions can be computed individually (logit lens, direct logit attribution).

### 6.1 The Logit Lens

Nostalgebraist (2020): unembed the residual stream after each layer to see what the model "would predict" at intermediate stages of processing.

```python
# For each layer l, project x^l into vocabulary space
intermediate_logits = model.unembed(model.ln_f(x_l))
intermediate_probs  = torch.softmax(intermediate_logits, dim=-1)
```

Typically: early layers predict syntactically plausible but semantically wrong tokens; middle layers converge on the right semantic class; final layers refine.

---

## 7. MLP Layers as Key-Value Memories

Geva et al. (2021) show that MLP layers act as key-value memories:

```
MLP(x) = W_out · ReLU(W_in · x)
```

- Each row of W_in is a **key**: it has high dot product with inputs that correspond to a specific factual association.
- The corresponding column of W_out is a **value**: it promotes certain output tokens when the key fires.

Example: the key for `"The Eiffel Tower is located in"` fires on inputs mentioning the Eiffel Tower, and the value promotes `Paris`.

This framing explains:
- Why factual knowledge is distributed across many MLP layers.
- Why fine-tuning on new facts can cause forgetting (overwriting existing key-value pairs).
- Why knowledge editing (ROME, MEMIT) targets MLP weights specifically.

---

## 8. Knowledge Editing

If MLP layers store factual associations as key-value pairs, we can *surgically edit* a specific fact without retraining the full model.

### 8.1 ROME (Rank-One Model Editing)

Meng et al. (2022): given a fact to change (e.g. `"Eiffel Tower → Rome"` instead of Paris), compute the MLP weight update as a rank-one edit:

```
W_new = W_old + Δ
Δ = (v* - W_old k) / (k^T C^{-1} k) · (C^{-1} k)^T
```

where k is the key vector for the subject, v* is the desired value vector, and C is the covariance of keys (precomputed).

ROME can change a single fact while leaving other knowledge intact — but edits can have unexpected side effects (ripple effects on related facts).

### 8.2 MEMIT

Meng et al. (2023): extends ROME to edit thousands of facts simultaneously by distributing the update across multiple MLP layers.

---

## 9. Probing

A **linear probe** tests whether a concept is linearly represented in a model's activations:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Collect activations and labels
X = np.stack([get_activation(model, text, layer=8) for text in dataset])
y = np.array([label for _, label in dataset])

probe = LogisticRegression(max_iter=1000)
probe.fit(X_train, y_train)
print(f"Probe accuracy: {probe.score(X_test, y_test):.3f}")
```

High probe accuracy means the concept is **linearly decodable** from the activations — it is represented as a direction in activation space.

### 9.1 Limitations of Probing

- High probe accuracy does not mean the model *uses* the representation for the task (correlation ≠ causation).
- A probe trained on layer l's activations may be picking up information that layer l merely *passes through*, not information it computed.
- Causal intervention (patching) is needed to confirm the representation is used.

---

## 10. Automated Interpretability

Manually interpreting thousands of neurons or SAE features is infeasible. **Automated interpretability** uses LLMs to label features:

### 10.1 Automated Explanation (Bills et al., 2023 — OpenAI)

1. Collect the top-activating tokens for a neuron.
2. Ask GPT-4 to generate a natural language explanation.
3. Ask GPT-4 to simulate what the neuron *would* activate on, given the explanation.
4. Measure correlation between simulated and actual activations.

### 10.2 Automated Circuit Discovery — ACDC

Conmy et al. (2023): **Automated Circuit DisCovery** — systematically patches each edge in the computational graph and retains only those whose removal degrades task performance above a threshold. Recovers circuits matching manually-identified ones (e.g. IOI) automatically.

---

## 11. Toy Models: Grokking

Grokking (Power et al., 2022) is a phenomenon where a model trained on modular arithmetic:

```
(a + b) mod p → result
```

initially memorises training examples (train accuracy = 100%, val accuracy = ~chance), then *suddenly* generalises (val accuracy jumps to 100%) after many more training steps — even though train loss was already near zero.

MI analysis (Nanda et al., 2023) reveals the mechanism:
1. Before grokking: the model uses a lookup table (memorisation).
2. After grokking: the model implements the **Fourier multiplication algorithm** — it represents numbers as Fourier modes and uses cosine identities to compute modular addition.
3. The transition occurs when the generalising circuit becomes more efficient (lower weight norm) than the memorising one under L2 regularisation.

This is a landmark result: the first complete reverse-engineering of a non-trivial algorithm learned by a neural network.

---

## 12. Universality

A striking finding across MI research: **the same circuits and features appear in different models trained independently** (Olah et al., 2020).

Examples:
- Curve detectors appear in every vision model.
- Induction heads appear in every language model trained on natural language.
- Multimodal neurons (responding to both images and words of the same concept) appear in both CLIP and GPT-4V.

If universality holds broadly, it suggests that neural networks converge on the *same* computational solutions to the same problems — making circuits transferable knowledge about how LLMs work in general.

---

## 13. Limitations and Open Problems

| Problem | Description |
|---------|-------------|
| Scalability | Most circuit analysis has been done on GPT-2 (117M). Does it generalise to 70B+ models? |
| Faithfulness | Are identified circuits faithful accounts of model behaviour, or post-hoc rationalisations? |
| Completeness | Circuits identified so far explain a tiny fraction of model behaviour. |
| Superposition depth | SAEs partially resolve superposition but many features remain polysemantic. |
| Causal completeness | Activation patching identifies *where* information is, not *why* it is computed. |
| Dynamic circuits | The same circuit may implement different algorithms on different inputs. |
| Emergent phenomena | It is unclear whether emergent capabilities have identifiable circuits or are globally distributed. |

---

## 14. Tools and Libraries

| Tool | Purpose |
|------|---------|
| [TransformerLens](https://github.com/neelnanda-io/TransformerLens) | Mechanistic interpretability toolkit for GPT-style models; hooks, caching, patching |
| [SAELens](https://github.com/jbloomAUS/SAELens) | Training and evaluating sparse autoencoders |
| [Neuronpedia](https://www.neuronpedia.org/) | Interactive browser for SAE features across models |
| [CircuitsVis](https://github.com/alan-cooney/CircuitsVis) | Visualising attention patterns and circuits |
| [baukit](https://github.com/davidbau/baukit) | General neural network probing and editing toolkit |
| [pyvene](https://github.com/stanfordnlp/pyvene) | Causal intervention library |

---

## 15. Practical This Week

See `practicals/week14_practical.py`:
- Load GPT-2 with TransformerLens and inspect the residual stream decomposition.
- Implement the logit lens across layers on a factual completion task.
- Run activation patching to identify which attention heads carry subject information in an IOI-style sentence.
- Train a linear probe to detect a syntactic property (e.g. whether the current token is a verb) from intermediate layer activations.
- Train a small sparse autoencoder on GPT-2 MLP activations and visualise the top-activating tokens for five learned features.

---

## 16. Further Reading

- Elhage et al. (2021) — "A Mathematical Framework for Transformer Circuits" — https://transformer-circuits.pub/2021/framework/index.html
- Elhage et al. (2022) — "Toy Models of Superposition" — https://transformer-circuits.pub/2022/toy_model/index.html
- Wang et al. (2022) — "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2" — https://arxiv.org/abs/2211.00593
- Nanda et al. (2023) — "Progress measures for grokking via mechanistic interpretability" — https://arxiv.org/abs/2301.05217
- Cunningham et al. (2023) — "Sparse Autoencoders Find Highly Interpretable Features in Language Models" — https://arxiv.org/abs/2309.08600
- Templeton et al. (2024) — "Scaling and Evaluating Sparse Autoencoders" (Anthropic) — https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- Bricken et al. (2023) — "Towards Monosemanticity" (Anthropic) — https://transformer-circuits.pub/2023/monosemantic-features/index.html
- Meng et al. (2022) — "Locating and Editing Factual Associations in GPT" (ROME) — https://arxiv.org/abs/2202.05262
- Geva et al. (2021) — "Transformer Feed-Forward Layers Are Key-Value Memories" — https://arxiv.org/abs/2012.14913
- Conmy et al. (2023) — "Towards Automated Circuit Discovery for Mechanistic Interpretability" — https://arxiv.org/abs/2304.14997
- Power et al. (2022) — "Grokking: Generalisation Beyond Overfitting" — https://arxiv.org/abs/2201.02177
- [TransformerLens documentation](https://neelnanda-io.github.io/TransformerLens/)
- [Anthropic Transformer Circuits thread](https://transformer-circuits.pub/)

---

## Discussion Questions

1. What is superposition and why does it make mechanistic interpretability difficult? How do sparse autoencoders attempt to address it?
2. Describe the induction circuit. What two heads compose to implement in-context copying, and what does each head contribute?
3. Explain the difference between activation patching and linear probing as tools for understanding model internals. What can each tell you, and what can each *not* tell you?
4. Grokking appears to be a transition from memorisation to a generalisable algorithm. What does the mechanistic analysis of modular arithmetic reveal about *when* and *why* this transition occurs?
5. A safety team wants to verify that a harmful capability has been removed by fine-tuning. How would you use mechanistic interpretability tools to check whether the capability is truly absent or merely suppressed?
