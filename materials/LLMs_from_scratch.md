# LLMs from scratch

- Lecture and practical on LLMs from scratch

- [LLMs from scratch Vizuara]

- [LLMs from scratch Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)

- [LLMs from scratch NanoGPT](https://www.youtube.com/watch?v=qra052AchPE)

- [The busy person's intro to LLMs by Andrej Karpathy](https://www.youtube.com/watch?v=zjkBMFhNj_g)

- [LLM from scratch](https://www.gilesthomas.com/2025/10/llm-from-scratch-22-finally-training-our-llm) and [here](https://dev.to/theirritainer/this-dev-built-his-own-llm-from-scratch-1i62)

- [LLMs NanoGPT by Andrej Karpathy](http://karpathy.github.io/2026/02/12/microgpt/)

- [code](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)



```python
"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
print(f"num params: {len(params)}")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer

# Repeat in sequence
num_steps = 1000 # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.

    # Backward the loss, calculating the gradients with respect to all model parameters
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

- Explanation line by line

# Karpathy MicroGPT: Line-by-Line Explanation

Here is a simple, line-by-line explanation of what the code is doing. I will keep the language plain and focus on the idea behind each line.

## What this program is

This is a tiny GPT-like language model written in pure Python.
It:

1. loads a list of text examples,
2. turns characters into tokens,
3. builds a very small transformer,
4. trains it with backpropagation,
5. then generates new text.

It is “micro” because it strips away all the usual libraries and writes almost everything manually.

---

## Header comment

```python
"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""
```

This is just a description of the code.
It says the file contains the full algorithm for training and using a GPT, with no external ML libraries.

---

## Imports and random seed

```python
import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos
```

* `os` is used to check whether a file exists.
* `math` gives access to basic math functions like logarithm and exponential.
* `random` is used for shuffling data, sampling text, and making random weights.

`random.seed(42)` makes the randomness repeatable.
That means you can run the code again and get the same results.

---

## Load the dataset

```python
# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
```

What this does:

* If `input.txt` is not already there, it downloads a file of names and saves it as `input.txt`.
* `docs` becomes a list of lines from that file.
* `line.strip()` removes extra spaces and newline characters.
* `if line.strip()` ignores empty lines.
* `random.shuffle(docs)` mixes the order of the names.
* It prints how many documents there are.

So here, each “document” is actually just a name.

---

## Build the tokenizer

```python
# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")
```

This builds a character-level vocabulary.

* `''.join(docs)` combines all names into one long string.
* `set(...)` keeps only unique characters.
* `sorted(...)` puts them in a stable order.
* `uchars` is the list of all unique characters in the dataset.
* Each character will later be represented by an integer index.

`BOS` means “Beginning of Sequence”.
It is a special token used to mark the start and end of a name.

`vocab_size` is the total number of tokens:

* all characters,
* plus one extra token for `BOS`.

---

## Manual autograd engine

```python
# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage
```

This defines a custom class called `Value`.

Each `Value` object will store:

* `data`: the numeric value,
* `grad`: the gradient,
* `_children`: the earlier values used to create it,
* `_local_grads`: how much this value depends on each child.

`__slots__` is a Python memory optimization.
It says only these four attributes are allowed.

---

### Constructor

```python
    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children
```

When a `Value` is created:

* `data` stores the actual number.
* `grad` starts at 0.
* `children` are the inputs that produced this value.
* `local_grads` stores the derivative information needed for backpropagation.

This is the core of the tiny autodiff system.

---

### Addition

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
```

This lets you write `a + b` with `Value` objects.

* If `other` is just a normal number, it wraps it in `Value`.
* The result is a new `Value` containing the sum.
* The local gradient of addition is 1 for both inputs.

So if `z = x + y`, then:

* `dz/dx = 1`
* `dz/dy = 1`

---

### Multiplication

```python
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
```

This defines multiplication.

If `z = x * y`, then:

* derivative with respect to `x` is `y`
* derivative with respect to `y` is `x`

That is why the local grads are `(other.data, self.data)`.

---

### Power, log, exp, relu

```python
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
```

These define more mathematical operations.

* `x ** other` raises `x` to a power.
* `log()` gives the natural logarithm.
* `exp()` gives the exponential.
* `relu()` returns `x` if `x > 0`, otherwise 0.

Each one also stores the derivative needed for backpropagation.

For example:

* `d/dx log(x) = 1/x`
* `d/dx exp(x) = exp(x)`
* `d/dx relu(x)` is 1 if `x > 0`, else 0

---

### Negation and operator shortcuts

```python
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
```

These make Python arithmetic work naturally.

* `-x` becomes multiplication by `-1`
* `other + self` works even if the number is on the left
* subtraction is built from addition and negation
* division is built from multiplication and inverse

This lets the rest of the code look natural, even though everything is being tracked for gradients.

---

### Backpropagation

```python
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad
```

This is the backward pass.

What it does:

1. It finds all nodes in the computation graph.
2. It sorts them in topological order.
3. It sets the gradient of the final output to 1.
4. It walks backward through the graph and sends gradients to earlier nodes.

The main idea is the chain rule.

If a value depends on another value, gradients are passed along that dependency.

This is what lets the model learn.

---

## Model size and parameters

```python
# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
```

These set the architecture size:

* `n_layer = 1`: one transformer block
* `n_embd = 16`: each token becomes a 16-dimensional vector
* `block_size = 16`: the model looks at up to 16 positions
* `n_head = 4`: attention is split into 4 heads
* `head_dim = n_embd // n_head`: each head gets 4 dimensions

This is a tiny model.

---

### Weight initialization helper

```python
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
```

This creates a matrix filled with random numbers.

* `nout` = number of rows
* `nin` = number of columns
* `random.gauss(0, std)` draws small random values from a normal distribution

These are the initial weights of the neural network.

---

### Store all parameters in a dictionary

```python
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
```

This creates the main parameter matrices:

* `wte`: word/token embeddings
* `wpe`: position embeddings
* `lm_head`: final output layer

Each is a matrix of `Value` objects.

---

### Add transformer layer weights

```python
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
```

For each transformer layer, it creates:

* `attn_wq`: query weights
* `attn_wk`: key weights
* `attn_wv`: value weights
* `attn_wo`: output projection after attention
* `mlp_fc1`: first MLP layer
* `mlp_fc2`: second MLP layer

These are the learned matrices that make the model work.

---

### Flatten all parameters into one list

```python
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
print(f"num params: {len(params)}")
```

This converts every matrix into one long list of parameters.

Why?

Because later, the optimizer will update all parameters one by one.

---

## Core neural network functions

```python
# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
```

This says the next functions define the GPT computation.

It is loosely based on GPT-2, but simplified.

---

### Linear layer

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

This applies a matrix multiplication.

* `x` is an input vector
* `w` is a matrix of weights
* each row of `w` gives one output neuron

So this computes a standard fully connected layer.

---

### Softmax

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

Softmax turns raw scores into probabilities.

Steps:

1. Find the maximum value for numerical stability.
2. Subtract it from each logit.
3. Exponentiate.
4. Divide by the sum.

The result is a probability distribution that adds up to 1.

---

### RMSNorm

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

This normalizes the vector.

* `ms` is the mean square of the values
* `scale` is roughly `1 / sqrt(ms)`
* each element is multiplied by this scale

This keeps activations from blowing up.

It is a simpler version of layer normalization.

---

## GPT forward pass

```python
def gpt(token_id, pos_id, keys, values):
```

This function computes the model’s output logits for one token at one position.

Inputs:

* `token_id`: the current character token
* `pos_id`: the current position in the sequence
* `keys`, `values`: caches used for attention

---

### Token and position embeddings

```python
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection
```

This converts the token ID into a vector and adds position information.

* `tok_emb` gives the embedding for the character
* `pos_emb` gives the embedding for the position
* they are added together

Then `rmsnorm` normalizes the result.

So now `x` is the model’s current hidden representation.

---

## Transformer layers

```python
    for li in range(n_layer):
```

Loop over each transformer layer.
Here there is only 1 layer, but the code is written generally.

---

### Attention block start

```python
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
```

* Save the old `x` as a residual connection.
* Normalize `x` before attention.

Residual connections help training.

---

### Create queries, keys, and values

```python
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
```

These are the standard attention vectors:

* `q`: query
* `k`: key
* `v`: value

They are different learned projections of the same input vector.

---

### Save keys and values for this position

```python
        keys[li].append(k)
        values[li].append(v)
```

This stores the current token’s key and value so future positions can attend to it.

That is how autoregressive attention works: later tokens can look back at earlier tokens.

---

### Prepare attention output

```python
        x_attn = []
```

This will collect the output from all heads.

---

### Multi-head attention loop

```python
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
```

This splits the vectors into heads.

For each head:

* `hs` is the starting index
* `q_h` is the query slice for this head
* `k_h` is the list of all past keys for this head
* `v_h` is the list of all past values for this head

So each head looks at a different slice of the hidden vector.

---

### Compute attention scores

```python
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
```

This measures how much the current token should pay attention to each earlier token.

For every past token `t`:

* take the dot product between query and key
* divide by `sqrt(head_dim)` to keep values stable

These are attention scores before softmax.

---

### Turn scores into weights

```python
            attn_weights = softmax(attn_logits)
```

Now the scores become probabilities.

Higher score means more attention.

---

### Weighted sum of values

```python
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
```

This combines the values using the attention weights.

In simple words:

* each previous token contributes some amount
* more relevant tokens contribute more
* the result is the output of one attention head

---

### Collect head outputs

```python
            x_attn.extend(head_out)
```

This appends the head’s result to the full attention output vector.

After all heads, `x_attn` contains all head outputs concatenated together.

---

### Output projection and residual connection

```python
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
```

* `linear(...)` mixes the heads together.
* Then the result is added back to the residual input.

That residual addition helps information flow through the network.

---

## MLP block

```python
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
```

This is the feed-forward part of the transformer.

Steps:

1. Save residual input.
2. Normalize.
3. Project to a larger hidden size.
4. Apply ReLU nonlinearity.
5. Project back down.
6. Add residual connection again.

The MLP helps the model transform features, not just mix context.

---

### Final output layer

```python
    logits = linear(x, state_dict['lm_head'])
    return logits
```

This maps the hidden vector to a score for every token in the vocabulary.

These scores are called `logits`.

The model returns them so the next character can be predicted.

---

## Adam optimizer setup

```python
# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer
```

This sets up Adam.

* `learning_rate`: how big each update step is
* `beta1`: momentum decay
* `beta2`: squared-gradient decay
* `eps_adam`: tiny number to avoid division by zero

`m` stores the moving average of gradients.
`v` stores the moving average of squared gradients.

---

## Training loop

```python
# Repeat in sequence
num_steps = 1000 # number of training steps
for step in range(num_steps):
```

Train for 1000 steps.

Each step uses one document and updates the weights.

---

### Pick a document and tokenize it

```python
    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
```

* Choose a document in sequence, looping around with `%`.
* Convert each character into an integer token.
* Add `BOS` at the start and end.
* `n` is the number of prediction steps to train on, capped by `block_size`.

So for a name like `"anna"`, tokens might look like:

`[BOS, 'a', 'n', 'n', 'a', BOS]`

---

### Build the computation graph for the whole sequence

```python
    # Forward the token sequence through the model, building up the computation graph all the way to the loss
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
```

* `keys` and `values` are reset for this new document.
* `losses` will store one loss per predicted token.

---

### Predict each next character

```python
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
```

For each position:

* `token_id` is the current input character.
* `target_id` is the correct next character.
* `gpt(...)` produces logits.
* `softmax(...)` turns logits into probabilities.
* `loss_t` is the negative log probability of the correct token.

This is cross-entropy loss, written manually.

Lower loss means better prediction.

---

### Average the losses

```python
    loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.
```

This averages the loss over all positions in the sequence.

The model is trained to minimize this value.

---

### Backpropagate

```python
    # Backward the loss, calculating the gradients with respect to all model parameters
    loss.backward()
```

This computes gradients for every parameter in the model.

These gradients tell the optimizer how to change the weights.

---

## Adam update

```python
    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
```

The learning rate slowly decreases over time.

This is called learning-rate decay.

---

### Update every parameter

```python
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0
```

For each parameter:

* update `m` with the gradient average
* update `v` with the squared gradient average
* compute bias-corrected versions `m_hat` and `v_hat`
* adjust the parameter in the opposite direction of the gradient
* reset gradient to zero for the next step

This is the Adam optimization algorithm.

---

### Print training progress

```python
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')
```

This shows training progress on one line.

It prints:

* current step
* total steps
* current loss

`end='\r'` keeps overwriting the same line.

---

## Inference / text generation

```python
# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
```

Now the model is used to generate new names.

* `temperature` controls randomness.
* Lower temperature = safer, more predictable output.
* Higher temperature = more random output.

---

### Generate 20 samples

```python
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
```

For each sample:

* reset attention caches
* start with the `BOS` token
* create an empty list for generated characters

---

### Generate one token at a time

```python
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

At each position:

1. Run the GPT.
2. Divide logits by temperature.
3. Convert to probabilities.
4. Sample one token from the distribution.

This does not always pick the highest-probability token.
It samples, so output varies.

---

### Stop at end token

```python
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
```

If the model predicts `BOS` again, generation stops.

Otherwise:

* convert token ID back into a character
* append it to the name

---

### Print the sample

```python
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

This prints the generated name.

`''.join(sample)` combines the characters into a string.

---


- This code trains a tiny character-level transformer from scratch, using its own manual autograd system, and then samples new names from it.

