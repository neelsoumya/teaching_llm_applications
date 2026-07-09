## Practicals

- [Francois course](https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer/)

- [CS336 Stanford coursework](https://cs336.stanford.edu/#coursework)

- [`nanoGPT`](https://github.com/karpathy/nanogpt)

- Google Colab and restrict to only a few open-source models

- Huggingface spaces


## Installation

```python
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## Quickstart (Week 1)

- Week 1

```python
python data/shakespeare_char/prepare.py
```


### Scaling laws using `nanoGPT`

- [Chinchilla paper](https://arxiv.org/pdf/2203.15556.pdf)

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