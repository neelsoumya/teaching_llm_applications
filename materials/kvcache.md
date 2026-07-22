# K-V Cache

- [K-V Cache](https://www.youtube.com/post/Ugkxkku-e-0CMebE9yn0sKWPAyBr6q73T4Dv)

![image](../images/kvcache.png)

- Paged caches and sliding windows

- [Why longer chats take more time, relation to K-V cache](https://www.youtube.com/shorts/UjbbwDnqpMw)

- The K-V cache grows and a new word has to be `matched` or attention computed with each K-V cache entry

-                     KV CACHE
        ┌───────────────────────────────┐
        │ K1 V1 │ K2 V2 │ K3 V3 │ ... │ Kt Vt │
        └───────────────────────────────┘
             ↑       ↑       ↑          ↑
             │       │       │          │
             └───────┴───────┴──────────┘
                         │
                         │ attention
                         ▼
                      Q(t+1)
                         │
                         ▼
                   next token


- _Concept_ 🧩 🚀 The KV cache saves computation by avoiding recomputation of the past, but it does not make the past disappear.

-           PREFILL
    ┌───────────────────┐
    │ The cat sat on... │
    └───────────────────┘
             │
             ▼
       KV Cache built
             │
             ▼
       DECODE PHASE

        token 1
           ↓
        token 2
           ↓
        token 3
           ↓
        token 4
           ↓
          ...

    KV cache grows →


_Concept_ 🧩 🚀

KV cache:

Memory       → O(T)

Attention per new token:

Computation  → O(T)

Total decode attention:

Computation  → O(T²)



- Python code to show K-V cache grows with tokens in context

```python
import numpy as np
import matplotlib.pyplot as plt

# Context lengths
tokens = np.arange(0, 100_001, 1_000)

# Assume one KV entry per token
kv_size = tokens

plt.figure(figsize=(8, 5))
plt.plot(tokens, kv_size)

plt.xlabel("Number of tokens in context")
plt.ylabel("Relative KV cache size")
plt.title("KV Cache Size Grows Linearly with Context Length")
plt.grid(True)

plt.show()
```

- ![image](../images/kvcache_growth.png)

