# CS 336 Stanford Language modelling from scratch

- [course](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&index=2)


## Week 1

- memory vs. compute

- DGX B200 8 GPUs connected 

- _Concept_ 🧩 🚀 GPU architecture

![image](https://github.com/stanford-cs336/lectures/blob/main/images/cpu-gpu.png)

- Kernel is a function that runs on GPU. When using PyTorch, each primitive operation launches a kernel

- ⚠️ moving data from memory is expensive

![image](https://github.com/stanford-cs336/lectures/blob/main/images/compute-memory.png)

- HBM: high bandwidth memory

- naive, fused

- operator fusion, tiling (FlashAttention)

- sharding (parameters, activations, gradients) across GPUs

- prefill phase

![image](https://github.com/stanford-cs336/lectures/blob/main/images/prefill-decode.png)


- The Prefill Phase in Large Language Models (LLMs)

When a Large Language Model (LLM) generates a response, the process (called inference) doesn't happen all at once. It is split into two distinct steps: the **Prefill Phase** and the **Decode Phase**. 

The **Prefill Phase** is the very first step where the model reads and processes your entire input prompt.

## What Happens During the Prefill Phase?

### 1. Parallel Processing of the Prompt
Unlike generating text (which happens one word at a time), the model reads your entire prompt all at once. Because the input tokens are already known, the GPU processes them in parallel. This step is highly efficient and relies heavily on the sheer computational power of the GPU (it is "compute-bound").

### 2. Building the KV Cache
As the model processes your prompt, it calculates and stores mathematical representations of every word to understand their context and relationships. It saves these calculations in what is known as the **Key-Value (KV) Cache**. This is a crucial step because it acts as the model's short-term memory for your prompt. 

### 3. Generating the First Token
The culmination of the prefill phase is the model predicting and outputting the very *first* token (word or sub-word) of its response.

---

## Contrast: The Decode Phase
Once the first token is generated, the prefill phase ends and the **Decode Phase** begins. 

* **Sequential Generation:** The model generates the rest of its response strictly one token at a time. 
* **Using the Cache:** To do this efficiently, it doesn't re-read your original prompt. Instead, it looks back at the **KV Cache** it built during the prefill phase. 
* **Speed:** Because it has to generate words one by one and constantly fetch data from the cache, the decode phase is much slower per token and is limited by how fast the GPU can move data around (it is "memory-bandwidth bound").

---

## A Simple Analogy
Imagine you are handed a piece of paper with a complex math word problem on it. 

* **The Prefill Phase:** You read the entire paragraph at once, understand the context, set up the equation in your head (building the cache), and write down the very first number of your answer. 
* **The Decode Phase:** You write down the rest of the numbers one by one to finish the equation, relying on your memory of the problem rather than re-reading the paragraph for every single digit.


## Divine Benevolence

![image](https://github.com/stanford-cs336/lectures/blob/main/images/divine-benevolence.png)