# Assessment

- Written component
- Software component/practical

## Written component

- Tradeoff between KV cache, memory management, saving sessions on Claude and restoring them from hard disk, L1 cache management, L2 cache, DRAM
- Engineering choices
- How does KV cache interact with architecture/hyperparameters? (such as optimizers, depth vs. width ). See [architecture](architecture.md)
- How does KV cache interact with software choices? 
- Distributed file systems

- also quantization

- How does this interact with the following points covered in the chapter on [GPU and Flash Attention](GPUs.md)?

- Prefill phase is memory bound
 and is one chip

- Decoding is compute bound
 and is another chip
- attention can go to one chip; MLP will go to another chip

- Come up with a new architecture where different parts of the Transformer such as attention, MLP, layer norm are placed on different chips and analyze the trade-offs in terms of performance and memory usage.

- theory exam on how many FLOPS for `ReLU` and how to speed it up

![image](../images/low_precision_RELU.png)


- can quantize activations after `ReLU`

- however more bang by quantizing `matmul`

- train a bigger model and then quantize it?


- 📝 how many cycles for computing sin^ x + cos^x ?

![image](../images/sines_cosines_cycles.png)


- backprop memory used

![image](../images/backprop_memory.png)

- _Concept_ 🧩 🚀 backpropagation intuition
![image](../images/backprop_intuition.png)

- 💡 in a world where memory is slower and compute is cheap/faster, you just recompute the activations!

![image](../images/backprop_memory_expensive.png)


- explain how you get this unexplained drop in throughput when you go from 98 tiles to 120 tiles on an A100 GPU

![image](../images/unexplained_drop.png)

- [🎥 How does ChatGPT deal with 900 million users?](https://www.youtube.com/watch?v=fVLmyuCEEy8)

- Assessment Task: Scaling LLMs to 900 Million Users

> To serve 900 million active users simultaneously on platforms like ChatGPT, Claude, or Sarvam, engineers must design an architecture capable of handling extreme, stateful concurrency. Global edge servers manage initial traffic via dynamic geo-load balancing, resolving edge routing and validating session tokens before handing off traffic to local regional clusters. Once inside the cluster, intelligent API gateways route incoming prompts to distributed databases to fetch conversational history, user profiles, and system prompts. To prevent persistent database bottlenecks during traffic spikes, localized read-only replica caches (e.g., Redis or distributed in-memory stores) serve high-frequency user metadata and system state with sub-millisecond latency.

> After state and history retrieval, the request reaches the core inference engines, where the most complex engineering challenges reside. Compute clusters rely on continuous batching and dynamic continuous sequence scheduling to maximize GPU utilization across tens of thousands of accelerator chips. Crucially, memory management uses techniques like PagedAttention to dynamically allocate and offload Key-Value (KV) cache chunks, preventing VRAM fragmentation during long context generation. By decoupling the lightweight API orchestration layer from the compute-heavy, memory-bound GPU clusters, the system maintains ultra-low latency while serving billions of daily tokens across a massive worldwide user base.


- Edge Routing & Load Balancing: Systems use global DNS/Anycast routing to direct traffic to edge servers, where dynamic load balancing, rate limiting, and SSL termination occur before handing off to regional clusters.
- Database & Read-Cache Layer: System prompts and chat histories are fetched via low-latency API gateways. High-frequency state data is served via distributed, read-only local caches (e.g., Redis) to decouple primary database read pressure from high-concurrency prompts.
- GPU Compute Optimization: Inference servers use continuous (iteration-level) batching to dynamically insert new requests into running GPU batches, maximizing FLOPS utilization across cluster nodes.
- KV Cache Management: To prevent VRAM out-of-memory errors and fragmenting, systems utilize virtual memory management (e.g., PagedAttention) to store key-value matrices in non-contiguous memory chunks, rapidly swapping or sharing context blocks across attention layers.

- Question on self attention

- Is self attention layer linear with respect to context length? Why or why not?

- Is self attention symmetric? See [here](https://theaisummer.com/self-attention/#self-attention-is-not-symmetric)

## Practical / coding component of assessment

- Implement an LLM-based application for low resource scenarios
    
- Stanford CS365 practical on GPUs, architecture choices and benchmarking metrics and reports [here](https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf)

- only use open-source models that will run on Google Colab such as `Qwen`

- [Stanford CS324 assignment](https://stanford-cs324.github.io/winter2022/projects/CS324_P1.pdf)