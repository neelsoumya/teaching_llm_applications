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


## Practical
- Implement an LLM-based application for low resource scenarios
    
