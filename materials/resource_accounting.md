# Resource accounting

- [Stanford CS336 course Lecture 2](https://youtu.be/kuYAsz7zspQ?si=TojWKWN3ORBHeYdF)

- [Lecture notes](https://cs336.stanford.edu/lectures/?trace=lecture_02_recording)

- float 32 representation

- Python code

- [🎮 Practicals](../practicals/practicals_resouce_accounting.py) and [here](https://cs336.stanford.edu/lectures/?trace=lecture_02_recording)

```python
    import torch
    x = torch.zeros(4, 8)  
    assert x.dtype == torch.float32  # Default type
    assert x.numel() == 4 * 8
    assert x.element_size() == 4  # Float is 4 bytes
    assert get_memory_usage(x) == 4 * 8 * 4  # 128 bytes
    # One matrix in the feedforward layer of GPT-3:
    assert get_memory_usage(torch.empty(12288 * 4, 12288)) == 2304 * 1024 * 1024  # 2.3 GB 

```

- reduce storage [quantization](quantization.md)

- [float 16 practicals](https://cs336.stanford.edu/lectures/?trace=lecture_02_recording#:~:text=122-,%5BWikipedia%5D,-Wikipedia)

```python
x = torch.zeros(4, 8, dtype=torch.float16)  
    assert x.element_size() == 2
    However, the dynamic range (especially for small numbers) isn't great.
    x = torch.tensor([1e-8], dtype=torch.float16)  
    assert x == 0  # Underflow!
    
```

- If this happens when you train, you can get instability.

- bfloat16

- fp8

- training with fp8, float16 and bfloat16 can lead to instabilities

- automatic mixed precision `AMP`

- cast into _bf16_ (safe for `matmul` not `exp`)

- _Concept_ 🧩 🚀 Train on high precision (bf16) and then quantize

- 🤔 Is quantization at training time or inference time?

- move tensors to GPUs

- [einops tutorial](https://einops.rocks/1-einops-basics/)

- 💡 generalized matrix multiplication where dimensions are named

```python
def einops_motivation():
    #Easy to mess up the dimensions (what is -2, -1?)...
    Traditional PyTorch code:
    x = torch.ones(2, 2, 3)      # batch seq hidden  
    y = torch.ones(2, 2, 3)      # batch seq hidden  
    z = x @ y.transpose(-2, -1)  # batch seq seq  
    #Easy to mess up the dimensions (what is -2, -1?)...


def einops_einsum():
    #Einsum is generalized matrix multiplication with good bookkeeping.
    x = torch.ones(3, 4)  # seq1 hidden 
    y = torch.ones(4, 3)  # hidden seq2 
    # Old way
    z = x @ y   # seq1 seq2  
    # New (einops) way
    z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")    

    # Let's try a more complex example...
    x = torch.ones(2, 3, 4)  # batch seq1 hidden 
    y = torch.ones(2, 3, 4)  # batch seq2 hidden 
    # Old way
    z = x @ y.transpose(-2, -1)  # batch seq1 seq2  
    # New (einops) way
    z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")  
    # Dimensions that are not named in the output are summed over.  
```

- `rearrange`

- floating point operation (FLOP)

- `x + y` is 1 FLOP

- $x + y + z$ is 2 FLOPs

- $x @ y$ is $mnp$ FLOPs for matrices of size $(m, n)$ and $(n, p)$

- Training `GPT-3` took 3.14e23 FLOPS

- Question: How long would it take to train a 70B parameter model on 15T tokens on 1024 B100s?


```python
    total_flops = 6 * 70e9 * 15e12  
    h100_flop_per_sec = 1979e12 / 2
    mfu = 0.5
    flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24  
    days = total_flops / flops_per_day  
```

- matrix multiplication is a very costly operation in AI/ML

- `B`: number of points, `D`: dimension, `K`: number of outputs

- `FLOPS = 2 * B * D * K`

- element wise operation on a matrix M x N = `O(MN)` FLOPS

- MFU: Model FLOPs Utilization - how close to the theoretical maximum are we?

- FLOPS for forward pass: `2 * num_parameters * num_tokens`

- Memory bound vs. Compute bound:

    >The theoretical maximum number of floating point operations per second of a processor is its theoretical maximum speed. For example, an NVIDIA H100 has a theoretical maximum speed of about 2000 teraflops (that is $2 \times 10^{15}$ flop/s), where a flop may be an addition, multiplication, division, etc.   
    
    > For matrix multiplication of two matrices of shape $M \times N$ and $N \times P$, we perform $M \times N \times P$ multiplications and $M \times N \times P - M \times P$ additions. This is approximately $2MNP$ flops.   
    - 

- `GeLU` is computationally more expensive than `ReLU`, in `GPT` architecture we use `GeLU`