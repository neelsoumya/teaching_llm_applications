# Resource accounting

- [Stanford CS336 course Lecture 2](https://youtu.be/kuYAsz7zspQ?si=TojWKWN3ORBHeYdF)

- [Lecture notes](https://cs336.stanford.edu/lectures/?trace=lecture_02_recording)

- float 32 representation

- Python code

- [🎮 Practicals](../practicals/practicals_resouce_accounting.py)

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

- [float 16](https://cs336.stanford.edu/lectures/?trace=lecture_02_recording#:~:text=122-,%5BWikipedia%5D,-Wikipedia)

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