'''
Practical for Resource Accounting
'''

# pyrefly: ignore [missing-import]
import torch

def get_memory_usage(x: torch.Tensor):
    return x.numel() * x.element_size()

x = torch.zeros(4, 8)  
assert x.dtype == torch.float32  # Default type
assert x.numel() == 4 * 8
assert x.element_size() == 4  # Float is 4 bytes
assert get_memory_usage(x) == 4 * 8 * 4  # 128 bytes
# One matrix in the feedforward layer of GPT-3:
assert get_memory_usage(torch.empty(12288 * 4, 12288)) == 2304 * 1024 * 1024  # 2.3 GB 
