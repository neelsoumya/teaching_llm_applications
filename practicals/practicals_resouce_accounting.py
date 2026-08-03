'''
Practical for Resource Accounting

Credits: Adapted from 

https://cs336.stanford.edu/lectures/?trace=lecture_02_recording

'''

# pyrefly: ignore [missing-import]
import enum
import torch
import timeit
import einops
from einops import einsum
from torch import nn
from facts import h100_flop_per_sec, h100_bytes_per_sec


# In PyTorch, the numel() method 
# returns the total number of elements in a tensor (short for "number of elements").
def get_memory_usage(x: torch.Tensor):
    ''' Get memory usage of a tensor
    '''
    # The total number of elements is multiplied by the byte size of an individual element
    return x.numel() * x.element_size()

x = torch.zeros(4, 8) # creates a tensor with 4 rows and 8 columns. 
assert x.dtype == torch.float32  # Default type
assert x.numel() == 4 * 8 # returns 32
assert x.element_size() == 4  # Float is 4 bytes
assert get_memory_usage(x) == 4 * 8 * 4  # 128 bytes
# One matrix in the feedforward layer of GPT-3:
assert get_memory_usage(torch.empty(12288 * 4, 12288)) == 2304 * 1024 * 1024  # 2.3 GB 



def time_matmul(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the number of seconds required to perform `a @ b`."""
    # Wait until previous CUDA threads are done
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    def run():
        # Perform the operation
        a @ b
        # Wait until CUDA threads are done
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    # Time the operation `num_trials` times
    num_trials = 5
    total_time = timeit.timeit(run, number=num_trials)
    return total_time / num_trials



def get_num_parameters(model: nn.Module) -> int:
    '''
    Get number of parameters
    '''
    return sum(param.numel() for param in model.parameters())


def einops_motivation():
    #Easy to mess up the dimensions (what is -2, -1?)...
    # Traditional PyTorch code:
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



def motivating_questions():
    # Question: How long would it take to train a 70B parameter model on 15T tokens on 1024 B100s?
    total_flops = 6 * 70e9 * 15e12  
    h100_flop_per_sec = 1979e12 / 2
    mfu = 0.5
    flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24  
    days = total_flops / flops_per_day  
    print("Days to train:", days)

    # Question: What's the largest model that can you can train on 8 H100s using AdamW (naively)?
    h100_bytes = 80e9  
    bytes_per_parameter = 2 + 2 + (4 + 4)  # parameters, gradients, optimizer state  
    num_parameters = (h100_bytes * 8) / bytes_per_parameter  
    print("Number of parameters:", num_parameters )
    # Caveat: activations are not accounted for (depends on batch size and sequence length).
    # This is a rough back-of-the-envelope calculation

def arithmetic_intensity_gelu():
    n = 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    y = F.gelu(x)  # GELU(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
    bytes = (2 * n) + (2 * n)  # Read x, write y (bf16 is 2 bytes/float)
    flops = 20 * n  # tanh can approximated in various ways (e.g., polynomial)
    arithmetic_intensity = flops / bytes  
    h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec  
    assert arithmetic_intensity < h100_accelerator_intensity

    # Note that GeLU does more work than ReLU per byte moved, so it has higher arithmetic intensity.
    # But still memory-bound!
    # In other words, ReLU is not faster than GeLU (when doing things in an isolated way).

def arithmetic_intensity_dot_product():
    n = 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    w = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    y = x @ w
    bytes = (2 * n) + (2 * n) + 2  # Read x, read w, write y
    flops = 2 * n - 1  # n multiplications, n-1 additions
    arithmetic_intensity = flops / bytes  # ~1/2 
    h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec  
    assert arithmetic_intensity < h100_accelerator_intensity
    # Memory-bound!

if __name__ == "__main__":
    motivating_questions()

    einops_einsum()
    
    einops_motivation()

    arithmetic_intensity_gelu()

    arithmetic_intensity_dot_product()
    