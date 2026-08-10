import torch

# Common dimensions
batch_size = 1
seq_len = 128
num_q_heads = 8
head_dim = 64

print("=== 1. MULTI-HEAD ATTENTION (MHA) ===")
num_kv_heads_mha = 8  # 1:1 ratio (8 Q heads, 8 KV heads)

q_mha = torch.randn(batch_size, num_q_heads, seq_len, head_dim)
k_mha = torch.randn(batch_size, num_kv_heads_mha, seq_len, head_dim)

print(f"Q shape: {list(q_mha.shape)}")
print(f"K shape: {list(k_mha.shape)} <-- Full memory footprint (8 heads)\n")

print("=== 2. GROUPED-QUERY ATTENTION (GQA) ===")
num_kv_heads_gqa = 2  # 4:1 ratio (8 Q heads share 2 KV heads)
group_size = num_q_heads // num_kv_heads_gqa

q_gqa = torch.randn(batch_size, num_q_heads, seq_len, head_dim)
k_gqa = torch.randn(batch_size, num_kv_heads_gqa, seq_len, head_dim)

print(f"Q shape:         {list(q_gqa.shape)}")
print(f"K shape (Cache): {list(k_gqa.shape)} <-- 75% smaller memory footprint!")

# Expand Key heads to match Query heads right before matrix multiplication
k_gqa_expanded = k_gqa.repeat_interleave(group_size, dim=1)
print(f"K shape (Math):  {list(k_gqa_expanded.shape)} <-- Expanded for dot-product")