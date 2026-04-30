import torch
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model.eval()
sd = model.state_dict()

# config
n_head = 16
n_embd = 1024
d_head = 64

# input
dummy = torch.tensor([[464, 3290, 318, 257, 922]])  # [1, 5]
seq_len = 5

# embeddings — same as your C++
wte = sd['transformer.wte.weight']
wpe = sd['transformer.wpe.weight']
x = wte[dummy[0]] + wpe[torch.arange(seq_len)]  # [5, 1024]
print("x[:3]:", x[0, :3].tolist())

# layer norm
ln1_w = sd['transformer.h.0.ln_1.weight']
ln1_b = sd['transformer.h.0.ln_1.bias']
x_ln = torch.layer_norm(x, [n_embd], ln1_w, ln1_b)
print("after ln1[:3]:", x_ln[0, :3].tolist())


# QKV projection — same as your addmm
c_attn_w = sd['transformer.h.0.attn.c_attn.weight']  # [1024, 3072]
c_attn_b = sd['transformer.h.0.attn.c_attn.bias']    # [3072]
qkv = x_ln @ c_attn_w + c_attn_b                     # [5, 3072]

q = qkv[:, :n_embd]
k = qkv[:, n_embd:2*n_embd]
v = qkv[:, 2*n_embd:]
# reshape to heads


q = q.view(seq_len, n_head, d_head).permute(1,0,2)  # [16, 5, 64]
k = k.view(seq_len, n_head, d_head).permute(1,0,2)
v = v.view(seq_len, n_head, d_head).permute(1,0,2)
print("q shape:", q.shape)
print("qkv[-1,:3]:", qkv[-1, :3].tolist())
print("q[-1,-1,:3]:", q[-1, -1, :3].tolist())  # last head, last token
print("k[-1,-1,:3]:", k[-1, -1, :3].tolist())

# attention
scale = 1.0 / (d_head ** 0.5)
scores = torch.bmm(q, k.transpose(1,2)) * scale     # [16, 5, 5]

# causal mask
mask = torch.full((seq_len, seq_len), -1e9).triu(1)
scores = scores + mask.unsqueeze(0)
attn = torch.softmax(scores, -1)
ctx = torch.bmm(attn, v).permute(1,0,2).contiguous().view(seq_len, n_embd)


print("scores[-1,-1,:3]:", scores[-1, -1, :3].tolist())
print("ctx[-1,:3]:", ctx[-1, :3].tolist())

# output proj
c_proj_w = sd['transformer.h.0.attn.c_proj.weight']  # [1024, 1024]
c_proj_b = sd['transformer.h.0.attn.c_proj.bias']
attn_out = ctx @ c_proj_w + c_proj_b
print("attn_out[-1,:3]:", attn_out[-1, :3].tolist())
print(sd['transformer.h.0.mlp.c_proj.weight'].shape)  # expect [4096, 1024]
# MLP
ln2_w = sd['transformer.h.0.ln_2.weight']
ln2_b = sd['transformer.h.0.ln_2.bias']
x2 = x + attn_out  # residual
x2_ln = torch.layer_norm(x2, [n_embd], ln2_w, ln2_b)
c_fc_w = sd['transformer.h.0.mlp.c_fc.weight']    # [1024, 4096]
c_fc_b = sd['transformer.h.0.mlp.c_fc.bias']
h = x2_ln @ c_fc_w + c_fc_b
print("mlp fc[-1,:3]:", h[-1, :3].tolist())
h = torch.nn.functional.gelu(h, approximate='tanh')
print("mlp gelu[-1,:3]:", h[-1, :3].tolist())
c_fc2_w = sd['transformer.h.0.mlp.c_proj.weight']  # [4096, 1024]
c_fc2_b = sd['transformer.h.0.mlp.c_proj.bias']
mlp_out = h @ c_fc2_w + c_fc2_b
print("mlp_out[-1,:3]:", mlp_out[-1, :3].tolist())