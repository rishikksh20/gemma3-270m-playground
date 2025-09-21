import torch
from torch import nn
from modules.attention import GQSWAttention
from torch.nn import functional as F

from modules.llm_utils import model_memory_size
from modules.positional_encoding import rope_rotate
from modules.rmsnorm import RMSNorm


class GatedFeedForward(nn.Module):
    def __init__(self, idim, hidden_dim, dtype):
        super().__init__()
        self.gate_proj = nn.Linear(idim, hidden_dim, dtype=dtype, bias=False)
        self.up_proj = nn.Linear(idim, hidden_dim, dtype=dtype, bias=False)
        self.down_proj = nn.Linear(hidden_dim, idim, dtype=dtype, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GemmaBlock(nn.Module):
    def __init__(self, i, dim, n_heads, num_groups, head_dim, mlp_dim, window_size, qk_norm, dtype):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        if (i + 1) % 6 == 0:
            sliding_window = False
        else:
            sliding_window = True
        self.attn = GQSWAttention(dim, n_heads = n_heads, num_groups=num_groups, head_dim = head_dim, dtype=dtype,  window_size=window_size,
                                   use_sliding_window=sliding_window, qk_norm=qk_norm)
        self.norm2 = RMSNorm(dim)
        self.ff = GatedFeedForward(dim, mlp_dim, dtype)
        self.norm3 = RMSNorm(dim)

    def forward(self, x, cos, sin, mask=None):

        x = self.attn(self.norm1(x), cos, sin, mask) + x
        x = self.norm3(self.ff(self.norm2(x))) + x
        return x


class Gemma3Model(nn.Module):
    def __init__(self, dim, depth, n_heads, num_groups, head_dim, mlp_dim, vocab_size, context_length, window_size,
                 qk_norm=False, dtype=torch.bfloat16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim, dtype=dtype)
        self.qwen3_blocks = nn.ModuleList([GemmaBlock(i, dim, n_heads, num_groups, head_dim, mlp_dim, window_size, qk_norm,dtype) for i in range(depth)])

        self.final_norm = RMSNorm(dim, eps=1e-6)
        self.final_proj = nn.Linear(dim, vocab_size, bias=False, dtype=dtype)

        cos, sin = rope_rotate(head_dim, context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.dtype = dtype


    def forward(self, inp):

        emb = self.tok_emb(inp)
        n = emb.shape[1]
        mask = torch.triu(torch.ones(n, n, device=inp.device, dtype=torch.bool), diagonal=1)
        x = emb

        for qwen3 in self.qwen3_blocks:
            x = qwen3(x, self.cos, self.sin, mask)

        x = self.final_norm(x)
        x = self.final_proj(x.to(self.dtype))
        return x
    
if __name__ == "__main__":
    GEMMA270M_CONFIG = {
        "vocab_size": 262_144,  # Vocabulary size
        "context_length": 32_768,  # Context length that was used to train the model
        "emb_dim": 640,  # Embedding dimension
        "n_heads": 4,  # Number of attention heads
        "n_layers": 18,  # Number of layers
        "hidden_dim": 2048,  # Size of the intermediate dimension in FeedForward
        "head_dim": 256,  # Size of the heads in GQA
        "qk_norm": True,  # Whether to normalize queries and keys in GQA
        "n_kv_groups": 1,  # Key-Value groups for grouped-query attention
        "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
        "sliding_window": 512,  # Sliding window attention size
        "dtype": torch.bfloat16
    }
    model = Gemma3Model(dim=GEMMA270M_CONFIG["emb_dim"], depth=GEMMA270M_CONFIG["n_layers"], n_heads=GEMMA270M_CONFIG["n_heads"],
                       num_groups=GEMMA270M_CONFIG["n_kv_groups"], head_dim=GEMMA270M_CONFIG["head_dim"],
                       mlp_dim=GEMMA270M_CONFIG["hidden_dim"],
                       vocab_size=GEMMA270M_CONFIG["vocab_size"], context_length=GEMMA270M_CONFIG["context_length"],
                       window_size=GEMMA270M_CONFIG["sliding_window"], dtype=GEMMA270M_CONFIG["dtype"])
    device = torch.device("cpu")
    out = model(torch.tensor([1, 2, 3]).unsqueeze(0)).to(device)

    print("Model output shape : ", out.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Account for weight tying
    total_params_normalized = total_params - model.tok_emb.weight.numel()
    print(f"\nTotal number of unique parameters: {total_params_normalized:,}")

    # print("\nModel : \n", model)

    print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
    print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")

