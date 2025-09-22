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
    def __init__(self, layer, dim, n_heads, num_groups, head_dim, mlp_dim, window_size, qk_norm, dtype):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        if layer == "full_attention":
            self.sliding_window = False
        else:
            self.sliding_window = True
        self.attn = GQSWAttention(dim, n_heads = n_heads, num_groups=num_groups, head_dim = head_dim, dtype=dtype,  window_size=window_size,
                                   use_sliding_window=self.sliding_window, qk_norm=qk_norm)
        self.norm3 = RMSNorm(dim)
        self.ff = GatedFeedForward(dim, mlp_dim, dtype)
        self.norm4 = RMSNorm(dim)

    def forward(self, x, cos, sin, cos_local, sin_local, mask):
        res = x
        x = self.norm1(x)
        if self.sliding_window:
            x = self.attn(x, cos_local, sin_local, mask)
        else:
            x = self.attn(x, cos, sin, mask)
        x = self.norm2(x) + res
        res = x
        x = self.norm3(x)
        x = self.ff(x)
        x = self.norm4(x) + res
        return x


class Gemma3Model(nn.Module):
    def __init__(self, dim, depth, n_heads, num_groups, head_dim, mlp_dim, vocab_size, context_length, window_size, layer_types,
                 qk_norm=True, dtype=torch.bfloat16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim, dtype=dtype)
        self.blocks = nn.ModuleList([GemmaBlock(layer, dim, n_heads, num_groups, head_dim, mlp_dim, window_size, qk_norm,dtype) for layer in layer_types])

        self.final_norm = RMSNorm(dim, eps=1e-6)
        self.final_proj = nn.Linear(dim, vocab_size, bias=False, dtype=dtype)

        cos, sin = rope_rotate(head_dim, context_length)
        self.window_size = window_size
        self.dim = dim
        # Reusable utilities    
        cos_local, sin_local = rope_rotate(
            head_dim, context_length, 10000.0
        )

        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.dtype = dtype


    def forward(self, inp):

        x = self.tok_emb(inp) * (self.dim ** 0.5)
        n = x.shape[1]
        mask = torch.triu(torch.ones(n, n, device=inp.device, dtype=torch.bool), diagonal=1)

        for gemma3 in self.blocks:
            x = gemma3(x, self.cos, self.sin, self.cos_local, self.sin_local, mask)

        x = self.final_norm(x)
        x = self.final_proj(x.to(self.dtype))
        return x
    
if __name__ == "__main__":
    GEMMA270M_CONFIG = {
        "vocab_size": 262_144,  # Vocabulary size
        "context_length": 32768,  # Context length that was used to train the model
        "emb_dim": 640,  # Embedding dimension
        "n_heads": 4,  # Number of attention heads
        "n_layers": 18,  # Number of layers
        "hidden_dim": 2048,  # Size of the intermediate dimension in FeedForward
        "head_dim": 256,  # Size of the heads in GQA
        "qk_norm": True,  # Whether to normalize queries and keys in GQA
        "n_kv_groups": 1,  # Key-Value groups for grouped-query attention
        "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
        "sliding_window": 512,  # Sliding window attention size
        "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention"
    ],
        "dtype": torch.bfloat16
    }
    model = Gemma3Model(dim=GEMMA270M_CONFIG["emb_dim"], depth=GEMMA270M_CONFIG["n_layers"], n_heads=GEMMA270M_CONFIG["n_heads"],
                       num_groups=GEMMA270M_CONFIG["n_kv_groups"], head_dim=GEMMA270M_CONFIG["head_dim"],
                       mlp_dim=GEMMA270M_CONFIG["hidden_dim"],
                       vocab_size=GEMMA270M_CONFIG["vocab_size"], context_length=GEMMA270M_CONFIG["context_length"],
                       window_size=GEMMA270M_CONFIG["sliding_window"], layer_types=GEMMA270M_CONFIG["layer_types"], dtype=GEMMA270M_CONFIG["dtype"])
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

    print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
    print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")

