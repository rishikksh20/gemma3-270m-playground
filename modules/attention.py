import math
from einops import rearrange
import torch
from torch import nn, einsum
from modules.rmsnorm import RMSNorm
from modules.positional_encoding import apply_rope



class GQAttention(nn.Module):

    def __init__(self, idim, n_heads, num_groups, head_dim, dtype):

        super(GQAttention, self).__init__()
        self.idim = idim
        self.n_heads = n_heads
        self.num_groups = num_groups
        self.head_dim = head_dim
        self.group_size = self.n_heads // self.num_groups
        self.n_kv_embed = self.head_dim * self.num_groups

        self.odim = self.n_heads * self.head_dim

        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.idim, self.odim, dtype=dtype, bias=False)
        self.k_proj = nn.Linear(self.idim, self.n_kv_embed, dtype=dtype, bias=False)
        self.v_proj = nn.Linear(self.idim, self.n_kv_embed, dtype=dtype, bias=False)
        self.o_proj = nn.Linear(self.odim, self.idim, dtype=dtype, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

    def forward(self, x, cos, sin, mask=None):

        b, L, dim = x.shape

        q = self.q_proj(x)              # (B, L, dim)
        k = self.k_proj(x)              # (B, L, n_kv_embed)
        v = self.v_proj(x)              # (B, L, n_kv_embed)

        q = rearrange(q, 'b l (n d) -> b n l d', n=self.n_heads)
        k = rearrange(k, 'b l (g d) -> b g l d', g=self.num_groups)
        v = rearrange(v, 'b l (g d) -> b g l d', g=self.num_groups)

        q = self.q_norm(q)
        k = self.k_norm(k)


        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)


        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = dots.masked_fill(mask, -torch.inf)

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.o_proj(out)



class GQSWAttention(nn.Module):

    def __init__(self, idim, n_heads, num_groups, head_dim, dtype, window_size=None, use_sliding_window=False, qk_norm=False,
                 query_pre_attn_scalar=None):
        super(GQSWAttention, self).__init__()
        self.idim = idim
        self.n_heads = n_heads
        self.num_groups = num_groups
        self.head_dim = head_dim
        self.group_size = self.n_heads // self.num_groups
        self.n_kv_embed = self.head_dim * self.num_groups

        self.odim = self.n_heads * self.head_dim

        if query_pre_attn_scalar is not None:
            self.scale = (query_pre_attn_scalar) ** -0.5
        else:
            self.scale = (head_dim) ** -0.5

        self.q_proj = nn.Linear(self.idim, self.odim, dtype=dtype, bias=False)
        self.k_proj = nn.Linear(self.idim, self.n_kv_embed, dtype=dtype, bias=False)
        self.v_proj = nn.Linear(self.idim, self.n_kv_embed, dtype=dtype, bias=False)
        self.o_proj = nn.Linear(self.odim, self.idim, dtype=dtype, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        self.window_size = window_size
        self.use_sliding_window = use_sliding_window

    def _sliding_window_mask(self, seq_len, device):
        # Mask shape: (seq_len, seq_len)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i+1] = False  # allow attending to window
        return mask  # True = masked, False = attend

    def forward(self, x, cos, sin, mask=None):
        b, L, dim = x.shape

        q = self.q_proj(x)              # (B, L, dim)
        k = self.k_proj(x)              # (B, L, n_kv_embed)
        v = self.v_proj(x)              # (B, L, n_kv_embed)

        q = rearrange(q, 'b l (n d) -> b n l d', n=self.n_heads)
        k = rearrange(k, 'b l (g d) -> b g l d', g=self.num_groups)
        v = rearrange(v, 'b l (g d) -> b g l d', g=self.num_groups)

        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Sliding window mask logic
        if self.use_sliding_window and self.window_size is not None:
            sw_mask = self._sliding_window_mask(L, x.device)  # (L, L)
            sw_mask = sw_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
            if mask is not None:
                mask = mask | sw_mask  # combine masks
            else:
                mask = sw_mask

        if mask is not None:
            dots = dots.masked_fill(mask, -torch.inf)

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.o_proj(out)

