import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func


# Helper: build cu_seqlens from a list of lengths
def build_cu_seqlens(lengths: list[int], device=None):
    """
    lengths: list[int] of sequence lengths [L0, L1, ..., L_{B-1}]
    returns:
      cu_seqlens: int32 tensor of shape (B+1,)
      max_seqlen: int (max length in batch)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(lens, dim=0)
    max_seqlen = int(lens.max().item())
    return cu_seqlens, max_seqlen


class FlashMHA(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int,
        dropout: float = 0.0, 
        causal: bool = False
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,           # (total_tokens, d_model)
        cu_seqlens: torch.Tensor,  # (B+1,), int32
        max_seqlen: int
    ) -> torch.Tensor:
        total_tokens, d_model = x.shape
        dtype = x.dtype
        assert d_model == self.d_model

        # Project to qkv
        qkv = self.qkv(x)  # (total_tokens, 3 * d_model)
        qkv = qkv.view(total_tokens, 3, self.num_heads, self.head_dim)

        # FlashAttention requires fp16/bf16 + CUDA
        qkv = qkv.to(dtype=torch.float16)  # or bfloat16 if you prefer

        out = flash_attn_varlen_qkvpacked_func(
            qkv,                     # (total, 3, H, D)
            cu_seqlens,              # (B+1,)
            max_seqlen,              # int
            self.dropout if self.training else 0.0,
            softmax_scale=None,
            causal=self.causal,
        )  # -> (total_tokens, H, D)
        out = out.to(dtype=dtype)

        out = out.reshape(total_tokens, self.d_model)
        out = self.out_proj(out)
        return out

class SelfAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = FlashMHA(d_model, num_heads, dropout=dropout, causal=causal)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x_cat: torch.Tensor,       # (total_tokens, d_model)
        cu_seqlens: torch.Tensor,  # (B+1,)
        max_seqlen: int,
    ) -> torch.Tensor:
        # Pre-norm
        h = self.ln1(x_cat)
        x_cat = x_cat + self.attn(h, cu_seqlens, max_seqlen)
        x_cat = x_cat + self.ff(self.ln2(x_cat))
        return x_cat

class SelfAttentionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 4,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            SelfAttentionTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                causal=causal,
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, sequences: list[torch.Tensor]):
        """
        sequences: list of 2D tensors [seq0, seq1, ..., seq_{B-1}]
                   seq_i.shape == (L_i, dim)
        Returns:
          logits_list: list of tensors, each (L_i, vocab_size)
        """
        device = sequences[0].device

        # Compute lengths + cu_seqlens
        lengths = [int(seq.shape[0]) for seq in sequences]
        cu_seqlens, max_seqlen = build_cu_seqlens(lengths, device=device)

        # Concatenate token embed
        x_cat = torch.cat(sequences, dim=0) # (total_tokens, dim)

        # Run all Transformer blocks in packed (varlen) form
        for layer in self.layers:
            x_cat = layer(x_cat, cu_seqlens, max_seqlen)

        # Split back into list per sequence using lengths
        out_embeds = torch.split(x_cat, lengths)

        return out_embeds


if __name__ == '__main__':
    import numpy as np

    dim = 768

    device = "cuda"
    model = SelfAttentionTransformer(
        d_model=dim,
        num_heads=8,
        d_ff=dim*4,
        num_layers=4,
        dropout=0.1,
        causal=False,
    ).to(device)

    n = 0
    for name, param in model.named_parameters():
        print(name, param.size())
        n += np.prod(param.size())
    print('nparams:', n)

    # Suppose you have variable-length tokenized sequences
    inputs = [torch.rand(10, dim), torch.rand(5, dim), torch.rand(9, dim)]
    for i in range(len(inputs)):
        inputs[i] = inputs[i].to(device)

    outputs = model(inputs)
    for v in outputs:
        print(v.size())
