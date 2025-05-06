import os
import math
import typing

import torch
from cs336_basics.utils import DEVICE
from torch.nn import Parameter, Linear, Embedding, Module, ModuleList
from cs336_systems.rms import RMSNormTritonFunc


class RMSNorm(Module):
    """RMSNorm implementation from Assignment 1"""
    def __init__(
        self, d_model: int, gain_init: torch.Tensor = None, eps: float = 1e-5
    ) -> None:
        super().__init__()
        if gain_init is not None:
             self.weight = Parameter(gain_init.clone().detach())
        else:
             self.weight = Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        H = a.shape[-1]
        weight_view_shape = [1] * (a.dim() - 1) + [-1]
        rms = torch.rsqrt(torch.mean(a.pow(2), dim=-1, keepdim=True) + self.eps)
        return a * rms * self.weight.view(*weight_view_shape)


class Gelu(Module):
    def __init__(self) -> None:
        super().__init__()
        self._sqrt_2 = torch.sqrt(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.erf(x / self._sqrt_2))


class FFN(Module):
    def __init__(self, d_model: int, d_ff: int = None) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.w1 = Linear(d_model, d_ff, bias=False)
        self.w2 = Linear(d_ff, d_model, bias=False)
        self.gelu = Gelu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.gelu(self.w1(x)))


def softmax(x: torch.Tensor, dim: int):
    max_val = torch.max(x, dim=dim, keepdim=True)[0]
    exp_val = torch.exp(x - max_val)
    return exp_val / exp_val.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    K: torch.Tensor,
    Q: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    p_drop: float = 0.0,
    is_training: bool = True,
) -> torch.Tensor:
    """
    Q: B x ... x S x D_k
    K: B x ... x S x D_k
    V: B x ... x S x D_v
    M: S x S (optional mask)
    """
    d_k = Q.shape[-1]
    pre_scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)
    if mask is not None:
        pre_scores = pre_scores.masked_fill(mask[..., :pre_scores.shape[-2], :pre_scores.shape[-1]], float("-inf"))
    scores = softmax(pre_scores, dim=-1)
    if p_drop > 0.0:
        scores = torch.nn.functional.dropout(scores, p=p_drop, training=is_training)
    return scores @ V


class CausalMultiheadSelfAttention(Module):
    def __init__(
        self, d_model: int, num_heads: int, attn_pdrop: float | None = None, max_seq_len: int = 512
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop if attn_pdrop is not None else 0.0
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, bias=False)
        self.k_proj = Linear(d_model, d_model, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=False)
        self.output_proj = Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        assert D == self.d_model, f"Input dimension {D} does not match model dimension {self.d_model}"

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        keys = keys.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        values = values.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        causal_mask = self.mask[:S, :S]

        attn_output = scaled_dot_product_attention(
            keys, queries, values,
            mask=causal_mask[None, None, :, :],
            p_drop=self.attn_pdrop,
            is_training=self.training
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        out = self.output_proj(attn_output)
        return out


class TransformerBlock(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float | None = None,
        residual_pdrop: float | None = None,
        is_parallel: bool = False,
        norm_type: typing.Literal["post", "pre", "none"] = "pre",
        use_layer_norm: bool = False,
        use_triton_rmsnorm: bool = False,
        max_seq_len: int = 512
    ) -> None:
        super().__init__()
        assert not (use_layer_norm and use_triton_rmsnorm), "Cannot use both LayerNorm and Triton RMSNorm flags"
        if use_triton_rmsnorm and RMSNormTritonFunc is None:
             raise ImportError("use_triton_rmsnorm=True but RMSNormTritonFunc could not be imported.")

        self.use_triton_rmsnorm = use_triton_rmsnorm
        self.is_parallel = is_parallel
        self.norm_type = norm_type

        self.ln1 = None
        self.ln2 = None
        self.ln1_w = None
        self.ln2_w = None

        if norm_type == "pre" or norm_type == "post":
            if use_triton_rmsnorm:
                 self.ln1_w = Parameter(torch.ones(d_model))
                 self.ln2_w = Parameter(torch.ones(d_model))
            elif use_layer_norm:
                self.ln1 = torch.nn.LayerNorm(d_model)
                self.ln2 = torch.nn.LayerNorm(d_model)
            else:
                self.ln1 = RMSNorm(d_model)
                self.ln2 = RMSNorm(d_model)
        elif norm_type == "none":
             self.ln1 = torch.nn.Identity()
             self.ln2 = torch.nn.Identity()

        self.attn = CausalMultiheadSelfAttention(d_model, num_heads, attn_pdrop, max_seq_len=max_seq_len)
        self.dropout = torch.nn.Dropout(residual_pdrop if residual_pdrop is not None else 0.0)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if self.norm_type == "pre":
            if self.use_triton_rmsnorm:
                normed_x = RMSNormTritonFunc.apply(x, self.ln1_w)
            elif self.ln1 is not None:
                normed_x = self.ln1(x)
            else:
                normed_x = x
        else:
            normed_x = x

        attn_output = self.attn(normed_x)
        attn_output = self.dropout(attn_output)

        if self.norm_type == "post":
            attn_output = residual + attn_output
            if self.use_triton_rmsnorm:
                normed_attn_output = RMSNormTritonFunc.apply(attn_output, self.ln1_w)
            elif self.ln1 is not None:
                normed_attn_output = self.ln1(attn_output)
            else:
                normed_attn_output = attn_output
        else:
            normed_attn_output = residual + attn_output

        residual_for_ffn = normed_attn_output

        if self.norm_type == "pre":
            if self.use_triton_rmsnorm:
                normed_ffn_input = RMSNormTritonFunc.apply(residual_for_ffn, self.ln2_w)
            elif self.ln2 is not None:
                normed_ffn_input = self.ln2(residual_for_ffn)
            else:
                normed_ffn_input = residual_for_ffn
        else:
             normed_ffn_input = residual_for_ffn

        ffn_output = self.ffn(normed_ffn_input)
        ffn_output = self.dropout(ffn_output)

        if self.norm_type == "post":
            ffn_output = residual_for_ffn + ffn_output
            if self.use_triton_rmsnorm:
                 final_output = RMSNormTritonFunc.apply(ffn_output, self.ln2_w)
            elif self.ln2 is not None:
                 final_output = self.ln2(ffn_output)
            else:
                final_output = ffn_output
        else:
            final_output = residual_for_ffn + ffn_output

        return final_output


class TransformerLM(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float | None = None,
        residual_pdrop: float | None = None,
        is_parallel: bool = False,
        norm_type: typing.Literal["post", "pre", "none"] = "pre",
        use_layer_norm: bool = False,
        use_triton_rmsnorm: bool = False
    ) -> None:
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.position_embeddings = Embedding(context_length, d_model)
        self.layers = ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    attn_pdrop=attn_pdrop,
                    residual_pdrop=residual_pdrop,
                    is_parallel=is_parallel,
                    norm_type=norm_type,
                    use_layer_norm=use_layer_norm,
                    use_triton_rmsnorm=use_triton_rmsnorm,
                    max_seq_len=context_length
                    )
                for _ in range(num_layers)
            ]
        )

        if norm_type != 'none':
            if use_layer_norm:
                self.ln_final = torch.nn.LayerNorm(d_model)
            else:
                self.ln_final = RMSNorm(d_model)
        else:
            self.ln_final = torch.nn.Identity()

        self.lm_head = Linear(d_model, vocab_size, bias=False)
        self.dropout = torch.nn.Dropout(residual_pdrop if residual_pdrop is not None else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len > self.position_embeddings.num_embeddings:
             raise ValueError(f"Input sequence length ({seq_len}) exceeds model context length ({self.position_embeddings.num_embeddings})")

        pos = torch.arange(seq_len, device=x.device)
        x = self.token_embeddings(x) + self.position_embeddings(pos).unsqueeze(0)
        x = self.dropout(x)

        for block in self.layers:
            x = block(x)

        x = self.ln_final(x)

        return self.lm_head(x)