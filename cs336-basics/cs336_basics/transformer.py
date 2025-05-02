import os
import math
import typing

import torch
import numpy as np
import numpy.typing as npt
from cs336_basics.utils import DEVICE
from torch.nn import Parameter, Linear, Embedding, Module, ModuleList


class RMSNorm(Module):
    def __init__(
        self, d_model: int, gain_init: torch.Tensor = None, eps: float = 1e-5
    ) -> None:
        super().__init__()
        if gain_init is None:
            gain_init = torch.ones(d_model)
        self.weight = Parameter(torch.zeros((d_model,)))
        self.eps = eps
        self.d_model = d_model

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        a: (..., d_model)
        """
        numerator = a * self.weight.view(*[1] * (len(a.shape) - 1), self.d_model)
        denominator = torch.sqrt(
            (1 / self.d_model) * torch.square(a).sum(-1, keepdim=True) + self.eps
        )
        return numerator / denominator


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
) -> torch.Tensor:
    """
    Q: B x ... x S x D_k
    K: B x ... x S x D_k
    V: B x ... x S x D_v
    M: S x S
    """

    # pre_scores: B x ... x S x S
    pre_scores = K @ Q.transpose(-1, -2) / math.sqrt(K.shape[-1])
    if mask is not None:
        pre_scores = pre_scores.masked_fill(mask, -torch.inf)
    # scores: B x ... x S x S
    scores = softmax(pre_scores, -1)
    if p_drop > 0:
        scores = torch.nn.functional.dropout(scores, p_drop)
    return scores @ V


class CausalMultiheadSelfAttention(Module):
    def __init__(
        self, d_model: int, num_heads: int, attn_pdrop: float | None = None, max_seq_len: int = 512
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        d_k = d_model // num_heads
        self.d_k = d_k

        self.q_proj = Linear(d_k * num_heads, d_model, bias=False)
        self.k_proj = Linear(d_k * num_heads, d_model, bias=False)
        self.v_proj = Linear(d_k * num_heads, d_model, bias=False)

        self.output_proj = Linear(d_k * num_heads, d_model, bias=False)
        self.mask = torch.triu(torch.ones((max_seq_len, max_seq_len)).bool(), diagonal=1).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B x ... x S x d_model
        """
        if x.dim() == 2:
            x.unsqueeze(0)

        B, S, _ = x.shape

        queries = x @ self.q_proj.weight.T
        keys = x @ self.k_proj.weight.T
        values = x @ self.v_proj.weight.T

        # q/k/v: B x d_k * num_heads x S

        # quries/keys/values: B x num_heads x S x d_k
        queries = queries.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        keys = keys.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        values = values.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        # mask: S x S
        attn = scaled_dot_product_attention(
            queries, keys, values, mask=self.mask[:S,:S], p_drop=self.attn_pdrop
        )
        # attn: B x h x S x d_k
        attn = attn.transpose(1, 2).reshape(B, S, -1)
        out = self.output_proj(attn)
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
        norm_type: typing.Literal["post", "pre", "none"] = "pre"
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = CausalMultiheadSelfAttention(d_model, num_heads, attn_pdrop)
        self.dropout = torch.nn.Dropout(residual_pdrop)
        self.ffn = FFN(d_model, d_ff)
        self.is_parallel = is_parallel
        self.norm_type = norm_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B x S x d_model
        """
        if self.is_parallel:
            assert self.norm_type == "pre", "Parallel transformer blocks only support pre-norm"
            return x + self.dropout(self.attn(self.ln1(x))) + self.dropout(self.ffn(self.ln2(x)))
        if self.norm_type == "post":
            y = self.ln1(x + self.dropout(self.attn(x)))
            return self.ln2(y + self.dropout(self.ffn(y)))
        if self.norm_type == "none":
            return x + self.dropout(self.attn(x)) + self.dropout(self.ffn(x))
        y = x + self.dropout(self.attn(self.ln1(x)))
        return y + self.dropout(self.ffn(self.ln2(y)))


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
        norm_type: typing.Literal["post", "pre", "none"] = "pre"
    ) -> None:
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.position_embeddings = Embedding(context_length, d_model)
        self.layers = ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop, is_parallel, norm_type)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)
        self.dropout = torch.nn.Dropout(residual_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(
            self.token_embeddings(x)
            + self.position_embeddings(torch.arange(x.shape[1]).to(DEVICE)).unsqueeze(0)
        )
        for block in self.layers:
            x = block(x)
        return self.lm_head(self.ln_final(x))
