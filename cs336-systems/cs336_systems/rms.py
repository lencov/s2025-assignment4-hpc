from typing import Any
import torch
import triton
import triton.language as tl


def rmsnorm_grad_weight(x, weight, grad_out, eps):
    """
    x: ... x H
    weight: H
    grad_out: ... x H
    """
    x = x.view(-1, x.shape[-1])
    grad_out = grad_out.view(-1, grad_out.shape[-1])
    norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    grad_weight = torch.sum(x * grad_out / norm, dim=0)
    return grad_weight


def rmsnorm_grad_x(x, weight, grad_out, eps):
    """
    x: ... x H
    weight: H
    grad_out: ... x H
    """
    x_shape = x.shape
    H = x_shape[-1]
    x = x.view(-1, x.shape[-1])
    grad_out = grad_out.view(-1, grad_out.shape[-1])
    norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)

    numerator = x * grad_out * weight.view(1, -1)
    second_term = torch.sum(numerator, dim=-1, keepdim=True) * x / (H * norm**3)

    first_term = grad_out * weight.view(1, -1) / norm
    return (first_term - second_term).view(*x_shape)


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-5):
        """
        x: ... x H
        weight: H
        """
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return (
            x
            * weight.view(1, -1)
            / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        )

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        return (
            rmsnorm_grad_x(x, weight, grad_out, eps),
            rmsnorm_grad_weight(x, weight, grad_out, eps),
            None,
        )


@triton.jit
def rmsnorm_forward(
    x_ptr: tl.pointer_type,
    weight_ptr: tl.pointer_type,
    x_row_stride: tl.uint32,
    output_ptr: tl.pointer_type,
    H: tl.uint32,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    weight_ptrs = weight_ptr + offsets
    mask = offsets < H
    row = tl.load(x_ptrs, mask=mask, other=0)
    weight = tl.load(weight_ptrs, mask=mask, other=0)
    norm_factor = tl.sqrt(tl.sum(row * row) / H + eps)
    output = row * weight / norm_factor
    output_ptrs = output_ptr + row_idx * x_row_stride + offsets
    tl.store(output_ptrs, output, mask=mask)


@triton.jit
def rmsnorm_backward(
    x_ptr: tl.pointer_type,
    weight_ptr: tl.pointer_type,
    grad_out_ptr: tl.pointer_type,
    x_row_stride: tl.uint32,
    grad_x_ptr: tl.pointer_type,
    grad_w_accum_ptr: tl.pointer_type,
    H: tl.uint32,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    access_range = row_idx * x_row_stride + offsets
    x_ptrs = x_ptr + access_range
    weight_ptrs = weight_ptr + offsets
    grad_out_ptrs = grad_out_ptr + access_range
    mask = offsets < H
    row = tl.load(x_ptrs, mask=mask, other=0)
    weight = tl.load(weight_ptrs, mask=mask, other=0)
    grad_out = tl.load(grad_out_ptrs, mask=mask, other=0)
    norm_factor = tl.sqrt(tl.sum(row * row) / H + eps)

    # Compute partial gradient w.r.t. weight
    grad_w_accum = row * grad_out / norm_factor
    tl.store(grad_w_accum_ptr + access_range, grad_w_accum, mask=mask)

    # Compute partial gradient w.r.t. x
    first_term = grad_out * weight / norm_factor
    second_term = (
        tl.sum(row * grad_out * weight)
        * row
        / (H * norm_factor * norm_factor * norm_factor)
    )
    grad_x = first_term - second_term
    tl.store(grad_x_ptr + access_range, grad_x, mask=mask)


class RMSNormTritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-5):
        """
        x: ... x H
        weight: H
        """
        ctx.save_for_backward(x, weight)
        ctx.eps = eps

        H = x.shape[-1]
        x_shape = x.shape

        x = x.view(-1, H)

        assert len(weight.shape) == 1 and weight.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        y = torch.empty_like(x, device=x.device)

        n_rows = x.shape[0]
        rmsnorm_forward[(n_rows,)](
            x, weight, x.stride(0), y, H, eps, num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE
        )
        return y.view(*x_shape)

    @staticmethod
    def backward(ctx: Any, grad_out) -> Any:
        x, weight = ctx.saved_tensors
        eps = ctx.eps

        H = x.shape[-1]
        x_shape = x.shape

        x = x.view(-1, H)
        n_rows = x.shape[0]

        # Allocate output tensors
        grad_x = torch.empty_like(x)
        partial_grad_weight = torch.empty_like(x)

        rmsnorm_backward[(n_rows,)](
            x,
            weight,
            grad_out,
            x.stride(0),
            grad_x,
            partial_grad_weight,
            H,
            eps,
            num_warps=16,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )
        return grad_x.view(*x_shape), partial_grad_weight.sum(dim=0), None


class RMSNormTriton(torch.nn.Module):
    def __init__(self, H):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(H))

    def forward(self, x):
        return RMSNormTritonFunc.apply(x, self.weight)

    @staticmethod
    def apply(x, weight, eps=1e-5):
        return RMSNormTritonFunc.apply(x, weight, eps)