import math
from typing import Tuple, Optional, Callable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        eps: float = 10e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get(
                    "t", 1
                )  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.square(grad)
                alpha = lr * (math.sqrt(1 - beta2**t) / (1 - beta1**t))
                p.data -= (
                    alpha * m / (torch.sqrt(v) + eps)
                )  # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss


def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1
            + math.cos(
                (it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters)
            )
        ) * (max_learning_rate - min_learning_rate)
    return min_learning_rate


def gradient_clipping(
    params: list[torch.nn.Parameter], max_l2_norm: float, eps: float = 10e-6
):
    l2 = torch.sqrt(torch.sum(torch.stack([torch.sum(p.grad ** 2) for p in params if p.grad is not None])))
    if l2 >= max_l2_norm:
        for p in params:
            if p.grad is None:
                continue
            p.grad *= max_l2_norm / (l2 + eps)
