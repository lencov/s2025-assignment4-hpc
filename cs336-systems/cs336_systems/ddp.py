from typing import Any
import torch
import torch.distributed as dist


class DDP:
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.handles = []
        self.sync_weights()
        self.world_size = dist.get_world_size()
        self.backend = dist.get_backend()
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self.sync_gradient)

    def sync_weights(self):
        for param in self.module.parameters():
            self.handles.append(dist.broadcast(param.data, 0, async_op=True))
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def sync_gradient(self, p: torch.Tensor):
        if self.backend == "nccl":
            self.handles.append(
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True)
            )
        else:
            self.handles.append(
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
            )

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        if self.backend == "gloo":
            for p in self.module.parameters():
                if p.requires_grad:
                    p.grad /= self.world_size

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.module(*args, **kwds)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)