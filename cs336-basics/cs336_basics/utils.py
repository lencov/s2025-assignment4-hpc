import os
import time
import typing
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import random
import numpy as np
import numpy.typing as npt

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@contextmanager
def timed_block(description="Block", logger=None):
    print_fn = logger.debug if logger else print

    print_fn(f"Starting {description}...")
    start_time = time.time()
    yield
    end_time = time.time()
    print_fn(f"{description} took {round(end_time - start_time, 3)} seconds")


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    N = len(dataset)
    start_indices = np.random.randint(0, N - context_length, size=(batch_size,))
    values = []
    targets = []
    for idx in start_indices:
        values.append(dataset[idx : idx + context_length].tolist())
        targets.append(dataset[idx + 1 : idx + context_length + 1].tolist())
    return torch.tensor(values, device=device).long(), torch.tensor(targets, device=device).long()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(out, "optimizer.pt"))
    torch.save(iteration, os.path.join(out, "iteration.pt"))


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
):
    model.load_state_dict(torch.load(os.path.join(src, "model.pt")))
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(os.path.join(src, "optimizer.pt")))
    return torch.load(os.path.join(src, "iteration.pt"))
