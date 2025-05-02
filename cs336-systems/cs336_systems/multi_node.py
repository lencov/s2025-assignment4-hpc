import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import logging
from datetime import timedelta
import argparse
from typing import Dict

# setup logging
logging.basicConfig(format="%(asctime)s (%(levelname)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TENSOR_SIZES: Dict[str, int] = {
    "512KB": int(512e3 / 4),
    "1MB": int(1e6 / 4),
    "10MB": int(10e6 / 4),
    "50MB": int(50e6 / 4),
    "100MB": int(100e6 / 4),
    "500MB": int(500e6 / 4),
    "1GB": int(1e9 / 4),
}


def setup(backend: str, use_cuda: bool) -> None:
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    assert os.environ["MASTER_ADDR"]
    assert os.environ["MASTER_PORT"]
    timeout = timedelta(seconds=60)

    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timeout)
    if backend == "nccl" or (backend == "gloo" and use_cuda):
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, local_world_size


def benchmark_all_reduce(backend: str, tensor_size_str: str, use_cuda: bool) -> None:
    rank, world_size, local_rank, local_world_size = setup(backend, use_cuda)
    if rank == 0:
        logger.debug(
            f"Rank: {rank}, World size: {world_size}, Local rank: {local_rank}, Local world size: {local_world_size}"
        )
    tensor_size = TENSOR_SIZES[tensor_size_str]
    device = torch.device(
        "cuda" if (backend == "nccl" or (backend == "gloo" and use_cuda)) else "cpu"
    )

    # Warmup steps
    data = torch.randn(tensor_size, device=device).to(torch.float32)
    for _ in range(5):
        dist.all_reduce(data, async_op=False)
        if device.type == "cuda":
            torch.cuda.synchronize()

    start_time = time.time()
    dist.all_reduce(data, async_op=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - start_time

    all_times = [None] * world_size
    dist.all_gather_object(all_times, total_time)
    if rank == 0:
        avg_time = sum(all_times) / world_size
        logger.debug(
            f"Backend: {backend}, Device: {device}, World size: {world_size}, Tensor size: {tensor_size_str}, Average Time taken: {avg_time} seconds"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--tensor_size", type=str, required=True)
    parser.add_argument("--backend", type=str, required=True)
    args = parser.parse_args()

    assert not (args.backend == "nccl" and not args.use_cuda)
    assert args.tensor_size in TENSOR_SIZES.keys()
    benchmark_all_reduce(args.backend, args.tensor_size, args.use_cuda)


if __name__ == "__main__":
    main()