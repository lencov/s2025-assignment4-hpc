import torch
import torch.nn as nn
import torch.distributed as dist
import logging
import os
from tqdm import tqdm
import argparse
from datetime import timedelta
from typing import Optional
from dataclasses import dataclass
import torch.multiprocessing as mp
from copy import deepcopy
from cs336_basics.optimizer import AdamW
from cs336_basics.loss import cross_entropy
from cs336_basics.transformer import TransformerLM
from cs336_systems.optimizer_sharding import OptimizerSharded
import timeit
from cs336_systems.ddp_bucket import DDPBucketed


# setup logging
logging.basicConfig(format="%(asctime)s (%(levelname)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ModelArgs:
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    vocab_size: int = 10000
    context_length: int = 128
    attn_pdrop: Optional[float] = 0.1
    residual_pdrop: Optional[float] = 0.05
    use_layer_norm: Optional[bool] = False
    use_triton_rmsnorm: Optional[bool] = False
    name: str = "small"


@dataclass
class TrainerArgs:
    batch_size: int = 16
    warmup_steps: int = 1
    train_steps: int = 5
    run_backward: bool = False
    mixed_precision: bool = False
    compile: bool = False


@dataclass
class OptimizerArgs:
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-9
    weight_decay: float = 0.1


MODEL_CONFIGS = {
    "small": ModelArgs(
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
    ),
    "medium": ModelArgs(
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=4096,
    ),
    "large": ModelArgs(
        d_model=1280,
        num_layers=36,
        num_heads=20,
        d_ff=5120,
    ),
    "xl": ModelArgs(
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
    ),
    "2.7B": ModelArgs(
        d_model=2560,
        num_layers=32,
        num_heads=32,
        d_ff=10240,
    ),
}


def validate_ddp_net_equivalence(net: nn.Module, rank: int):
    # Helper to validate synchronization of nets across ranks.
    net_module_states = list(net.state_dict().values())
    # Check that all tensors in module's state_dict() are equal.
    for t in net_module_states:
        tensor_list = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, t)
        for tensor in tensor_list:
            assert torch.allclose(tensor, t)
    if rank == 0:
        logger.info("All parameters are equal across all ranks")


def setup_singlenode(
    backend: str = str, rank: int = None, world_size: int = None
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "14322"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)


def setup_multinode(backend: str) -> None:
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    assert os.environ["MASTER_ADDR"]
    assert os.environ["MASTER_PORT"]
    timeout = timedelta(seconds=60)

    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timeout)
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, local_world_size


def cleanup():
    dist.destroy_process_group()


def ddp_main(
    rank: int,
    world_size: int,
    backend: str,
    data: torch.Tensor,
    labels: torch.Tensor,
    model_args: ModelArgs,
    optimizer_args: OptimizerArgs,
    batched: bool = False,
    shard_optim: bool = False,
    profile_memory: bool = False,
    dd_bucket: bool = False,
    multinode: bool = False,
):
    if rank == -1:
        rank, world_size, _, _ = setup_multinode(backend)
    else:
        setup_singlenode(backend, rank, world_size)

    DEVICE = "cuda" if backend == "nccl" else "cpu"

    if rank == 0:
        logger.info(
            f"Running benchmark with model config: {model_args.name}\n{model_args}"
        )

    torch.manual_seed(rank)
    partition = data.size(0) // world_size
    start_index = rank * partition
    end_index = start_index + partition

    warmup_steps = 5
    num_steps = 21
        
    data = data.to(DEVICE)
    labels = labels.to(DEVICE)

    model = TransformerLM(
        vocab_size=model_args.vocab_size,
        context_length=model_args.context_length,
        d_model=model_args.d_model,
        num_layers=model_args.num_layers,
        num_heads=model_args.num_heads,
        d_ff=model_args.d_ff,
        attn_pdrop=model_args.attn_pdrop,
        residual_pdrop=model_args.residual_pdrop,
        norm_type="pre" if model_args.use_layer_norm else "none",
    ).to(DEVICE)

    if shard_optim:
        optimizer = OptimizerSharded(
            model.parameters(),
            AdamW,
            lr=optimizer_args.lr,
            betas=optimizer_args.betas,
            eps=optimizer_args.eps,
            weight_decay=optimizer_args.weight_decay,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=optimizer_args.lr,
            betas=optimizer_args.betas,
            eps=optimizer_args.eps,
            weight_decay=optimizer_args.weight_decay,
        )
    model.train()

    comm_time = 0

    # Broadcast rank 0 model
    start = timeit.default_timer()
    if dd_bucket:
        model = DDPBucketed(model, bucket_size_mb=64)
    elif batched:
        params = torch.nn.utils.parameters_to_vector(model.parameters())
        dist.broadcast(params, 0, async_op=False)
        dist.barrier()
        torch.cuda.synchronize()
        torch.nn.utils.vector_to_parameters(params, model.parameters())
    else:
        for param in model.parameters():
            dist.broadcast(param.data, 0, async_op=False)
        dist.barrier()
    comm_time += timeit.default_timer() - start

    if profile_memory and rank == 0:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    step_timer = timeit.default_timer()
    for step in tqdm(range(num_steps)):
        if step == warmup_steps - 1:
            step_timer = timeit.default_timer()
        out = model(data[start_index:end_index])
        loss = loss = cross_entropy(out, labels[start_index:end_index])
        loss.backward()

        start = timeit.default_timer()
        if dd_bucket:
            model.finish_gradient_synchronization()
        elif batched:
            params = torch._utils._flatten_dense_tensors(
                [param.grad for param in model.parameters()]
            )
            if backend == "nccl":
                dist.all_reduce(tensor=params, op=dist.ReduceOp.AVG, async_op=False)
            else:
                dist.all_reduce(tensor=params, op=dist.ReduceOp.SUM, async_op=False)
                params /= world_size
            dist.barrier()
            torch.cuda.synchronize()
            for param, grad in zip(
                model.parameters(),
                torch._utils._unflatten_dense_tensors(
                    params, [p.grad for p in model.parameters()]
                ),
            ):
                param.grad = grad
        else:
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                if backend == "nccl":
                    dist.all_reduce(
                        tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False
                    )
                else:
                    dist.all_reduce(
                        tensor=param.grad, op=dist.ReduceOp.SUM, async_op=False
                    )
                    param.grad /= world_size
            dist.barrier()
        if step >= warmup_steps:
            comm_time += timeit.default_timer() - start

        optimizer.step()
        optimizer.zero_grad()

        logger.debug(f"Rank {rank}: step = {step}, loss = {loss.item()}")

    dist.barrier()
    if rank == 0:
        logger.info(
            f"Time taken per steps: {(timeit.default_timer() - step_timer) / (num_steps - warmup_steps)}"
        )
        logger.info(f"Time taken for communication: {comm_time}")
        if profile_memory:
            torch.cuda.memory._dump_snapshot(
                f"memory_snapshot-{model_args.name}-optim-sharding-{shard_optim}-multinode-{multinode}.pickle"
            )
            torch.cuda.memory._record_memory_history(enabled=None)

    validate_ddp_net_equivalence(model, rank)
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True)
    parser.add_argument("--multinode", action="store_true", default=False)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--shard-optim", action="store_true", default=False)
    parser.add_argument("--profile-memory", action="store_true", default=False)
    parser.add_argument("--ddp-bucket", action="store_true", default=False)
    parser.add_argument(
        "--model-config",
        type=str,
        default="small",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
    )
    parser.add_argument("--batched", action="store_true", default=False)
    args = parser.parse_args()

    model_cfg = (
        MODEL_CONFIGS.keys() if args.model_config == "all" else [args.model_config]
    )

    for model_config in model_cfg:
        model_args = MODEL_CONFIGS[model_config]
        model_args.name = args.model_config
        optimizer_args = OptimizerArgs()
        args = parser.parse_args()

        # Generate data
        batch_size = 16
        torch.manual_seed(0)
        data = torch.randint(
            0,
            model_args.vocab_size,
            (batch_size * args.world_size, model_args.context_length),
        )
        labels = torch.randint(
            0,
            model_args.vocab_size,
            (batch_size * args.world_size, model_args.context_length),
        )

        if args.multinode:
            ddp_main(
                -1,
                -1,
                args.backend,
                data,
                labels,
                model_args,
                optimizer_args,
                args.batched,
                args.shard_optim,
                args.profile_memory,
                args.ddp_bucket,
                args.multinode,
            )
        else:
            mp.spawn(
                ddp_main,
                args=(
                    args.world_size,
                    args.backend,
                    data,
                    labels,
                    model_args,
                    optimizer_args,
                    args.batched,
                    args.shard_optim,
                    args.profile_memory,
                    args.ddp_bucket,
                    args.multinode,
                ),
                nprocs=args.world_size,
                join=True,
            )


if __name__ == "__main__":
    main()