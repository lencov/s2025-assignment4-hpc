#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.distributed as dist
import logging
import os
from tqdm import tqdm
import argparse
from datetime import timedelta
from typing import Optional, Dict # Added Dict
from dataclasses import dataclass
import torch.multiprocessing as mp
from copy import deepcopy
import timeit # For timing

# Import from your Assignment 1 code
from cs336_basics.transformer import TransformerLM # Assuming this is your model class
from cs336_basics.optimizer import AdamW
from cs336_basics.loss import cross_entropy # Assuming this is your loss function

# Setup logging
logging.basicConfig(
    format="%(asctime)s (%(levelname)s) [%(funcName)s:%(lineno)d] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Model configurations (same as your benchmark.py)
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
    use_layer_norm: Optional[bool] = False # For TransformerLM init
    use_triton_rmsnorm: Optional[bool] = False # For TransformerLM init
    name: str = "small"


MODEL_CONFIGS: Dict[str, ModelArgs] = {
    "small": ModelArgs(d_model=768, num_layers=12, num_heads=12, d_ff=3072),
    "medium": ModelArgs(d_model=1024, num_layers=24, num_heads=16, d_ff=4096),
    "large": ModelArgs(d_model=1280, num_layers=36, num_heads=20, d_ff=5120),
    # Add other configs if needed, ensure they match what TransformerLM expects
}
OPTIMIZER_ARGS = {"lr": 1e-4, "betas": (0.9, 0.99), "eps": 1e-9, "weight_decay": 0.1}

# Global for passing use_cuda to setup_singlenode via mp.spawn
_global_use_cuda_for_spawn = False

def setup_distributed(rank: int, world_size: int, backend: str, multinode: bool, use_cuda_for_setup: bool):
    if multinode:
        if rank == -1: 
             return None, None 
        
        actual_rank = int(os.environ["SLURM_PROCID"])
        actual_world_size = int(os.environ["SLURM_NTASKS"])
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
        
        logger.info(f"Rank {actual_rank}/{actual_world_size} on {os.uname()[1]}: Initializing process group (multinode). Master: {master_addr}:{master_port}")
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=actual_rank,
            world_size=actual_world_size,
            timeout=timedelta(seconds=120)
        )
        if use_cuda_for_setup and backend == "nccl":
            torch.cuda.set_device(int(os.environ["SLURM_LOCALID"]))
        return actual_rank, actual_world_size
    else: # Single-node
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "14322") # Default port for this script
        
        logger.info(f"Rank {rank}/{world_size} on {os.uname()[1]}: Initializing process group (single-node). Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        if use_cuda_for_setup and torch.cuda.is_available():
            target_device = rank
            if backend == "nccl" and world_size == 2: # Your MIG fix
                target_device = 0 if rank == 0 else 2 
            elif backend == "nccl" and world_size == 4: # Your MIG fix for 4
                target_device = rank 
            
            if rank < torch.cuda.device_count(): # General check
                 torch.cuda.set_device(target_device)
                 if rank == 0: logger.debug(f"Rank {rank} (backend {backend}) set to CUDA device {target_device}")
            else:
                 if rank == 0: logger.error(f"Rank {rank} wants device {target_device} but only {torch.cuda.device_count()} devices available.")
        return rank, world_size

def ddp_benchmark_worker(
    rank: int,
    world_size: int,
    backend: str,
    use_cuda_passed: bool,
    model_config_name: str,
    multinode: bool,
    batched_allreduce: bool # Added for flat DDP
):
    global _global_use_cuda_for_spawn
    _global_use_cuda_for_spawn = use_cuda_passed

    actual_rank, actual_world_size = setup_distributed(rank, world_size, backend, multinode, use_cuda_passed)
    if multinode and actual_rank is None:
        return

    DEVICE = "cuda" if use_cuda_passed and torch.cuda.is_available() else "cpu"
    if actual_rank == 0: logger.info(f"Using device: {DEVICE}, Rank: {actual_rank}, WorldSize: {actual_world_size}, BatchedAllreduce: {batched_allreduce}")

    cfg_data = MODEL_CONFIGS[model_config_name]
    # Ensure all required args for your TransformerLM are passed
    model = TransformerLM(
        vocab_size=cfg_data.vocab_size,
        context_length=cfg_data.context_length,
        d_model=cfg_data.d_model,
        num_layers=cfg_data.num_layers,
        num_heads=cfg_data.num_heads,
        d_ff=cfg_data.d_ff,
        attn_pdrop=cfg_data.attn_pdrop,
        residual_pdrop=cfg_data.residual_pdrop,
        # Pass norm flags if your TransformerLM expects them
        use_layer_norm=cfg_data.use_layer_norm,
        use_triton_rmsnorm=cfg_data.use_triton_rmsnorm
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), **OPTIMIZER_ARGS)
    model.train()

    # Broadcast initial model parameters from rank 0
    for param in model.parameters():
        dist.broadcast(param.data, 0, async_op=False)
    dist.barrier()

    batch_size_per_gpu = 16 
    local_batch_size = batch_size_per_gpu
    inputs = torch.randint(0, cfg_data.vocab_size, (local_batch_size, cfg_data.context_length), device=DEVICE)
    targets = torch.randint(0, cfg_data.vocab_size, (local_batch_size, cfg_data.context_length), device=DEVICE)
    
    num_warmup_steps = 5
    num_benchmark_steps = 10 # As per assignment PDF example for naive_ddp_benchmarking

    for step in range(num_warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, cfg_data.vocab_size), targets.view(-1))
        loss.backward()
        # Naive/Flat DDP gradient sync
        if batched_allreduce:
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if grads:
                flat_grads = torch._utils._flatten_dense_tensors(grads)
                op = dist.ReduceOp.AVG if backend == "nccl" else dist.ReduceOp.SUM
                dist.all_reduce(tensor=flat_grads, op=op, async_op=False)
                if backend == "gloo" and op == dist.ReduceOp.SUM:
                    flat_grads /= actual_world_size
                for grad_orig, grad_flat_unflattened in zip(grads, torch._utils._unflatten_dense_tensors(flat_grads, grads)):
                    grad_orig.copy_(grad_flat_unflattened)
        else:
            for param in model.parameters():
                if param.grad is not None:
                    op = dist.ReduceOp.AVG if backend == "nccl" else dist.ReduceOp.SUM
                    dist.all_reduce(tensor=param.grad, op=op, async_op=False)
                    if backend == "gloo" and op == dist.ReduceOp.SUM:
                        param.grad /= actual_world_size
        optimizer.step()

    if use_cuda_passed: torch.cuda.synchronize()
    dist.barrier()

    total_step_times = []
    total_comm_times = []

    for step in range(num_benchmark_steps):
        if use_cuda_passed: torch.cuda.synchronize()
        dist.barrier()
        step_start_time = timeit.default_timer()

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, cfg_data.vocab_size), targets.view(-1))
        loss.backward()
        
        if use_cuda_passed: torch.cuda.synchronize() # Ensure grads are ready
        comm_start_time = timeit.default_timer()
        if batched_allreduce:
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if grads:
                flat_grads = torch._utils._flatten_dense_tensors(grads)
                op = dist.ReduceOp.AVG if backend == "nccl" else dist.ReduceOp.SUM
                dist.all_reduce(tensor=flat_grads, op=op, async_op=False)
                if backend == "gloo" and op == dist.ReduceOp.SUM:
                    flat_grads /= actual_world_size
                # Unflatten (important for optimizer.step())
                for grad_orig, grad_flat_unflattened in zip(grads, torch._utils._unflatten_dense_tensors(flat_grads, grads)):
                    grad_orig.copy_(grad_flat_unflattened)
        else:
            for param in model.parameters():
                if param.grad is not None:
                    op = dist.ReduceOp.AVG if backend == "nccl" else dist.ReduceOp.SUM
                    dist.all_reduce(tensor=param.grad, op=op, async_op=False)
                    if backend == "gloo" and op == dist.ReduceOp.SUM:
                        param.grad /= actual_world_size
        if use_cuda_passed: torch.cuda.synchronize() # Ensure comm is done
        comm_time_per_step = timeit.default_timer() - comm_start_time
        total_comm_times.append(comm_time_per_step)
        
        optimizer.step()

        if use_cuda_passed: torch.cuda.synchronize()
        dist.barrier()
        step_time = timeit.default_timer() - step_start_time
        total_step_times.append(step_time)

    if actual_rank == 0:
        avg_step_time = sum(total_step_times) / len(total_step_times)
        avg_comm_time = sum(total_comm_times) / len(total_comm_times)
        logger.info(f"Model: {model_config_name}, Backend: {backend}, CUDA: {use_cuda_passed}, WorldSize: {actual_world_size}, Batched: {batched_allreduce}")
        logger.info(f"Avg Total Time/Step: {avg_step_time:.4f} s")
        logger.info(f"Avg Comm Time/Step: {avg_comm_time:.4f} s")

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, required=True, choices=MODEL_CONFIGS.keys())
    parser.add_argument("--backend", type=str, required=True, choices=["gloo", "nccl"])
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--multinode", action="store_true", default=False)
    parser.add_argument("--batched", action="store_true", default=False, help="Enable batched (flat) gradient all-reduce")
    # Add profiling args if/when needed for problem 3.4.2 b/c
    # parser.add_argument("--enable_profiling", action="store_true", default=False)
    # parser.add_argument("--profile_output_file", type=str, default="trace.json")
    args = parser.parse_args()

    if args.multinode:
        ddp_benchmark_worker(-1, -1, args.backend, args.use_cuda, args.model_config, True, args.batched)
    else:
        assert "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ, \
            "For single-node mp.spawn, MASTER_ADDR and MASTER_PORT must be set in environment."
        mp.spawn(
            ddp_benchmark_worker,
            args=(args.world_size, args.backend, args.use_cuda, args.model_config, False, args.batched),
            nprocs=args.world_size,
            join=True,
        )

if __name__ == "__main__":
    main()