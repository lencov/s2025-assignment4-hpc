#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.distributed as dist
import logging
import os
import argparse
from datetime import timedelta
import torch.multiprocessing as mp
from copy import deepcopy
import timeit

from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.loss import cross_entropy
from cs336_systems.ddp import DDP # Import your DDP class

from torch.profiler import profile, ProfilerActivity
from contextlib import nullcontext

logging.basicConfig(
    format="%(asctime)s (%(levelname)s) [%(funcName)s:%(lineno)d] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "small": {"d_model": 768, "num_layers": 12, "num_heads": 12, "d_ff": 3072, "vocab_size": 10000, "context_length": 128, "attn_pdrop": 0.1, "residual_pdrop": 0.05},
    "medium": {"d_model": 1024, "num_layers": 24, "num_heads": 16, "d_ff": 4096, "vocab_size": 10000, "context_length": 128, "attn_pdrop": 0.1, "residual_pdrop": 0.05},
    "large": {"d_model": 1280, "num_layers": 36, "num_heads": 20, "d_ff": 5120, "vocab_size": 10000, "context_length": 128, "attn_pdrop": 0.1, "residual_pdrop": 0.05},
}
OPTIMIZER_ARGS = {"lr": 1e-4, "betas": (0.9, 0.99), "eps": 1e-9, "weight_decay": 0.1}

use_cuda_flag_from_args = False # Global for setup_distributed when using mp.spawn

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
        if backend == "nccl": # NCCL implies CUDA
            torch.cuda.set_device(int(os.environ["SLURM_LOCALID"]))
        return actual_rank, actual_world_size
    else: # Single-node
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        
        logger.info(f"Rank {rank}/{world_size} on {os.uname()[1]}: Initializing process group (single-node). Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        if use_cuda_for_setup and torch.cuda.is_available():
            target_device = rank
            if backend == "nccl" and world_size == 2: # MIG fix
                target_device = 0 if rank == 0 else 2
            logger.info(f"Rank {rank} attempting to use CUDA device {target_device}")
            torch.cuda.set_device(target_device)
        return rank, world_size

def ddp_overlap_benchmark_worker(
    rank: int,
    world_size: int,
    backend: str,
    use_cuda_passed: bool,
    model_config_name: str,
    multinode: bool,
    enable_profiling: bool,
    profile_output_file: str
):
    global use_cuda_flag_from_args
    use_cuda_flag_from_args = use_cuda_passed

    actual_rank, actual_world_size = setup_distributed(rank, world_size, backend, multinode, use_cuda_passed)
    if multinode and actual_rank is None:
        return

    DEVICE = "cuda" if use_cuda_passed and torch.cuda.is_available() else "cpu"
    if actual_rank == 0: logger.info(f"Using device: {DEVICE}, Rank: {actual_rank}, WorldSize: {actual_world_size}")

    cfg = MODEL_CONFIGS[model_config_name]
    model = TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        attn_pdrop=cfg.get("attn_pdrop"),
        residual_pdrop=cfg.get("residual_pdrop"),
        # Ensure your TransformerLM handles these or has sensible defaults
        use_layer_norm=False, # Assuming default is RMSNorm from your transformer.py
        use_triton_rmsnorm=False
    ).to(DEVICE)

    ddp_model = DDP(model)
    optimizer = AdamW(ddp_model.parameters(), **OPTIMIZER_ARGS)

    batch_size_per_gpu = 16
    local_batch_size = batch_size_per_gpu
    inputs = torch.randint(0, cfg["vocab_size"], (local_batch_size, cfg["context_length"]), device=DEVICE)
    targets = torch.randint(0, cfg["vocab_size"], (local_batch_size, cfg["context_length"]), device=DEVICE)
    
    num_warmup_steps = 5
    num_benchmark_steps = 10

    for step in range(num_warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = ddp_model(inputs)
        loss = cross_entropy(logits.view(-1, cfg["vocab_size"]), targets.view(-1))
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()

    if use_cuda_passed: torch.cuda.synchronize()
    dist.barrier()

    total_step_times = []
    
    profiler_context = nullcontext()
    if enable_profiling and actual_rank == 0:
        logger.info(f"Rank {actual_rank}: Enabling PyTorch profiler for {num_benchmark_steps} steps.")
        profiler_context = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])

    with profiler_context as prof:
        for step in range(num_benchmark_steps):
            if use_cuda_passed: torch.cuda.synchronize()
            dist.barrier()
            step_start_time = timeit.default_timer()

            optimizer.zero_grad(set_to_none=True)
            logits = ddp_model(inputs)
            loss = cross_entropy(logits.view(-1, cfg["vocab_size"]), targets.view(-1))
            loss.backward()
            ddp_model.finish_gradient_synchronization()
            optimizer.step()

            if use_cuda_passed: torch.cuda.synchronize()
            dist.barrier()
            step_time = timeit.default_timer() - step_start_time
            
            if not enable_profiling or actual_rank == 0:
                total_step_times.append(step_time)
                logger.debug(f"Rank {actual_rank} Step {step} time: {step_time:.4f}s")

    if enable_profiling and prof is not None and actual_rank == 0:
        try:
            prof.export_chrome_trace(profile_output_file)
            logger.info(f"Rank {actual_rank}: Saved profiler trace to {profile_output_file}")
        except Exception as e:
            logger.error(f"Rank {actual_rank}: Failed to export profiler trace: {e}")
    
    if len(total_step_times) > 0 and actual_rank == 0 :
        avg_step_time = sum(total_step_times) / len(total_step_times)
        logger.info(f"Model: {model_config_name}, Backend: {backend}, CUDA: {use_cuda_passed}, WorldSize: {actual_world_size}")
        logger.info(f"Avg Total Time/Step (DDP Overlap): {avg_step_time:.4f} s")

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, required=True, choices=MODEL_CONFIGS.keys())
    parser.add_argument("--backend", type=str, required=True, choices=["gloo", "nccl"])
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--multinode", action="store_true", default=False)
    parser.add_argument("--enable_profiling", action="store_true", default=False, help="Enable PyTorch Profiler")
    parser.add_argument("--profile_output_file", type=str, default="trace_overlap.json", help="Name for the output Chrome trace file")
    args = parser.parse_args()

    if args.multinode:
        ddp_overlap_benchmark_worker(
            -1, -1, # rank, world_size are placeholders, ignored in multinode setup_distributed
            args.backend, args.use_cuda, args.model_config, True,
            args.enable_profiling, args.profile_output_file
        )
    else:
        assert "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ, \
            "For single-node mp.spawn, MASTER_ADDR and MASTER_PORT must be set in environment."
        mp.spawn(
            ddp_overlap_benchmark_worker,
            args=(args.world_size, args.backend, args.use_cuda, args.model_config, False,
                  args.enable_profiling, args.profile_output_file),
            nprocs=args.world_size,
            join=True,
        )

if __name__ == "__main__":
    main()