import torch
import torch.nn as nn
import torch.distributed as dist
import logging
import os
import argparse
from datetime import timedelta
import torch.multiprocessing as mp
from copy import deepcopy

# setup logging
logging.basicConfig(format="%(asctime)s (%(levelname)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global to pass use_cuda to setup_singlenode when using mp.spawn
# This is a common workaround for mp.spawn's argument serialization limits
_global_use_cuda_for_spawn = False

def validate_ddp_net_equivalence(net: nn.Module, rank: int):
    # Helper to validate synchronization of nets across ranks.
    net_module_states = list(net.state_dict().values())
    # Check that all tensors in module's state_dict() are equal.
    for t in net_module_states:
        tensor_list = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, t)
        for tensor_item in tensor_list: # Renamed to avoid conflict
            assert torch.allclose(tensor_item, t)
    if rank == 0:
        logger.info("All parameters are equal across all ranks")


class _FC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 50, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        return x


class ToyModel(nn.Module):
    def __init__(self, in_features: int = 10, out_features: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.fc2 = _FC2()
        self.fc3 = nn.Linear(50, out_features, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_fixed_param = nn.Parameter(
            torch.tensor([2.0, 2.0]), requires_grad=False
        )

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def setup_singlenode(
    backend: str, rank: int, world_size: int
) -> None:
    global _global_use_cuda_for_spawn # Use the global flag

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "12356") # Default for this script

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    if rank == 0:
        logger.debug(f"Rank {rank}: Initializing process group. Master: {master_addr}:{master_port}, World: {world_size}, Backend: {backend}")

    dist.init_process_group(backend, rank=rank, world_size=world_size)

    if _global_use_cuda_for_spawn and torch.cuda.is_available(): # Check flag
        target_device = rank # Default for Gloo+CUDA or NCCL with WS > 2 or non-MIG
        if backend == "nccl": # Specific MIG handling for NCCL
            # Assuming 4 MIG instances are visible as devices 0,1,2,3
            # And 2 physical GPUs are underlying these.
            if world_size == 2:
                target_device = 0 if rank == 0 else 2 # Map to MIGs on different physical GPUs
                logger.debug(f"Rank {rank} (NCCL, WS=2, MIG): Mapping to CUDA device {target_device}")
            elif world_size == 4: # If using all 4 MIGs on one node
                 target_device = rank
                 logger.debug(f"Rank {rank} (NCCL, WS=4, MIG): Mapping to CUDA device {target_device}")
            # Add other world_size specific mappings if needed, or default to rank
        
        if rank < torch.cuda.device_count():
             torch.cuda.set_device(target_device)
             if rank == 0: logger.debug(f"Rank {rank} (backend {backend}) set to CUDA device {target_device}")
        else:
             if rank == 0: logger.error(f"Rank {rank} wants device {target_device} but only {torch.cuda.device_count()} devices available.")


def setup_multinode(backend: str, use_cuda: bool): # Added use_cuda
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
    if use_cuda and backend == "nccl": # Check use_cuda
        torch.cuda.set_device(int(os.environ["SLURM_LOCALID"]))
    return actual_rank, actual_world_size, int(os.environ["SLURM_LOCALID"]), int(os.environ["SLURM_NTASKS_PER_NODE"])


def cleanup():
    dist.destroy_process_group()


def single_process_reference( # Added use_cuda argument
    use_cuda: bool, # Explicitly pass use_cuda
    reference_model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
):
    DEVICE = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    logger.info(f"Reference model running on: {DEVICE}")

    reference_model = reference_model.to(DEVICE)
    data = data.to(DEVICE)
    labels = labels.to(DEVICE)

    optimizer = torch.optim.SGD(reference_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for step in range(5):
        out = reference_model(data).squeeze()
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step == 0 and rank == 0: # Log only once from rank 0
             logger.debug(f"Reference: step = {step}, loss = {loss.item()}")
    return reference_model


def ddp_main(
    rank: int,
    world_size: int,
    backend: str,
    use_cuda_arg: bool, # Renamed to avoid conflict with global
    data: torch.Tensor,
    labels: torch.Tensor,
    out_features: int,
    multinode: bool # Added multinode flag
):
    global _global_use_cuda_for_spawn # Set the global for setup_singlenode
    _global_use_cuda_for_spawn = use_cuda_arg

    actual_rank = rank
    actual_world_size = world_size

    if multinode: # rank == -1 was for a different script structure
        actual_rank, actual_world_size, _, _ = setup_multinode(backend, use_cuda_arg)
    else:
        setup_singlenode(backend, actual_rank, actual_world_size)

    DEVICE = "cuda" if use_cuda_arg and torch.cuda.is_available() else "cpu"
    if actual_rank == 0: logger.info(f"DDP main using device: {DEVICE}")

    torch.manual_seed(actual_rank) # Seed for consistent data sharding if data is generated per rank
    
    # Data sharding
    num_samples_total = data.size(0)
    samples_per_rank = num_samples_total // actual_world_size
    start_index = actual_rank * samples_per_rank
    end_index = start_index + samples_per_rank
    
    local_data = data[start_index:end_index].to(DEVICE)
    local_labels = labels[start_index:end_index].to(DEVICE)
    in_features = data.size(1)

    model = ToyModel(in_features, out_features).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Broadcast rank 0 model parameters
    for param in model.parameters():
        dist.broadcast(param.data, 0, async_op=False)

    # Single process reference model (only rank 0 needs to run this for comparison)
    # But all ranks need a consistent reference_model state if asserts are done by all
    reference_model_for_assert = deepcopy(model) # All ranks have same initial model
    
    loss_fn = nn.MSELoss()

    for step in range(5):
        out = model(local_data) # Use local shard
        loss = loss_fn(out, local_labels) # Use local shard
        loss.backward()

        for param in model.parameters():
            if param.grad is None: # Check if grad exists
                continue
            if backend == "nccl" and use_cuda_arg:
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
            else: # Gloo (CPU or GPU)
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM, async_op=False)
                param.grad /= actual_world_size

        optimizer.step()
        optimizer.zero_grad(set_to_none=True) # Use set_to_none=True

        if actual_rank == 0:
            # Run reference step on rank 0 for comparison
            # Create a fresh reference model for each step to ensure isolated updates
            # Or, more simply, just compare final models.
            # For this test, the original script compared params at each step.
            # We need a consistent reference_model across all ranks if all ranks assert.
            # Let's simplify: only rank 0 runs the reference and asserts.
            # For a more robust test, all ranks should compare against a broadcasted reference.
            # For now, we'll stick to the original script's assert logic on rank 0.

            # Create a temporary reference model and optimizer for this step's comparison
            temp_ref_model = deepcopy(reference_model_for_assert) # Use the consistent initial state
            temp_ref_optimizer = torch.optim.SGD(temp_ref_model.parameters(), lr=1e-3)
            
            # Simulate full batch for reference on rank 0
            ref_out = temp_ref_model(data.to(DEVICE)) # Full data to device
            ref_loss = loss_fn(ref_out, labels.to(DEVICE)) # Full labels to device
            ref_loss.backward()
            temp_ref_optimizer.step()
            # Update the persistent reference model for the next step's comparison basis
            reference_model_for_assert.load_state_dict(temp_ref_model.state_dict())


            for p_ddp, p_ref in zip(model.parameters(), reference_model_for_assert.parameters()):
                assert torch.allclose(p_ddp.data, p_ref.data, atol=1e-5), \
                    f"Step {step}: DDP param {p_ddp.data} != Ref param {p_ref.data}"
            if step == 4 : # Log only at the end
                logger.info("Model parameters are equal to reference model parameters after all steps.")
        
        if actual_rank == 0: logger.debug(f"Rank {actual_rank}: step = {step}, DDP loss = {loss.item()}")
        dist.barrier() # Ensure all ranks complete step before next

    dist.barrier()
    validate_ddp_net_equivalence(model, actual_rank)
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True, choices=["gloo", "nccl"])
    parser.add_argument("--multinode", action="store_true", default=False)
    parser.add_argument("--world_size", type=int, default=1, help="Number of processes for single-node, or total for multi-node if SLURM not used")
    parser.add_argument("--use_cuda", action="store_true", default=False, help="Flag to run on CUDA if available") # Added this
    args = parser.parse_args()

    samples = 1024
    num_dim = 10
    out_features = 5
    torch.manual_seed(0) # Global seed for consistent data generation
    data = torch.randn(samples, num_dim)
    labels = torch.randn(samples, out_features)

    if args.multinode:
        # In multinode, srun launches script on each process. rank/world_size from SLURM env.
        ddp_main(-1, -1, args.backend, args.use_cuda, data, labels, out_features, True)
    else:
        # For single-node, mp.spawn launches processes.
        # MASTER_ADDR and MASTER_PORT must be set in the environment.
        assert "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ, \
            "For single-node mp.spawn, MASTER_ADDR and MASTER_PORT must be set in environment."
        mp.spawn(
            ddp_main,
            args=(args.world_size, args.backend, args.use_cuda, data, labels, out_features, False), # Pass multinode=False
            nprocs=args.world_size,
            join=True,
        )

if __name__ == "__main__":
    main()