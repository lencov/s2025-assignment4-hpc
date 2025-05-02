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
    backend: str = str, rank: int = None, world_size: int = None
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
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


def single_process_reference(
    backend: str,
    reference_model: nn.Module,
    world_size: int,
    data: torch.Tensor,
    labels: torch.Tensor,
):
    DEVICE = "cuda" if backend == "nccl" else "cpu"

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

        logger.debug(f"Reference: step = {step}, loss = {loss.item()}")

    return reference_model


def ddp_main(
    rank: int,
    world_size: int,
    backend: str,
    data: torch.Tensor,
    labels: torch.Tensor,
    out_features: int,
    reference: bool = True,
):
    if rank == -1:
        rank, _, world_size, _ = setup_multinode(backend)
    else:
        setup_singlenode(backend, rank, world_size)

    DEVICE = "cuda" if backend == "nccl" else "cpu"

    torch.manual_seed(rank)
    partition = data.size(0) // world_size
    in_features = data.size(1)
    start_index = rank * partition
    end_index = start_index + partition
    data = data.to(DEVICE)
    labels = labels.to(DEVICE)

    model = ToyModel(in_features, out_features).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Broadcast rank 0 model
    for param in model.parameters():
        dist.broadcast(param.data, 0, async_op=False)

    reference_model = deepcopy(model)
    ref_optimizer = torch.optim.SGD(reference_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for step in range(5):
        out = model(data[start_index:end_index])
        loss = loss_fn(out, labels[start_index:end_index])
        loss.backward()

        for param in model.parameters():
            if not param.requires_grad:
                continue
            if backend == "nccl":
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
            else:
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM, async_op=False)
                param.grad /= world_size

        optimizer.step()
        optimizer.zero_grad()

        if rank == 0:
            dist.barrier()
            ref_out = reference_model(data)
            ref_loss = loss_fn(ref_out, labels)
            ref_loss.backward()
            ref_optimizer.step()
            ref_optimizer.zero_grad()
            for param, reference_param in zip(
                model.parameters(), reference_model.parameters()
            ):
                assert torch.allclose(param, reference_param)
            logger.info("Model parameters are equal to reference model parameters")

        logger.debug(f"Rank {rank}: step = {step}, loss = {loss.item()}")

    dist.barrier()

    validate_ddp_net_equivalence(model, rank)
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True)
    parser.add_argument("--multinode", action="store_true", default=False)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    # Generate data
    samples = 1024
    num_dim = 10
    out_features = 5
    torch.manual_seed(0)
    data = torch.randn(samples, num_dim)
    labels = torch.randn(samples, out_features)

    if args.multinode:
        ddp_main(-1, -1, args.backend, data, labels, out_features)
    else:
        mp.spawn(
            ddp_main,
            args=(args.world_size, args.backend, data, labels, out_features),
            nprocs=args.world_size,
            join=True,
        )


if __name__ == "__main__":
    main()