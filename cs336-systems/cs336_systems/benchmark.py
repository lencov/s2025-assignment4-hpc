import torch
import timeit
import logging
import argparse
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple
from cs336_basics.optimizer import AdamW
from cs336_basics.loss import cross_entropy
from cs336_basics.transformer import TransformerLM
from torch.profiler import profile, record_function, ProfilerActivity


# setup logging
logging.basicConfig(format="%(asctime)s (%(levelname)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEVICE = "cuda:0"


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


def run_step(
    model: TransformerLM,
    inputs: torch.Tensor,
    optimizer: AdamW,
    enable_backward: bool,
    mixed_precision: bool = False,
    profile: bool = True,
) -> Tuple[float, float, float]:
    forward_time, backward_time, optimizer_time = 0.0, 0.0, 0.0
    with record_function("forward_pass") if profile else nullcontext():
        with torch.autocast(device_type="cuda") if mixed_precision else nullcontext():
            start = timeit.default_timer()
            out = model(inputs)
            torch.cuda.synchronize()
            forward_time = timeit.default_timer() - start
    if enable_backward:
        with record_function("backward_pass") if profile else nullcontext():
            with (
                torch.autocast(device_type="cuda") if mixed_precision else nullcontext()
            ):
                start = timeit.default_timer()
                loss = cross_entropy(out, inputs)
            loss.backward()
            torch.cuda.synchronize()
            backward_time = timeit.default_timer() - start
        with record_function("optimizer") if profile else nullcontext():
            start = timeit.default_timer()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            optimizer_time = timeit.default_timer() - start
    return forward_time, backward_time, optimizer_time


def main(
    model_args: ModelArgs,
    trainer_args: TrainerArgs,
    optimizer_args: OptimizerArgs,
    do_profile: bool = True,
    profile_memory: bool = False,
):
    logger.info(f"Do profile: {do_profile}, profile memory: {profile_memory}")
    assert not (
        not do_profile and profile_memory
    ), "Cannot profile memory without profiling"

    model = TransformerLM(
        vocab_size=model_args.vocab_size,
        context_length=model_args.context_length,
        d_model=model_args.d_model,
        num_layers=model_args.num_layers,
        num_heads=model_args.num_heads,
        d_ff=model_args.d_ff,
        attn_pdrop=model_args.attn_pdrop,
        residual_pdrop=model_args.residual_pdrop,
        use_layer_norm=model_args.use_layer_norm, # Make sure your TransformerLM uses these
        use_triton_rmsnorm=model_args.use_triton_rmsnorm,
    ).to(DEVICE)

    if trainer_args.compile:
        model = torch.compile(model)
    optimizer = AdamW(
        model.parameters(),
        lr=optimizer_args.lr,
        betas=optimizer_args.betas,
        eps=optimizer_args.eps,
        weight_decay=optimizer_args.weight_decay,
    )
    model.train()
    dummy_data = torch.randint(
        0, model_args.vocab_size, (trainer_args.batch_size, model_args.context_length)
    ).to(DEVICE)

    for _ in range(trainer_args.warmup_steps):
        run_step(
            model,
            dummy_data,
            AdamW(model.parameters()), # Use separate optimizer for warmup state
            trainer_args.run_backward,
            mixed_precision=trainer_args.mixed_precision,
            profile=False,
        )
    torch.cuda.synchronize()

    profiler_context = profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # Use schedule=None when profiling memory
            schedule=None if profile_memory else (
                torch.profiler.schedule(
                    wait=0, warmup=0, active=trainer_args.train_steps, repeat=1
                )
            ),
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
            record_shapes=True,
            profile_memory=profile_memory,
            with_stack=True,
        ) if do_profile else nullcontext()

    logger.info("Starting profiling loop...")
    with profiler_context as prof:
        forward_times = []
        backward_times = []
        optimizer_times = []
        for i in tqdm(range(trainer_args.train_steps)):
            # prof.step() is not called when schedule is None
            f, b, o = run_step(
                model,
                dummy_data,
                optimizer,
                trainer_args.run_backward,
                mixed_precision=trainer_args.mixed_precision,
                profile=do_profile,
            )
            forward_times.append(f)
            backward_times.append(b)
            optimizer_times.append(o)

    torch.cuda.synchronize()
    logger.info("Profiling loop finished.")

    if do_profile and prof:
        logger.info("Exporting profiler results...")
        try:
             prof.export_stacks("lm_profiler_stacks.txt", "self_cuda_time_total")
             logger.info("Exported profiler stacks.")
             print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
        except Exception as e:
             logger.error(f"Failed to export/print stacks/table: {e}")

        if profile_memory:
            try:
                logger.info("Attempting to export memory timeline HTML...")
                prof.export_memory_timeline(
                    f"timeline-{model_args.name}-run-backward-{trainer_args.run_backward}.html",
                    device=DEVICE,
                )
                logger.info("Exported memory timeline HTML successfully.")
            except ValueError as e:
                 logger.error(f"Failed to export memory timeline HTML: {e}. Likely no memory events recorded by timeline mechanism.")
            except Exception as e:
                logger.error(f"An unexpected error occurred during HTML export: {e}")

    if profile_memory: # Keep the pickle dump separate
        try:
             logger.info("Attempting to dump memory snapshot pickle...")
             if "cuda" in DEVICE:
                 torch.cuda.synchronize(device=DEVICE)
             torch.cuda.memory._dump_snapshot(
                 f"memory_snapshot-{model_args.name}-run-backward-{trainer_args.run_backward}.pickle"
             )
             logger.info("Dumped memory snapshot pickle successfully.")
        except Exception as e:
            logger.error(f"Failed to dump memory snapshot pickle: {e}")

    print(f"Forward time: {np.mean(forward_times):.4f} s (std: {np.std(forward_times):.4f} s)")
    print(f"Backward time: {np.mean(backward_times):.4f} s (std: {np.std(backward_times):.4f} s)")
    print(f"Optimizer time: {np.mean(optimizer_times):.4f} s (std: {np.std(optimizer_times):.4f} s)")

# Keep the if __name__ == "__main__": block as it was
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        type=str,
        default="small",
        choices=list(MODEL_CONFIGS.keys()),
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=5)
    parser.add_argument("--run-backward", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)
    parser.add_argument("--use-triton-rmsnorm", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--no-profile", action="store_true", default=False)
    parser.add_argument("--profile-memory", action="store_true", default=False)
    args = parser.parse_args()
    model_args = MODEL_CONFIGS[args.model_config]
    model_args.use_layer_norm = args.use_layer_norm
    model_args.use_triton_rmsnorm = args.use_triton_rmsnorm
    model_args.name = args.model_config
    logger.info(
        f"Running benchmark with model config: {args.model_config}\n{model_args}"
    )
    trainer_args = TrainerArgs(
        warmup_steps=args.warmup_steps,
        train_steps=args.train_steps,
        run_backward=args.run_backward,
        mixed_precision=args.mixed_precision,
        compile=args.compile,
    )
    optimizer_args = OptimizerArgs()
    logger.info(f"Trainer args: {trainer_args}")
    main(
        model_args,
        trainer_args,
        optimizer_args,
        do_profile=not args.no_profile,
        profile_memory=args.profile_memory,
    )
