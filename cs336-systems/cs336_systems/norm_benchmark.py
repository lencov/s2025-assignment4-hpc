from cs336_basics.model import RMSNorm
from cs336_systems.rms import RMSNormTriton
import torch
import time


ITERS = 1000
ROWS = 50000
COLS = [1024, 2048, 4096, 8192]


def main(run_backward: bool = False):
    for col in COLS:
        data = torch.randn(ROWS, col).to("cuda")
        result = torch.randn(ROWS, col).to("cuda")
        layer_norm = torch.nn.LayerNorm(col).to("cuda")
        rms_norm = RMSNorm(col).to("cuda")

        # warmup
        for _ in range(10):
            layer_norm(data)
            torch.cuda.synchronize()
        forward = []
        backward = []
        for _ in range(ITERS):
            start = time.time()
            output = layer_norm(data)
            torch.cuda.synchronize()
            forward.append(time.time() - start)
            if run_backward:
                start = time.time()
                output.backward(result)
                torch.cuda.synchronize()
                backward.append(time.time() - start)
                data.grad = None
        print(f"LayerNorm: {col} cols took {sum(forward) / ITERS} seconds")
        if run_backward:
            print(
                f"LayerNorm backward: {col} cols took {sum(backward) / ITERS} seconds"
            )
            print(
                f"Combined average time for LayerNorm (forward + backward): {sum([f+b for f, b in zip(forward, backward)]) / ITERS} seconds"
            )

        # warmup
        for _ in range(10):
            rms_norm(data)
            torch.cuda.synchronize()
        forward_rms = []
        backward_rms = []
        for _ in range(ITERS):
            start_rms = time.time()
            output = rms_norm(data)
            torch.cuda.synchronize()
            forward_rms.append(time.time() - start_rms)
            if run_backward:
                start_rms = time.time()
                output.backward(result)
                torch.cuda.synchronize()
                backward_rms.append(time.time() - start_rms)
                data.grad = None
        print(f"RMSNorm: {col} cols took {sum(forward_rms) / ITERS} seconds")
        if run_backward:
            print(
                f"RMSNorm backward: {col} cols took {sum(backward_rms) / ITERS} seconds"
            )
            print(
                f"Combined average time for RMSNorm (forward + backward): {sum([f+b for f, b in zip(forward_rms, backward_rms)]) / ITERS} seconds"
            )

        triton_norm = RMSNormTriton(col).to("cuda")
        # warmup
        for _ in range(10):
            triton_norm(data)
            torch.cuda.synchronize()
        forward_triton = []
        backward_triton = []
        for _ in range(ITERS):
            start_triton = time.time()
            output = triton_norm(data)
            torch.cuda.synchronize()
            forward_triton.append(time.time() - start_triton)
            if run_backward:
                start_triton = time.time()
                output.backward(result)
                torch.cuda.synchronize()
                backward_triton.append(time.time() - start_triton)
                data.grad = None
        print(f"RMSNormTriton: {col} cols took {sum(forward_triton) / ITERS} seconds")
        if run_backward:
            print(
                f"RMSNormTriton backward: {col} cols took {sum(backward_triton) / ITERS} seconds"
            )
            print(
                f"Combined average time for RMSNormTriton (forward + backward): {sum([f+b for f, b in zip(forward_triton, backward_triton)]) / ITERS} seconds"
            )

        compiled_triton_norm = torch.compile(rms_norm)
        # warmup
        for _ in range(10):
            compiled_triton_norm(data)
            torch.cuda.synchronize()
        forward_compiled = []
        backward_compiled = []
        for _ in range(ITERS):
            start_compiled = time.time()
            output = compiled_triton_norm(data)
            torch.cuda.synchronize()
            forward_compiled.append(time.time() - start_compiled)
            if run_backward:
                start_compiled = time.time()
                output.backward(result)
                torch.cuda.synchronize()
                backward_compiled.append(time.time() - start_compiled)
                data.grad = None
        print(
            f"Compiled RMSNorm: {col} cols took {sum(forward_compiled) / ITERS} seconds"
        )
        if run_backward:
            print(
                f"Compiled RMSNorm backward: {col} cols took {sum(backward_compiled) / ITERS} seconds"
            )
            print(
                f"Combined average time for Compiled RMSNorm (forward + backward): {sum([f+b for f, b in zip(forward_compiled, backward_compiled)]) / ITERS} seconds"
            )


if __name__ == "__main__":
    main(run_backward=True)