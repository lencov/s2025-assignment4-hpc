import subprocess
import argparse
import tempfile
import os
import time

job_script = """#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node={nprocs}
#SBATCH --nodes=2
#SBATCH --mem=8G
#SBATCH --time=00:02:00
#SBATCH --gpus-per-node={nprocs}
#SBATCH --output=sbatch/{name}.out
#SBATCH --error=sbatch/{name}.err

eval "$(conda shell.bash hook)"
# Change conda environment name, if necessary
conda activate cs336_systems

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_PORT: ${{MASTER_PORT}}"
echo "MASTER_ADDR: ${{MASTER_ADDR}}"

# Execute command for each task
srun python cs336-systems/cs336_systems/multi_node.py --backend {backend} --tensor_size {tensor_size} {use_cuda}
"""

TENSOR_SIZES = {
    "512KB": int(512e3 / 4),
    "1MB": int(1e6 / 4),
    "10MB": int(10e6 / 4),
    "50MB": int(50e6 / 4),
    "100MB": int(100e6 / 4),
    "500MB": int(500e6 / 4),
    "1GB": int(1e9 / 4),
}

def launch_slurm_jobs(backend, use_cuda, processes_per_node):

    use_cuda_flag = "--use_cuda" if use_cuda else ""

    for tensor_size in TENSOR_SIZES:
        for nprocs in processes_per_node:
            name = f"{backend}_{tensor_size}_{nprocs}_{use_cuda}"
            with open("tmp.sh", "w+") as tmpfile:
                job_script_formatted = job_script.format(nprocs=nprocs, backend=backend, tensor_size=tensor_size, use_cuda=use_cuda_flag, name=name)
                tmpfile.write(job_script_formatted)

            cmd = [
                "sbatch",
                f"--partition=a2",
                "tmp.sh"
            ]
            print(f"Launching job with command: {' '.join(cmd)} for tensor size {tensor_size} and {nprocs} processes per node.")
            subprocess.run(cmd)
            time.sleep(1)
            #os.unlink("tmp.sh")  # Clean up the temporary file after submitting the job

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch SLURM jobs for distributed training.")
    parser.add_argument("--backend", type=str, required=True, help="Backend for distributed training (gloo, nccl)")
    parser.add_argument("--use_cuda", action='store_true', help="Flag to use CUDA for distributed training")
    parser.add_argument("--processes_per_node", type=int, required=True)
    args = parser.parse_args()

    assert args.backend in ["gloo", "nccl"], "Backend must be one of 'gloo' or 'nccl'"
    assert args.processes_per_node in [1, 2, 3], "Processes per node must be one of 1, 2, or 3"
    processes_per_node = [args.processes_per_node]
    
    launch_slurm_jobs(args.backend, args.use_cuda, processes_per_node)