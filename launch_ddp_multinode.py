import subprocess
import argparse
import tempfile
import os
import time

job_script = """#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node={nprocs}
#SBATCH --nodes={nodes}
#SBATCH --mem=50G
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
srun python cs336-systems/cs336_systems/naive_ddp_benchmarking.py --backend nccl --multinode --world_size {world_size} --model-config {model_config} --ddp-bucket
"""


def launch_slurm_jobs():

    procs_per_node = 1
    nodes = 2
    for model_cfg in ["small", "medium", "large", "xl", "2.7B"]:
        name = f"ddp_non_sharded_benchmark_{model_cfg}"
        with open("tmp.sh", "w+") as tmpfile:
            job_script_formatted = job_script.format(nprocs=procs_per_node, nodes=nodes, world_size=procs_per_node * nodes, model_config=model_cfg, name=name)
            tmpfile.write(job_script_formatted)

        cmd = [
            "sbatch",
            f"--partition=a2",
            "tmp.sh"
        ]
        print(f"Launching job with command: {' '.join(cmd)} for model cfg {model_cfg} and {procs_per_node} processes per node.")
        subprocess.run(cmd)
        time.sleep(1)
        #os.unlink("tmp.sh")  # Clean up the temporary file after submitting the job

if __name__ == "__main__":
    launch_slurm_jobs()