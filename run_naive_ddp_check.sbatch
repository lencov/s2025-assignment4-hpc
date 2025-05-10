#!/bin/bash
#SBATCH --job-name=naive_ddp_check_gloo_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2    # For world_size=2 (CPU processes)
#SBATCH --cpus-per-task=1      # 1 CPU per process is fine for this toy model
#SBATCH --mem=8G               # Modest memory for the toy model
#SBATCH --time=00:10:00        # Should be very quick
#SBATCH --partition=shared     # Can use 'shared' or 'gpu' partition. 'shared' might be faster if no GPUs needed.
#SBATCH --gres=gpu:0           # Explicitly request 0 GPUs for CPU run
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

echo "--- Job Info ---"
echo "Job ID: $SLURM_JOBID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "------------------"

module purge
module load lang/Anaconda3/2024.02-1
module load system/CUDA/12.2.0 
module list
source activate HW4 

cd $SLURM_SUBMIT_DIR

# Set MASTER_ADDR and MASTER_PORT for the DDP processes
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

echo "Running naive_ddp.py check with Backend: gloo, WorldSize: 2 (CPU)"
python cs336-systems/cs336_systems/naive_ddp.py \
    --backend gloo \
    --world_size 2
    # --use_cuda is not passed, so it will default to False, running on CPU

echo "Job Finished."