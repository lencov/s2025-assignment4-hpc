#!/bin/bash
#SBATCH --job-name=ddp_2N_naive_${MODEL_CFG}
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1    # 1 process per node
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G              # Memory PER NODE
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1      # 1 GPU per node
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

echo "--- Job Info (2-Node Naive DDP) ---"
echo "Job ID: $SLURM_JOBID; Job Name: $SLURM_JOB_NAME"
echo "Node list: $SLURM_JOB_NODELIST; Model Cfg: ${MODEL_CFG}"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE; GPUs per node: $SLURM_GPUS_ON_NODE"
echo "---------------------------------------"

module purge
module load lang/Anaconda3/2024.02-1
module load system/CUDA/12.2.0
module list
source activate HW4

cd $SLURM_SUBMIT_DIR

# MASTER_ADDR and MASTER_PORT for multi-node (srun will use these)
# Rank 0 (SLURM_PROCID == 0) typically hosts.
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

echo "Running naive_ddp_benchmarking.py (2-node, 1-GPU/node, NCCL+CUDA)..."
srun python cs336-systems/cs336_systems/naive_ddp_benchmarking.py \
    --backend nccl \
    --use_cuda \
    --multinode \
    --world_size 2 \
    --model-config ${MODEL_CFG}
    # --world_size will be overridden by Slurm variables in multinode setup in script
    # if script's setup_multinode() is used properly.
    # Or, explicitly, script can use SLURM_NTASKS for world_size.
    # The provided naive_ddp_benchmarking.py should handle this.

echo "Job Finished."