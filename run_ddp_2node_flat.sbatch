#!/bin/bash
#SBATCH --job-name=ddp_2N_flat_${MODEL_CFG}
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1    # 1 process per node
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G              # Memory PER NODE
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1      # 1 GPU per node
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

echo "--- Job Info (2-Node Flat DDP) ---"
echo "Job ID: $SLURM_JOBID; Job Name: $SLURM_JOB_NAME"
echo "Node list: $SLURM_JOB_NODELIST; Model Cfg: ${MODEL_CFG}"
echo "------------------------------------"

module purge
module load lang/Anaconda3/2024.02-1
module load system/CUDA/12.2.0
module list
source activate HW4

cd $SLURM_SUBMIT_DIR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

echo "Running naive_ddp_benchmarking.py with --batched (2-node, 1-GPU/node, NCCL+CUDA)..."
srun python cs336-systems/cs336_systems/naive_ddp_benchmarking.py \
    --backend nccl \
    --use_cuda \
    --multinode \
    --world_size 2 \
    --model-config ${MODEL_CFG} \
    --batched # Key change: added this flag

echo "Job Finished."