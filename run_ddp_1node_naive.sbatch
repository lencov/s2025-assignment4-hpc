#!/bin/bash
#SBATCH --job-name=ddp_1N_naive_${MODEL_CFG}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2    # 2 processes for 2 GPUs
#SBATCH --cpus-per-task=4      # CPUs per process
#SBATCH --mem=64G              # Memory per node (adjust if model needs more/less)
#SBATCH --time=00:30:00        # Adjust as needed per model size
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2           # Request 2 GPU instances
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

echo "--- Job Info (1-Node Naive DDP) ---"
echo "Job ID: $SLURM_JOBID; Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname); Model Cfg: ${MODEL_CFG}"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "---------------------------------------"

module purge
module load lang/Anaconda3/2024.02-1
module load system/CUDA/12.2.0
module list
source activate HW4

cd $SLURM_SUBMIT_DIR

# Set MASTER_ADDR and MASTER_PORT for mp.spawn used by the script
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

echo "Running naive_ddp_benchmarking.py (1-node, 2-GPU, Gloo+CUDA)..."
python cs336-systems/cs336_systems/naive_ddp_benchmarking.py \
    --backend gloo \
    --use_cuda \
    --world_size 2 \
    --model-config ${MODEL_CFG}
    # Note: --multinode is NOT set, so script uses mp.spawn

echo "Job Finished."