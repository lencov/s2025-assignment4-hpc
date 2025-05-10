#!/bin/bash
#SBATCH --job-name=allreduce_bench # Base name, will be updated by --export
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4    # Max processes for WS=4 (matches 4 MIGs)
#SBATCH --cpus-per-task=2      # CPUs per process
#SBATCH --mem=32G              # Memory for the node
#SBATCH --time=00:45:00        # Time for one full run of single_node.py
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4           # Request all 4 MIG instances
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

echo "--- Job Info ---"
echo "Job ID: $SLURM_JOBID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "PyBackend: ${PY_BACKEND}, PyUseCUDAFlag: ${PY_USE_CUDA_FLAG:-}"
echo "------------------"

module purge
module load lang/Anaconda3/2024.02-1
module load system/CUDA/12.2.0
module list
source activate HW4 # Your confirmed conda environment name

cd $SLURM_SUBMIT_DIR # Change to directory where sbatch was run

export MASTER_ADDR=$(hostname)
# Generate a unique port based on Job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

echo "Running single_node.py..."
python cs336-systems/cs336_systems/single_node.py \
    --backend ${PY_BACKEND} \
    ${PY_USE_CUDA_FLAG:-} 
    # The python script's internal loops will handle world_size [2,4] and tensor_sizes

echo "Job Finished."