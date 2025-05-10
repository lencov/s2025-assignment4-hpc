#!/bin/bash
#SBATCH --job-name=test_ddp_overlap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2    # Test with 2 processes
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2           # Request 2 GPU instances for the test
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

echo "--- Job Info (DDP Overlap Pytest) ---"
echo "Job ID: $SLURM_JOBID; Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "---------------------------------------"

module purge
module load lang/Anaconda3/2024.02-1
module load system/CUDA/12.2.0 # Your CUDA module
module list
source activate HW4 # Your conda environment name

cd $SLURM_SUBMIT_DIR # Change to directory where sbatch was run

# MASTER_ADDR and MASTER_PORT are needed for torch.multiprocessing.spawn
# which is likely used by the test utilities or adapters.
export MASTER_ADDR=$(hostname)
# Using a unique port for the test run
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4) + $SLURM_PROCID) 
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

echo "Running pytest for DDP overlap..."
export PYTHONUNBUFFERED=1 # For live output
python -m pytest cs336-systems/tests/test_ddp_individual_parameters.py -v -s

echo "Pytest Finished."