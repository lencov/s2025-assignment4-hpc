#!/bin/bash
#SBATCH --job-name=ddp_1N_naive_prof_${MODEL_CFG}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2 
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G 
#SBATCH --time=00:20:00 # Profiling can be shorter
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

echo "--- Job Info (Naive DDP Profiling) ---"
echo "Job ID: $SLURM_JOBID; Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname); Model Cfg: ${MODEL_CFG}"
echo "---------------------------------------"

module purge
module load lang/Anaconda3/2024.02-1
module load system/CUDA/12.2.0 # Your CUDA module
module list
source activate HW4 # Your conda environment name

cd $SLURM_SUBMIT_DIR

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Running naive_ddp_benchmarking.py with --batched --enable_profiling for Model: ${MODEL_CFG}"
python cs336-systems/cs336_systems/naive_ddp_benchmarking.py \
    --backend gloo --use_cuda \
    --world_size 2 \
    --model-config ${MODEL_CFG} \
    --batched \
    --enable_profiling \
    --profile_output_file "trace_naive_batched_${MODEL_CFG}.json"
    # Ensure naive_ddp_benchmarking.py is modified to handle these profiling args
    
echo "Job Finished."