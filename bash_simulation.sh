#!/bin/bash
# SLURM job array script to run multiple Python scripts with uv

#SBATCH --job-name=benchmark
#SBATCH --partition=cpu_long
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --array=1-10
#SBATCH --output=logs/uv_%A_%a.out
#SBATCH --error=logs/uv_%A_%a.err
#SBATCH --mail-type=FAIL --mail-type=END --mail-type=ARRAY_TASKS


# Load modules or activate environment if needed
# module load python/3.10
# source ~/venvs/uvenv/bin/activate

# Define the list of scripts
scripts=(
  "perfect_foresight.py"
  "stochastic_planner.py"
  "deterministic_planner.py"
)

# Define common parameters
# Define common parameters
params=(
  --name "simulation_name"
  --nombres_simu 64
  --carbon_tax n # Default value No
  --coal_phase_out n # Default value No
  --kgbound 50000
  --step_g 500
  --kwbound 70000
  --ksbound 70600
  --step_s 1000
  --step_w 1000
  --kappa_weight 0 # Default value 0
  --nu_weight 0 # Default value 0
  --lambda_weight 0 # Default value 0
)

# Select the script based on SLURM_ARRAY_TASK_ID
script=${scripts[$SLURM_ARRAY_TASK_ID-1]}

echo "Running task $SLURM_ARRAY_TASK_ID: uv run $script ${params[*]}"

# Execute the command
srun uv run "$script" "${params[@]}"
