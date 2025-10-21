#!/bin/bash -x
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4#!/bin/bash -x
#SBATCH -N 1                           # Number of nodes (increase if you want multi-node)
#SBATCH --partition=gpu                # GPU partition
#SBATCH --ntasks-per-node=2            # Number of tasks per node (number of CPU tasks)
#SBATCH --cpus-per-task=24             # Number of CPU cores per task matching CPU cores for performance
#SBATCH --gres=gpu:2                   # Request 2 GPUs per node (max 2 per GPU node)
#SBATCH --mem=80G                     # Memory per node (match GPU node memory or your requirement)
#SBATCH --job-name=NN_training         # Job name
#SBATCH --time=00:01:00              # Walltime (2 days, adjust if you want max 4 days)
#SBATCH --output=%j.out                # Stdout file
#SBATCH --error=%j.err                 # Stderr file
#SBATCH --exclusive                    # Exclusive access to nodes for max performance


cd IRS


module purge
module load mldl_modules/miniconda


source $(conda info --base)/etc/profile.d/conda.sh
conda activate NN_training
