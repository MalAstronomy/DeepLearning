#!/bin/bash
#
#SBATCH --job-name=Eagle
#SBATCH --output=./slurm-%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=10G
##SBATCH --exclude=alan-compute-[01]
##SBATCH --nodelist=alan-compute-04
#SBATCH --time="5-00:00:00"

#source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
python kde1.py 
