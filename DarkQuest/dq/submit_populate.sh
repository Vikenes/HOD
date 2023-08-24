#!/bin/bash -l
#
# Batch script for bash users
#
#SBATCH -n 16 
#SBATCH -J pop 
#SBATCH -o logs/pop_%J.dump
#SBATCH -e logs/pop_%J.err
#SBATCH -p cosma6
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 80 ## minutes run time


python create_xi_dataset.py #--num_threads $SLURM_NTASKS

