#!/bin/bash
#SBATCH -J YZ
#SBATCH -p wzidnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#BATCH --mem=90G

module list

module purge
module load compiler/dtk/24.04

module list

make clean 
make -j

srun ./build/conv2ddemo 64 256 14 14 256 3 3 1 1 1 1