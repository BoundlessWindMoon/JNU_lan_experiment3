#!/bin/bash
#SBATCH -p wzidnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1

module purge
module load compiler/dtk/24.04

make -j

dccobjdump --inputs=./build/conv2ddemo --output=./assembly