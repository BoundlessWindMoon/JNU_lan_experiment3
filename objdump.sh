#!/bin/bash
#SBATCH -p wzidnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=dcu:1

module purge
module load compiler/dtk/24.04

make -j TEST=y

dccobjdump --inputs=./conv2dfp16demo --architecture=gfx928 --output=./build