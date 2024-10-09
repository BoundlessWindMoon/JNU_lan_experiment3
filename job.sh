#!/bin/bash
#SBATCH -p wzidnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=dcu:1

module purge
module load compiler/dtk/24.04


make -j
preliminary_1="64 256 14 14 256 3 3 1 1 1 1"
preliminary_2="256 192 14 14 192 3 3 1 1 1 1"
preliminary_3="16 256 26 26 512 3 3 1 1 1 1"
preliminary_4="32 256 14 14 256 3 3 1 1 1 1"
preliminary_5="2 1280 16 16 1280 3 3 1 1 1 1"
preliminary_6="2 960 64 64 32 3 3 1 1 1 1"

final_1="16 128 64 64 27 3 3 1 1 1 1"
final_2="16 256 32 32 256 3 3 1 1 1 1"
final_3="16 64 128 128 64 3 3 1 1 1 1"
final_4="2 1920 32 32 640 3 3 1 1 1 1"
final_5="2 640 64 64 640 3 3 1 1 1 1"
final_6="2 320 64 64 4 3 3 1 1 1 1"

./conv2dfp16demo $preliminary_1
./conv2dfp16demo $preliminary_2
./conv2dfp16demo $preliminary_3
./conv2dfp16demo $preliminary_4
./conv2dfp16demo $preliminary_5
./conv2dfp16demo $preliminary_6