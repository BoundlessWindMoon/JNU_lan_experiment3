#!/bin/bash
#SBATCH -p wzidnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=dcu:1


module purge
module load compiler/dtk/24.04

make clean
make -j TEST=y

TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
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

hipprof  --pmc ./conv2dfp16demo $preliminary_1 && mv pmc_results_*  "prof/${TIMESTAMP}_pmc_results_preliminary_1.txt"
hipprof  --pmc ./conv2dfp16demo $preliminary_2 && mv pmc_results_*  "prof/${TIMESTAMP}_pmc_results_preliminary_2.txt"
hipprof  --pmc ./conv2dfp16demo $preliminary_3 && mv pmc_results_*  "prof/${TIMESTAMP}_pmc_results_preliminary_3.txt"
hipprof  --pmc ./conv2dfp16demo $preliminary_4 && mv pmc_results_*  "prof/${TIMESTAMP}_pmc_results_preliminary_4.txt"
hipprof  --pmc ./conv2dfp16demo $preliminary_5 && mv pmc_results_*  "prof/${TIMESTAMP}_pmc_results_preliminary_5.txt"
hipprof  --pmc ./conv2dfp16demo $preliminary_6 && mv pmc_results_*  "prof/${TIMESTAMP}_pmc_results_preliminary_6.txt"

hipprof  --hip-trace -o "prof/${TIMESTAMP}_hip_trace_preliminary_1.csv" -d prof ./conv2dfp16demo $preliminary_1
hipprof  --hip-trace -o "prof/${TIMESTAMP}_hip_trace_preliminary_2.csv" -d prof ./conv2dfp16demo $preliminary_2
hipprof  --hip-trace -o "prof/${TIMESTAMP}_hip_trace_preliminary_3.csv" -d prof ./conv2dfp16demo $preliminary_3
hipprof  --hip-trace -o "prof/${TIMESTAMP}_hip_trace_preliminary_4.csv" -d prof ./conv2dfp16demo $preliminary_4
hipprof  --hip-trace -o "prof/${TIMESTAMP}_hip_trace_preliminary_5.csv" -d prof ./conv2dfp16demo $preliminary_5
hipprof  --hip-trace -o "prof/${TIMESTAMP}_hip_trace_preliminary_6.csv" -d prof ./conv2dfp16demo $preliminary_6