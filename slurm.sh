#!/bin/bash

# Invoke predict.py on data preprocessed by prepare.sh

#SBATCH --job-name=hplt-registers
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi250:8
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err
#SBATCH --account=project_462000353
#SBATCH --partition=small-g

# If run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    sbatch "$0" "$@"
    exit
fi

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 DUMP" >&2
    echo >&2
    echo "example: $0 fineweb/CC-MAIN-2024-18" >&2
    exit 1
fi
