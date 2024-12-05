#!/bin/bash

#SBATCH --job-name=fw-registers
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi250:8
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err
#SBATCH --account=project_462000642
#SBATCH --partition=small-g

set -euo pipefail

source common.sh

SUBSET="$1"
PARQUET_DIR="$ROOT_DIR/$DATA_DIR/$SUBSET"
PREDICT_DIR="$ROOT_DIR/$PREDICT_DIR/$SUBSET"
LOG_DIR="$ROOT_DIR/$LOG_DIR/$SUBSET"

files="$2"

# Create logging function
log_event() {
    local file="$1"
    local event="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$timestamp - $event" >> "$LOG_DIR/${file}.log"
}

# Convert comma-separated list to array
IFS=',' read -ra file_array <<< "$files"

# Process each file
for i in "${!file_array[@]}"; do
    filename="${file_array[i]}"
    full_path="$PARQUET_DIR/$filename"
    
    # Log start time
    log_event "$filename" "START"
    
    # Launch the Python script using srun
    (
        srun \
            --ntasks=1 \
            --gres=gpu:mi250:1 \
            --mem=16G \
            python3 process_parquet_file.py \
            "$full_path" \
            "$PREDICT_DIR" \
            &
        
        exit_status=$?
        
        # Only log completion if successful
        if [ $exit_status -eq 0 ]; then
            log_event "$filename" "SUCCESS"
        else
            log_event "$filename" "FAIL: $exit_status"
        fi
    ) &
done

# Wait for all background processes to complete
wait

# Exit with success
exit 0