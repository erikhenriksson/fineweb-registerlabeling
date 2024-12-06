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

#SCRIPT_DIR=$(dirname "$(readlink -f "$0")")



set -euo pipefail

module use /appl/local/csc/modulefiles
module load pytorch/2.4
SCRIPT_DIR="$3"
source "$SCRIPT_DIR/common.sh"
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
    input_path="$PARQUET_DIR/$filename"
    output_path="$PREDICT_DIR/$filename"
    
    # Log start time
    log_event "$filename" "START"
    log_event "$filename" "srun --ntasks=1 --gres=gpu:mi250:1 --mem=16G python3 $SCRIPT_DIR/process_parquet_file.py $input_path $output_path"

    
    srun \
        --ntasks=1 \
        --gres=gpu:mi250:1 \
        --mem=16G \
        python3 "$SCRIPT_DIR/process_parquet_file.py" \
        "$input_path" \
        "$output_path" \
        && log_event "$filename" "SUCCESS" \
        || log_event "$filename" "FAIL: $?"
    done

# Wait for all background processes to complete
wait

# Exit with success
exit 0