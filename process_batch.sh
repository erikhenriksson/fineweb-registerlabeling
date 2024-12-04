#!/bin/bash
#SBATCH --job-name=parquet-processing
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi250:8
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Get the directory and file list from arguments
PARQUET_DIR="$1"
files="$2"
LOG_DIR="logs/$(basename "$PARQUET_DIR")"
PREDICT_DIR="/pfs/lustrep3/scratch/project_462000642/FINEWEB/predictions"

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
    log_event "$filename" "Started processing"
    
    # Launch the Python script using srun
    (
        srun \
            --ntasks=1 \
            --gres=gpu:mi250:1 \
            --mem=16G \
            python3 process_file.py \
            "$full_path" \
            "$PREDICT_DIR" \
            &
        
        exit_status=$?
        
        # Only log completion if successful
        if [ $exit_status -eq 0 ]; then
            log_event "$filename" "Successfully completed processing"
        else
            log_event "$filename" "Failed processing (exit code: $exit_status)"
        fi
    ) &
done

# Wait for all background processes to complete
wait

# Exit with success
exit 0