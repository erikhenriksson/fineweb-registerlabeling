#!/bin/bash

set -euo pipefail

source common.sh

# Common has:
# ROOT_DIR, DATA_DIR, PREDICT_DIR, TEST_MODE

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 SUBSET" >&2
    echo >&2
    echo "example: $0 CC-MAIN-2013-20" >&2
    exit 1
fi
SUBSET="$1"

PARQUET_DIR="$ROOT_DIR/$DATA_DIR/$SUBSET"
PREDICT_DIR="$ROOT_DIR/$PREDICT_DIR/$SUBSET"
LOG_DIR="$ROOT_DIR/$LOG_DIR/$SUBSET"

# Check if directory exists
if [ ! -d "$PARQUET_DIR" ]; then
    echo "Error: Directory $PARQUET_DIR does not exist"
    exit 1
fi

# Get list of all parquet files (just the filenames, not the full path)
cd "$PARQUET_DIR"
files=(*.parquet)

# Check if any parquet files were found
if [ ! -e "${files[0]}" ]; then
    echo "Error: No parquet files found in $PARQUET_DIR"
    exit 1
fi

mkdir -p $(dirname "$PREDICT_DIR")
mkdir -p $(dirname "$LOG_DIR")

total_files=${#files[@]}
batch_size=8  # Number of GPUs/files to process in parallel

# Calculate number of batches needed
num_batches=$(( (total_files + batch_size - 1) / batch_size ))

echo "Found $total_files files in $PARQUET_DIR"
echo "Will process in $num_batches batches of up to $batch_size files each"
echo "Logs will be stored in: $LOG_DIR"
echo "------------------------"

for ((batch=0; batch<num_batches; batch++)); do
    # Calculate start and end indices for this batch
    start=$((batch * batch_size))
    end=$((start + batch_size - 1))
    
    # Ensure we don't exceed the array bounds
    if [ $end -ge $total_files ]; then
        end=$((total_files - 1))
    fi
    
    # Create a comma-separated list of files for this batch
    file_list=""
    echo "Batch $batch would process:"
    for ((i=start; i<=end; i++)); do
        echo "  - ${files[i]} (would use GPU $((i-start)))"
        if [ -n "$file_list" ]; then
            file_list="${file_list},"
        fi
        file_list="${file_list}${files[i]}"
    done
    
    echo "Command that would be executed:"
    echo "------------------------"
    echo "sbatch process_batch.sh \"$PARQUET_DIR\" \"$file_list\""
    echo "------------------------"
    echo ""
    
    if [ "$TEST_MODE" = false ]; then
        sbatch process_batch.sh "$PARQUET_DIR" "$file_list"
        echo "Submitted batch $batch (files $start to $end)"
    fi
done

if [ "$TEST_MODE" = true ]; then
    echo "TEST MODE: No jobs were actually submitted"
fi