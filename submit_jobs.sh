#!/bin/bash
# submit_jobs.sh
# Testing version - shows what would be submitted without actual submission

# Function to print usage
usage() {
    echo "Usage: $0 [-t|--test] <parquet_directory>"
    echo "  -t, --test    Run in test mode (print commands without executing)"
    exit 1
}

# Parse command line arguments
TEST_MODE=false
PARQUET_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test)
            TEST_MODE=true
            shift
            ;;
        *)
            PARQUET_DIR="$1"
            shift
            ;;
    esac
done

# Check if directory is provided
if [ -z "$PARQUET_DIR" ]; then
    usage
fi

# Check if directory exists
if [ ! -d "$PARQUET_DIR" ]; then
    echo "Error: Directory $PARQUET_DIR does not exist"
    exit 1
fi

# Create a directory for logs if it doesn't exist
if [ "$TEST_MODE" = false ]; then
    mkdir -p logs
else
    echo "[TEST] Would create logs directory if it doesn't exist"
fi

# Get list of all parquet files
cd "$PARQUET_DIR"
files=(*.parquet)

# Check if any parquet files were found
if [ ! -e "${files[0]}" ]; then
    echo "Error: No parquet files found in $PARQUET_DIR"
    exit 1
fi

total_files=${#files[@]}
batch_size=8  # Number of GPUs/files to process in parallel

# Calculate number of batches needed
num_batches=$(( (total_files + batch_size - 1) / batch_size ))

echo "Found $total_files files in $PARQUET_DIR"
echo "Will process in $num_batches batches of up to $batch_size files each"
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
    
    echo "SBATCH command that would be executed:"
    echo "------------------------"
    echo "sbatch \\"
    echo "    --job-name=\"parquet_batch_${batch}\" \\"
    echo "    --output=\"logs/batch_${batch}_%j.out\" \\"
    echo "    --error=\"logs/batch_${batch}_%j.err\" \\"
    echo "    --gres=gpu:8 \\"
    echo "    process_batch.sh \"$file_list\""
    echo "------------------------"
    echo ""
    
    if [ "$TEST_MODE" = false ]; then
        sbatch \
            --job-name="parquet_batch_${batch}" \
            --output="logs/batch_${batch}_%j.out" \
            --error="logs/batch_${batch}_%j.err" \
            --gres=gpu:8 \
            process_batch.sh "$file_list"
        echo "Submitted batch $batch (files $start to $end)"
    fi
done

if [ "$TEST_MODE" = true ]; then
    echo "TEST MODE: No jobs were actually submitted"
fi