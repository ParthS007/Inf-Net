#!/bin/bash
# Helper script to submit all array jobs from the generated scripts
# Usage: ./submit_all_array_jobs.sh [output_dir]

OUTPUT_DIR="${1:-./slurm_jobs_unified}"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Directory $OUTPUT_DIR does not exist"
    echo "Please run the submission script first to generate array jobs"
    exit 1
fi

echo "Submitting all array jobs from $OUTPUT_DIR"
echo "=========================================="

# Find all array_job.sh files and submit them
for script in $(find "$OUTPUT_DIR" -name "array_job.sh" | sort); do
    script_dir=$(dirname "$script")
    partition=$(basename "$script_dir")
    
    echo ""
    echo "Submitting array job for partition: $partition"
    echo "  Script: $script"
    
    cd "$script_dir"
    sbatch array_job.sh
    cd - > /dev/null
done

echo ""
echo "=========================================="
echo "All array jobs submitted!"
echo "Check status with: squeue -u \$USER"

