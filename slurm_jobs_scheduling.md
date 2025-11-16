# Inf-Net Training with SLURM Array Jobs

This directory contains a unified training system that uses SLURM array jobs to efficiently manage hundreds of training experiments.

## Overview

Instead of creating individual SLURM scripts for each experiment, we use array jobs which:
- Reduce scheduler load (one job submission instead of hundreds)
- Are easier to manage and monitor
- Provide better resource utilization
- Allow controlled concurrency

## Files

- `MyTrain_LungInf_Unified.py`: Unified training script supporting all three networks (Inf_Net, UNet, NestedUNet) with all configurations
- `submit_training_jobs_unified.py`: Script to generate array job files
- `submit_all_array_jobs.sh`: Helper script to submit all generated array jobs
- `slurm_jobs_unified/`: Directory containing generated array job scripts and command files

## Experimental Configuration

The system generates experiments for:

- 3 Networks: Inf_Net GroupNorm, UNet_GroupNorm, NestedUNet_GroupNorm
- 3 Run Types:
  1. Base (no DP, no Morph)
  2. Base with/without Morph
  3. Base with/without DP
- Batch Sizes: 24, 48, 64
- Epsilon Values: 8, 200 (for DP experiments)
- Clipping Strategies: base, automatic, psac, nsgd (for DP experiments)
- Morph Operation: both, open and close
- Epochs: 100
- Max Grad Norm: 1.2
- Runs: 3 per configuration

## Usage

### Step 1: Generate Array Job Scripts

First, generate the array job scripts and command files:

```bash
cd code/inf-net

# Dry-run to see what will be generated
python submit_training_jobs_unified.py --dry-run

# Generate for all networks (default)
python submit_training_jobs_unified.py

# Generate for specific networks
python submit_training_jobs_unified.py --networks Inf_Net UNet

# Specify partitions
python submit_training_jobs_unified.py --partitions a100-80g gpu

# Adjust max concurrent tasks (default: 20)
python submit_training_jobs_unified.py --max-concurrent 30
```

This will create:
- `slurm_jobs_unified/` directory
- Subdirectories for each partition (e.g., `a100-80g/`, `gpu/`)
- `commands.cmd` file in each partition directory (one command per line)
- `array_job.sh` script in each partition directory
- `experiments_summary.txt` with full experiment listing

### Step 2: Submit Array Jobs

Option A: Submit all at once (recommended)
```bash
./submit_all_array_jobs.sh
```

Option B: Submit individually
```bash
cd slurm_jobs_unified/a100-80g
sbatch array_job.sh
cd ../gpu
sbatch array_job.sh
```

Option C: Submit from the main script
The script can also submit automatically (without `--dry-run` flag).

### Step 3: Monitor Jobs

```bash
# Check all jobs
squeue -u $USER

# Check specific array job
squeue -j <JOB_ID>

# Cancel an array job
scancel <JOB_ID>

# Cancel all tasks in an array
scancel <JOB_ID>_*
```

## Directory Structure

After generation, you'll have:

```
slurm_jobs_unified/
├── a100-80g/
│   ├── commands.cmd          # All training commands (one per line)
│   └── array_job.sh          # SLURM array job script
├── gpu/
│   ├── commands.cmd
│   └── array_job.sh
└── experiments_summary.txt   # Complete experiment listing with task IDs
```

## How Array Jobs Work

1. commands.cmd: Contains all training commands, one per line
   ```
   python MyTrain_LungInf_Unified.py --network Inf_Net --batchsize 24 --run 1 ...
   python MyTrain_LungInf_Unified.py --network Inf_Net --batchsize 24 --run 2 ...
   ...
   ```

2. array_job.sh: SLURM script with `--array=1-N%M` directive
   - `N` = total number of tasks (lines in commands.cmd)
   - `M` = max concurrent tasks (default: 20)
   - Each task reads its command using `SLURM_ARRAY_TASK_ID`

3. Execution: Task 1 executes line 1, task 2 executes line 2, etc.

## Log Files

Log files are saved to `logs/train/` with naming:
- `infnet_<partition>_<JOB_ID>_<TASK_ID>.out`
- `infnet_<partition>_<JOB_ID>_<TASK_ID>.err`

Where:
- `<JOB_ID>` = Array job ID
- `<TASK_ID>` = Task number within the array

## Customization

### Adjust Concurrency
```bash
python submit_training_jobs_unified.py --max-concurrent 30
```

### Filter Experiments
```bash
# Only specific networks
python submit_training_jobs_unified.py --networks UNet

# Only specific batch sizes
python submit_training_jobs_unified.py --batch-sizes 24 48

# Only 1 run per config (for testing)
python submit_training_jobs_unified.py --num-runs 1
```

### Time Limits
```bash
python submit_training_jobs_unified.py --time-limit 12:00:00
```

## Example Workflow

```bash
# 1. Generate scripts (dry-run first)
python submit_training_jobs_unified.py --dry-run

# 2. Review generated files
cat slurm_jobs_unified/a100-80g/commands.cmd | head -5
cat slurm_jobs_unified/a100-80g/array_job.sh

# 3. Generate for real
python submit_training_jobs_unified.py

# 4. Submit all array jobs
./submit_all_array_jobs.sh

# 5. Monitor
squeue -u $USER

# 6. Check logs
tail -f logs/train/infnet_a100-80g_*.out
```

## Advantages Over Individual Scripts

| Individual Scripts | Array Jobs |
|-------------------|------------|
| 100s of `sbatch` calls | 1-3 `sbatch` calls |
| Hard to manage | Easy to manage |
| High scheduler load | Low scheduler load |
| Many small log files | Organized log files |
| Hard to cancel all | Easy to cancel all |

## Troubleshooting

Q: Jobs not starting?
- Check partition availability: `sinfo -p <partition>`
- Check QoS limits: `sacctmgr show qos`
- Reduce `--max-concurrent` if hitting limits

Q: Need to cancel all tasks?
```bash
scancel <JOB_ID>_*
```

Q: Want to resubmit failed tasks?
- Edit `commands.cmd` to only include failed tasks
- Regenerate array script with new task count
- Submit again

Q: Check which task failed?
```bash
# Check error logs
ls -lh logs/train/*.err | sort -k5 -h

# Check specific task
cat logs/train/infnet_a100-80g_<JOB_ID>_<TASK_ID>.err
```

## Notes

- Array jobs automatically handle task scheduling
- Each task runs independently
- Failed tasks don't affect other tasks
- Results are saved to `./Snapshots/save_weights/` with organized directory structure
- Check `experiments_summary.txt` for complete experiment mapping

