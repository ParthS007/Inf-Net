#!/bin/bash
# -*- coding: utf-8 -*-

"""
Convenience script for submitting training jobs with various presets
Usage: ./train_submission.sh [preset] [options]
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PRESET="all"
DRY_RUN=false
TIME_LIMIT="12:00:00"
SCRIPT_DIR="./job_scripts"

function print_banner() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║         Inf-Net Training Job Submission Manager                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

function print_usage() {
    cat << EOF
Usage: ./train_submission.sh [PRESET] [OPTIONS]

PRESETS:
  standard          - Standard Inf-Net only (no morphology, no DP)
  morph             - Inf-Net with morphology only (no DP)
  dp                - Inf-Net with DP only (3 epsilon values)
  dpmorph           - Inf-Net with DP + Morphology combined
  all               - All configurations (default)
  quick             - Quick test: batch 32 × (standard+morph+dp) × 1 run
  small             - Small run: batch 32 × all variants × 3 runs
  full              - Full run: all configs × 3 runs

OPTIONS:
  --dry-run         - Generate scripts without submitting
  --time HOURS      - Job time limit (default: 12 hours)
  --batch-sizes BS  - Specific batch sizes (e.g., 32 64)
  --morph-ops OPS   - Specific morph operations (e.g., open close)
  --script-dir DIR  - Directory for job scripts
  --help            - Show this help message

EXAMPLES:
  # Submit all standard Inf-Net jobs
  ./train_submission.sh standard

  # Submit DP variants with epsilon 1, 8, 200
  ./train_submission.sh dp

  # Dry run with DP+Morphology
  ./train_submission.sh dpmorph --dry-run

  # Quick test with batch 32 only
  ./train_submission.sh quick

  # Full run with longer time limit
  ./train_submission.sh full --time 24

EOF
}

function submit_preset() {
    local preset=$1
    local args=""

    print_banner

    case $preset in
        standard)
            echo -e "${YELLOW}[INFO]${NC} Submitting standard Inf-Net configurations..."
            args="--configs standard"
            ;;
        morph)
            echo -e "${YELLOW}[INFO]${NC} Submitting Inf-Net with morphology configurations..."
            args="--configs morph"
            ;;
        dp)
            echo -e "${YELLOW}[INFO]${NC} Submitting Inf-Net with differential privacy configurations..."
            args="--configs dp"
            ;;
        dpmorph)
            echo -e "${YELLOW}[INFO]${NC} Submitting Inf-Net with DP+Morphology configurations..."
            args="--configs dpmorph"
            ;;
        all)
            echo -e "${YELLOW}[INFO]${NC} Submitting all configurations..."
            args="--configs all"
            ;;
        quick)
            echo -e "${YELLOW}[INFO]${NC} Submitting quick test (batch 32 only, 1 run)..."
            args="--configs all --batch-sizes 32 --num-runs 1"
            ;;
        small)
            echo -e "${YELLOW}[INFO]${NC} Submitting small run (batch 32 only)..."
            args="--configs all --batch-sizes 32"
            ;;
        full)
            echo -e "${YELLOW}[INFO]${NC} Submitting full run (all configurations)..."
            args="--configs all --batch-sizes 32 64 128 --num-runs 3"
            TIME_LIMIT="24:00:00"
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Unknown preset: $preset"
            print_usage
            exit 1
            ;;
    esac

    # Add optional parameters
    if [ "$DRY_RUN" = true ]; then
        args="$args --dry-run"
    fi

    args="$args --time-limit $TIME_LIMIT"
    args="$args --script-dir $SCRIPT_DIR"

    # Run Python submission script
    echo -e "${BLUE}Command:${NC} python3 submit_training_jobs.py $args"
    echo ""

    python3 submit_training_jobs.py $args
}

function main() {
    if [ $# -eq 0 ]; then
        print_usage
        exit 1
    fi

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --time)
                TIME_LIMIT="${2}:00:00"
                shift 2
                ;;
            --batch-sizes)
                shift
                BATCH_SIZES=""
                while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                    BATCH_SIZES="$BATCH_SIZES $1"
                    shift
                done
                ;;
            --morph-ops)
                shift
                MORPH_OPS=""
                while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                    MORPH_OPS="$MORPH_OPS $1"
                    shift
                done
                ;;
            --script-dir)
                SCRIPT_DIR="$2"
                shift 2
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                PRESET="$1"
                shift
                ;;
        esac
    done

    # Verify we're in the right directory
    if [ ! -f "MyTrain_LungInf_Morph.py" ]; then
        echo -e "${RED}[ERROR]${NC} MyTrain_LungInf_Morph.py not found!"
        echo "Please run this script from the inf-net directory."
        exit 1
    fi

    # Submit jobs
    submit_preset "$PRESET"

    echo ""
    if [ "$DRY_RUN" = true ]; then
        echo -e "${GREEN}[SUCCESS]${NC} Dry run completed!"
        echo "Scripts are ready in: $SCRIPT_DIR"
        echo ""
        echo "To submit all jobs now, run:"
        echo -e "  ${BLUE}sbatch $SCRIPT_DIR/*.sh${NC}"
        echo ""
    else
        echo -e "${GREEN}[SUCCESS]${NC} Jobs submitted!"
        echo ""
        echo "To check job status, run:"
        echo -e "  ${BLUE}squeue -u \$USER${NC}"
        echo ""
        echo "To cancel all jobs, run:"
        echo -e "  ${BLUE}scancel -u \$USER${NC}"
    fi
}

main "$@"
