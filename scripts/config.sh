#!/bin/bash
# filepath: /mnt/windows_e/workplace/task_generation/scripts/config.sh

# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Set base directories
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$BASE_DIR/data"
SRC_DIR="$BASE_DIR/src"
SCRIPTS_DIR="$SRC_DIR/scripts"

# Create run-specific directories
RUN_DIR="$DATA_DIR/run_$TIMESTAMP"
CACHE_DIR="$RUN_DIR/cache"
IMAGES_DIR="$RUN_DIR/images"
RESULTS_DIR="$RUN_DIR/results"

# Create necessary directories
mkdir -p "$CACHE_DIR"
mkdir -p "$IMAGES_DIR/image4rename"
mkdir -p "$IMAGES_DIR/image4outcomebase"
mkdir -p "$IMAGES_DIR/image4interaction"
mkdir -p "$RESULTS_DIR"

# Input data paths (these are suggestions, not required)
DATASET_ROOT="$DATA_DIR/datasets/replica_dataset"
RENAMING_DICT_TEMPLATE="$DATA_DIR/templates/renaming_dict.json"
VOTING_PROMPTS="$DATA_DIR/templates/voting_prompts.json"
BENCHMARK_PROMPTS="$DATA_DIR/templates/benchmark_prompts.json"
REFLECTION_PROMPTS="$DATA_DIR/templates/reflection_prompts.json"

# Intermediate output paths (these will be used if we want to override defaults)
PARSED_JSON="$CACHE_DIR/dataset_parsed.json"
SCENE_GRAPH_PKL="$CACHE_DIR/scene_graph.pkl"
RENAME_DICT_JSON="$CACHE_DIR/rename_dict.json"
PROCESS_TASKS_PKL="$CACHE_DIR/process_based_tasks.pkl"
PROCESS_TASKS_TXT="$CACHE_DIR/process_based_tasks_ins.txt"
OUTCOME_TASKS_TXT="$CACHE_DIR/outcome_based_tasks_ins.txt"
REFLECTION_NOTES="$CACHE_DIR/reflection_notes.txt"
BENCHMARK_RESULTS="$RESULTS_DIR/benchmark_results.txt"

# Log file
LOG_FILE="$RUN_DIR/execution.log"

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Check if a Python script exists
check_script() {
    if [ ! -f "$1" ]; then
        log_error "Script not found: $1"
        exit 1
    fi
}

# Check if an input file/directory exists (optional check)
check_input_optional() {
    if [ ! -e "$1" ]; then
        log_warn "Input file/directory not found: $1 (will use config defaults)"
        return 1
    fi
    return 0
}

# General function to run Python scripts
run_python_script() {
    local script_name="$1"
    local script_path="$SCRIPTS_DIR/$script_name"
    shift
    
    log_step "Running $script_name"
    check_script "$script_path"
    
    cd "$BASE_DIR"
    export PYTHONPATH="$BASE_DIR:$PYTHONPATH"
    
    log_info "Running script: $script_path with PYTHONPATH=$BASE_DIR"

    if python "$script_path" "$@" 2>&1 | tee -a "$LOG_FILE"; then
        log_info "$script_name completed successfully"
        return 0
    else
        log_error "$script_name failed"
        return 1
    fi
}

# Build arguments array for scripts (only add non-empty arguments)
build_args() {
    local args=()
    
    # Add argument only if value is provided and not empty
    while [[ $# -gt 0 ]]; do
        local key="$1"
        local value="$2"
        if [[ -n "$value" && "$value" != "" ]]; then
            args+=("$key" "$value")
        fi
        shift 2
    done
    
    echo "${args[@]}"
}

# Print configuration information
print_config() {
    log_info "=== Run Configuration ==="
    log_info "Timestamp: $TIMESTAMP"
    log_info "Run Directory: $RUN_DIR"
    log_info "Base Directory: $BASE_DIR"
    log_info "Log File: $LOG_FILE"
    log_info "Using configuration file defaults unless overridden"
    log_info "========================="
}