#!/bin/bash
# filepath: /mnt/windows_e/workplace/task_generation/scripts/run_02a_process_tasks.sh

# Load configuration
source "$(dirname "$0")/config.sh"

print_config

log_step "Starting 02a_gen_process_based_tasks.py"

cd "$BASE_DIR"

log_info "Current working directory: $BASE_DIR"

# Build optional arguments
args=(
    --output_dir "$RUN_DIR"
)

if [[ -n "$CONFIG_FILE" ]]; then
    args+=(--config "$CONFIG_FILE")
    log_info "Using config file in bash: $CONFIG_FILE"
fi




# Run process-based task generation script
run_python_script "02a_gen_process_based_tasks.py" "${args[@]}"

if [ $? -eq 0 ]; then
    log_info "Process-based task generation completed successfully"
    
    # List actually generated files
    if [ -f "$PROCESS_TASKS_PKL" ]; then
        log_info "  - Tasks PKL: $PROCESS_TASKS_PKL"
    fi
    if [ -f "$PROCESS_TASKS_TXT" ]; then
        log_info "  - Tasks TXT: $PROCESS_TASKS_TXT"
    fi
else
    log_error "Process-based task generation failed"
    exit 1
fi