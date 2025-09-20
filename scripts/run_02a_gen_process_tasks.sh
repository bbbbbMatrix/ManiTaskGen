#!/bin/bash
# filepath: /mnt/windows_e/workplace/task_generation/scripts/run_02a_process_tasks.sh

# Load configuration
source "$(dirname "$0")/config.sh"

print_config

log_step "Starting 02a_gen_process_based_tasks.py"

# Build optional arguments
args=(
    --output_dir "$RUN_DIR"
)

# Only add arguments if we want to override defaults or if files exist in our run directory
if [[ -f "$SCENE_GRAPH_PKL" ]]; then
    args+=(--scene_graph_pkl_load_path "$SCENE_GRAPH_PKL")
fi

if [[ -f "$RENAME_DICT_JSON" ]]; then
    args+=(--rename_dict_path "$RENAME_DICT_JSON")
fi

if [[ -n "$PROCESS_TASKS_PKL" ]]; then
    args+=(--process_tasks_pkl_path "$PROCESS_TASKS_PKL")
fi

if [[ -n "$PROCESS_TASKS_TXT" ]]; then
    args+=(--process_tasks_txt_path "$PROCESS_TASKS_TXT")
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