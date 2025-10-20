#!/bin/bash
# filepath: /mnt/windows_e/workplace/task_generation/scripts/run_02b_gen_outcome_tasks.sh


source "$(dirname "$0")/config.sh"

print_config

log_step "Starting 02b_gen_outcome_based_tasks.py"

cd "$BASE_DIR"

log_info "Current working directory: $BASE_DIR"

args=(
    --output_dir "$RUN_DIR"
)


check_input_optional "$SCENE_GRAPH_PKL"
check_input_optional "$RENAME_DICT_JSON"

if [[ -n "$CONFIG_FILE" ]]; then
    args+=(--config "$CONFIG_FILE")
    log_info "Using config file in bash: $CONFIG_FILE"
fi



run_python_script "02b_gen_outcome_based_tasks.py" "${args[@]}"

if [ $? -eq 0 ]; then
    log_info "Outcome-based task generation completed successfully"
    log_info "Generated files:"
    log_info "  - Tasks TXT: $OUTCOME_TASKS_TXT"
else
    log_error "Outcome-based task generation failed"
    exit 1
fi