#!/bin/bash
# filepath: /mnt/windows_e/workplace/task_generation/scripts/run_02b_gen_outcome_tasks.sh


source "$(dirname "$0")/config.sh"

print_config

log_step "Starting 02b_gen_outcome_based_tasks.py"


check_input_optional "$SCENE_GRAPH_PKL"
check_input_optional "$RENAME_DICT_JSON"


run_python_script "02b_gen_outcome_based_tasks.py" \
    --scene_graph_pkl_load_path "$SCENE_GRAPH_PKL" \
    --rename_dict_path "$RENAME_DICT_JSON" \
    --manitaskot_pattern_file "$VOTING_PROMPTS" \
    --outcome_based_task_txt_save_path "$OUTCOME_TASKS_TXT" \
    --vlm_list "openai/gpt-4.1" "anthropic/claude-3.5-haiku" "google/gemini-2.5-flash-lite-preview-06-17"

if [ $? -eq 0 ]; then
    log_info "Outcome-based task generation completed successfully"
    log_info "Generated files:"
    log_info "  - Tasks TXT: $OUTCOME_TASKS_TXT"
else
    log_error "Outcome-based task generation failed"
    exit 1
fi