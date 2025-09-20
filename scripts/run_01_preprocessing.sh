#!/bin/bash
# filepath: /mnt/windows_e/workplace/task_generation/scripts/run_01_preprocessing.sh

# Load configuration
source "$(dirname "$0")/config.sh"

print_config

log_step "Starting 01_preprocessings.py"

# Build optional arguments
args=(
    --output_dir "$RUN_DIR"
)

# Only add specific arguments if we want to override defaults
if [[ -n "$CONFIG_FILE" ]]; then
    args+=(--config "$CONFIG_FILE")
fi

# Add output paths to use our run-specific directory
if [[ -n "$PARSED_JSON" ]]; then
    args+=(--output_json_path "$PARSED_JSON")
fi

if [[ -n "$SCENE_GRAPH_PKL" ]]; then
    args+=(--scene_graph_pkl_save_path "$SCENE_GRAPH_PKL")
fi

if [[ -n "$RENAME_DICT_JSON" ]]; then
    args+=(--rename_dict_path "$RENAME_DICT_JSON")
fi

if [[ -n "$IMAGES_DIR" ]]; then
    args+=(--image4rename_path "$IMAGES_DIR/image4rename")
fi

# Run preprocessing script with optional arguments
run_python_script "01_preprocessings.py" "${args[@]}"

if [ $? -eq 0 ]; then
    log_info "Preprocessing completed successfully"
    log_info "Output directory: $RUN_DIR"
    
    # List actually generated files
    if [ -f "$SCENE_GRAPH_PKL" ]; then
        log_info "  - Scene graph: $SCENE_GRAPH_PKL"
    fi
    if [ -f "$PARSED_JSON" ]; then
        log_info "  - Parsed JSON: $PARSED_JSON"
    fi
    if [ -f "$RENAME_DICT_JSON" ]; then
        log_info "  - Rename dict: $RENAME_DICT_JSON"
    fi
    if [ -d "$IMAGES_DIR/image4rename" ]; then
        log_info "  - Images: $IMAGES_DIR/image4rename/ ($(ls -1 "$IMAGES_DIR/image4rename" 2>/dev/null | wc -l) files)"
    fi
else
    log_error "Preprocessing failed"
    exit 1
fi