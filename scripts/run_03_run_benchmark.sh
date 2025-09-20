#!/bin/bash
# filepath: /mnt/windows_e/workplace/task_generation/scripts/run_03_benchmark.sh

# 加载配置
source "$(dirname "$0")/config.sh"

print_config

log_step "Starting 03_run_benchmark.py"

# 检查必要的输入文件
check_input "$OUTPUT_JSON"
check_input "$ENTITY_JSON"
check_input_optional "$SCENE_GRAPH_PKL"
check_input_optional "$ATOMIC_TASK_PKL"

# 设置基准测试参数
TASK_NUM=${TASK_NUM:-10}
MODE=${MODE:-"online"}
MODEL_NAME=${MODEL_NAME:-"openai/gpt-4-1-mini"}

log_info "Benchmark configuration:"
log_info "  - Task number: $TASK_NUM"
log_info "  - Mode: $MODE"
log_info "  - Model: $MODEL_NAME"
log_info "  - Output JSON: $OUTPUT_JSON"
log_info "  - Entity JSON: $ENTITY_JSON"

# 运行基准测试脚本
run_python_script "03_run_benchmark.py" \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --task_num "$TASK_NUM" \
    --mode "$MODE" \
    --model_name "$MODEL_NAME" \
    --scene_graph_pkl_save_path "$SCENE_GRAPH_PKL" \
    --atomic_task_pkl_load_path "$ATOMIC_TASK_PKL"

if [ $? -eq 0 ]; then
    log_info "Benchmark execution completed successfully"
    log_info "Results saved to:"
    log_info "  - Output directory: $OUTPUT_DIR"
    log_info "  - Result file: ${RESULT_FILE:-$OUTPUT_DIR/benchmark_results.txt}"
    
    # 显示基准测试结果摘要（如果结果文件存在）
    if [ -f "${RESULT_FILE:-$OUTPUT_DIR/benchmark_results.txt}" ]; then
        log_info "Benchmark Results Summary:"
        tail -n 5 "${RESULT_FILE:-$OUTPUT_DIR/benchmark_results.txt}" | while read line; do
            log_info "  $line"
        done
    fi
else
    log_error "Benchmark execution failed"
    exit 1
fi