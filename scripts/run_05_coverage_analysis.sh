#!/bin/bash
# Process-based coverage analysis: ManiTaskGen (sampled) vs GPT taskgen.
# Usage:
#   MANITASKGEN_PKL=... SCENE_GRAPH_PKL=... [GPT_JSON=...] bash scripts/run_05_coverage_analysis.sh
source "$(dirname "$0")/config.sh"
print_config
log_step "Starting 05_coverage_analysis.py"
cd "$BASE_DIR"

: "${MANITASKGEN_PKL:=$CACHE_DIR/process_based_task.pkl}"
: "${SCENE_GRAPH_PKL:=$CACHE_DIR/scene_graph.pkl}"

args=(--manitaskgen_pkl "$MANITASKGEN_PKL" --scene_graph_pkl "$SCENE_GRAPH_PKL" --sample_size "${SAMPLE_SIZE:-100}" --seed "${SEED:-0}")
[[ -n "$GPT_JSON" ]] && args+=(--gpt_json "$GPT_JSON")
[[ -n "$OUT_DIR" ]] && args+=(--out "$OUT_DIR")

run_python_script "05_coverage_analysis.py" "${args[@]}"
[[ $? -eq 0 ]] && log_info "Coverage report -> ${OUT_DIR:-runs/output/coverage}" || { log_error "coverage failed"; exit 1; }
