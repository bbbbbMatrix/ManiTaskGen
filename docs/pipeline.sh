#!/bin/bash
# run_pipeline.sh

set -e

show_usage() {
    echo "Usage: $0 <pipeline_type> [options]"
    echo ""
    echo "Pipeline types:"
    echo "  process    - Run 01 + 02a + 03a (process-based pipeline)"
    echo "  process-reflect - Run 01 + 02a + 03b (process-based with reflection)"
    echo "  outcome    - Run 01 + 02b (outcome-based pipeline)"
    echo "  clean      - Clean all run directories"
    echo ""
    echo "Options:"
    echo "  --config FILE    - Configuration file (default: configs/default.yaml)"
    echo "  --dataset PATH   - Dataset path override"
    echo "  --name NAME      - Custom run name suffix"
}

# 解析参数
PIPELINE_TYPE="$1"
shift

CONFIG_FILE="configs/default.yaml"
DATASET_PATH=""
RUN_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --name)
            RUN_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 验证pipeline类型
case $PIPELINE_TYPE in
    process|process-reflect|outcome|clean)
        ;;
    *)
        echo "Error: Invalid pipeline type '$PIPELINE_TYPE'"
        show_usage
        exit 1
        ;;
esac

# 清理操作
if [ "$PIPELINE_TYPE" == "clean" ]; then
    echo "Cleaning all run directories..."
    rm -rf data/runs/run_*
    rm -f data/runs/latest
    echo "Done."
    exit 0
fi

# 运行管道
python3 -c "
import sys
sys.path.append('.')
from src.utils.run_context import RunContext
import subprocess
import glog

# 创建管道上下文
pipeline_name = '${PIPELINE_TYPE}' + ('_${RUN_NAME}' if '${RUN_NAME}' else '')

with RunContext.pipeline_context(pipeline_name) as run_dir:
    print(f'=== Running ${PIPELINE_TYPE} pipeline in: {run_dir} ===')
    
    # 准备通用参数
    common_args = [
        '--config', '${CONFIG_FILE}',
        '--run_dir', run_dir
    ]
    
    if '${DATASET_PATH}':
        common_args.extend(['--dataset_path', '${DATASET_PATH}'])
    
    # 步骤1：预处理（所有管道都需要）
    print('--- Step 1: Preprocessing ---')
    subprocess.run(['python3', 'scripts/01_preprocessing.py'] + common_args, check=True)
    
    # 根据管道类型执行后续步骤
    if '${PIPELINE_TYPE}' in ['process', 'process-reflect']:
        print('--- Step 2a: Process-based Task Generation ---')
        subprocess.run(['python3', 'scripts/02a_process_based_tasks.py'] + common_args, check=True)
        
        if '${PIPELINE_TYPE}' == 'process':
            print('--- Step 3a: Benchmark Execution ---')
            subprocess.run(['python3', 'scripts/03a_benchmark_executor.py'] + common_args, check=True)
        else:  # process-reflect
            print('--- Step 3b: Benchmark Execution with Reflection ---')
            subprocess.run(['python3', 'scripts/03b_benchmark_executor.py'] + common_args, check=True)
    
    elif '${PIPELINE_TYPE}' == 'outcome':
        print('--- Step 2b: Outcome-based Task Generation ---')
        subprocess.run(['python3', 'scripts/02b_outcome_based_tasks.py'] + common_args, check=True)
    
    print(f'=== Pipeline ${PIPELINE_TYPE} completed successfully ===')
    print(f'Results saved in: {run_dir}')
"