#!/bin/bash

set -e  # Exit on error

echo "=== ManiTaskGen Environment Installation Script ==="

# Check if in a conda environment
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "Warning: Please activate a conda environment first"
    echo "For example: conda activate your_env_name"
    exit 1
fi

echo "Current environment: ${CONDA_DEFAULT_ENV}"

# Add conda-forge channel
echo "Adding conda-forge channel..."
conda config --add channels conda-forge

# Install pip packages
echo "Installing pip dependencies..."
pip install -r config/requirements_minimal.txt

# Install VLMEvalKit
echo "Installing VLMEvalKit..."
if [ -d "src/vlm_interaction/VLMEvalKit" ]; then
    cd src/vlm_interaction/VLMEvalKit
    pip install -e .
    pip install torchvision==0.21.0
    cd ../../..
else
    echo "Warning: VLMEvalKit directory does not exist"
fi

echo "=== Installation Complete! ==="
echo "Testing core package imports..."

python -c "
try:
    import mani_skill
    import sapien  
    import trimesh
    print('✓ All core packages imported successfully!')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    exit(1)
"

echo "Environment setup complete, you can start using it now!"
