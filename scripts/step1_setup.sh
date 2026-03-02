#!/bin/bash
# Step 1: Setup environment
# Run from the project root: bash scripts/step1_setup.sh

set -e
CONDA_ENV="pd-tau-bench"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Activating conda environment: $CONDA_ENV ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "=== Installing tau2-bench ==="
pip install -e "$PROJECT_DIR/tau2-bench"

echo "=== Installing remaining dependencies ==="
pip install pyyaml tqdm

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Set your DashScope API key:"
echo "     export DASHSCOPE_API_KEY=your_key_here"
echo ""
echo "  2. Verify tau2-bench loads:"
echo "     python -c 'import tau2; print(tau2.__version__)'"
echo ""
echo "  3. Run fork test (requires API key):"
echo "     conda activate $CONDA_ENV"
echo "     cd $PROJECT_DIR"
echo "     python -m src.predictive_decoding.test_env_fork --domain retail --task-id 0"
