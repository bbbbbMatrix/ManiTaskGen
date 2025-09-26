
#!/bin/bash
# Submodule management script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

case "${1:-help}" in
    "init")
        echo "Initializing submodules..."
        git submodule update --init --recursive
        echo "✓ Submodules initialized"
        ;;
    
    "update")
        echo "Updating submodules to latest..."
        git submodule update --remote --recursive
        echo "✓ Submodules updated"
        ;;
    
    "status")
        echo "Submodule status:"
        git submodule status --recursive
        ;;
    
    "push-changes")
        echo "Pushing submodule changes..."
        cd src/vlm_interaction/VLMEvalKit
        if [[ -n $(git status --porcelain) ]]; then
            git add .
            git commit -m "${2:-Update VLMEvalKit modifications}"
            git push origin task-generation-integration
            cd "$PROJECT_ROOT"
            git add src/vlm_interaction/VLMEvalKit
            git commit -m "Update VLMEvalKit submodule"
            echo "✓ Changes pushed to both repositories"
        else
            echo "No changes to push"
        fi
        ;;
    
    "help"|*)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  init         - Initialize submodules after clone"
        echo "  update       - Update submodules to latest remote commits"
        echo "  status       - Show submodule status"
        echo "  push-changes - Commit and push submodule changes"
        echo "  help         - Show this help"
        ;;
esac
