#!/bin/zsh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_FILE="$SCRIPT_DIR/main.py"
REPORT_FILE="$SCRIPT_DIR/report.html"
VENV_DIR="$SCRIPT_DIR/.ibi_group_project_venv"

cd "$SCRIPT_DIR"

echo "IBI Group Project launcher"
echo "Project folder: $SCRIPT_DIR"
echo

if [ ! -f "$PROJECT_FILE" ]; then
    echo "Error: main.py was not found next to this launcher."
    echo "Please keep this launcher, main.py, and report files in the same folder."
    echo
    read "?Press Enter to close..."
    exit 1
fi

if [ ! -f "$REPORT_FILE" ]; then
    echo "Error: report.html was not found next to this launcher."
    echo "Please keep this launcher, main.py, and report files in the same folder."
    echo
    read "?Press Enter to close..."
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python - <<'PY'
import importlib.util
import subprocess
import sys

required_packages = ["numpy", "scipy", "matplotlib"]
missing_packages = [
    package for package in required_packages
    if importlib.util.find_spec(package) is None
]

if missing_packages:
    print("Installing missing requirements:", ", ".join(missing_packages))
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        *missing_packages,
    ])
else:
    print("All requirements are already installed in the virtual environment.")
PY

echo
echo "Running project..."
python "$PROJECT_FILE"

echo
if [ -f "$REPORT_FILE" ]; then
    echo "Opening HTML report..."
    open "$REPORT_FILE"
fi

echo
echo "Finished. Output figures and report are saved in the ibi_outputs folder."
read "?Press Enter to close..."
