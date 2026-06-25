#!/bin/zsh
set -euo pipefail

PID="${1:-quick}"
DEMOS="${2:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python procedure_flappybird.py -c 0 -id "$PID" -n "$DEMOS"
python procedure_lunarlander.py -c 0 -id "$PID" -n "$DEMOS"
python procedure_robot.py -c 0 -id "$PID" -n "$DEMOS"
