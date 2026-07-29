#!/usr/bin/env bash
# LoRA Distillery - launch the UI (run ./setup.sh once first)
set -uo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "[ERROR] No virtual environment found (.venv is missing)."
    echo "        Run ./setup.sh first - it installs everything this needs."
    exit 1
fi
# Not `exec`: we want to report on a non-zero exit. The overwhelmingly common cause
# is an ImportError after a `git pull` added a dependency, so name that fix.
.venv/bin/python app.py
status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "[ERROR] LoRA Distillery exited with an error (status $status). The cause"
    echo "        is in the output above."
    echo
    echo "  Most common cause: you just ran 'git pull' and the new version needs a"
    echo "  dependency you do not have yet. Fix it by re-running:"
    echo "        ./setup.sh"
    echo
    echo "  For a full check of this install:"
    echo "        .venv/bin/python cli.py doctor"
fi
exit "$status"
