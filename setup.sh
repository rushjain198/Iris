#!/bin/bash
# One-time setup: Python venv + dependencies. Models (~41 MB) download on first run into models/.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3.12}
command -v "$PY" >/dev/null || PY=python3
"$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
echo "Ready. Run: ./go.sh glasses   (or ./go.sh webcam)"
