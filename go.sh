#!/bin/bash
# ./go.sh glasses | webcam [N] | file <path> [--no-speech] [--stub]
cd "$(dirname "$0")"
PY=.venv/bin/python; [ -x "$PY" ] || { echo "Run ./setup.sh first"; exit 1; }
case "$1" in
  glasses) shift; SRC=glasses ;;
  webcam)  shift; SRC="webcam:${1:-0}"; [[ "$1" =~ ^[0-9]+$ ]] && shift ;;
  file)    shift; SRC="$1"; shift ;;
  test)    for t in tests/test_*.py; do echo "$t"; "$PY" -B "$t" 2>&1 | grep -E "^(Ran|OK|FAILED)"; done; exit ;;
  *) echo "usage: ./go.sh glasses | webcam [N] | file <path> | test"; exit 1 ;;
esac
echo "Iris dashboard: http://127.0.0.1:8810   HUD: /hud   Caregiver: /caregiver"
exec "$PY" -B app.py --source "$SRC" "$@"
