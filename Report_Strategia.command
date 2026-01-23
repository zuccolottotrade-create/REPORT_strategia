#!/bin/bash
set -euo pipefail

# Root suite: arriva dalla pipeline; fallback se lanci standalone
SUITE_DIR="${PYCHAM_SUITE_DIR:-/Users/claudio 1/Py_SUITE_TRADING}"

# Cartella del progetto (dove c'è scripts/report_strategia.py)
PROJECT_DIR="$SUITE_DIR/4. REPORT strategia"

# Python da usare (venv del progetto)
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

# Script da lanciare
SCRIPT_PATH="$PROJECT_DIR/scripts/report_strategia.py"

# (Opzionale) argomenti CLI per il tuo script
ARGS=""

cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR"
export PYTHONUNBUFFERED=1

echo "== PyCham / Report Strategia =="
echo "Project: $PROJECT_DIR"
echo "Python : $PYTHON_BIN"
echo "Script : $SCRIPT_PATH"
echo ""

# Check rapidi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "❌ ERRORE: python venv non trovato o non eseguibile: $PYTHON_BIN"
  if [ "${PIPELINE_MODE:-0}" != "1" ]; then
    read -r
  fi
  exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
  echo "❌ ERRORE: script non trovato: $SCRIPT_PATH"
  if [ "${PIPELINE_MODE:-0}" != "1" ]; then
    read -r
  fi
  exit 1
fi

"$PYTHON_BIN" "$SCRIPT_PATH" $ARGS

echo ""
echo "Fatto."
if [ "${PIPELINE_MODE:-0}" != "1" ]; then
  echo "Premi INVIO per passare al PROSSIMO MODULO..."
  read -r
fi
exit 0
