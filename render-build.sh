#!/usr/bin/env bash
set -euo pipefail

echo "Upgrading pip, setuptools and wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing Python dependencies..."
if [ -f Backend/requirements.txt ]; then
  python -m pip install -r Backend/requirements.txt
elif [ -f requirements.txt ]; then
  python -m pip install -r requirements.txt
else
  echo "No requirements file found (Backend/requirements.txt or requirements.txt)"
  exit 1
fi

echo "Dependencies installed successfully."
