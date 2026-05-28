#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
VENV=".venv/bin/python"

if [[ ! -x "$VENV" ]]; then
  echo "Missing venv. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [[ ! -d "/Volumes/FileStore/ASTRO/AARO/NINA/LIGHT" ]]; then
  echo "FileStore not mounted; attempting auto-mount..."
  "$(dirname "$0")/../.cursor/hooks/ensure-filestore.sh" || true
fi

if [[ ! -d "/Volumes/FileStore/ASTRO/AARO/NINA/LIGHT" ]]; then
  echo "FileStore not mounted at /Volumes/FileStore/ASTRO/AARO/NINA/LIGHT"
  exit 1
fi

echo "==> Analyzing LIGHT frames..."
"$VENV" night_analysis.py

echo "==> Generating plots..."
"$VENV" night_visualization.py

echo
echo "Done. Outputs:"
find ../output/night-activity -type f | sort
