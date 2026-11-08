#!/bin/bash
# Start the local Kronos research UI. This script does not install packages.

set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is required."
    exit 1
fi

if [ ! -f "app.py" ]; then
    echo "Run this script from the webui directory."
    exit 1
fi

echo "Kronos Web UI"
echo "Install dependencies yourself from webui/requirements.txt before starting."
echo "The server binds to 127.0.0.1 and does not enable the debugger."
echo "Forecasts are research outputs, not paper orders or portfolio returns."
echo

exec python3 app.py
