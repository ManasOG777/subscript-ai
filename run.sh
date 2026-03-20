#!/bin/bash
cd "$(dirname "$0")"
export PATH="/Users/manasprotimghosh/Library/Python/3.9/bin:/opt/homebrew/bin:$PATH"
echo "Starting SubScript on http://localhost:5050"
open "http://localhost:5050"
python3 app.py
