#!/usr/bin/env bash
# Forward to scripts/run_web_dashboard.sh
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$DIR/scripts/run_web_dashboard.sh" "$@"
