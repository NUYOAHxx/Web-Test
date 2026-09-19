#!/usr/bin/env bash
# Forward to scripts/run_10dof_rviz.sh
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$DIR/scripts/run_10dof_rviz.sh" "$@"
