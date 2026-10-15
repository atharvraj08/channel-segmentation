#!/bin/bash
# Usage: bash agents/run_agent.sh <agent_name>
NAME="$1"
cd "$(dirname "$0")/.." || exit 1
echo "=== $NAME started $(date) ===" | tee "agents/logs/$NAME.log"
claude -p "$(cat agents/tasks/$NAME.md)" --dangerously-skip-permissions 2>&1 | tee -a "agents/logs/$NAME.log"
echo "=== $NAME exited code ${PIPESTATUS[0]} at $(date) ===" | tee -a "agents/logs/$NAME.log"
touch "agents/status/$NAME.finished"
