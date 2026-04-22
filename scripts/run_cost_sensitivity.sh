#!/bin/bash
# 兼容入口：与 run_constraint_sweep.sh 相同的 5 点 ε 扫描（原 0.01/0.1/0.5 已弃用）
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/run_constraint_sweep.sh" "$@"
