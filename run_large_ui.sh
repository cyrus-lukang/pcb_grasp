#!/usr/bin/env bash
# 兼容旧入口：转发到全流程 UI
exec "$(cd "$(dirname "$0")" && pwd)/run_ui.sh" "$@"
