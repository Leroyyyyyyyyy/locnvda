#!/usr/bin/env bash
# 停止 deploy.sh 启动的 vLLM。
set -euo pipefail
source "$(dirname "$0")/lib.sh"
pidfile="$LOG_DIR/vllm.pid"

[[ -f "$pidfile" ]] || { log "没有 pid 文件，vLLM 没在运行"; exit 0; }
pid="$(cat "$pidfile")"
if kill -0 "$pid" 2>/dev/null; then
  log "停止 vLLM (pid $pid)"
  kill "$pid"
  # vLLM 会带起多个子进程（TP 时每张卡一个 worker），等它们都退出、显存释放
  for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && { log "30s 未退出，强制结束"; kill -9 "$pid"; }
fi
rm -f "$pidfile"
