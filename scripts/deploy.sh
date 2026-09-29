#!/usr/bin/env bash
# 一键部署：装环境 → 下模型 → 后台启动 vLLM → 等待就绪 → 冒烟测试。
# 用法: scripts/deploy.sh configs/<方案>.env
# 日志: logs/vllm-<方案>.log    停止: scripts/stop.sh
set -euo pipefail
source "$(dirname "$0")/lib.sh"
CONFIG="${1:-}"
load_config "$CONFIG"
cd "$REPO_ROOT"

name="$(basename "$CONFIG" .env)"
mkdir -p "$LOG_DIR"
logfile="$LOG_DIR/vllm-$name.log"
pidfile="$LOG_DIR/vllm.pid"

if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
  die "已有 vLLM 在运行 (pid $(cat "$pidfile"))，先运行 scripts/stop.sh"
fi

[[ -d "$VENV_DIR" ]] || scripts/setup.sh
scripts/download.sh "$CONFIG"

log "后台启动 vLLM，日志: $logfile"
nohup scripts/serve_vllm.sh "$CONFIG" >"$logfile" 2>&1 &
echo $! >"$pidfile"

# 启动要加载权重、profile 显存、编译 CUDA graph，大模型可能要几分钟
timeout="${STARTUP_TIMEOUT:-900}"
log "等待服务就绪（最多 ${timeout}s）"
start=$SECONDS
until curl -sf "http://127.0.0.1:${PORT:-8000}/health" >/dev/null; do
  if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    tail -50 "$logfile" >&2
    die "vLLM 进程退出了，日志见 $logfile"
  fi
  (( SECONDS - start > timeout )) && die "等待超时，日志见 $logfile"
  sleep 5
done
log "服务就绪，用时 $((SECONDS - start))s"

# 启动日志里最值得看的几行：权重占用、KV cache 大小、最大并发
grep -iE "model weights took|kv cache|maximum concurrency|cuda graph" "$logfile" | tail -10 >&2 || true
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv >&2

scripts/smoke_test.sh "$CONFIG"
log "部署完成: http://${HOST:-127.0.0.1}:${PORT:-8000}/v1"
