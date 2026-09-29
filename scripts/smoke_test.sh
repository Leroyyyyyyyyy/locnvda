#!/usr/bin/env bash
# 冒烟测试：确认服务活着、模型列表正确、能生成一段回答。
# 用法: scripts/smoke_test.sh configs/<方案>.env
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config "${1:-}"

base="http://127.0.0.1:${PORT:-8000}"
auth=()
[[ -n "${API_KEY:-}" ]] && auth=(-H "Authorization: Bearer $API_KEY")

log "GET /health"
curl -sf "$base/health" >/dev/null && echo ok

log "GET /v1/models"
curl -sf ${auth[@]+"${auth[@]}"} "$base/v1/models"; echo

# enable_thinking=false：Qwen3.5 默认先输出 <think> 思考过程，冒烟测试不需要
log "POST /v1/chat/completions"
curl -sf ${auth[@]+"${auth[@]}"} "$base/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$SERVED_MODEL_NAME\",
    \"messages\": [{\"role\": \"user\", \"content\": \"用一句话介绍你自己\"}],
    \"max_tokens\": 100,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }"
echo
