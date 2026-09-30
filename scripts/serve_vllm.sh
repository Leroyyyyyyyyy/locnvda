#!/usr/bin/env bash
# 按配置启动 vLLM OpenAI 兼容服务（前台运行）。
# 用法: scripts/serve_vllm.sh configs/<方案>.env
#       DRY_RUN=1 scripts/serve_vllm.sh configs/<方案>.env   # 只打印命令，不启动
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config "${1:-}"

# 用本地下载好的路径启动；--served-model-name 让 API 里的 model 字段仍然是好读的名字
args=(
  serve "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "${HOST:-127.0.0.1}"
  --port "${PORT:-8000}"
  --dtype "${DTYPE:-auto}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --max-model-len "${MAX_MODEL_LEN:-8192}"
  --max-num-seqs "${MAX_NUM_SEQS:-64}"
  --kv-cache-dtype "${KV_CACHE_DTYPE:-auto}"
)
# 可选参数：配置里为空就不传，交给 vLLM 默认值
[[ -n "${QUANTIZATION:-}" ]]         && args+=(--quantization "$QUANTIZATION")
[[ "${LANGUAGE_MODEL_ONLY:-0}" == 1 ]] && args+=(--language-model-only)
[[ "${ENFORCE_EAGER:-0}" == 1 ]]       && args+=(--enforce-eager)
[[ -n "${REASONING_PARSER:-}" ]]     && args+=(--reasoning-parser "$REASONING_PARSER")
[[ -n "${API_KEY:-}" ]]              && args+=(--api-key "$API_KEY")
# 临时加参数不改配置：EXTRA_ARGS="--enable-prefix-caching" scripts/serve_vllm.sh ...
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]]           && args+=($EXTRA_ARGS)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'vllm'; printf ' %q' "${args[@]}"; echo
  exit 0
fi

activate_venv
[[ -f "$MODEL_PATH/config.json" ]] || die "模型不在 $MODEL_PATH，先运行 scripts/download.sh $1"
log "启动: vllm ${args[*]}"
save_deployed_config "vllm ${args[*]}"
exec vllm "${args[@]}"
