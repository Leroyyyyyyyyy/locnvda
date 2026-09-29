#!/usr/bin/env bash
# 公共函数，被其他脚本 source。不单独执行。

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
# 模型目录：Vast 实例的大磁盘一般挂在 /workspace，有就用它
if [[ -z "${MODELS_DIR:-}" ]]; then
  if [[ -d /workspace && -w /workspace ]]; then MODELS_DIR=/workspace/models; else MODELS_DIR="$REPO_ROOT/models"; fi
fi
LOG_DIR="$REPO_ROOT/logs"

log() { echo "[$(date '+%H:%M:%S')] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# 读取部署配置：set -a 让 source 进来的变量自动 export，子进程（vllm）也能看到
load_config() {
  local config="${1:-}"
  [[ -n "$config" ]] || die "用法: $0 configs/<方案>.env"
  [[ -f "$config" ]] || die "配置文件不存在: $config"
  # 根目录 .env 放密钥（HF_TOKEN 等），不进 git
  if [[ -f "$REPO_ROOT/.env" ]]; then set -a; source "$REPO_ROOT/.env"; set +a; fi
  # 命令行环境变量优先于配置文件，方便做参数实验而不改文件：
  #   GPU_MEMORY_UTILIZATION=0.5 scripts/deploy.sh configs/qwen3.5-0.8b.env
  local var overrides=()
  for var in $(grep -oE '^[A-Z_][A-Z0-9_]*=' "$config" | tr -d '='); do
    [[ -n "${!var+x}" ]] && overrides+=("$var=${!var}")
  done
  set -a; source "$config"; set +a
  local kv name
  for kv in ${overrides[@]+"${overrides[@]}"}; do
    name="${kv%%=*}"
    [[ "${!name}" != "${kv#*=}" ]] && log "覆盖配置: $kv"
    export "$kv"
  done
  [[ -n "${MODEL_ID:-}" ]] || die "$config 里没有设置 MODEL_ID"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_ID}"
  # Qwen/Qwen3.5-0.8B -> $MODELS_DIR/Qwen__Qwen3.5-0.8B
  MODEL_PATH="$MODELS_DIR/${MODEL_ID//\//__}"
}

activate_venv() {
  [[ -f "$VENV_DIR/bin/activate" ]] || die "虚拟环境不存在，先运行 scripts/setup.sh"
  source "$VENV_DIR/bin/activate"
}
