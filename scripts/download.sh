#!/usr/bin/env bash
# 从 Hugging Face 下载配置里的模型。
# 用法: scripts/download.sh configs/<方案>.env
# 断点续传：中断后重跑会接着下，已下完的文件跳过。
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config "${1:-}"
activate_venv

mkdir -p "$MODELS_DIR"
log "下载 $MODEL_ID -> $MODEL_PATH"
df -h "$MODELS_DIR" | tail -1 >&2   # 下载前看一眼剩余磁盘

# 只下推理需要的文件，跳过其他格式的权重（如 .bin / .pth / gguf）
hf download "$MODEL_ID" --local-dir "$MODEL_PATH" \
  --include "*.json" "*.safetensors" "*.txt" "*.model" "*.jinja" "*.py"

du -sh "$MODEL_PATH" >&2
log "下载完成"
