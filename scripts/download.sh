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
# 注意：hf CLI 2.x 的 --include 每次只接一个模式，必须重复写；
# 写成 --include a b c 时 b、c 会被当成文件名，且 --include 整体被忽略
includes=()
for pattern in "*.json" "*.safetensors" "*.txt" "*.model" "*.jinja" "*.py"; do
  includes+=(--include "$pattern")
done
hf download "$MODEL_ID" --local-dir "$MODEL_PATH" "${includes[@]}"

# 下载命令成功不代表文件齐全，检查 vLLM 启动必需的文件
for f in config.json tokenizer_config.json; do
  [[ -f "$MODEL_PATH/$f" ]] || die "下载不完整，缺少 $MODEL_PATH/$f"
done
ls "$MODEL_PATH"/*.safetensors >/dev/null 2>&1 || die "下载不完整，没有 .safetensors 权重文件"

du -sh "$MODEL_PATH" >&2
log "下载完成"
