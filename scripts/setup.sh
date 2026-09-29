#!/usr/bin/env bash
# 装环境：uv + Python 虚拟环境 + vLLM + HF 下载工具。
# 用法: scripts/setup.sh
# 重复运行是安全的（已装的会跳过）。
set -euo pipefail
source "$(dirname "$0")/lib.sh"

# 版本写死：保证每次新开实例装出来的环境一样，压测结果才可比
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

log "检查 GPU"
command -v nvidia-smi >/dev/null || die "找不到 nvidia-smi，这台机器没有 NVIDIA 驱动"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# uv：比 pip 快很多的 Python 包管理器，vLLM 官方推荐用它安装
if ! command -v uv >/dev/null; then
  log "安装 uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "创建虚拟环境 $VENV_DIR (Python $PYTHON_VERSION)"
  uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --torch-backend=auto：uv 根据本机 CUDA 驱动自动选择匹配的 PyTorch 版本
log "安装 vllm==$VLLM_VERSION"
uv pip install "vllm==$VLLM_VERSION" --torch-backend=auto
uv pip install "huggingface_hub[cli]"

log "版本信息"
python -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"
log "setup 完成"
