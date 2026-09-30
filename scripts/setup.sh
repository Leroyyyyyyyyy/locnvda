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

# 按驱动支持的最高 CUDA 版本选 vLLM 构建：
#   - PyPI 上的默认 wheel 按 CUDA 13 编译，驱动必须 >= 13.0
#   - 驱动是 12.x 时用 GitHub Release 上的 +cu129 wheel，配 cu129 的 PyTorch
#     （CUDA 同一大版本内“小版本兼容”，12.9 编译的程序可以跑在 12.8 驱动上）
driver_cuda="$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | awk '{print $3}')"
[[ -n "$driver_cuda" ]] || die "读不到驱动的 CUDA 版本"
driver_major="${driver_cuda%%.*}"
log "驱动最高支持 CUDA $driver_cuda"

if (( driver_major >= 13 )); then
  vllm_spec="vllm==$VLLM_VERSION"
  torch_backend="cu130"
elif (( driver_major == 12 )); then
  vllm_spec="vllm @ https://github.com/vllm-project/vllm/releases/download/v$VLLM_VERSION/vllm-$VLLM_VERSION+cu129-cp38-abi3-manylinux_2_28_$(uname -m).whl"
  torch_backend="cu129"
else
  die "驱动 CUDA $driver_cuda 太旧，租 Max CUDA >= 12.8 的机器"
fi

# --torch-backend：让 uv 从对应 CUDA 版本的 PyTorch 源安装 torch
log "安装 vLLM $VLLM_VERSION (torch backend: $torch_backend)"
uv pip install "$vllm_spec" --torch-backend="$torch_backend"
uv pip install "huggingface_hub[cli]"

log "版本信息"
python -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"
log "setup 完成"
