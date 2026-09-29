# LLM 推理服务部署练习

在 Vast.ai GPU 实例上用 vLLM / SGLang 部署 Qwen3.5 系列，提供 OpenAI 兼容 API，并做压测、量化和框架对比。

背景、目标和进度见 [HANDOFF.md](HANDOFF.md)。

## 目录

| 目录 | 内容 |
|---|---|
| `configs/` | 每个部署方案一份 `.env`，从 `_template.env` 复制 |
| `scripts/` | 装环境、下模型、启动服务、一键部署 |
| `bench/` | 压测与对比报告 |
| `eval/` | 量化效果对比用的评测集 |
| `gateway/` | vLLM 前的 API 网关（鉴权、限流、日志） |
| `results/` | 压测结果 JSON（destroy 实例前提交） |

## 新实例恢复

```bash
git clone https://github.com/Leroyyyyyyyyy/locnvda.git && cd locnvda
scripts/deploy.sh configs/qwen3.5-0.8b.env    # 装环境 → 下模型 → 启动 → 冒烟测试
```

| 脚本 | 作用 |
|---|---|
| `scripts/deploy.sh <config>` | 一键部署（下面几步串起来，后台运行 vLLM） |
| `scripts/setup.sh` | 装 uv、Python venv、vLLM（版本固定） |
| `scripts/download.sh <config>` | 从 HF 下载模型到 `/workspace/models`（没有则 `./models`） |
| `scripts/serve_vllm.sh <config>` | 前台启动 vLLM；`DRY_RUN=1` 只打印命令 |
| `scripts/smoke_test.sh <config>` | 调 `/health`、`/v1/models`、`/v1/chat/completions` |
| `scripts/stop.sh` | 停止后台 vLLM |

临时改参数不用改配置文件，环境变量优先：

```bash
GPU_MEMORY_UTILIZATION=0.5 MAX_MODEL_LEN=4096 scripts/deploy.sh configs/qwen3.5-0.8b.env
```

服务默认只监听 `127.0.0.1`，从本地电脑用 SSH 隧道访问（端口和地址用 Vast 控制台给的）：

```bash
ssh -p <vast-ssh-port> root@<vast-ip> -L 8000:127.0.0.1:8000
```
