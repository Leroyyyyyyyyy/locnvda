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

## API 网关（阶段 4a / 4b / 4c）

`gateway/` 已实现异步 JSON / SSE 代理、API key 鉴权、上游凭证隔离、按调用方令牌桶限流和全局在途推理请求保护；可用 CPU 模拟上游验证，无需租 GPU。
启动时必须设置 `GATEWAY_API_KEYS`，三个 `/v1/...` 路由都要求 Bearer key，`/health` 可匿名访问。
频率超限返回 429，推理并发已满返回 503；不排队，SSE 结束或断连后释放名额。
安装、启动、curl 示例与参数原理见 [gateway/README.md](gateway/README.md)。
**目前使用单进程内存状态，运行一个 worker，仍只应监听本机。**

## 压测

服务用 `deploy.sh` 启动后，在实例上运行（压测客户端和服务在同一台机器，避免网络延迟混进 TTFT）：

```bash
bench/run_suite.sh                          # 标准套件：chat(1K/256) + long(7.5K/256)，自动生成对比报告
python bench/bench.py --input-len 2048 --output-len 128 --concurrency 1,8,32   # 单次自定义
python bench/report.py results/raw/qwen3.5-9b*.json results/raw/qwen3.5-27b*.json --out results/9b-vs-27b.md
```

- 部署时的生效配置（含环境变量覆盖）自动记录在 `logs/deployed.env`，压测结果据此标注模型、量化、TP 等元数据。
- 原始结果在 `results/raw/*.json`，报告在 `results/report-*.md`。实例上没有 GitHub 凭证，**destroy 前在本地电脑把结果拉回来再提交**：

```bash
scp -i ~/.ssh/vast_ed25519 -P <vast-ssh-port> -r "root@<vast-ip>:/workspace/locnvda/results/*" results/
```

服务默认只监听 `127.0.0.1`，从本地电脑用 SSH 隧道访问（端口和地址用 Vast 控制台给的）：

```bash
ssh -p <vast-ssh-port> root@<vast-ip> -L 8000:127.0.0.1:8000
```
