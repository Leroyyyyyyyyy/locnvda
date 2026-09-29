# HANDOFF：LLM 推理服务部署学习项目

> 用途：新会话 / 新实例接手时先读这份文档。记录目标、约束、设计决定和当前进度。
> 最后更新：2026-09-29

---

## 1. 背景

- 我是后端工程师，目标是学会**生产级** LLM 推理服务部署（不是个人本地聊天）。
- 本地没有 NVIDIA 显卡，GPU 从 **Vast.ai** 租用。
- **最终目标：部署 Qwen3.5-27B**。9B 及更小的模型只用来练手。
- 学习优先：每一步都要解释关键参数的作用和原理，不只给命令。

## 2. 技术选型

| 项目 | 选择 |
|---|---|
| 推理框架 | vLLM（先学）→ 之后用 SGLang 对比 |
| 接口 | OpenAI 兼容 API（`/v1/chat/completions`、`/v1/completions`、`/v1/models`） |
| 模型路线 | Qwen3.5-4B / 0.8B 跑通流程 → Qwen3.5-9B 练全套 → Qwen3.5-27B |

> ⚠️ **部署前必须先在 Hugging Face 上确认准确仓库名**（包括官方 AWQ / FP8 量化版本是否存在、命名是否一致），不要凭记忆写死。确认结果记录到第 8 节。

## 3. 分阶段目标

| # | 阶段 | 产出 | 硬件 |
|---|---|---|---|
| 1 | 部署 | 一键部署脚本：装环境 → 下模型 → 启动 vLLM | 24GB 单卡（3090/4090） |
| 2 | 压测 | 压测脚本：TTFT、tokens/s、不同并发吞吐，输出对比表 | 同上 |
| 3 | 量化对比 | BF16 vs AWQ / FP8：显存占用、速度、效果 | 同上 |
| 4 | API 网关 | vLLM 前的自研服务：鉴权、限流、日志 | 同上 |
| 5 | 框架对比 | 同模型、同压测脚本跑 SGLang，重点看前缀缓存（RadixAttention） | 同上 |
| 6 | 部署 27B | 三种方案都要支持（见下） | 见下 |

### 阶段 6：27B 三种方案（BF16 权重约 54GB）

| 方案 | 硬件 | 量化 | 并行 | 定位 / 取舍 |
|---|---|---|---|---|
| a | 单卡 24GB | AWQ 4bit | TP=1 | 最便宜；KV cache 空间小，并发和上下文受限 |
| b | 2×24GB | FP8 或 AWQ | TP=2 | **重点练多卡**（tensor parallel、NCCL 通信） |
| c | 单卡 80GB（A100/H100）或 2×48GB | BF16 原版 | TP=1 / TP=2 | 效果基准（quality baseline） |

## 4. 设计要求

1. **全部参数化**：模型名、量化方式、`--tensor-parallel-size`、`--max-model-len` 等全部走配置，**换模型不改代码**。
2. **一个配置 = 一个部署方案**：9B、27B-a/b/c 各一份配置文件。
3. 压测脚本必须能直接输出：
   - 9B vs 27B 对比
   - 27B 各方案（a/b/c）之间的对比
4. 压测结果带元数据（模型、量化、TP、GPU 型号、max-model-len、框架及版本），便于跨实例汇总。

## 5. 约束

- 所有脚本和配置放 **GitHub 仓库**；Vast 实例用完即 destroy，新实例 `git clone` 即恢复。
- **模型不备份**，每次从 Hugging Face 重新下载。
- 压测结果需要在 destroy 前提交回仓库（或拉回本地），否则会丢。
- 实例磁盘：9B 及以下 **50–80GB**；27B **150GB**。

## 6. 建议仓库结构（待实现）

```
.
├── HANDOFF.md
├── README.md
├── configs/                  # 每个部署方案一份
│   ├── qwen3.5-0.8b.env
│   ├── qwen3.5-4b.env
│   ├── qwen3.5-9b-bf16.env
│   ├── qwen3.5-9b-awq.env
│   ├── qwen3.5-9b-fp8.env
│   ├── qwen3.5-27b-a-awq-1x24g.env
│   ├── qwen3.5-27b-b-tp2-2x24g.env
│   └── qwen3.5-27b-c-bf16.env
├── scripts/
│   ├── setup.sh              # 装环境（python/venv、vllm、sglang）
│   ├── download.sh           # 从 HF 下载模型
│   ├── serve_vllm.sh         # 读 config 启动 vLLM
│   ├── serve_sglang.sh       # 读 config 启动 SGLang
│   └── deploy.sh             # 一键：setup → download → serve
├── bench/
│   ├── bench.py              # 压测：TTFT / TPOT / tokens/s / 并发吞吐
│   ├── report.py             # 汇总多次结果 → 对比表（markdown）
│   └── prompts/              # 固定测试集（含共享前缀场景，用于阶段 5）
├── eval/                     # 量化效果对比用的小评测集
├── gateway/                  # 阶段 4：鉴权、限流、日志
└── results/                  # 压测结果 JSON（提交进仓库）
```

## 7. 需要讲清原理的关键参数（每步用到时详细解释）

| 参数 | 一句话 |
|---|---|
| `--gpu-memory-utilization` | vLLM 可用的显存比例；扣掉权重和激活后剩下的全给 KV cache |
| `--max-model-len` | 单请求最大上下文（输入+输出），决定单序列 KV cache 上限 |
| `--tensor-parallel-size` | 把每层权重切到 N 张卡上，卡间每层都要通信 |
| `--quantization` | awq / fp8 等；影响权重显存、计算速度、精度 |
| `--kv-cache-dtype` | KV cache 精度（如 fp8），省显存换并发 |
| `--max-num-seqs` | 同时调度的最大序列数（并发上限） |
| `--max-num-batched-tokens` | 单步最多处理 token 数，影响 prefill/decode 调度 |
| `--enable-prefix-caching` | 前缀缓存；与 SGLang RadixAttention 对比 |
| `--enable-chunked-prefill` | 长 prompt 分块 prefill，降低对 decode 的阻塞 |

压测指标：**TTFT**（首 token 延迟）、**TPOT/ITL**（每 token 间隔）、**输出 tokens/s**（单请求 & 总吞吐）、**请求吞吐 req/s**、P50/P90/P99。

> 注意：Qwen3.5 的具体架构（层数、KV head 数、是否含线性注意力层等）会影响 KV cache 估算，部署前从模型 `config.json` 确认后再算显存账。

## 8. 待确认事项

- [ ] HF 仓库名：Qwen3.5-0.8B / 4B / 9B / 27B 的准确 repo id
- [ ] 是否有官方 AWQ / FP8 版本；没有的话用哪个社区版本或自己量化
- [ ] vLLM / SGLang 当前版本对 Qwen3.5 的支持情况（最低版本要求）
- [ ] 模型是否需要 HF token（gated）
- [ ] Vast 镜像选择（官方 vLLM 镜像 vs CUDA 基础镜像 + 自装）

## 9. 当前进度

- [x] 需求整理、本 handoff 文档
- [x] 本地 git 仓库（main 分支）+ 目录骨架 + `configs/_template.env`
- [ ] 建 GitHub 仓库并推送初始结构
- [ ] 阶段 1：部署脚本（先用 0.8B/4B 跑通）
- [ ] 阶段 2：压测脚本
- [ ] 阶段 3：量化对比
- [ ] 阶段 4：API 网关
- [ ] 阶段 5：SGLang 对比
- [ ] 阶段 6：27B 方案 a / b / c

## 10. 下一步

1. 在 HF 上确认模型仓库名，填入第 8 节。
2. 初始化 git 仓库，建第 6 节目录骨架，推到 GitHub。
3. 写 `configs/qwen3.5-0.8b.env` + `scripts/deploy.sh`，在 Vast 24GB 实例上跑通并用 `curl` 调通 OpenAI 接口。
