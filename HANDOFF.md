# HANDOFF：LLM 推理服务部署学习项目

> 用途：新会话 / 新实例接手时先读这份文档。记录目标、约束、设计决定和当前进度。
> 最后更新：2026-10-02

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
│   ├── qwen3.5-27b-b-fp8-tp2-2x24g.env
│   ├── qwen3.5-27b-b-awq-tp2-2x24g.env
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

> 注意：Qwen3.5 是混合注意力架构，KV cache 估算见 8.2。

## 8. 已确认事项（2026-09-29 查 HF API）

### 8.1 仓库名与权重大小

全部 **非 gated、Apache-2.0**，不需要 HF token（但带 token 下载限速更宽松）。

| 用途 | repo id | 权重 | 备注 |
|---|---|---|---|
| 流程跑通 | `Qwen/Qwen3.5-0.8B` | 1.7 GB | 官方 |
| 流程跑通 | `Qwen/Qwen3.5-4B` | 9.3 GB | 官方 |
| 9B BF16 | `Qwen/Qwen3.5-9B` | 19.3 GB | 官方；24GB 卡上 KV cache 只剩约 2GB |
| 9B AWQ | `cyankiwi/Qwen3.5-9B-AWQ-4bit` | 9.1 GB | 社区；备选 `QuantTrio/Qwen3.5-9B-AWQ`（12.4 GB） |
| 9B FP8 | `RedHatAI/Qwen3.5-9B-FP8-dynamic` | 14.0 GB | 社区；**官方没有 9B FP8/Int4** |
| 27B BF16（方案 c） | `Qwen/Qwen3.5-27B` | 55.6 GB | 官方 |
| 27B FP8（方案 b） | `Qwen/Qwen3.5-27B-FP8` | 30.9 GB | 官方 |
| 27B AWQ（方案 a/b） | `cyankiwi/Qwen3.5-27B-AWQ-4bit` | 20.1 GB | 社区；备选 `QuantTrio/Qwen3.5-27B-AWQ`（21.9 GB） |
| 27B GPTQ | `Qwen/Qwen3.5-27B-GPTQ-Int4` | 30.2 GB | 官方，但**放不进 24GB**（见下） |

- 官方**没有 AWQ**，只有 FP8 和 GPTQ-Int4（且只有 27B 及以上才有）。
- 所有 `-Base` 版本是预训练底座，部署服务用不带 `-Base` 的版本。
- 9B 社区量化的格式（2026-10-01 查 config.json）：两个都是 **compressed-tensors**，配置里 `QUANTIZATION` 必须留空让 vLLM 自动识别。
  - `cyankiwi/Qwen3.5-9B-AWQ-4bit`：名字叫 AWQ，实际是 W4A16 int4、group_size=32；除 lm_head 和线性注意力 `in_proj_a/b` 外全部 Linear 都量化。
  - `RedHatAI/Qwen3.5-9B-FP8-dynamic`：W8A8（权重按通道静态、激活按 token 动态）；**线性注意力层的投影全部保留 BF16**，只量化 full attention 和 MLP，所以 14GB。3090 上没有 FP8 单元，会退化为 W8A16。

### 8.2 关键发现：架构与显存

- **是多模态模型**：架构 `Qwen3_5ForConditionalGeneration`（HF 标签 image-text-to-text），带视觉编码器。只做文本服务时 vLLM 可加 `--language-model-only` 省显存（模型卡给出的参数）。
- **混合注意力**：27B 共 64 层，只有 16 层是标准 full attention，48 层是线性注意力（`full_attention_interval=4`）。9B 为 32 层中 8 层 full attention。
  - 只有 full attention 层产生随长度增长的 KV cache；线性注意力层每个序列一份**固定大小**的状态。
  - KV cache 估算（BF16）：每 token = full 层数 × 2(K,V) × 4 KV heads × 256 head_dim × 2 字节
    - 27B：16 × 2 × 4 × 256 × 2 = **64 KB/token** → 1 GB ≈ 16K tokens
    - 9B：8 × 2 × 4 × 256 × 2 = **32 KB/token** → 1 GB ≈ 32K tokens
  - 比同尺寸纯 Transformer 小很多，这是方案 a 仍有可能跑起来的原因。
- **官方 GPTQ-Int4 为什么 30GB**：它的量化配置排除了所有 `attn`（包括线性注意力）、视觉、MTP 层，只量化 MLP，所以比社区 AWQ 大 10GB。
- **方案 a 风险**：社区 AWQ 权重 20–22GB，24GB 卡上留给 KV cache 和激活的只有 1–2GB。需要 `--language-model-only`、较高的 `--gpu-memory-utilization`、较小的 `--max-model-len`，可能还要 `--enforce-eager`（不用 CUDA graph，省显存但会变慢）。能否跑通要实测。
- 模型卡建议上下文 ≥128K 以保留 thinking 能力；24GB 方案做不到，压测时统一用非 thinking 模式或固定输出长度。

### 8.3 服务相关

- **默认 thinking 模式**（输出 `<think>...</think>`）。vLLM 加 `--reasoning-parser qwen3` 把思考内容拆到单独字段；请求里用 `"chat_template_kwargs": {"enable_thinking": false}` 关闭。**压测必须固定这一项**，否则输出长度不可比。
- 工具调用：`--enable-auto-tool-choice --tool-call-parser qwen3_coder`。
- 自带 MTP 投机解码：`--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`，可作为后续优化对比项。
- 框架版本：模型卡发布时（2026-02）要求 vLLM / SGLang 的 main 分支；当前 PyPI 最新是 **vLLM 0.30.0、SGLang 0.5.20**，应已支持，在实例上首次部署时验证并把版本号固定到脚本里。

### 8.4 仍待确认

- [x] Vast 镜像：用 Vast 官方 CUDA 基础模板（SSH 启动），不用 vLLM / PyTorch 模板。vLLM 0.30.0 依赖 torch 2.13.0（有 cu126/cu129/cu130/cu132 版本），CUDA 运行时由 pip 包自带，关键看宿主机驱动：筛选 Max CUDA ≥ 13.0（最低 12.9）的机器。
- [ ] 社区量化版本的效果（阶段 3 用 eval 集对比官方 BF16）

## 9. 当前进度

- [x] 需求整理、本 handoff 文档
- [x] 本地 git 仓库（main 分支）+ 目录骨架 + `configs/_template.env`
- [x] GitHub 仓库（public）：https://github.com/Leroyyyyyyyyy/locnvda
- [x] 阶段 1a：部署脚本已写好（`scripts/`，配置 `configs/qwen3.5-0.8b.env`、`qwen3.5-4b.env`），本地 dry run 通过
- [x] 阶段 1b-1：Vast 3090 上跑通 0.8B（vLLM 0.30.0 + `--language-model-only` 均可用），数据见 `results/stage1-baseline.md`
- [x] 阶段 1b-2：跑通 4B（Max CUDA 13 机器），基线表已补全
- [x] 阶段 1c：参数实验（gpu-memory-utilization、max-model-len、language-model-only、kv-cache-dtype），结论见 `results/stage1-baseline.md`
- [x] 阶段 2a：压测脚本（`bench/bench.py`、`report.py`、`run_suite.sh`），本地用模拟服务端测试通过
- [x] 阶段 2b：0.8B / 4B 标准套件跑通，分析见 `results/stage2-analysis.md`
- [ ] 阶段 3：量化对比
  - [x] 3a：9B 三份配置 `configs/qwen3.5-9b-{bf16,awq,fp8}.env`，除模型外参数完全相同，本地 dry run 通过
  - [ ] 3b：同一台实例依次部署三份配置，记录显存、跑标准压测套件
  - [ ] 3c：效果对比（`eval/` 评测集，含 FP8 KV cache 的精度影响）
- [ ] 阶段 4：API 网关（当前学习重点）
  - [x] 4a：`gateway/app.py` + `config.py` 异步代理，支持 models / completions / chat completions，普通 JSON 与 SSE 原始字节转发。
  - [x] 4a 本地验收：真实 localhost TCP 模拟服务覆盖请求/响应透传、SSE 不缓冲、断连清理（含等待响应头阶段）、错误/超时、压缩和连接池释放。CPU 演示服务为 `gateway/mock_upstream.py`，安装/运行/原理见 `gateway/README.md`。
  - [x] 4a 用户手动验证：模型列表、SSE、停止模拟上游后的 502、上游重启后恢复 200，网关无需重启。
  - [ ] 4a 真实 vLLM 验收与直连 / 网关开销对比（当前没有限流或在途并发保护，只监听本机；独立 `.venv-gateway`，不修改推理环境依赖）。
  - [x] 4b 实现：`gateway/auth.py` 静态 key 注册表，`GATEWAY_API_KEYS` 为必填 JSON（调用方 ID → key），无/错/重复 Authorization 返回 401，认证先于读请求体；内部只记录 `request.state.caller_id`。客户端 Authorization 删除，上游独立使用 `GATEWAY_UPSTREAM_API_KEY`，未配置就不发送授权头；`/health` 可匿名访问。
  - [x] 4b 自动验收：含身份区分/防伪造、拒绝请求不上游、凭证隔离、配置失败时拒绝启动、错误/对象 repr 不回显密钥；全套 31 项测试通过（含 4a 与 setup 回归），无需 GPU。
  - [x] 4b 用户已手动验证：无 key 返回 401，正确 key 返回 200 和模型列表。
  - [ ] 4b 其余手动验收：错误 key 的 401、带 key 的 SSE；真实 vLLM 上游鉴权仍待验收（对应本地自动测试已覆盖）。
  - [ ] 4c：按 key 限流 + 全局在途推理请求保护。
  - [ ] 4d：结构化日志与网关压测。
- [ ] 阶段 5：SGLang 对比
- [ ] 阶段 6：27B 方案 a / b / c
  - 当前执行顺序调整：用户选择先做 27B 方案 b（同机 2×4090，TP=2），先 FP8 后 AWQ；9B 配置保留，BF16 基准需另租大显存卡。
  - [x] 方案 b 两份配置已写好，本地 dry run 通过；HF config.json 确认官方 FP8 为 `fp8` / 128×128 block，社区 AWQ 为 `compressed-tensors` / int4 / group_size=32，两者都自动识别。
  - 实例验机确认：2×RTX 4090，各 24564MiB，驱动 615.71.09；GPU0/GPU1 拓扑为 NODE，同一 NUMA 节点。系统可见内存 503GiB（可能为宿主机总量），磁盘可用约 150G；通信性能、量化 kernel 及显存余量待实测。
  - setup 退出的实际原因已确认：驱动 615.71.09 的表头改为 `CUDA UMD Version: 13.4`，不是原来的 `CUDA Version`（最初误判为空白格式问题）。解析已兼容新旧字段和空白，无效值与命令失败都会明确报错；13.4 驱动选择 cu130 构建。
  - 回归测试 `tests/test_setup.py`：用模拟命令运行 setup，不联网、不安装包，覆盖实例原始表头、新旧字段、空白、CUDA 12/13、N/A、字段缺失及 nvidia-smi 失败。单独运行 `python3 -m unittest discover -s tests -p test_setup.py -v`（全套测试使用 `.venv-gateway/bin/python -m unittest discover -s tests -v`）；实例已安装成功：vLLM 0.30.0、torch 2.13.0+cu130、CUDA 13.0、gpus=2。
  - [x] 27B 官方 FP8 / TP=2 部署和冒烟测试成功，启动基线见 `results/stage3-27b-baseline.md`；每卡 nvidia-smi 占用 22072MiB。TP1 报告 weights+non-torch=14.39GiB、激活=0.75GiB、CUDA graph=0.12GiB、KV cache=6.1GiB；EngineCore 报告容量 122398 token，满 8192 上下文最大并发 14.94x（整个 TP=2 服务，不能再乘 2）；实际 kernel 已确认是 Marlin FP8 weight-only（非原生 W8A8）。
  - [x] 27B FP8 标准压测：chat 528/528、long 96/96 成功，终端摘要分析见 `results/stage3-27b-fp8-analysis.md`；chat 吞吐 48.5→319.0 tok/s（并发 1→64），32→64 仅 +1.6%。报告 `results/report-20261001-151122.md` 和两份原始 JSON 已拉回并核查，尾延迟与日志证据见运行核查文档。
  - [x] AWQ / TP=2 相同参数部署和冒烟测试成功，15:23:22 完成；TP0 报告 weights+non-torch=9.24GiB、激活=0.75GiB、CUDA graph=0.12GiB、KV cache=11.25GiB，每卡 nvidia-smi 占用 22086MiB。对比记录见启动基线。
  - [x] AWQ 缓存容量补录：226484 token，满 8192 上下文最大并发 27.65x，比 FP8 token 容量增加约 85%。
  - [x] AWQ 标准压测：chat 528/528、long 96/96 成功；与 FP8 的终端摘要对比见 `results/stage3-27b-quant-analysis.md`。AWQ chat 并发 1 吞吐 +39.6%、并发 64 +10.2%；long 并发 16 +6.8%。并发 64 的 TTFT 更低但 TPOT 更高，不能宣称所有延迟指标更好。
  - [x] 四份 JSON、两轮报告、合并报告和两份服务日志已拉回本地。1248/1248 成功，所有档位输入输出长度精确、无短输出，参数一致，合并报告重生成逐字节相同。
  - [x] 运行核查见 `results/stage3-27b-runtime-findings.md`：FP8=`MarlinFP8ScaledMMLinearKernel` weight-only，AWQ=`MarlinLinearKernel`；P2P 不可用或测试失败禁用自定义 all-reduce，实际 PYNCCL/NCCL 2.29.7。chat 并发 64 的采样最大 Running 为 FP8 50 / AWQ 64，缓存最高 99.2% / 68.7%；未发现抢占日志，不能证明无抢占。
  - [x] 尾延迟补录：chat 并发 16 TTFT p99 约 7.8～8s，64 为 FP8 45.50s / AWQ 33.19s；long 并发 16 为 60.88s / 55.81s。不能按 p50 单独推荐生产并发。
  - [ ] 量化质量评测，及可选的通信 / kernel 支持专项实验。

## 10. 下一步

当前用户选择先做阶段 4。4a 本地手动验证已完成，4b 鉴权实现与自动测试已通过，用户已验证无 key 的 401 和正确 key 的 200。下一步进入 4c：按调用方限流和全局在途并发保护，仍可用本地模拟上游完成；错误 key / 带 key 的 SSE 手动验证、真实 vLLM 验收、质量评测与多卡专项仍保留为待办。

1. 两轮结果、报告和日志已经安全保存到本地；如不继续使用 GPU，可 destroy 停止计费。原始日志在忽略的 `logs/`，关键证据摘录已写入 `results/stage3-27b-runtime-findings.md`。
2. 准备独立效果评测；当前只完成显存、速度与运行路径核查，不能仅凭性能决定量化选型。
3. 后续若研究多卡效率，检查实际 P2P 支持和 NCCL 性能；若研究原生 FP8，先确认量化格式与 kernel 支持，不要只按 GPU 型号推断。
