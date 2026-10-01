# 27B 量化对比：方案 b 启动基线

## 环境与配置

- 2026-10-01 实例启动输出；原始 JSON、报告和日志已拉回本地并核查，未直接访问远端实例。
- 同机 2×RTX 4090，各 24564MiB；GPU 互联为 NODE，同一 NUMA 节点，无 NVLink。
- 驱动 615.71.09，nvidia-smi 表头 `CUDA UMD Version: 13.4`。
- vLLM 0.30.0，PyTorch 2.13.0+cu130，CUDA runtime 13.0，PyTorch 识别到 2 张 GPU。
- FP8 配置：`configs/qwen3.5-27b-b-fp8-tp2-2x24g.env`。
- TP=2，gpu-memory-utilization=0.90，max-model-len=8192，max-num-seqs=64，language-model-only=1，KV cache dtype=auto（BF16），启用 CUDA graph。
- 仓库 `Qwen/Qwen3.5-27B-FP8`：官方 block FP8 权重；日志确认 `MarlinFP8ScaledMMLinearKernel`，本轮为 weight-only FP8 compression，并非原生 W8A8 计算。

## FP8 启动结果

以下预算细分最初来自 `Worker_TP1`；已拉回的日志确认 TP0/TP1 报告相同，nvidia-smi 总占用也相同。

| 项目 | 数值 | 范围 / 含义 |
|---|---|---|
| vLLM 报告总显存 | 23.6 GiB | 单卡，区别于 nvidia-smi 的物理显存总量 |
| 启动时空闲显存 | 22.99 GiB | 单卡 |
| 目标显存预算 | 21.24 GiB | 单卡，util=0.90 |
| weights + non-torch | 14.39 GiB | 单卡消耗，不是纯权重大小 |
| 激活峰值 | 0.75 GiB | 单卡 |
| CUDA graph | 0.12 GiB | 单卡 |
| 当前 KV cache memory | 6.1 GiB | 单卡，日志原值 |
| nvidia-smi memory.used | 22072 MiB | GPU0、GPU1 各自占用 |
| KV cache token 容量 | 122,398 tokens | EngineCore 报告的整个 TP=2 服务的逻辑容量，不再乘以卡数 |
| 满 8192 上下文最大并发 | 14.94x | 122398 ÷ 8192；缓存预算下约 14 个完整满上下文请求，非调度器或性能上限 |

日志同时建议 `--kv-cache-memory=6255602074`（5.83 GiB）以满足请求预算，或 `8142180352`（7.58 GiB）以充分利用显存。当前缓存 6.1 GiB 与这些建议值不完全一致，保留原始数据，不将缓存归约为简单减法的精确结果；本轮不改参数，避免改变对比条件。

补充日志：两个 worker 都提示启用了 CUDA graph memory profiling；util=0.90 在该估算机制下，相当于关闭该机制时的 0.8819。建议 0.9181 是为了恢复旧机制下的缓存容量，并非运行报错；本轮保持 0.90，不因该提示改变参数。

标准 long workload 每请求 7680+256=7936 token，并发 16 时合计 126976 token，超过日志报告的 122398 token 容量。请求未必同时达到最大长度，但此档可能出现缓存压力、排队或抢占；需结合压测和服务日志观察，不预先断言请求失败。短请求还受混合架构每请求固定状态等约束，不能仅用总 token 容量除以输入输出长度推导实际并发。

## FP8 冒烟测试

- `/health` 成功。
- `/v1/models` 返回 `qwen3.5-27b-fp8`，max_model_len=8192。
- `/v1/chat/completions` 成功，finish_reason=stop；输入 16 token，输出 44 token，reasoning_tokens=0。
- 响应 system_fingerprint：`vllm-0.30.0-tp2-09e1adb8`。
- 部署完成时间：实例日志 14:44:24。未提供完整启动时间，不估算耗时。

说明：冒烟测试证明双卡服务能返回正常响应，不代表量化精度、吞吐或通信效率已经验证。

## AWQ 启动结果与显存对比

配置 `configs/qwen3.5-27b-b-awq-tp2-2x24g.env`，仓库 `cyankiwi/Qwen3.5-27B-AWQ-4bit`。与 FP8 相同部署参数；以下 AWQ 细分来自 TP0，两张卡的 nvidia-smi 总占用相同。

| 单卡项目 | FP8 | AWQ |
|---|---|---|
| 目标显存预算 | 21.24 GiB | 21.24 GiB |
| weights + non-torch | 14.39 GiB | 9.24 GiB |
| 激活峰值 | 0.75 GiB | 0.75 GiB |
| CUDA graph | 0.12 GiB | 0.12 GiB |
| 当前 KV cache memory | 6.1 GiB | 11.25 GiB |
| nvidia-smi memory.used | 22072 MiB | 22086 MiB |
| KV cache token 容量（整个 TP=2 服务） | 122398 | 226484 |
| 满 8192 上下文最大并发 | 14.94x | 27.65x |

- AWQ weights+non-torch 每卡少 5.15 GiB（约 -35.8%），KV cache 多 5.15 GiB（约 +84.4%）；这说明省下的空间主要变成缓存，而不是 nvidia-smi 总占用降低。
- AWQ EngineCore 实测 token 容量 226484，满 8192 上下文最大并发 27.65x；逻辑容量比 FP8 增加约 85.0%。long 并发 16 的最大 token 总量 126976 小于该容量，但仍需实测调度和运行时开销，不能据此断言没有缓存压力或抢占。
- AWQ 日志建议固定缓存为 `11789920666` 字节（10.98 GiB）以满足请求预算，或 `13676498944` 字节（12.74 GiB）以充分利用显存。保留原始值，本轮不改变参数。
- `/health`、`/v1/models`、`/v1/chat/completions` 均成功；API model=`qwen3.5-27b-awq`，max_model_len=8192；输入 16 token，输出 43 token，finish_reason=stop，reasoning_tokens=0。
- system_fingerprint=`vllm-0.30.0-tp2-1259f6bb`，部署完成时间 15:23:22。
- 两份回答措辞不同不能说明哪种量化更好；速度与吞吐对比已记录于 `stage3-27b-quant-analysis.md`，质量待独立评测。

## 已完成核查与下一步

- 两版本标准套件均完成，各 624 个正式请求全部成功。四份 JSON、报告和两份服务日志已保存到本地，输入输出长度及合并报告可复现性已核查。
- 性能对比见 [stage3-27b-quant-analysis.md](stage3-27b-quant-analysis.md)，kernel、NCCL/P2P 条件、运行时缓存采样和尾延迟见 [stage3-27b-runtime-findings.md](stage3-27b-runtime-findings.md)。两版本均为 Marlin weight-only 路径；自定义 all-reduce 因 P2P 不可用或测试失败关闭，使用 PYNCCL。
- 尚未做质量评测、通信 profiling 或读取抢占计数。BF16 基准需要另租大显存硬件。
