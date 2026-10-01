# 27B 双卡量化：原始结果与日志核查

2026-10-01，结果和日志已从实例复制到本地 `/Users/dld/locnvda/`。原始运行日志在 `logs/`（git 忽略），以下保存关键摘录与结论，便于实例销毁后复查。

## 文件与数据完整性

- 四份 27B JSON 均能解析；FP8/AWQ 的 chat 各 528 个、long 各 96 个正式请求，合计 1248 个，失败为 0。
- 所有档位 mean_prompt_tokens 与目标 1024/7680 一致，mean_output_tokens=256，short_outputs=0。
- vLLM 均为 0.30.0，TP=2，CONFIG_OVERRIDES 为空；除配置名、仓库、API 名称和启动命令中的模型路径外，记录的部署参数一致。
- 两轮独立报告和合并报告已拉回；用当前 `bench/report.py` 从四份 JSON 重生成合并报告，与 `results/stage3-27b-awq-vs-fp8.md` 逐字节相同。
- FP8/AWQ 两份日志、AWQ deployed.env 均已保存。deployed.env 会被下一次部署覆盖，FP8 生效配置已包含在其 JSON 内。

## 实际量化 kernel：本轮不是原生 FP8 计算对比

FP8 日志关键摘录：

```text
Selected MarlinFP8ScaledMMLinearKernel for Fp8LinearMethod
Your GPU does not have native support for FP8 computation but FP8 quantization is being used. Weight-only FP8 compression will be used leveraging the Marlin kernel. This may degrade performance for compute-heavy workloads.
```

来源：`logs/vllm-qwen3.5-27b-b-fp8-tp2-2x24g.log`，启动日志行 35、57。

AWQ 日志：

```text
Using MarlinLinearKernel for CompressedTensorsWNA16
```

来源：`logs/vllm-qwen3.5-27b-b-awq-tp2-2x24g.log`，启动日志行 34、37。

结论：**这份官方 block FP8 权重在本机、当前 vLLM 配置下选用了 Marlin weight-only 路径**；AWQ 也是 Marlin W4A16。不能把该结果当作原生 FP8 W8A8 Tensor Core 计算性能的结论。

更正前期“4090 会走原生 FP8”的绝对说法：硬件的 FP8 能力并不保证任意量化格式都有受支持的原生 kernel。这里应以实际选中的 kernel 和 fallback 警告为准，也不能由该通用警告推断所有 4090 FP8 模型都没有原生支持。

两版本都记录 `FLASH_ATTN` attention backend、Triton/FLA GDN prefill kernel，默认开启 chunked prefill 和 prefix caching；运行采样中的 prefix cache hit rate 始终为 0.0%，符合本套件不测试共享前缀的目标。

## 通信：自定义 all-reduce 不可用，使用 NCCL

两版本共同的启动摘录：

```text
vLLM is using nccl==2.29.7
Custom allreduce is disabled because your platform lacks GPU P2P capability or P2P test failed.
Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0'
```

此外 SymmMemCommunicator 不支持 device capability 8.9，FlashInfer All Reduce 不支持 world_size=2，均被禁用。

结论：本机的 vLLM P2P 检测不可用或失败，自定义 all-reduce 关闭，实际 TP all-reduce 走 PYNCCL。**没有 NVLink 与没有可用 P2P 不是同一概念**；警告并未区分硬件能力缺失和测试失败，也不能仅凭此确定 NCCL 的具体传输路径或量化通信开销。

通信条件可能限制多卡效率，但缺少通信 microbenchmark/profiling，不能断言它是唯一或主要瓶颈。

## 调度与缓存采样

按压测时间窗提取日志周期性指标（约 10s 一次）；以下最大值可能出现在不同采样时刻，不是全过程的严格峰值：

| 场景 | 版本 | 样本数 | 最大 Running | 最大 Waiting | 最大 KV cache usage |
|---|---|---|---|---|---|
| chat 并发 64 | FP8 | 20 | 50 | 48 | 99.2% |
| chat 并发 64 | AWQ | 18 | 64 | 45 | 68.7% |
| long 并发 16 | FP8 | 27 | 16 | 13 | 86.2% |
| long 并发 16 | AWQ | 26 | 16 | 13 | 46.9% |

- FP8 chat 并发 64 存在明显缓存压力；AWQ 观察到更多同时 Running 请求。这支持更大缓存改变实际驻留并发的解释，也说明为什么 AWQ 的 TTFT 和 TPOT 不一定同步改善。
- long 并发 16：两个版本都观察到 Running=16，但 AWQ 缓存使用率只有约 47%，仍有很高的生成延迟。仅缓存容量不足不能解释其性能，prefill/decode 干扰、计算与通信等应继续考察。
- 日志未发现 preemption/recompute 消息；这是“没有这类日志证据”，**不是已证明抢占次数为 0**。此前基于最大 token 总量的缓存边界预测不等同于实测发生抢占。
- FP8 停服时记录强制清理 EngineCore 和 multiprocessing 资源警告，发生在两组压测完成之后；不把它当作本轮测试请求失败，但停止脚本的进程清理以后值得检查。

## 尾延迟：比 p50 更影响容量判断

| 场景 | FP8 TTFT p99 | AWQ TTFT p99 |
|---|---|---|
| chat 并发 4 | 1.93s | 1.99s |
| chat 并发 8 | 3.84s | 3.95s |
| chat 并发 16 | 7.99s | 7.81s |
| chat 并发 64 | 45.50s | 33.19s |
| long 并发 16 | 60.88s | 55.81s |

AWQ chat 并发 16 的 ITL p99=760.3ms，并发 64 为 951.4ms；long 并发 16 为 810.9ms。TPOT 是平均生成间隔尺度，可能掩盖接近一秒的单次流式输出停顿。

如果举例要求 chat TTFT p99<2s，本轮只有并发 1/2/4 档符合，AWQ 并发 4 已很贴边；不能把此前仅依据 p50 的并发 8～16 候选当成生产推荐。低并发档位仅 16 个样本，闭环测试且有同步突发，p99 稳定性与开放式流量表现还需更多测试。

质量评测尚未完成；当前验证的是显存、性能和运行路径，不是量化效果完整结论。
