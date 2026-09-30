# 阶段 1 基线：vLLM 启动显存数据

环境：Vast.ai RTX 3090 24GB，镜像 `vastai/base-image_cuda-12.8.1`，vLLM 0.30.0。
参数：`GPU_MEMORY_UTILIZATION=0.90`，`LANGUAGE_MODEL_ONLY=1`，`KV_CACHE_DTYPE=auto`，`MAX_NUM_SEQS=64`。

| 模型 | max-model-len | 显存预算 | 权重+非torch | 激活峰值 | CUDA graph | KV cache | KV cache 容量 (tokens) | 实测 KB/token | 理论 KB/token | 最大并发 | 启动耗时 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B BF16 | 8192 | 21.2 GiB | 1.75 GiB | 0.63 GiB | 0.08 GiB | 18.83 GiB | 1,125,655 | 17.5 | 12.0 | 137.4x | 172 s（首次） |
| Qwen3.5-4B BF16 | 16384 | 21.2 GiB | 8.15 GiB | 1.37 GiB | 0.09 GiB | 11.68 GiB | 312,158 | 39.2 | 32.0 | 19.1x | 未记录 |

## 说明

- **理论 KB/token** 只算 full attention 层：`full 层数 × 2(K,V) × KV heads × head_dim × 2 字节`。
  - 0.8B：24 层中 6 层 full attention，2 个 KV head，head_dim 256 → 6×2×2×256×2 = 12 KB。
  - 4B：32 层中 8 层 full attention，4 个 KV head，head_dim 256 → 8×2×4×256×2 = 32 KB。
- 实测比理论值大：0.8B 多 46%，4B 多 23%。推测来自线性注意力层的状态和混合注意力分页对齐（未验证）。估算大模型显存时按 1.2–1.5 倍余量算。
- **9B 预测**：9B 的层结构与 4B 相同（32 层 / 8 层 full / 4 KV head），每 token KV 开销应接近 4B（约 39 KB）。BF16 权重约 18 GiB，KV cache 只剩约 2 GiB → 约 5 万 token。部署 9B 时验证。
- **最大并发** = KV cache 容量 ÷ max-model-len，是“每个请求都用满上下文”时的显存上限。0.8B 的实际并发上限是 `MAX_NUM_SEQS=64`，瓶颈在调度器而不是显存；4B 在满上下文时只够 19 个请求，瓶颈已经变成显存。
- `nvidia-smi` 显示已用 21502 MiB：vLLM 启动时按预算预分配，不代表模型本身需要这么多。
- 启动 172 s 主要是首次编译 / 预热，结果缓存在 `~/.cache/vllm`，实例销毁后丢失。

## 采集命令

```bash
grep -iE "KV cache size|maximum concurrency|model weights took|memory utilization is" logs/vllm-<方案>.log
```
