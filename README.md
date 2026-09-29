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
git clone <repo-url> && cd <repo>
# 之后：scripts/deploy.sh configs/<方案>.env（待实现）
```
