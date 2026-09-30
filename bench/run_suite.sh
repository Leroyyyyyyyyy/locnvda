#!/usr/bin/env bash
# 标准压测套件：所有模型 / 方案都跑同一组 workload 和并发档位，结果才能直接对比。
# 用法: bench/run_suite.sh [bench.py 的额外参数，如 --label xxx]
# 前提: 服务已通过 scripts/deploy.sh 启动（读取 logs/deployed.env）
set -euo pipefail
source "$(dirname "$0")/../scripts/lib.sh"
activate_venv
cd "$REPO_ROOT"

# 并发档位可通过环境变量改：CONCURRENCY=1,8,32 bench/run_suite.sh
CONCURRENCY="${CONCURRENCY:-1,2,4,8,16,32,64}"
LONG_CONCURRENCY="${LONG_CONCURRENCY:-1,4,16}"

files=()
run() {
  local out
  out="$(python bench/bench.py "$@")"
  if [[ -n "$out" ]]; then files+=("$out"); fi   # 空 = 被跳过（超出 max-model-len）
}

# chat：典型对话，1K 输入 / 256 输出，看并发从 1 到 64 吞吐和延迟怎么变
run --workload-name chat --input-len 1024 --output-len 256 --concurrency "$CONCURRENCY" "$@"
# long：长输入（RAG / 长文档），prefill 占大头，看 TTFT 和 KV cache 压力
# 7680 + 256 = 7936，max-model-len 8192 的配置也能跑；更小的配置会自动跳过
run --workload-name long --input-len 7680 --output-len 256 --concurrency "$LONG_CONCURRENCY" "$@"

[[ ${#files[@]} -gt 0 ]] || die "没有产生任何结果"
report="results/report-$(date +%Y%m%d-%H%M%S).md"
python bench/report.py "${files[@]}" --out "$report"
log "报告: $report"
