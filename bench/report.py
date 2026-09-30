#!/usr/bin/env python3
"""把多次压测结果汇总成 Markdown 对比表。

用法:
  python bench/report.py results/raw/*.json                     # 所有结果
  python bench/report.py results/raw/qwen3.5-9b*.json results/raw/qwen3.5-27b*.json --out results/9b-vs-27b.md

输出三部分:
  1. 运行环境：每次运行的模型、量化、TP、GPU、框架版本
  2. 对比表：同一 workload 下，各运行在每个并发档位的关键指标并排（9B vs 27B、27B 各方案之间）
  3. 明细表：每次运行每个并发档位的完整指标
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def load_runs(patterns):
    runs = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            if Path(path).is_file():
                run = json.loads(Path(path).read_text())
                run["_file"] = Path(path).name
                runs.append(run)
    # 同一标签 + workload 跑了多次：标签后面加时间区分
    seen = defaultdict(int)
    for r in runs:
        seen[(r["label"], r["workload"]["workload_name"])] += 1
    for r in runs:
        if seen[(r["label"], r["workload"]["workload_name"])] > 1:
            r["label"] = f"{r['label']} @{r['timestamp'][5:16]}"
    return runs


def fmt(v, digits=1):
    if v is None:
        return "—"
    return f"{v:,.{digits}f}" if isinstance(v, float) else str(v)


def get(level, path):
    """get(level, "ttft_ms.p50") -> level["ttft_ms"]["p50"]"""
    v = level
    for key in path.split("."):
        v = v.get(key) if isinstance(v, dict) else None
    return v


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def workload_key(w):
    return (w["workload_name"], w["api"], w["input_len"], w["output_len"], w["shared_prefix_len"])


def workload_title(w):
    s = f"{w['workload_name']}：输入 {w['input_len']} / 输出 {w['output_len']} token，api={w['api']}"
    return s + (f"，共享前缀 {w['shared_prefix_len']}" if w["shared_prefix_len"] else "")


def env_section(runs):
    rows = []
    for r in runs:
        d, s, h = r["deployment"], r["server"], r["host"]
        quant = d.get("QUANTIZATION") or "auto"
        rows.append([r["label"], f"{s.get('framework')} {s.get('framework_version') or ''}".strip(),
                     d.get("MODEL_ID", "—"), quant, d.get("KV_CACHE_DTYPE", "—"),
                     d.get("TENSOR_PARALLEL_SIZE", "—"), fmt(s.get("served_max_model_len")),
                     d.get("GPU_MEMORY_UTILIZATION", "—"), "<br>".join(h.get("gpus") or ["—"]),
                     h.get("git_commit") or "—", r["timestamp"]])
    return table(["运行", "框架", "模型", "量化", "KV dtype", "TP", "max-model-len",
                  "显存比例", "GPU", "commit", "时间"], rows)


COMPARE_METRICS = [
    ("输出吞吐 (tok/s，越高越好)", "output_tok_s", 1),
    ("TTFT p50 (ms，越低越好)", "ttft_ms.p50", 1),
    ("TTFT p99 (ms，越低越好)", "ttft_ms.p99", 1),
    ("TPOT p50 (ms，越低越好)", "tpot_ms.p50", 1),
    ("单请求生成速度 (tok/s，越高越好)", "per_request_tok_s", 1),
]


def compare_section(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[workload_key(r["workload"])].append(r)
    out = []
    for group in groups.values():
        out.append(f"### {workload_title(group[0]['workload'])}")
        concurrencies = sorted({lv["concurrency"] for r in group for lv in r["levels"]})
        for title, path, digits in COMPARE_METRICS:
            rows = []
            for c in concurrencies:
                row = [str(c)]
                for r in group:
                    lv = next((x for x in r["levels"] if x["concurrency"] == c), None)
                    row.append(fmt(get(lv, path), digits) if lv else "—")
                rows.append(row)
            out.append(f"**{title}**\n\n" + table(["并发"] + [r["label"] for r in group], rows))
    return "\n\n".join(out)


def detail_section(runs):
    out = []
    for r in runs:
        rows = []
        for lv in r["levels"]:
            rows.append([str(lv["concurrency"]), f"{lv['completed']}/{lv['num_requests']}",
                         fmt(lv["request_throughput"], 2), fmt(lv["output_tok_s"]), fmt(lv["total_tok_s"]),
                         fmt(get(lv, "ttft_ms.p50")), fmt(get(lv, "ttft_ms.p99")),
                         fmt(get(lv, "tpot_ms.p50")), fmt(get(lv, "tpot_ms.p99")),
                         fmt(get(lv, "itl_ms.p99")), fmt(lv["per_request_tok_s"]),
                         fmt(get(lv, "e2e_s.p50"), 2), fmt(get(lv, "e2e_s.p99"), 2)])
        out.append(f"### {r['label']} — {workload_title(r['workload'])}\n\n" + table(
            ["并发", "成功", "req/s", "输出 tok/s", "总 tok/s", "TTFT p50", "TTFT p99",
             "TPOT p50", "TPOT p99", "ITL p99", "单请求 tok/s", "E2E p50 (s)", "E2E p99 (s)"], rows))
    return "\n\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="结果 JSON 文件或 glob")
    p.add_argument("--out", help="写入 Markdown 文件，默认打印到终端")
    args = p.parse_args()

    runs = load_runs(args.files)
    if not runs:
        raise SystemExit("没有找到结果文件")
    md = "\n\n".join([
        "# 压测报告",
        "## 运行环境\n\n" + env_section(runs),
        "## 对比\n\n" + compare_section(runs),
        "## 明细\n\n时间单位 ms（E2E 为 s）。\n\n" + detail_section(runs),
    ]) + "\n"
    if args.out:
        Path(args.out).write_text(md)
        print(f"已写入 {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
