#!/usr/bin/env python3
"""压测 OpenAI 兼容推理服务（vLLM / SGLang / 自己的网关），测 TTFT、TPOT、吞吐。

用法（在实例上、服务运行时）:
  python bench/bench.py --input-len 1024 --output-len 256 --concurrency 1,4,16,64

默认从 logs/deployed.env 读取当前部署的配置（deploy.sh 启动时写入），
结果写到 results/raw/<标签>__<workload>__<时间>.json，最后一行 stdout 打印结果文件路径。

指标定义:
  TTFT  首 token 延迟：发出请求 → 收到第一个输出 token。主要由排队 + prefill 决定
  TPOT  每输出 token 耗时：(E2E - TTFT) / (输出 token 数 - 1)。主要由 decode 决定
  ITL   相邻两次流式返回的间隔（一次返回可能包含多个 token）
  E2E   整个请求的耗时
  输出吞吐  所有请求的输出 token 总数 / 该并发档位的总耗时，衡量整机产能
"""

import argparse
import asyncio
import json
import os
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp

REPO_ROOT = Path(__file__).resolve().parent.parent


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def load_env_file(path):
    """读 KEY=VALUE 文件；兼容 configs/*.env（带引号和行尾注释）和 deployed.env（bash %q 转义）。"""
    env = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, rest = line.split("=", 1)
        parts = shlex.split(rest, comments=True)
        env[key.strip()] = parts[0] if parts else ""
    return env


# ---------------------------------------------------------------- prompt 生成

class PromptGenerator:
    """用随机 token 拼 prompt，精确控制输入长度。

    每个请求用不同的随机种子，保证 prompt 互不相同：vLLM 默认开启前缀缓存，
    重复的 prompt 会命中缓存、跳过 prefill，TTFT 就不真实了。
    shared_prefix_len > 0 时所有请求共享同一段前缀，用来专门测前缀缓存（阶段 5）。
    """

    def __init__(self, model_path, model_id, seed):
        from tokenizers import Tokenizer

        path = Path(model_path) / "tokenizer.json"
        if not path.exists():
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(model_id, "tokenizer.json"))
        self.tokenizer = Tokenizer.from_file(str(path))
        raw = json.loads(path.read_text())
        # 排除特殊 token（<|im_start|>、<think> 等），只从普通词表里采样
        self.special_ids = {t["id"] for t in raw.get("added_tokens", [])}
        self.vocab_size = len(raw["model"]["vocab"])
        self.seed = seed

    def _random_ids(self, rng, n):
        ids = []
        while len(ids) < n:
            i = rng.randrange(self.vocab_size)
            if i not in self.special_ids:
                ids.append(i)
        return ids

    def make(self, input_len, shared_prefix_len, key):
        prefix = self._random_ids(random.Random(f"{self.seed}-prefix"), shared_prefix_len)
        body = self._random_ids(random.Random(f"{self.seed}-{key}"), input_len - shared_prefix_len)
        return prefix + body

    def decode(self, ids):
        return self.tokenizer.decode(ids)


# ---------------------------------------------------------------- 发请求

def build_request(args, model, ids, gen):
    common = {
        "model": model,
        "max_tokens": args.output_len,
        # 忽略结束符，强制生成满 output_len 个 token，不同模型之间输出长度一致才可比
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},  # 最后一个 chunk 带 token 统计
    }
    if args.api == "completions":
        # 直接传 token id，服务端不再分词，输入长度完全精确
        return "/v1/completions", {**common, "prompt": ids}
    return "/v1/chat/completions", {
        **common,
        "messages": [{"role": "user", "content": gen.decode(ids)}],
        # Qwen3.5 默认 thinking，压测统一关闭
        "chat_template_kwargs": {"enable_thinking": False},
    }


async def send_one(session, url, payload, headers):
    rec = {"ok": False, "ttft": None, "e2e": None, "itl": [],
           "prompt_tokens": 0, "output_tokens": 0, "error": None}
    start = last = time.perf_counter()
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                return rec
            # SSE 流：每行 "data: {json}"，以 "data: [DONE]" 结束
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    rec["prompt_tokens"] = chunk["usage"].get("prompt_tokens", 0)
                    rec["output_tokens"] = chunk["usage"].get("completion_tokens", 0)
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    text = (choice.get("text") or delta.get("content")
                            or delta.get("reasoning_content") or delta.get("reasoning"))
                    if not text:
                        continue
                    now = time.perf_counter()
                    if rec["ttft"] is None:
                        rec["ttft"] = now - start
                    else:
                        rec["itl"].append(now - last)
                    last = now
        rec["e2e"] = time.perf_counter() - start
        rec["ok"] = rec["ttft"] is not None
        if not rec["ok"]:
            rec["error"] = "没有收到任何输出 token"
    except Exception as e:  # 超时、连接断开等
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


# ---------------------------------------------------------------- 统计

def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def dist(values, scale=1.0):
    if not values:
        return None
    v = [x * scale for x in values]
    return {"mean": sum(v) / len(v), "p50": pct(v, 50), "p90": pct(v, 90), "p99": pct(v, 99)}


def summarize(recs, duration, concurrency, output_len):
    ok = [r for r in recs if r["ok"]]
    tpots = [(r["e2e"] - r["ttft"]) / (r["output_tokens"] - 1) for r in ok if r["output_tokens"] > 1]
    itls = [x for r in ok for x in r["itl"]]
    in_tok = sum(r["prompt_tokens"] for r in ok)
    out_tok = sum(r["output_tokens"] for r in ok)
    tpot = dist(tpots, 1000)
    return {
        "concurrency": concurrency,
        "num_requests": len(recs),
        "completed": len(ok),
        "failed": len(recs) - len(ok),
        "errors": sorted({r["error"] for r in recs if r["error"]})[:5],
        "duration_s": duration,
        "request_throughput": len(ok) / duration,
        "input_tok_s": in_tok / duration,
        "output_tok_s": out_tok / duration,
        "total_tok_s": (in_tok + out_tok) / duration,
        "mean_prompt_tokens": in_tok / len(ok) if ok else None,
        "mean_output_tokens": out_tok / len(ok) if ok else None,
        "short_outputs": sum(1 for r in ok if r["output_tokens"] < output_len),
        "ttft_ms": dist([r["ttft"] for r in ok], 1000),
        "tpot_ms": tpot,
        "itl_ms": dist(itls, 1000),
        "e2e_s": dist([r["e2e"] for r in ok]),
        # 单个请求感受到的生成速度（tokens/s），= 1000 / TPOT
        "per_request_tok_s": 1000 / tpot["mean"] if tpot else None,
    }


async def run_level(session, base, headers, args, model, gen, concurrency, n):
    reqs = [build_request(args, model, gen.make(args.input_len, args.shared_prefix_len,
                                                f"{args.workload_name}-c{concurrency}-{i}"), gen)
            for i in range(n)]
    sem = asyncio.Semaphore(concurrency)

    async def worker(path, payload):
        # 闭环并发：始终保持最多 concurrency 个请求在途，一个完成再发下一个
        async with sem:
            return await send_one(session, base + path, payload, headers)

    t0 = time.perf_counter()
    recs = await asyncio.gather(*(worker(p, body) for p, body in reqs))
    return summarize(recs, time.perf_counter() - t0, concurrency, args.output_len)


# ---------------------------------------------------------------- 元数据

async def get_json(session, url, headers):
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as r:
            return await r.json(content_type=None) if r.status == 200 else None
    except Exception:
        return None


async def server_meta(session, base, headers):
    meta = {"framework": None, "framework_version": None, "served_max_model_len": None}
    sg = await get_json(session, base + "/get_server_info", headers)
    if sg:
        meta.update(framework="sglang", framework_version=sg.get("version"))
    else:
        v = await get_json(session, base + "/version", headers)
        if v:
            meta.update(framework="vllm", framework_version=v.get("version"))
    models = await get_json(session, base + "/v1/models", headers)
    if models and models.get("data"):
        meta["served_model"] = models["data"][0].get("id")
        meta["served_max_model_len"] = models["data"][0].get("max_model_len")
    return meta


def local_meta():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                                  cwd=REPO_ROOT).stdout.strip()
        except Exception:
            return None
    gpus = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    return {"gpus": gpus.splitlines() if gpus else None,
            "git_commit": run(["git", "rev-parse", "--short", "HEAD"])}


# ---------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "logs/deployed.env"),
                   help="部署配置，默认读当前运行中服务的 logs/deployed.env")
    p.add_argument("--base-url", help="服务地址，默认 http://127.0.0.1:<PORT>；压网关时改这里")
    p.add_argument("--label", help="结果标签，默认 <配置名>[覆盖参数] · <框架>")
    p.add_argument("--workload-name", default="custom")
    p.add_argument("--api", choices=["completions", "chat"], default="completions",
                   help="completions 直接传 token id，输入长度精确；chat 走对话模板，更接近真实业务")
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--shared-prefix-len", type=int, default=0, help="所有请求共享的前缀长度（测前缀缓存）")
    p.add_argument("--concurrency", default="1,4,16,64", help="逗号分隔的并发档位")
    p.add_argument("--num-prompts", type=int, help="每档请求数，默认 max(4 × 并发, 16)")
    p.add_argument("--warmup", type=int, default=4, help="正式测试前的预热请求数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=int, default=1800, help="单个请求超时秒数")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results/raw"))
    return p.parse_args()


async def main():
    args = parse_args()
    if not Path(args.config).exists():
        sys.exit(f"找不到 {args.config}：服务没有通过 deploy.sh 启动？用 --config configs/<方案>.env 指定")
    cfg = load_env_file(args.config)
    base = (args.base_url or f"http://127.0.0.1:{cfg.get('PORT', '8000')}").rstrip("/")
    api_key = os.environ.get("API_KEY")
    if not api_key and (REPO_ROOT / ".env").exists():
        api_key = load_env_file(REPO_ROOT / ".env").get("API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    model_id = cfg["MODEL_ID"]
    models_dir = os.environ.get("MODELS_DIR") or (
        "/workspace/models" if os.access("/workspace", os.W_OK) else str(REPO_ROOT / "models"))
    gen = PromptGenerator(Path(models_dir) / model_id.replace("/", "__"), model_id, args.seed)
    levels = [int(c) for c in args.concurrency.split(",")]

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        meta = await server_meta(session, base, headers)
        if not meta.get("served_model"):
            sys.exit(f"{base}/v1/models 无响应，服务没启动？")
        model = meta["served_model"]
        max_len = meta.get("served_max_model_len") or int(cfg.get("MAX_MODEL_LEN", 0) or 0)
        if max_len and args.input_len + args.output_len > max_len:
            log(f"SKIP: input {args.input_len} + output {args.output_len} 超过 max-model-len {max_len}")
            return

        overrides = cfg.get("CONFIG_OVERRIDES", "")
        label = args.label or (f"{cfg.get('CONFIG_NAME', model)}"
                               f"{f'[{overrides}]' if overrides else ''} · {meta['framework'] or 'unknown'}")
        log(f"{label} | {args.workload_name}: api={args.api} input={args.input_len} "
            f"output={args.output_len} prefix={args.shared_prefix_len} 并发={levels}")

        if args.warmup:
            log(f"预热 {args.warmup} 个请求")
            await asyncio.gather(*(send_one(session, base + path, body, headers) for path, body in (
                build_request(args, model, gen.make(args.input_len, 0, f"warmup-{i}"), gen)
                for i in range(args.warmup))))

        results = []
        for c in levels:
            n = args.num_prompts or max(4 * c, 16)
            r = await run_level(session, base, headers, args, model, gen, c, n)
            results.append(r)
            ttft, tpot = r["ttft_ms"] or {}, r["tpot_ms"] or {}
            log(f"并发 {c:>3}: {r['completed']}/{r['num_requests']} 成功, "
                f"输出 {r['output_tok_s']:8.1f} tok/s, TTFT p50 {ttft.get('p50', 0):7.1f} ms, "
                f"TPOT p50 {tpot.get('p50', 0):6.1f} ms, 耗时 {r['duration_s']:.1f}s")
            if r["failed"]:
                log(f"  失败示例: {r['errors']}")
            if r["short_outputs"]:
                log(f"  警告: {r['short_outputs']} 个请求输出不足 {args.output_len} token（ignore_eos 未生效？）")

    run = {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "workload": {k: getattr(args, k) for k in
                     ("workload_name", "api", "input_len", "output_len", "shared_prefix_len", "seed")},
        "server": {"base_url": base, **meta},
        "deployment": {k: v for k, v in cfg.items() if not k.startswith("MIN_")},
        "host": local_meta(),
        "levels": results,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
    out = out_dir / f"{safe}__{args.workload_name}__{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(run, ensure_ascii=False, indent=2))
    log(f"结果: {out}")
    print(out)


if __name__ == "__main__":
    asyncio.run(main())
