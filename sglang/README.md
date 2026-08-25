# SGLang — Qwen3.8-27B NVFP4 + DFlash 2 (production)

SGLang 0.5.18 serving `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` with the
`z-lab/Qwen3.8-27B-DFlash2` drafter on an RTX PRO 6000 (96 GB, SM120).

- `sglang-qwen38-27b.service` — the production systemd unit
- `overlay/` — surgical DFlash 2 backport for pip sglang 0.5.18 (required —
  0.5.18 has no DFlash 2; see its README)

## Install sketch

```bash
python3.13 -m venv /srv/vllm/venv-sglang
/srv/vllm/venv-sglang/bin/pip install sglang==0.5.18 torch==2.13.0 flashinfer-python==0.6.17 httpx
# outlines_core 0.1.26 has no cp313 manylinux wheel on PyPI — sglang's dep
# chain (outlines 0.1.11) builds it from source otherwise (needs Rust).
# Either preinstall a built wheel or have cargo on PATH.
# Then apply the overlay:
cp -r overlay/{srt,kernels} /srv/vllm/venv-sglang/lib/python3.13/site-packages/sglang/
```

CUDA >= 12.9 toolkit with `nvcc` on PATH (SM120 FlashInfer JIT).

## Flag rationale (see unit for the full set)

| Flag | Value | Why |
|---|---|---|
| `--mem-fraction-static` | 0.90 | Card is dedicated (no co-tenant). Lands on ~959k-token KV pool, `max_running_requests=8` (not capped by the mamba state cache, unlike 0.70 which caps at 6) |
| `--kv-cache-dtype` | fp8_e4m3 | Best supported KV precision for this hybrid-GDN model (4-bit KV paths don't build for it in this version) |
| `--context-length` | 262144 | Native 256K window, no YaRN |
| `--speculative-algorithm` | DFLASH | + `z-lab/Qwen3.8-27B-DFlash2` draft, 8 draft tokens/step, unquant, flashinfer backend |
| `--attention-backend` | flashinfer | SM120 |
| `--reasoning-parser` / `--tool-call-parser` | qwen3 / qwen3_coder | Thinking mode + tool calling |
| `--chunked-prefill-size` / `--max-prefill-tokens` | 4096 | Prefill chunking for long contexts |
| `--enable-metrics` | — | Prometheus `/metrics` |

The `ExecStartPre` gate waits until GPU memory used < 2000 MiB before
starting: SGLang computes `--mem-fraction-static` against *total* VRAM but
allocates from what's free at startup — a predecessor still releasing memory
would OOM the launch or silently shrink the KV pool.

Single-stream decode is insensitive to mem-fraction (identical tok/s at 0.70
and 0.90); only the KV pool and request cap move. If you must co-locate a
second tenant, 0.70 works — expect `max_running_requests` to cap at 6.

## Bench reference

See the top-level README for the full table. Headline: 242.7 / 270.7 / 278.3
tok/s on artifact workloads A1/A2/A3 (vLLM DFlash2-k5: 197.6 / 212.6 / 219.2),
1203.8 tok/s aggregate at 8 concurrent streams, steady-state VRAM ~92–94 GiB.
