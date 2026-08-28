# llm-settings

Production LLM serving configurations from [Sub-Net](https://sub-net.at), with
measured benchmarks. Single host, two Blackwell GPUs:

| GPU | Model (HF checkpoint) | Speculative drafter | Stack |
|---|---|---|---|
| RTX PRO 6000 Blackwell (96 GB, SM120) | [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) (125B-A6B MoE + PLE + MTP, hybrid GDN/QSA, vision) | native NEXTN (in-checkpoint MTP head), 3 steps / 4 draft tokens | `sglang/flash-next/` (prod, SM120 SGLang fork) |
| RTX 5090 (32 GB, SM120) | [ornith-ai/Ornith-1.5-35B-A3B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4) | MTP k=3 with the repaired head from [shisa-ai/Ornith-1.5-35B-A3B-MTP](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP) | `vllm/ornith-35b/` |

Retired 2026-08-28 (superseded by Flash-Next, kept as rollback):
[RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead)
+ DFlash2 — `sglang/` (SGLang 0.5.18 + overlay) and `vllm/qwen38-27b/`.

Both models serve OpenAI-compatible APIs behind LiteLLM + a routing proxy,
with speculative decoding enabled and tuned on our own agent traffic.

## What's interesting here

- **Qwen3.8-Flash-Next on a single 96 GB card** — 125B-A6B MoE with a 51B
  PLE embedding table offloaded to host RAM, NVFP4 weights, native NEXTN
  speculative decoding: 162–178 tok/s single-stream, ~460–474 tok/s at 4
  streams, 524K context. Runs on an SM120-qualified SGLang *fork* because
  upstream support is still an open PR and the PR head is SM100-tuned —
  TRTLLM-Gen kernels are hard-impossible on SM120 (SM100-only cubins).
  See `sglang/flash-next/`.
- **SGLang + DFlash 2 on pip 0.5.18** — DFlash 2 landed upstream 2026-08-19
  (PR #35371 + #35496) but no pip wheel ships it yet. `sglang/overlay/` is a
  *surgical* backport: copying the community reference overlay wholesale
  breaks 0.5.18; only 8 files are actually needed. Full story in
  `sglang/overlay/README.md`.
- **vLLM master + DFlash 2 k=5** — DFlash 2 needs a vLLM master build
  (PR #52816). We A/B'd k=7 vs k=5 on production traffic: k=7's extra two
  draft positions accept ~nothing, so k=5 does the same work with 29 % fewer
  draft tokens.
- **Ornith MTP with a repaired head** — the stock Ornith-1.5-35B-A3B MTP head
  is broken (~34 % acceptance); the shisa-ai fixed head
  (`shisa-ai/Ornith-1.5-35B-A3B-MTP`) restores it. See `vllm/ornith-35b/`.
- **Real A/B numbers**, same card, same bench harness — see below.

## Measured (2026-08-28, RTX PRO 6000, full card)

Flash-Next (SGLang fork, NVFP4, NEXTN accept-len 2.4–3.0): **162–178 tok/s
single-stream decode, 459.9/473.7 tok/s 4-way aggregate, 2,463–4,810 tok/s
prefill @3.5K, 824K-token FP8 KV pool**. llama.cpp GGUF UD-Q4_K_XL on the
same card: 77–79 tok/s, no concurrency. Full shootout:
[llm-benchmark `flash-next/`](https://git.sub-net.at/Sub-Net-Public/llm-benchmark).

## Measured (2026-08-25, RTX PRO 6000, full card unless noted) — retired 27B setup

Custom harness: workload A = repetitive artifact generation (write a full
HTML dashboard, then modify it twice with the previous output fed back),
workload B = six diverse prompts, sequential + 4-way concurrent.

| Decode tok/s | vLLM DFlash2 k=5 | SGLang 0.5.18 + DFlash2 |
|---|---|---|
| A1 generate (cold) | 197.6 | **242.7** |
| A2 modify | 212.6 | **270.7** |
| A3 modify (warm) | 219.2 | **278.3** |
| B sequential (6 cases) | 110.8–199.6 | **126.9–215.7** |
| B 4-way concurrent aggregate | **387.4** | 368.8 |
| 8-way probe aggregate | — | **1203.8** |

SGLang wins ~20–27 % single-stream, ties at 4-way, scales cleanly to 8
streams. The vLLM config ran at `--gpu-memory-utilization 0.75` with a ~19 GB
VLM co-tenant; SGLang runs `--mem-fraction-static 0.90` with the card to
itself (`max_running_requests=8`, ~959k-token KV pool, 262K context).

Ornith-35B on the 5090 (vLLM, MTP k=3, fixed head): 407–423 tok/s on workload
A, mean acceptance length 2.46–3.14.

Also evaluated and rejected: SuffixDecoding (arXiv 2411.04975) — only wins
with a fully warm suffix cache, 2–4× slower on cold/diverse traffic, and
incompatible with `--async-scheduling` in current vLLM.

## Layout

- `sglang/flash-next/` — current production: Flash-Next launcher, unit file,
  free-VRAM start gate, build contract + SM120 notes
- `sglang/` — retired 27B production: unit file, flag rationale, DFlash 2 overlay
- `vllm/qwen38-27b/` — previous 27B production, kept as one-command rollback
- `vllm/ornith-35b/` — 5090 instance: MTP k=3 + fixed MTP head

Units are systemd templates from our ansible repo with site-specific values
rendered in; paths (`/srv/vllm`, `/srv/local-ai`) are ours — adjust to taste.
All services bind dual-stack and sit behind a local nginx TLS proxy.
