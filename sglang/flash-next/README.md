# sglang/flash-next — Qwen3.8-Flash-Next (NVFP4 + NEXTN) on one RTX PRO 6000

Production daily driver on gpu02's GPU 0 since 2026-08-28. Qwen3.8-Flash-Next
(125B-A6B MoE, 51B PLE n-gram embeddings, 4B MTP head, hybrid GDN + QSA
attention, vision, 262K native ctx) served by the SM120-qualified SGLang fork
[jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000)
— upstream support is still an open PR
([sgl-project/sglang#36497](https://github.com/sgl-project/sglang/pull/36497)),
and the PR head alone is SM100-tuned; the fork's SM120 deltas are required.

Measured on this exact setup (full report + raw numbers:
[Sub-Net-Public/llm-benchmark](https://git.sub-net.at/Sub-Net-Public/llm-benchmark)
`flash-next/`): **162–178 tok/s single-stream decode, ~460–474 tok/s
aggregate at 4 streams, 824K-token FP8 KV pool, 524K context (YaRN ×2),
working vision.** That is 2.2× the llama.cpp GGUF path (77–79 tok/s) with
real continuous batching on top.

## Files

- `serve-flash-next.sh` — the launcher. Adapted from the fork's
  `configs/pennyroyal/serve-flash-next.sh`: NIXL/HiCache persistence stripped
  (host-RAM HiCache kept), venv `bin/` added to `PATH` (JIT needs the `ninja`
  *binary* — a venv package alone crashes startup after the full weight
  load), bound to `[::1]:8001` behind a local nginx TLS proxy.
- `sglang-flashnext.service` — systemd unit. `TimeoutStartSec=1800` covers
  the 8–12 min cold weight load (135 GB checkpoint) + JIT/graph capture;
  warm restarts take ~4–5 min.
- `gpu0-free-gate.sh` — `ExecStartPre` free-VRAM gate. SGLang sizes its
  KV/state pools from free VRAM **once** at startup; a still-releasing
  predecessor or an orphaned scheduler (SGLang's `kill_process_tree` can
  orphan them on a crash) silently shrinks the pool or OOMs the load. Do
  not skip this.

## Build contract (qualified on the fork's host; deviations noted)

| Component | Qualified | Ours |
|---|---|---|
| GPU | RTX PRO 6000 96 GB, SM120 | same |
| Driver | 610.57.04 | same |
| CUDA / NVCC | 13.3 / 13.3.73 | same |
| GCC | 15.3.1 | **14.2 (Debian trixie)** — works; the pin matters for their published JIT fingerprint identity, not correctness |
| PyTorch | 2.13.0+cu130 | same |
| FlashInfer | 0.6.17 | same |
| Checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4` rev `7b719225` | same |

Build = checkout the fork tag `sglang-rtxpro6000-20260827`, uv venv with
Python 3.12.13, `uv pip install --prerelease=allow --index-strategy
unsafe-best-match --extra-index-url https://docs.sglang.ai/whl/cu130/
--no-build-isolation -e python` (preinstall `setuptools wheel ninja cmake
packaging wheel_stub` into the venv first — `--no-build-isolation` surfaces
them one at a time otherwise). `TORCH_CUDA_ARCH_LIST=12.0`.

## SM120 notes (why the flags look like this)

- TRTLLM-Gen MoE kernels are SM100-only; on SM120 the stack resolves to
  FlashInfer CUTLASS MoE and XQA for QSA decode — automatic with the fork,
  broken upstream.
- `--linear-attn-{decode,prefill}-backend flashinfer` must be explicit;
  auto-selection misses FlashInfer GDN on SM120.
- `--gdn-mtp-cache-mode none` + RecoverSSM + `--mem-fraction-static 0.981`
  recover a 1 GiB speculative SSM pool the upstream KV estimator still
  reserves — that is the 824K-token KV pool.
- The vision path needs the fork's fused-kernel mRoPE fix; other SM120
  builds silently misposition vision tokens.
- The RadixArk chat template only supports `enable_thinking`;
  `reasoning_effort` is silently ignored. Tiny `max_tokens` gets eaten by
  thinking → empty content with `finish_reason=length`.
- Keep it single-GPU: TP across PRO 6000 + RTX 5090 over PCIe measured
  *slower* (48–52 tok/s) than one card alone (llama.cpp numbers, same
  model).
