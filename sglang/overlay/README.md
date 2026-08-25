# DFlash 2 overlay for pip sglang 0.5.18

This directory is the **surgical** variant of the DFlash 2 backport overlay.
Apply it over the matching paths in your venv's
`.../lib/python3.13/site-packages/sglang/` after every pip (re)install.
Derived from the community reference overlay
([Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark](https://github.com/MiaAILab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark),
itself a backport of upstream sgl-project/sglang PR #35371 + #35496, landed
2026-08-19) adapted for pip **sglang 0.5.18** — do NOT copy the reference
overlay wholesale; see below.

## Why surgical and not wholesale

The reference overlay targets the cookbook docker image
(`0.0.0.dev0+qwen38.27b.g561c8f3`). Its "replaced wholesale with upstream main"
files mostly work on 0.5.18, but its appended-onto files are image-vintage:
copying its `srt/layers/moe/utils.py` over 0.5.18's broke the install
(`ImportError: cannot import name 'xpu_moe_ld_padding_elems'`). Meanwhile
0.5.18 already provides `sample_simulated_acc_len` (spec_utils),
`draft_model_build_scope` (moe/utils) and `compute_spec_logprobs`
(logprob_processor) — the three other functions the reference overlay appends.

So this overlay ships:

| File | What it is |
|---|---|
| `srt/models/dflash.py` | upstream main — adds `DFlash2DraftModel`, `CandidateSelector`, `DFlashGroupedConv` |
| `kernels/ops/speculative/dflash.py` | upstream main — adds `selector_walk_triton` |
| `srt/speculative/dflash_utils.py` | upstream main — adds `is_dense_head_weight`, `table_qk_norm_rope_` |
| `srt/speculative/dflash_worker_v2.py` | upstream main — DFlash 2 worker |
| `srt/speculative/dflash_info.py` | upstream main — DFlash 2 verify input |
| `srt/speculative/dflash_info_v2.py` | upstream main — DFlash 2 draft input |
| `srt/speculative/draft_worker_common.py` | upstream main — DFlash 2 draft worker plumbing |
| `srt/mem_cache/allocation_sizing.py` | **stock 0.5.18 file + appended** `page_aligned_decode_alloc_lens` (marked with a `# --- DFlash2 backport overlay (upstream main) ---` comment) |

Validated on an RTX PRO 6000 (SM120): server log shows
`Initialized DFLASH draft runner. ..., model=DFlash2DraftModel, block_size=8`,
`DFLASH fused KV materialization enabled.`, draft CUDA graphs captured;
bench numbers in the repo's top-level README.

## Updating

Once a pip sglang release ships DFlash 2 natively, delete this overlay and
run stock `sglang.launch_server --speculative-algorithm DFLASH`. To move to a
newer upstream before that: re-fetch the seven upstream files from
sgl-project/sglang main and re-append `page_aligned_decode_alloc_lens` to the
then-current pip `allocation_sizing.py` (never overwrite it wholesale).
