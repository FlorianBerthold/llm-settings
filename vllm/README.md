# vLLM configurations

Our previous (27B) and current (Ornith) vLLM production setups. Both run a
vLLM **master build** (`0.26.1rc1.dev1053+ge85d1b69c`) — DFlash 2 support
(PR #52816) is not in any release; build from source with CUDA >= 12.9 for
SM120.

## qwen38-27b/ — Qwen3.8-27B on RTX PRO 6000 (previous prod, now rollback)

`unsloth/Qwen3.8-27B-NVFP4` + DFlash 2 (`incoai/Qwen3.8-27B-DFlash2`),
`num_speculative_tokens=5`.

- **k=5, not k=7**: A/B on production traffic showed identical decode speed
  and tokens/step, but k=7's extra two draft positions accept ~nothing
  (acceptance 23.6 %→33.3 % after the switch). k=5 does the same work with
  29 % fewer draft tokens.
- DFlash2-k5 measured acceptance: mean accepted length 5.3 on repetitive
  artifact workloads (85–87 % draft acceptance), 3.3–3.9 on diverse traffic.
- `--gpu-memory-utilization 0.75` leaves room for a ~19 GB VLM co-tenant
  (moondream QA); the `ExecStartPre` gate therefore waits for < 22000 MiB
  used. Dedicated card? Raise util and drop the gate to ~2000.
- Sits on port 8083 behind an nginx TLS proxy; the SGLang config in
  `../sglang/` serves the same port + model name so cutover/rollback is a
  unit swap with zero proxy changes.

## ornith-35b/ — Ornith-1.5-35B-A3B on RTX 5090 (current prod)

`ornith-ai/Ornith-1.5-35B-A3B-NVFP4` + MTP k=3 — **with a repaired MTP head**.

The stock Ornith MTP head is broken (~34 % acceptance). The shisa-ai repack
[`shisa-ai/Ornith-1.5-35B-A3B-MTP`](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP)
ships a fixed `model-mtp.safetensors`. The `--speculative-config` model path
is a small local dir assembled as:

```
ornith-fixed-mtp/
├── config.json              # MTP draft config (see repo)
├── model-mtp.safetensors    # from shisa-ai/Ornith-1.5-35B-A3B-MTP
├── tokenizer.json           # symlinks/copies from the base NVFP4 checkpoint
├── vocab.json / merges.txt / tokenizer_config.json / generation_config.json
```

With the fixed head, MTP k=3 measures 407–423 tok/s on artifact workloads
(5090), mean acceptance length 2.46–3.14 (48–71 % draft acceptance).

- `--language-model-only`: Ornith is a VL model
  (`Qwen3_5MoeForConditionalGeneration`, has `vision_config`) but we serve it
  text-only — drop this flag to enable the vision tower.
- `--tool-call-parser qwen3_xml` (not qwen3_coder — that's the 27B's parser).
