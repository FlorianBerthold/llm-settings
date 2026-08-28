#!/bin/bash
# free-VRAM gate: wait (max 5 min) until GPU0 is below 5000 MiB.
# KV/pool sizing happens once at startup; a still-releasing predecessor
# (llama-flashnext, sglang-qwen38-27b, bake workers) silently shrinks it.
for i in {1..60}; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$used" -lt 5000 ] && exit 0
  sleep 5
done
echo "GPU0 still busy after 5 min (${used} MiB)" >&2
exit 1
