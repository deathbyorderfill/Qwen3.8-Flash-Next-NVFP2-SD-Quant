#!/usr/bin/env bash
# SDnvfp2 — Qwen3.8-Flash-Next with 2-bit experts on ONE DGX Spark (GB10, 128 GB), served by sglang.
# Usage:  CKPT=/path/to/Qwen3.8-Next-SDnvfp2 ./run.sh [extra sglang args]
# Env:    CODES (default $CKPT/dense_codes)  IMAGE  PORT (8934)  NAME (sdnvfp2)  MEMFRAC (0.80)  ACC (0.7)  CTX (32768)
#         VISION=1 loads the vision tower too (default 0 = --language-only; all published numbers are text-only)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CKPT=${CKPT:?set CKPT=<dir with the SDnvfp2 safetensors>}
CODES=${CODES:-$CKPT/dense_codes}
IMAGE=${IMAGE:-lmsysorg/sglang:qwen38flashnext}
PORT=${PORT:-8934}; NAME=${NAME:-sdnvfp2}; MEMFRAC=${MEMFRAC:-0.80}; ACC=${ACC:-0.7}; CTX=${CTX:-32768}; VISION=${VISION:-0}
S=/sgl-workspace/sglang/python/sglang
[ -d "$CODES" ] || { echo "dense codes dir not found: $CODES"; exit 1; }
if pgrep -f 'sglang::scheduler' >/dev/null; then echo "an sglang scheduler is already running on this box"; exit 2; fi
# Serving view: symlinks to the shards (container paths) + the config pair sglang's modelopt_fp4 parser expects.
VIEW=$HERE/.view; rm -rf "$VIEW"; mkdir -p "$VIEW"
for f in "$CKPT"/*; do b=$(basename "$f"); case "$b" in config.json|hf_quant_config.json) continue;; esac; ln -s "/ckpt/$b" "$VIEW/$b"; done
cp "$HERE/checkpoint_config/config.json" "$HERE/checkpoint_config/hf_quant_config.json" "$VIEW/"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --network host --ipc=host --shm-size 32g \
  -e SGLANG_NVFP2_LM20=1 -e SGLANG_DENSE_FP8=gdn,attn,lm_head -e SGLANG_DENSE_NVFP4=gdn,attn,lm_head \
  -e SGLANG_DENSE_NVFP4_BACKEND=cute-dsl -e SGLANG_DENSE_NVFP4_CODES_DIR=/codes -e SGLANG_QWEN4_PLE_NVFP4=1 \
  -e SGLANG_FP2_TILES=base -e SGLANG_PLE_PREFETCH=auto -e SGLANG_DENSE_HC=off -e SGLANG_DRAFT_UNQUANT=1 \
  -e SGLANG_DRAFT_LMHEAD_SLICE=/sidecars/draft_vocab_top65536.pt \
  -v "$CKPT":/ckpt:ro -v "$VIEW":/models/sdnvfp2 -v "$CODES":/codes:ro -v "$HERE/sidecars":/sidecars:ro \
  -v "$HERE/model/qwen4_exp.py":$S/srt/models/qwen4_exp.py:ro \
  -v "$HERE/model/qwen4_exp_mtp.py":$S/srt/models/qwen4_exp_mtp.py:ro \
  -v "$HERE/flashnext/weight_utils.py":$S/srt/model_loader/weight_utils.py:ro \
  -v "$HERE/flashnext/sparse_attn.py":$S/srt/layers/attention/qsa/sparse_attn.py:ro \
  -v "$HERE/flashnext/qwen_sparse_attn_backend.py":$S/srt/layers/attention/qwen_sparse_attn_backend.py:ro \
  -v "$HERE/flashnext/flash_fwd.py":/usr/local/lib/python3.12/dist-packages/flash_attn/cute/flash_fwd.py:ro \
  -v "$HERE/sglang_patched/fused_moe_triton_kernels.py":$S/kernels/ops/moe/fused_moe_triton_kernels.py:ro \
  -v "$HERE/sglang_patched/fused_moe.py":$S/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:ro \
  -v "$HERE/sglang_patched/triton.py":$S/srt/layers/moe/moe_runner/triton.py:ro \
  -v "$HERE/sglang_patched/modelopt_quant.py":$S/srt/layers/quantization/modelopt_quant.py:ro \
  "$IMAGE" \
  python3 -m sglang.launch_server --model-path /models/sdnvfp2 --trust-remote-code $([ "$VISION" = 1 ] || echo --language-only) \
  --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass --moe-runner-backend triton \
  --kv-cache-dtype fp8_e4m3 --page-size 64 --mamba-scheduler-strategy extra_buffer --mamba-track-interval 64 \
  --chunked-prefill-size 8192 --max-prefill-tokens 32768 --max-running-requests 8 \
  --max-mamba-cache-size 24 --mamba-ssm-dtype bfloat16 --context-length "$CTX" \
  --mem-fraction-static "$MEMFRAC" --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 --speculative-draft-model-quantization unquant \
  --speculative-accept-threshold-acc "$ACC" --disable-overlap-schedule \
  --host 0.0.0.0 --port "$PORT" --served-model-name sdnvfp2 "$@"
echo "started container $NAME on :$PORT — first load takes ~10-12 min (the PLE table is packed at load); watch: docker logs -f $NAME"
echo "ready when: curl -s localhost:$PORT/health"
