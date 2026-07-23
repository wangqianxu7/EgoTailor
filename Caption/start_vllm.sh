#!/usr/bin/env bash
# vLLM OpenAI-compatible server for fine-grained video captioning.
#
# Model : Qwen3-Omni-30B-A3B-Instruct  (MoE, ~3B active)
# Server: 2x A800 (80G)
#
# Deviations from the single-card template, all so a caption request can carry
# enough frames to actually be "fine-grained":
#   --tensor-parallel-size 2   use both A800s (was 1)
#   --max-model-len 32768      room for ~24-32 image frames + a long caption
#                              (8192 only fits ~6-8 frames -> coarse captions)
#   --limit-mm-per-prompt      let one prompt hold many frames
# enforce-eager is kept: Omni's multimodal path is safest without cuda graphs.
set -euo pipefail

MODEL_PATH="/mnt/data/workspace/outputs/Qwen3-Omni-30B-A3B-Instruct"
SERVED_NAME="Qwen3-Omni-30B-A3B-Instruct"
HOST="0.0.0.0"
PORT=8080
LOG_FILE="/root/vllm_server.log"

nohup python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 32}' \
    --dtype bfloat16 \
    --enforce-eager \
    --trust-remote-code \
    > "${LOG_FILE}" 2>&1 &

echo "vLLM server starting, PID=$!"
echo "Log: ${LOG_FILE}"
echo "API: http://${HOST}:${PORT}/v1/chat/completions"
echo "Check: tail -f ${LOG_FILE}"
