#!/bin/bash
# Qwen3-VL-8B-Instruct vLLM OpenAI API Server (frame-based captioning).
#
# The lighter alternative to start_vllm.sh (Qwen3-Omni-30B). Single card, no
# audio — fine here, since caption_videos.py samples frames anyway.
#
# One addition to the plain template: --limit-mm-per-prompt 'image=32'. vLLM
# defaults to ONE image per prompt; the captioner sends a batch of frames per
# call (--batch-frames, default 12), so without this the server rejects them.

MODEL_PATH="/mnt/data_oss/models/Qwen3-VL-8B-Instruct"
SERVED_NAME="Qwen3-VL-8B-Instruct"
HOST="0.0.0.0"
PORT=8080
LOG_FILE="/root/vllm_server.log"

nohup python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 32}' \
    --enforce-eager \
    --trust-remote-code \
    > "${LOG_FILE}" 2>&1 &

echo "vLLM server starting, PID=$!"
echo "Log: ${LOG_FILE}"
echo "API: http://${HOST}:${PORT}/v1/chat/completions"
echo "Check: tail -f ${LOG_FILE}"
