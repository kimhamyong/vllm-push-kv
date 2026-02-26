#!/bin/bash
set -xe

MODEL_NAME="meta-llama/Llama-3.2-1B"
PREFILL_IP="172.31.2.19"
DECODE_IP="172.31.0.191"

export VLLM_NIXL_SIDE_CHANNEL_HOST=$PREFILL_IP
export VLLM_NIXL_SIDE_CHANNEL_PORT=14580
export UCX_TLS=cuda_copy,tcp
export UCX_NET_DEVICES=ens5
export UCX_TCP_PORT_RANGE=40000-40009
echo "=== Starting Prefill Instance (KV Producer) with NixlConnector ==="
echo "Model: $MODEL_NAME"
echo "Prefill IP: $PREFILL_IP"
echo "NIXL Side Channel: $PREFILL_IP:14580"

vllm serve $MODEL_NAME \
    --host 0.0.0.0 \
    --port 8100 \
    --max-model-len 24576 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
