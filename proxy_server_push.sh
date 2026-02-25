#!/bin/bash
set -xe

PREFILL_IP="172.31.2.19"
DECODE_IP="172.31.0.191"

echo "=== Starting Push-based NIXL Proxy Server on port 8000 ==="
echo "Prefill: http://$PREFILL_IP:8100"
echo "Decode: http://$DECODE_IP:8200"
echo "Mode: Push (prefill WRITE per-layer)"

python3 /home/ubuntu/vllm/push_proxy_server.py \
    --port 8000 \
    --prefiller-hosts $PREFILL_IP \
    --prefiller-ports 8100 \
    --decoder-hosts $DECODE_IP \
    --decoder-ports 8200
