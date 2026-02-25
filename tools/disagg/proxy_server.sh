#!/bin/bash
set -xe

PREFILL_IP="172.31.2.19"
DECODE_IP="172.31.0.191"

echo "=== Starting NIXL Proxy Server on port 8000 ==="
echo "Prefill: http://$PREFILL_IP:8100"
echo "Decode: http://$DECODE_IP:8200"

python3 ~/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
    --port 8000 \
    --prefiller-hosts $PREFILL_IP \
    --prefiller-ports 8100 \
    --decoder-hosts $DECODE_IP \
    --decoder-ports 8200
