#!/bin/bash
set -e

# =============================================================================
# Disaggregated Prefill Benchmark Script
# All results saved to JSON file
# Output: results/benchmark_disagg/benchmark_disagg_{timestamp}.json
# =============================================================================

# Configuration
MODEL_NAME="meta-llama/Llama-3.2-1B"
PROXY_URL=${PROXY_URL:-http://127.0.0.1:8000}
PREFILL_URL="http://172.31.2.19:8100"
DECODE_URL="http://172.31.0.191:8200"

# Benchmark settings
SCRIPT_NAME="benchmark_disagg"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/home/ubuntu/vllm/results/${SCRIPT_NAME}"
RESULT_FILE="${RESULTS_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.json"
NUM_PROMPTS=${NUM_PROMPTS:-10}
OUTPUT_LEN=${OUTPUT_LEN:-200}
# PROMPT_SET lets us switch between the large prompt bundles (set only when needed).
PROMPT_SET=${PROMPT_SET:-}
PROMPT_TOKEN_CHECK=${PROMPT_TOKEN_CHECK:-0}
MODEL_MAX_LEN=${MODEL_MAX_LEN:-131072}

# Default prompt set is the short ~800 token accuracy prompt bundle (10 requests).
SEQ_PROMPT_FILE_DEFAULT="/home/ubuntu/vllm/disagg_prompts/seq_800.txt"
CONC_PROMPT_FILE_DEFAULT="/home/ubuntu/vllm/disagg_prompts/conc_800.txt"
PROMPT_SET_LABEL="base_800"

case "$PROMPT_SET" in
    0)
        SEQ_PROMPT_FILE_DEFAULT="/home/ubuntu/vllm/disagg_prompts/seq_48k.txt"
        CONC_PROMPT_FILE_DEFAULT="/home/ubuntu/vllm/disagg_prompts/conc_24k.txt"
        PROMPT_SET_LABEL="0(48k/24k)"
        ;;
    1)
        SEQ_PROMPT_FILE_DEFAULT="/home/ubuntu/vllm/disagg_prompts/seq_24k.txt"
        CONC_PROMPT_FILE_DEFAULT="/home/ubuntu/vllm/disagg_prompts/conc_12k.txt"
        PROMPT_SET_LABEL="1(24k/12k)"
        ;;
    2)
        SEQ_PROMPT_FILE_DEFAULT="/home/ubuntu/vllm/disagg_prompts/seq_18k.txt"
        CONC_PROMPT_FILE_DEFAULT="/home/ubuntu/vllm/disagg_prompts/conc_9k.txt"
        PROMPT_SET_LABEL="2(18k/9k)"
        ;;
esac

SEQ_PROMPT_FILE=${SEQ_PROMPT_FILE:-$SEQ_PROMPT_FILE_DEFAULT}
CONC_PROMPT_FILE=${CONC_PROMPT_FILE:-$CONC_PROMPT_FILE_DEFAULT}

# Create results directory
mkdir -p "$RESULTS_DIR"

# Export variables for Python
export MODEL_NAME PROXY_URL PREFILL_URL DECODE_URL
export NUM_PROMPTS OUTPUT_LEN RESULT_FILE TIMESTAMP SEQ_PROMPT_FILE CONC_PROMPT_FILE
export PROMPT_TOKEN_CHECK MODEL_MAX_LEN PROMPT_SET

echo "=============================================="
echo "Disaggregated Prefill Benchmark"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Proxy URL: $PROXY_URL"
if [ -n "$PROMPT_SET" ]; then
    echo "Prompt Files: seq=$SEQ_PROMPT_FILE conc=$CONC_PROMPT_FILE (PROMPT_SET=$PROMPT_SET_LABEL)"
else
    echo "Prompt Files: seq=$SEQ_PROMPT_FILE conc=$CONC_PROMPT_FILE (PROMPT_SET=$PROMPT_SET_LABEL)"
fi
echo "Result File: $RESULT_FILE"
echo "=============================================="

# =============================================================================
# nsys setup: NSYS_PROFILE=1 이면 프리필 서버를 nsys 아래에서 시작
#   벤치마크(섹션 1-3) 실행 중의 실제 cudaMemcpy 시간을 측정
# =============================================================================
NSYS_RESULT_TEMP="/tmp/nsys_result_${TIMESTAMP}.json"
NSYS_ENABLED=0

if [ "${NSYS_PROFILE}" = "1" ] && command -v nsys &> /dev/null; then
    echo ""
    echo "=============================================="
    echo "nsys: Starting prefill server under nsys profiling"
    echo "=============================================="

    NSYS_OUTPUT="/tmp/nsys_nixl_${TIMESTAMP}"
    PREFILL_IP="172.31.2.19"
    DECODE_IP="172.31.0.191"
    SSH_KEY="~/.ssh/hayoung-cluster.pem"
    VLLM_BIN="${VLLM_BIN:-/home/ubuntu/vllm/.venv/bin/vllm}"

    # Kill existing prefill server (패턴을 구체적으로 지정하여 proxy 서버를 건드리지 않음)
    echo "[nsys] Killing existing prefill server..."
    pkill -f "vllm serve.*--port 8100" 2>/dev/null || true
    sleep 3
    if pgrep -f "vllm serve.*--port 8100" > /dev/null 2>&1; then
        pkill -9 -f "vllm serve.*--port 8100" 2>/dev/null || true
        sleep 2
    fi

    # Start prefill server under nsys
    echo "[nsys] Starting prefill server under nsys..."
    # 모델이 이미 로컬 캐시에 있으면 HF 네트워크 요청 건너뛰기 (gated model 401 방지)
    export HF_HUB_OFFLINE=1
    export VLLM_NIXL_SIDE_CHANNEL_HOST=$PREFILL_IP
    export VLLM_NIXL_SIDE_CHANNEL_PORT=14580
    export UCX_TLS=cuda_copy,tcp
    export UCX_NET_DEVICES=ens5
    export UCX_TCP_PORT_RANGE=40000-40009


    nsys profile \
        --trace=cuda \
        --sample=none \
        --output=${NSYS_OUTPUT} \
        --force-overwrite=true \
        ${VLLM_BIN} serve ${MODEL_NAME} \
            --host 0.0.0.0 \
            --port 8100 \
            --max-model-len 24576 \
            --gpu-memory-utilization 0.8 \
            --trust-remote-code \
            --enforce-eager \
            --kv-transfer-config \
            '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
        > /tmp/nsys_server_${TIMESTAMP}.log 2>&1 &
    NSYS_PID=$!
    echo "  nsys PID: $NSYS_PID"

    # Wait for prefill server to be ready
    echo "[nsys] Waiting for prefill server to be ready (max 60)..."
    for i in $(seq 1 60); do
        # Check if nsys process is still alive
        if ! kill -0 $NSYS_PID 2>/dev/null; then
            echo "  ERROR: nsys process died. Last log lines:"
            tail -5 /tmp/nsys_server_${TIMESTAMP}.log 2>/dev/null
            break
        fi
        if curl -s -o /dev/null -w "%{http_code}" ${PREFILL_URL}/health 2>/dev/null | grep -q "200"; then
            echo "  Prefill server ready after ${i}s"
            NSYS_ENABLED=1
            break
        fi
        # Print progress every 10s
        if [ $((i % 10)) -eq 0 ]; then
            echo "  ...waiting ${i}s (last log: $(tail -1 /tmp/nsys_server_${TIMESTAMP}.log 2>/dev/null | cut -c1-80))"
        fi
        sleep 1
    done

    if [ "$NSYS_ENABLED" = "0" ]; then
        echo "  ERROR: Prefill server did not start under nsys."
        echo "  Last 10 log lines:"
        tail -10 /tmp/nsys_server_${TIMESTAMP}.log 2>/dev/null
        kill $NSYS_PID 2>/dev/null; wait $NSYS_PID 2>/dev/null
    else
        echo "  Prefill server running under nsys. Benchmark requests will be profiled."

        # Decode 서버도 재시작 (새 prefill과 NIXL 연결 수립을 위해)
        echo ""
        echo "[nsys] Restarting decode server on ${DECODE_IP} via SSH..."
        ssh -i ${SSH_KEY} -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@${DECODE_IP} bash -s << 'DECODE_SSH'
pkill -f "vllm serve.*--port 8200" 2>/dev/null || true
sleep 3
if pgrep -f "vllm serve.*--port 8200" > /dev/null 2>&1; then
    pkill -9 -f "vllm serve.*--port 8200" 2>/dev/null || true
    sleep 2
fi

export HF_HUB_OFFLINE=1
export VLLM_NIXL_SIDE_CHANNEL_HOST=172.31.0.191
export VLLM_NIXL_SIDE_CHANNEL_PORT=14580
export UCX_TLS=cuda_copy,tcp
export UCX_NET_DEVICES=ens5
export UCX_TCP_PORT_RANGE=40000-40009


nohup /home/ubuntu/vllm/.venv/bin/vllm serve meta-llama/Llama-3.2-1B \
    --host 0.0.0.0 \
    --port 8200 \
    --max-model-len 24576 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
    > /tmp/decode_server_nsys.log 2>&1 &
echo "  Decode server PID: $!"
DECODE_SSH

        # Wait for decode server to be ready
        echo "[nsys] Waiting for decode server to be ready (max 60s)..."
        for i in $(seq 1 60); do
            if curl -s -o /dev/null -w "%{http_code}" ${DECODE_URL}/health 2>/dev/null | grep -q "200"; then
                echo "  Decode server ready after ${i}s"
                break
            fi
            if [ $((i % 10)) -eq 0 ]; then
                echo "  ...waiting ${i}s"
            fi
            sleep 1
        done

        if ! curl -s -o /dev/null -w "%{http_code}" ${DECODE_URL}/health 2>/dev/null | grep -q "200"; then
            echo "  WARNING: Decode server not ready after 60s. Benchmark may fail."
        fi
    fi

elif [ "${NSYS_PROFILE}" = "1" ]; then
    echo "nsys not found. Skipping CUDA memcpy profiling."

else
    # ==========================================================================
    # Normal mode: start prefill & decode servers (without nsys)
    # ==========================================================================
    PREFILL_IP="172.31.2.19"
    DECODE_IP="172.31.0.191"
    SSH_KEY="$HOME/.ssh/hayoung-cluster.pem"
    VLLM_BIN="/home/ubuntu/vllm/.venv/bin/vllm"

    echo ""
    echo "=============================================="
    echo "Starting prefill & decode servers"
    echo "=============================================="

    # --- Kill existing servers ---
    pkill -f "vllm serve.*--port 8100" 2>/dev/null || true
    ssh -i ${SSH_KEY} -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@${DECODE_IP} \
        'pkill -f "vllm serve.*--port 8200" 2>/dev/null || true' 2>/dev/null || true
    sleep 3
    pkill -9 -f "vllm serve.*--port 8100" 2>/dev/null || true
    ssh -i ${SSH_KEY} -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@${DECODE_IP} \
        'pkill -9 -f "vllm serve.*--port 8200" 2>/dev/null || true' 2>/dev/null || true
    sleep 2

    # --- Free GPU memory from orphaned processes ---
    for pid in $(fuser /dev/nvidia* 2>/dev/null | tr -s ' '); do
        kill -9 $pid 2>/dev/null || true
    done
    ssh -i ${SSH_KEY} -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@${DECODE_IP} \
        'for pid in $(fuser /dev/nvidia* 2>/dev/null | tr -s " "); do kill -9 $pid 2>/dev/null || true; done' 2>/dev/null || true
    sleep 2

    # --- Start decode server on remote via SSH ---
    echo "[normal] Starting decode server on ${DECODE_IP}..."
    ssh -i ${SSH_KEY} -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@${DECODE_IP} bash -s << 'DECODE_NORMAL_SSH'
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_NIXL_SIDE_CHANNEL_HOST=172.31.0.191
export VLLM_NIXL_SIDE_CHANNEL_PORT=14580
export UCX_TLS=cuda_copy,tcp
export UCX_NET_DEVICES=ens5
export UCX_TCP_PORT_RANGE=40000-40009


nohup /home/ubuntu/vllm/.venv/bin/vllm serve meta-llama/Llama-3.2-1B \
    --host 0.0.0.0 \
    --port 8200 \
            --max-model-len 24576 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
    > /tmp/decode_server.log 2>&1 &
echo "  Decode server PID: $!"
DECODE_NORMAL_SSH

    # --- Start prefill server locally ---
    echo "[normal] Starting prefill server on ${PREFILL_IP}..."
    export HF_HUB_OFFLINE=1
    export VLLM_LOGGING_LEVEL=DEBUG
    export VLLM_NIXL_SIDE_CHANNEL_HOST=$PREFILL_IP
    export VLLM_NIXL_SIDE_CHANNEL_PORT=14580
    export UCX_TLS=cuda_copy,tcp
    export UCX_NET_DEVICES=ens5
    export UCX_TCP_PORT_RANGE=40000-40009


    ${VLLM_BIN} serve ${MODEL_NAME} \
        --host 0.0.0.0 \
        --port 8100 \
    --max-model-len 24576 \
        --gpu-memory-utilization 0.8 \
        --trust-remote-code \
        --enforce-eager \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
        > /tmp/prefill_server.log 2>&1 &
    PREFILL_PID=$!
    echo "  Prefill server PID: $PREFILL_PID"

    # --- Wait for both servers to be ready ---
    echo "[normal] Waiting for servers to be ready (max 60s)..."
    SERVERS_READY=0
    for i in $(seq 1 60); do
        P_OK=$(curl -s -o /dev/null -w "%{http_code}" http://${PREFILL_IP}:8100/health 2>/dev/null || true)
        D_OK=$(curl -s -o /dev/null -w "%{http_code}" http://${DECODE_IP}:8200/health 2>/dev/null || true)
        if [ "$P_OK" = "200" ] && [ "$D_OK" = "200" ]; then
            echo "  Both servers ready after ${i}s"
            SERVERS_READY=1
            break
        fi
        if [ $((i % 15)) -eq 0 ]; then
            echo "  ...waiting ${i}s (prefill=${P_OK}, decode=${D_OK})"
        fi
        sleep 1
    done

    if [ "$SERVERS_READY" = "0" ]; then
        echo "  WARNING: Servers not ready after 60s. Benchmark may fail."
        echo "  Prefill log tail:"
        tail -5 /tmp/prefill_server.log 2>/dev/null
        echo "  Decode log tail (remote):"
        ssh -i ${SSH_KEY} -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@${DECODE_IP} \
            'tail -5 /tmp/decode_server.log' 2>/dev/null
    fi
fi

export NSYS_RESULT_TEMP

# Run all benchmarks in one Python script, save to ONE file
python3 << 'PYTHON_ALL'
import json
import time
import requests
import statistics
import os
import httpx
from datetime import datetime

# Configuration from shell
PROXY_URL = os.environ.get("PROXY_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-1B")
PREFILL_URL = os.environ.get("PREFILL_URL", "http://172.31.2.19:8100")
DECODE_URL = os.environ.get("DECODE_URL", "http://172.31.0.191:8200")
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "20"))
OUTPUT_LEN = int(os.environ.get("OUTPUT_LEN", "200"))
SEQ_USE_BASE = os.environ.get("SEQ_USE_BASE", "0") == "1"
PROMPT_TOKEN_CHECK = os.environ.get("PROMPT_TOKEN_CHECK", "0") == "1"
MODEL_MAX_LEN = os.environ.get("MODEL_MAX_LEN", "24576")
SEQ_PROMPT_FILE = os.environ.get(
    "SEQ_PROMPT_FILE",
    "/home/ubuntu/vllm/disagg_prompts/seq_96k.txt",
)
CONC_PROMPT_FILE = os.environ.get(
    "CONC_PROMPT_FILE",
    "/home/ubuntu/vllm/disagg_prompts/conc_48k.txt",
)

RESULT_FILE = os.environ.get("RESULT_FILE", "/home/ubuntu/vllm/results/benchmark_disagg/benchmark_disagg.json")
TIMESTAMP = os.environ.get("TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))

STREAM_HEADERS = {
    "Accept": "text/event-stream",
    "Cache-Control": "no-cache",
    "Accept-Encoding": "identity",
}
RAW_SSE_PREVIEW_BYTES = 2048
STREAM_CLIENT = httpx.Client(
    timeout=None,
    headers=STREAM_HEADERS,
    trust_env=False,
)

# SSE helpers
def iter_sse_data_bytes(byte_iter):
    """Yield raw data payloads from SSE responses as bytes."""
    buffer = b""
    for chunk in byte_iter:
        if not chunk:
            continue
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.lstrip(b"\r")
            if line.startswith(b"data: "):
                yield line[6:]
            elif line.startswith(b"data:"):
                yield line[5:]
    buffer = buffer.lstrip(b"\r")
    if buffer.startswith(b"data: "):
        yield buffer[6:]
    elif buffer.startswith(b"data:"):
        yield buffer[5:]

def _validate_completion(chunks):
    """Validate that the response is well-formed and non-empty."""
    if not chunks:
        return False, "no_chunks"
    # If any chunk has non-empty text, consider it valid.
    for ch in chunks:
        try:
            txt = ch.get("choices", [{}])[0].get("text", "")
            if txt:
                return True, None
        except Exception:
            continue
    # Otherwise, if a finish_reason exists, still accept as valid but empty.
    for ch in chunks:
        try:
            fr = ch.get("choices", [{}])[0].get("finish_reason")
            if fr is not None:
                return False, "empty_text_finish"
        except Exception:
            continue
    return False, "invalid_chunks"

# Helper function to calculate percentiles
def percentile(data, p):
    if not data:
        return 0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]

# Helper function to fetch NIXL metrics from a server
def get_nixl_metrics_from(url):
    """Fetch NIXL metrics from a server's /metrics endpoint"""
    try:
        resp = requests.get(f"{url}/metrics", timeout=10)
        if resp.status_code != 200:
            return None

        metrics = {}
        for line in resp.text.split('\n'):
            if line.startswith('vllm:nixl_') and ('_sum{' in line or '_count{' in line):
                parts = line.split(' ')
                if len(parts) >= 2:
                    name = parts[0].split('{')[0]
                    value = float(parts[1])
                    metrics[name] = value
        return metrics
    except Exception as e:
        print(f"  Warning: Could not fetch NIXL metrics from {url}: {e}")
        return None

def get_nixl_metrics_both():
    """Fetch NIXL metrics from both decode AND prefill servers.
    In pull mode, decode has the metrics. In push mode, prefill has them."""
    decode_m = get_nixl_metrics_from(DECODE_URL) or {}
    prefill_m = get_nixl_metrics_from(PREFILL_URL) or {}
    # Merge: take the max of each metric across both servers
    merged = {}
    for key in set(list(decode_m.keys()) + list(prefill_m.keys())):
        merged[key] = max(decode_m.get(key, 0), prefill_m.get(key, 0))
    return merged if merged else None

# Initialize ONE consolidated result object
# Detect transfer mode from the process actually listening on proxy port
import subprocess
try:
    # ss -tlnp shows the listener; lsof -i shows the PID owning the port
    _lsof = subprocess.run(
        ["lsof", "-iTCP:8000", "-sTCP:LISTEN", "-t"],
        capture_output=True, text=True, timeout=5
    ).stdout.strip()
    if _lsof:
        _pid = _lsof.split('\n')[0]
        _cmdline = open(f"/proc/{_pid}/cmdline", "rb").read().decode(errors="replace").replace('\x00', ' ')
    else:
        _cmdline = ""
except Exception:
    _cmdline = ""
if "push_proxy" in _cmdline:
    transfer_mode = "push_sync" if os.environ.get("VLLM_PUSH_SYNC_ACK") == "1" else "push_async"
else:
    transfer_mode = "pull"

result = {
    "transfer_mode": transfer_mode,
    "summary": {},  # Will be filled at the end with key metrics
    "benchmark_id": TIMESTAMP,
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "configuration": {
        "proxy_url": PROXY_URL,
        "prefill_url": PREFILL_URL,
        "decode_url": DECODE_URL,
        "ucx_tls": "cuda_copy,tcp",
        "output_len": OUTPUT_LEN,
        "transfer_mode": transfer_mode,
    },
    "connectivity_test": {},
    "benchmark": {},
    "accuracy": {}
}

# =============================================================================
# 1. Connectivity Test
# =============================================================================
print("\n=== 1. Connectivity Test ===")
try:
    resp = requests.post(f"{PROXY_URL}/v1/completions",
        json={"model": MODEL_NAME, "prompt": "Hello", "max_tokens": 5, "temperature": 0}, timeout=60)
    if resp.status_code == 200:
        result["connectivity_test"] = {"status": "passed", "response_code": 200}
        print("Connectivity test PASSED")
    else:
        result["connectivity_test"] = {"status": "failed", "response_code": resp.status_code}
        print(f"Connectivity test FAILED: {resp.status_code}")
except Exception as e:
    result["connectivity_test"] = {"status": "error", "error": str(e)}
    print(f"Connectivity test ERROR: {e}")

# =============================================================================
# 2. Benchmark (Latency, Throughput, TTFT, TPOT, NIXL Transfer)
# =============================================================================
print(f"\n=== 2. Sequential Benchmark (output_len={OUTPUT_LEN}) ===")

# 긴 프롬프트 (~800 tokens each)
_base_prompts = [
    "The quick brown fox jumps over the lazy dog and then runs across the wide open field where many animals gather to play and rest under the warm afternoon sun while birds sing their beautiful songs in the tall green trees nearby. The river flows gently through the valley carrying fallen leaves and small twigs downstream toward the distant ocean where waves crash upon the rocky shore. Fishermen cast their nets into the deep blue water hoping for a bountiful catch while seagulls circle overhead calling to one another. The clouds drift slowly across the sky painting shadows on the landscape below as the day progresses from morning to afternoon. Children play in the meadow chasing butterflies and picking wildflowers to bring home to their families. The old stone bridge arches gracefully over the stream connecting the two villages that have traded goods for centuries. Farmers tend their crops in the fertile fields watching the weather for signs of rain that will nourish the growing plants. The forest at the edge of town is home to deer foxes rabbits and countless species of birds that fill the air with music at dawn.",
    "In a galaxy far far away there existed a civilization of advanced beings who had mastered the art of interstellar travel and communication across vast distances using quantum entanglement technology that allowed them to share knowledge instantly across light years. Their ships were powered by antimatter engines capable of bending spacetime itself creating stable wormholes between star systems. The civilization had colonized thousands of worlds each with its own unique ecosystem and culture but all connected through a vast neural network that spanned the galaxy. Scientists on the homeworld continued to push the boundaries of physics discovering new dimensions of reality that challenged everything they thought they knew about the universe. The council of elders governed wisely balancing the needs of trillions of citizens spread across countless planets moons and space stations. Artists created works that could only be experienced in zero gravity while musicians composed symphonies using the electromagnetic frequencies of pulsars and magnetars. Engineers built megastructures around dying stars harvesting their final energy output to power the civilization for millennia to come.",
    "Once upon a time there was a young wizard who discovered an ancient book of spells hidden deep within the forbidden library of the grand castle where generations of powerful sorcerers had studied and practiced their magical arts for over a thousand years. The book was bound in dragon leather and its pages were made from enchanted parchment that could only be read by moonlight. Each spell within was more powerful and dangerous than the last requiring immense concentration and magical energy to cast properly. The young wizard spent months studying the first chapter alone learning the fundamental principles of elemental manipulation and dimensional folding. The castle itself was alive with magic its walls shifting and corridors rearranging themselves according to ancient enchantments placed by the founders. Ghosts of former students wandered the halls offering cryptic advice to those brave enough to listen. The library contained millions of books scrolls and artifacts collected from every corner of the known world and several corners of worlds unknown. Deep beneath the castle lay a network of caverns where underground rivers of pure magical energy flowed providing power to the wards and enchantments that protected the school."
]

def _load_prompts(path: str, n: int):
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return None
        return [lines[i % len(lines)] for i in range(n)]
    except Exception:
        return None

def get_prompt_token_count(prompt: str):
    if not PROMPT_TOKEN_CHECK:
        return None, None
    try:
        resp = requests.post(
            f"{PROXY_URL}/v1/completions/render",
            json={"model": MODEL_NAME, "prompt": prompt, "max_tokens": OUTPUT_LEN, "temperature": 0, "stream": False},
            timeout=30,
        )
        if resp.status_code != 200:
            return None, None
        data = resp.json()
        if isinstance(data, list) and data:
            item = data[0]
            token_ids = item.get("prompt_token_ids")
            if token_ids is not None:
                prompt_tokens = len(token_ids)
            else:
                prompt_tokens = None
        else:
            prompt_tokens = None
        remaining = None
        if prompt_tokens is not None and MODEL_MAX_LEN:
            try:
                remaining = int(MODEL_MAX_LEN) - prompt_tokens
            except ValueError:
                remaining = None
        return prompt_tokens, remaining
    except Exception:
        return None, None
if SEQ_USE_BASE:
    benchmark_prompts = [_base_prompts[0]] * NUM_PROMPTS
    print("Using sequential prompts from: _base_prompts[0] (SEQ_USE_BASE=1)")
else:
    benchmark_prompts = _load_prompts(SEQ_PROMPT_FILE, NUM_PROMPTS)
    if benchmark_prompts is None:
        # Fallback to built-in prompts
        benchmark_prompts = [
            _base_prompts[i % len(_base_prompts)] for i in range(NUM_PROMPTS)
        ]
    else:
        print(f"Using sequential prompts from: {SEQ_PROMPT_FILE}")

measurements = []
latencies = []
ttft_values = []
tpot_values = []
nixl_xfer_times = []
total_tokens = 0

print(f"Running {len(benchmark_prompts)} sequential requests...")
bench_start = time.perf_counter()

# Sanity check: run one request with same params as accuracy (long prompt)
print("\n[Sanity] Single request with accuracy-style prompt/max_tokens")
_sanity_prompt = _base_prompts[0]
_sanity_max_tokens = 200
try:
    _s_start = time.perf_counter()
    _s_parts = []
    with STREAM_CLIENT.stream(
        "POST",
        f"{PROXY_URL}/v1/completions",
        json={"model": MODEL_NAME, "prompt": _sanity_prompt, "max_tokens": _sanity_max_tokens, "temperature": 0, "stream": True},
        timeout=300,
    ) as _s_resp:
        if _s_resp.status_code == 200:
            for _data in iter_sse_data_bytes(_s_resp.iter_bytes()):
                _chunk_data = _data.decode("utf-8", errors="ignore")
                if _chunk_data.strip() == "[DONE]":
                    break
                try:
                    _chunk = json.loads(_chunk_data)
                    _txt = _chunk.get("choices", [{}])[0].get("text", "")
                    if _txt:
                        _s_parts.append(_txt)
                except (json.JSONDecodeError, KeyError):
                    pass
        _s_elapsed = (time.perf_counter() - _s_start) * 1000
    _s_preview = "".join(_s_parts)[:120]
    print(f"[Sanity] status={_s_resp.status_code} elapsed_ms={_s_elapsed:.2f} preview='{_s_preview}'")
except Exception as _e:
    print(f"[Sanity] ERROR - {_e}")

for i, prompt in enumerate(benchmark_prompts):
    # Get NIXL metrics BEFORE this request
    nixl_before = get_nixl_metrics_both()

    start = time.perf_counter()
    ttft = None
    tokens = 0
    chunk_times_ms = []
    chunk_count = 0
    sse_samples = []
    sse_content_type = None
    sse_first_bytes = None
    first_byte_ms = None
    first_event_ms = None
    last_event_ms = None
    parsed_chunks = []
    parse_errors = 0
    http_ok = False
    valid_text = False
    usage_tokens = None
    completion_parts = []
    raw_bytes = bytearray()
    non_stream_fallback = False
    prompt_tokens = None
    remaining_tokens_estimate = None
    if PROMPT_TOKEN_CHECK:
        prompt_tokens, remaining_tokens_estimate = get_prompt_token_count(prompt)
    try:
        # Use streaming to measure REAL TTFT
        def _capture_iter(byte_iter):
            for _chunk in byte_iter:
                if _chunk and len(raw_bytes) < RAW_SSE_PREVIEW_BYTES:
                    take = RAW_SSE_PREVIEW_BYTES - len(raw_bytes)
                    raw_bytes.extend(_chunk[:take])
                yield _chunk

        with STREAM_CLIENT.stream(
            "POST",
            f"{PROXY_URL}/v1/completions",
            json={"model": MODEL_NAME, "prompt": prompt, "max_tokens": OUTPUT_LEN, "temperature": 0, "stream": True},
            timeout=300,
        ) as resp:
            sse_content_type = resp.headers.get("content-type")
            http_ok = resp.status_code == 200
            if http_ok:
                for data in iter_sse_data_bytes(_capture_iter(resp.iter_bytes())):
                    if first_event_ms is None:
                        first_event_ms = (time.perf_counter() - start) * 1000
                    chunk_data = data.decode("utf-8", errors="ignore")
                    if chunk_data.strip() == "[DONE]":
                        break
                    if len(sse_samples) < 3:
                        sse_samples.append(chunk_data)
                    if sse_first_bytes is None:
                        sse_first_bytes = chunk_data[:128]
                    chunk_count += 1
                    now_ms = (time.perf_counter() - start) * 1000
                    if first_byte_ms is None:
                        first_byte_ms = now_ms
                    chunk_times_ms.append(now_ms)
                    last_event_ms = now_ms
                    try:
                        chunk = json.loads(chunk_data)
                        parsed_chunks.append(chunk)
                        if ttft is None:
                            ttft = (time.perf_counter() - start) * 1000
                        choice = chunk.get("choices", [{}])[0]
                        txt = choice.get("text", "")
                        if txt:
                            valid_text = True
                            tokens += 1
                            completion_parts.append(txt)
                        # Check for finish_reason
                        usage = chunk.get("usage")
                        if usage and usage.get("completion_tokens"):
                            usage_tokens = usage["completion_tokens"]
                    except (json.JSONDecodeError, KeyError):
                        parse_errors += 1
                        pass

            if not parsed_chunks and raw_bytes:
                raw_preview = raw_bytes.decode("utf-8", errors="replace").strip()
                if raw_preview.startswith("{"):
                    try:
                        parsed = json.loads(raw_preview)
                        parsed_chunks.append(parsed)
                        choice = parsed.get("choices", [{}])[0]
                        txt = choice.get("text", "")
                        if txt:
                            completion_parts.append(txt)
                            valid_text = True
                        usage = parsed.get("usage") or {}
                        completion_tokens = usage.get("completion_tokens")
                        if completion_tokens is not None:
                            usage_tokens = completion_tokens
                        non_stream_fallback = True
                    except json.JSONDecodeError:
                        pass

            latency = (time.perf_counter() - start) * 1000

            # Get NIXL metrics AFTER this request
            nixl_after = get_nixl_metrics_both()

            nixl_xfer_ms = 0
            nixl_bytes_kb = 0
            if nixl_before and nixl_after:
                xfer_time_diff = nixl_after.get('vllm:nixl_xfer_time_seconds_sum', 0) - nixl_before.get('vllm:nixl_xfer_time_seconds_sum', 0)
                bytes_diff = nixl_after.get('vllm:nixl_bytes_transferred_sum', 0) - nixl_before.get('vllm:nixl_bytes_transferred_sum', 0)
                nixl_xfer_ms = xfer_time_diff * 1000
                nixl_bytes_kb = bytes_diff / 1024

            if usage_tokens is not None:
                tokens = usage_tokens
            if ttft is None:
                ttft = first_event_ms if first_event_ms is not None else (first_byte_ms if first_byte_ms is not None else latency)
            if len(chunk_times_ms) >= 2:
                stream_span = chunk_times_ms[-1] - chunk_times_ms[0]
                tpot = stream_span / max(tokens - 1, 1)
                tpot_fallback = None
            else:
                stream_span = 0
                tpot = None
                tpot_fallback = (latency - ttft) / max(tokens - 1, 1)

            is_valid, invalid_reason = _validate_completion(parsed_chunks)
            status = "success" if http_ok else "failed"
            if completion_parts and not valid_text:
                valid_text = True
            raw_preview = raw_bytes.decode("utf-8", errors="replace") if raw_bytes else None
            latencies.append(latency)
            ttft_values.append(ttft)
            tpot_values.append(tpot if tpot is not None else tpot_fallback)
            nixl_xfer_times.append(nixl_xfer_ms)
            total_tokens += tokens
            measurements.append({
                "request_id": i+1,
                "status": status,
                "http_ok": http_ok,
                "valid_text": valid_text,
                "invalid_reason": invalid_reason,
                "parse_errors": parse_errors,
                "latency_ms": round(latency, 2),
                "ttft_ms": round(ttft, 2),
                "tpot_ms": round(tpot, 2) if tpot is not None else None,
                "tpot_fallback_ms": round(tpot_fallback, 2) if tpot_fallback is not None else None,
                "chunk_count": chunk_count,
                "stream_span_ms": round(stream_span, 2),
                "sse_samples": sse_samples,
                "sse_content_type": sse_content_type,
                "sse_first_bytes": sse_first_bytes,
                "sse_raw_first_bytes": raw_preview[:256] if raw_preview else None,
                "first_byte_ms": round(first_byte_ms, 2) if first_byte_ms is not None else None,
                "first_event_ms": round(first_event_ms, 2) if first_event_ms is not None else None,
                "last_event_ms": round(last_event_ms, 2) if last_event_ms is not None else None,
                "nixl_xfer_ms": round(nixl_xfer_ms, 3),
                "nixl_bytes_kb": round(nixl_bytes_kb, 2),
                "prompt_tokens": prompt_tokens,
                "remaining_tokens_estimate": remaining_tokens_estimate,
                "completion_tokens": tokens,
                "completion_preview": "".join(completion_parts)[:120],
                "non_stream_fallback": non_stream_fallback
            })
            tpot_display = f"{tpot:.2f}ms" if tpot is not None else f"fallback {tpot_fallback:.2f}ms"
            print(f"  Request {i+1}: status={status}, latency={latency:.2f}ms, TTFT={ttft:.2f}ms, TPOT={tpot_display}, stream_span={stream_span:.2f}ms, chunks={chunk_count}, NIXL_xfer={nixl_xfer_ms:.2f}ms, tokens={tokens}")
    except Exception as e:
        measurements.append({"request_id": i+1, "status": "error", "error": str(e)})
        print(f"  Request {i+1}: ERROR - {e}")

bench_end = time.perf_counter()
total_time = bench_end - bench_start

# Calculate NIXL xfer time statistics
nixl_stats = {}
if nixl_xfer_times:
    nixl_stats = {
        "avg": round(statistics.mean(nixl_xfer_times), 3),
        "min": round(min(nixl_xfer_times), 3),
        "max": round(max(nixl_xfer_times), 3),
        "p50": round(percentile(nixl_xfer_times, 50), 3),
        "p90": round(percentile(nixl_xfer_times, 90), 3),
        "p99": round(percentile(nixl_xfer_times, 99), 3),
        "stddev": round(statistics.stdev(nixl_xfer_times), 3) if len(nixl_xfer_times) > 1 else 0
    }

lat_stats = {}
ttft_stats = {}
tpot_stats = {}
if latencies:
    lat_stats = {
        "avg": round(statistics.mean(latencies), 2),
        "min": round(min(latencies), 2),
        "max": round(max(latencies), 2),
        "p50": round(percentile(latencies, 50), 2),
        "p90": round(percentile(latencies, 90), 2),
        "p99": round(percentile(latencies, 99), 2),
        "stddev": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0
    }
if ttft_values:
    ttft_stats = {
        "avg": round(statistics.mean(ttft_values), 2),
        "p50": round(percentile(ttft_values, 50), 2),
        "p90": round(percentile(ttft_values, 90), 2),
        "p99": round(percentile(ttft_values, 99), 2)
    }
if tpot_values:
    tpot_stats = {
        "avg": round(statistics.mean(tpot_values), 2),
        "p50": round(percentile(tpot_values, 50), 2),
        "p90": round(percentile(tpot_values, 90), 2),
        "p99": round(percentile(tpot_values, 99), 2)
    }

result["benchmark"] = {
    "measurements": measurements,
    "summary": {
        "total_requests": len(benchmark_prompts),
        "successful_requests": len(latencies),
        "valid_text_requests": sum(1 for m in measurements if m.get("valid_text")),
        "invalid_text_requests": sum(1 for m in measurements if m.get("http_ok") and not m.get("valid_text")),
        "failed_requests": len(benchmark_prompts) - len(latencies),
        "total_time_sec": round(total_time, 2),
        "throughput_req_per_sec": round(len(latencies) / total_time, 2) if latencies else 0,
        "throughput_tokens_per_sec": round(total_tokens / total_time, 2) if latencies else 0,
        "latency_ms": lat_stats,
        "nixl_xfer_ms": nixl_stats,
        "ttft_ms": ttft_stats,
        "tpot_ms": tpot_stats
    }
}
s = result["benchmark"]["summary"]
print(f"\n--- Summary ---")
if latencies:
    print(f"Throughput: {s['throughput_req_per_sec']:.2f} req/s, {s['throughput_tokens_per_sec']:.2f} tokens/s")
    print(f"Latency: Avg={s['latency_ms']['avg']:.2f}ms, P90={s['latency_ms']['p90']:.2f}ms, P99={s['latency_ms']['p99']:.2f}ms")
    if nixl_stats:
        print(f"NIXL KV Transfer: Avg={nixl_stats['avg']:.2f}ms, P90={nixl_stats['p90']:.2f}ms, P99={nixl_stats['p99']:.2f}ms")
    if ttft_stats:
        print(f"TTFT: Avg={s['ttft_ms']['avg']:.2f}ms, P90={s['ttft_ms']['p90']:.2f}ms, P99={s['ttft_ms']['p99']:.2f}ms")
    if tpot_stats:
        print(f"TPOT: Avg={s['tpot_ms']['avg']:.2f}ms, P90={s['tpot_ms']['p90']:.2f}ms, P99={s['tpot_ms']['p99']:.2f}ms")

# =============================================================================
# 2-B. Concurrent Benchmark
# =============================================================================
print(f"\n=== 2-B. Concurrent Benchmark ({NUM_PROMPTS} requests) ===")

import concurrent.futures

conc_measurements = []
conc_latencies = []
conc_ttft_values = []
conc_tpot_values = []
conc_total_tokens = 0

def run_one_request(idx, prompt):
    """Send one request with streaming to measure real TTFT."""
    nixl_before = get_nixl_metrics_both()
    start = time.perf_counter()
    ttft = None
    tokens = 0
    sse_samples = []
    sse_content_type = None
    sse_first_bytes = None
    first_byte_ms = None
    first_event_ms = None
    last_event_ms = None
    parsed_chunks = []
    raw_bytes = bytearray()
    non_stream_fallback = False
    parse_errors = 0
    http_ok = False
    valid_text = False
    usage_tokens = None
    completion_parts = []
    prompt_tokens = None
    remaining_tokens_estimate = None
    if PROMPT_TOKEN_CHECK:
        prompt_tokens, remaining_tokens_estimate = get_prompt_token_count(prompt)
    try:
        def _capture_iter(byte_iter):
            for _chunk in byte_iter:
                if _chunk and len(raw_bytes) < RAW_SSE_PREVIEW_BYTES:
                    take = RAW_SSE_PREVIEW_BYTES - len(raw_bytes)
                    raw_bytes.extend(_chunk[:take])
                yield _chunk

        with STREAM_CLIENT.stream(
            "POST",
            f"{PROXY_URL}/v1/completions",
            json={"model": MODEL_NAME, "prompt": prompt, "max_tokens": OUTPUT_LEN, "temperature": 0, "stream": True},
            timeout=300,
        ) as resp:
            sse_content_type = resp.headers.get("content-type")
            http_ok = resp.status_code == 200
            if http_ok:
                for data in iter_sse_data_bytes(_capture_iter(resp.iter_bytes())):
                    if first_event_ms is None:
                        first_event_ms = (time.perf_counter() - start) * 1000
                    chunk_data = data.decode("utf-8", errors="ignore")
                    if chunk_data.strip() == "[DONE]":
                        break
                    if len(sse_samples) < 3:
                        sse_samples.append(chunk_data)
                    if sse_first_bytes is None:
                        sse_first_bytes = chunk_data[:128]
                    try:
                        chunk = json.loads(chunk_data)
                        parsed_chunks.append(chunk)
                        if ttft is None:
                            ttft = (time.perf_counter() - start) * 1000
                        choice = chunk.get("choices", [{}])[0]
                        txt = choice.get("text", "")
                        if txt:
                            valid_text = True
                            tokens += 1
                            completion_parts.append(txt)
                        usage = chunk.get("usage")
                        if usage and usage.get("completion_tokens"):
                            usage_tokens = usage["completion_tokens"]
                    except (json.JSONDecodeError, KeyError):
                        parse_errors += 1
                        pass
                    now_ms = (time.perf_counter() - start) * 1000
                    if first_byte_ms is None:
                        first_byte_ms = now_ms
                    last_event_ms = now_ms

            if not parsed_chunks and raw_bytes:
                raw_preview = raw_bytes.decode("utf-8", errors="replace").strip()
                if raw_preview.startswith("{"):
                    try:
                        parsed = json.loads(raw_preview)
                        parsed_chunks.append(parsed)
                        choice = parsed.get("choices", [{}])[0]
                        txt = choice.get("text", "")
                        if txt:
                            completion_parts.append(txt)
                            valid_text = True
                        usage = parsed.get("usage") or {}
                        completion_tokens = usage.get("completion_tokens")
                        if completion_tokens is not None:
                            usage_tokens = completion_tokens
                        non_stream_fallback = True
                    except json.JSONDecodeError:
                        pass

            latency = (time.perf_counter() - start) * 1000

            nixl_after = get_nixl_metrics_both()
            nixl_xfer_ms = 0
            nixl_bytes_kb = 0
            if nixl_before and nixl_after:
                xfer_time_diff = nixl_after.get('vllm:nixl_xfer_time_seconds_sum', 0) - nixl_before.get('vllm:nixl_xfer_time_seconds_sum', 0)
                bytes_diff = nixl_after.get('vllm:nixl_bytes_transferred_sum', 0) - nixl_before.get('vllm:nixl_bytes_transferred_sum', 0)
                nixl_xfer_ms = xfer_time_diff * 1000
                nixl_bytes_kb = bytes_diff / 1024

            if usage_tokens is not None:
                tokens = usage_tokens
            if ttft is None:
                ttft = first_event_ms if first_event_ms is not None else (first_byte_ms if first_byte_ms is not None else latency)
            if first_event_ms is not None and last_event_ms is not None and last_event_ms > first_event_ms:
                tpot = (last_event_ms - first_event_ms) / max(tokens - 1, 1)
                tpot_fallback = None
            else:
                tpot = None
                tpot_fallback = (latency - ttft) / max(tokens - 1, 1)
            is_valid, invalid_reason = _validate_completion(parsed_chunks)
            status = "success" if http_ok else "failed"
            if completion_parts and not valid_text:
                valid_text = True
            raw_preview = raw_bytes.decode("utf-8", errors="replace") if raw_bytes else None
            return {
                "request_id": idx+1, "status": status,
                "http_ok": http_ok,
                "valid_text": valid_text,
                "invalid_reason": invalid_reason,
                "parse_errors": parse_errors,
                "latency_ms": round(latency, 2), "ttft_ms": round(ttft, 2),
                "tpot_ms": round(tpot, 2) if tpot is not None else None,
                "tpot_fallback_ms": round(tpot_fallback, 2) if tpot_fallback is not None else None,
                "sse_samples": sse_samples,
                "sse_content_type": sse_content_type,
                "sse_first_bytes": sse_first_bytes,
                "sse_raw_first_bytes": raw_preview[:256] if raw_preview else None,
                "first_byte_ms": round(first_byte_ms, 2) if first_byte_ms is not None else None,
                "first_event_ms": round(first_event_ms, 2) if first_event_ms is not None else None,
                "last_event_ms": round(last_event_ms, 2) if last_event_ms is not None else None,
                "nixl_xfer_ms": round(nixl_xfer_ms, 3),
                "nixl_bytes_kb": round(nixl_bytes_kb, 2), "completion_tokens": tokens,
                "prompt_tokens": prompt_tokens,
                "remaining_tokens_estimate": remaining_tokens_estimate,
                "completion_preview": "".join(completion_parts)[:120],
                "non_stream_fallback": non_stream_fallback
            }
    except Exception as e:
        return {"request_id": idx+1, "status": "error", "error": str(e)}

conc_prompts = _load_prompts(CONC_PROMPT_FILE, NUM_PROMPTS)
if conc_prompts is None:
    conc_prompts = [
        _base_prompts[i % len(_base_prompts)] for i in range(NUM_PROMPTS)
    ]
else:
    print(f"Using concurrent prompts from: {CONC_PROMPT_FILE}")
print(f"Sending {len(conc_prompts)} concurrent requests...")
conc_start = time.perf_counter()

with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROMPTS) as executor:
    futures = [executor.submit(run_one_request, i, p) for i, p in enumerate(conc_prompts)]
    for f in concurrent.futures.as_completed(futures):
        m = f.result()
        conc_measurements.append(m)
        if m["status"] == "success":
            conc_latencies.append(m["latency_ms"])
            conc_ttft_values.append(m["ttft_ms"])
            conc_tpot_values.append(m["tpot_ms"] if m.get("tpot_ms") is not None else m.get("tpot_fallback_ms"))
            conc_total_tokens += m["completion_tokens"]
            print(f"  Request {m['request_id']}: status=success, latency={m['latency_ms']:.2f}ms, tokens={m['completion_tokens']}")
        else:
            print(f"  Request {m['request_id']}: status={m['status']}")

conc_end = time.perf_counter()
conc_total_time = conc_end - conc_start
conc_measurements.sort(key=lambda x: x["request_id"])

if conc_latencies:
    conc_nixl_times = [m["nixl_xfer_ms"] for m in conc_measurements if m["status"] == "success"]
    conc_nixl_stats = {}
    if conc_nixl_times:
        conc_nixl_stats = {
            "avg": round(statistics.mean(conc_nixl_times), 3),
            "min": round(min(conc_nixl_times), 3),
            "max": round(max(conc_nixl_times), 3),
            "p50": round(percentile(conc_nixl_times, 50), 3),
            "p90": round(percentile(conc_nixl_times, 90), 3),
            "p99": round(percentile(conc_nixl_times, 99), 3),
            "stddev": round(statistics.stdev(conc_nixl_times), 3) if len(conc_nixl_times) > 1 else 0
        }

    conc_tpot_non_null = [v for v in conc_tpot_values if v is not None]
    if not conc_tpot_non_null:
        conc_tpot_non_null = []

    conc_lat_stats = {}
    conc_ttft_stats = {}
    conc_tpot_stats = {}
    if conc_latencies:
        conc_lat_stats = {
            "avg": round(statistics.mean(conc_latencies), 2),
            "min": round(min(conc_latencies), 2),
            "max": round(max(conc_latencies), 2),
            "p50": round(percentile(conc_latencies, 50), 2),
            "p90": round(percentile(conc_latencies, 90), 2),
            "p99": round(percentile(conc_latencies, 99), 2),
            "stddev": round(statistics.stdev(conc_latencies), 2) if len(conc_latencies) > 1 else 0
        }
    if conc_ttft_values:
        conc_ttft_stats = {
            "avg": round(statistics.mean(conc_ttft_values), 2),
            "p50": round(percentile(conc_ttft_values, 50), 2),
            "p90": round(percentile(conc_ttft_values, 90), 2),
            "p99": round(percentile(conc_ttft_values, 99), 2)
        }
    if conc_tpot_non_null:
        conc_tpot_stats = {
            "avg": round(statistics.mean(conc_tpot_non_null), 2),
            "p50": round(percentile(conc_tpot_non_null, 50), 2),
            "p90": round(percentile(conc_tpot_non_null, 90), 2),
            "p99": round(percentile(conc_tpot_non_null, 99), 2)
        }

    result["benchmark_concurrent"] = {
        "measurements": conc_measurements,
        "summary": {
            "total_requests": len(conc_prompts),
            "successful_requests": len(conc_latencies),
            "valid_text_requests": sum(1 for m in conc_measurements if m.get("valid_text")),
            "invalid_text_requests": sum(1 for m in conc_measurements if m.get("http_ok") and not m.get("valid_text")),
            "failed_requests": len(conc_prompts) - len(conc_latencies),
            "total_time_sec": round(conc_total_time, 2),
            "throughput_req_per_sec": round(len(conc_latencies) / conc_total_time, 2) if conc_latencies else 0,
            "throughput_tokens_per_sec": round(conc_total_tokens / conc_total_time, 2) if conc_latencies else 0,
            "latency_ms": conc_lat_stats,
            "nixl_xfer_ms": conc_nixl_stats,
            "ttft_ms": conc_ttft_stats,
            "tpot_ms": conc_tpot_stats
        }
    }
    cs = result["benchmark_concurrent"]["summary"]
    print(f"\n--- Concurrent Summary ---")
    print(f"Wall time: {conc_total_time:.2f}s")
    print(f"Throughput: {cs['throughput_req_per_sec']:.2f} req/s, {cs['throughput_tokens_per_sec']:.2f} tokens/s")
    print(f"Latency: Avg={cs['latency_ms']['avg']:.2f}ms, P90={cs['latency_ms']['p90']:.2f}ms, P99={cs['latency_ms']['p99']:.2f}ms")

# =============================================================================
# 3. Accuracy Verification
# =============================================================================
print("\n=== 3. Accuracy Verification (streaming, same as benchmark) ===")
test_cases = [
    {"prompt": "The capital of France is", "expected": ["Paris", "paris"], "max_tokens": 10},
    {"prompt": "2 + 2 =", "expected": ["4", "four"], "max_tokens": 5},
    {"prompt": _base_prompts[0], "expected": ["__NON_EMPTY__"], "max_tokens": 200},
    {"prompt": _base_prompts[1], "expected": ["__NON_EMPTY__"], "max_tokens": 200},
    {"prompt": _base_prompts[2], "expected": ["__NON_EMPTY__"], "max_tokens": 200}
]
accuracy_tests = []
passed = 0

for i, tc in enumerate(test_cases):
    try:
        # Use streaming (same code path as benchmark)
        with STREAM_CLIENT.stream(
            "POST",
            f"{PROXY_URL}/v1/completions",
            json={"model": MODEL_NAME, "prompt": tc["prompt"], "max_tokens": tc["max_tokens"], "temperature": 0, "stream": True},
            timeout=60,
        ) as resp:
            if resp.status_code == 200:
                completion_parts = []
                for data in iter_sse_data_bytes(resp.iter_bytes()):
                    chunk_data = data.decode("utf-8", errors="ignore")
                    if chunk_data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_data)
                        txt = chunk.get("choices", [{}])[0].get("text", "")
                        if txt:
                            completion_parts.append(txt)
                    except (json.JSONDecodeError, KeyError):
                        pass
                completion = "".join(completion_parts).strip()
            else:
                completion = ""
            if "__NON_EMPTY__" in tc["expected"]:
                match = len(completion.strip()) > 0
            else:
                match = any(e.lower() in completion.lower() for e in tc["expected"])
            accuracy_tests.append({
                "test_id": i+1, "prompt": tc["prompt"], "completion": completion,
                "expected": tc["expected"], "passed": match, "status": "success"
            })
            if match:
                passed += 1
                print(f"  Test {i+1}: PASS - '{completion[:40]}'")
            else:
                print(f"  Test {i+1}: FAIL - got '{completion[:40]}'")
    except Exception as e:
        accuracy_tests.append({"test_id": i+1, "passed": False, "status": "error", "error": str(e)})
        print(f"  Test {i+1}: ERROR - {e}")

result["accuracy"] = {
    "test_cases": accuracy_tests,
    "summary": {
        "total_tests": len(test_cases),
        "passed": passed,
        "failed": len(test_cases) - passed,
        "accuracy_percent": round(passed / len(test_cases) * 100, 1)
    }
}
print(f"\nAccuracy: {passed}/{len(test_cases)} ({result['accuracy']['summary']['accuracy_percent']}%)")

# =============================================================================
# Build summary at the TOP of the JSON
# =============================================================================
bench_summary = result.get("benchmark", {}).get("summary", {})
conc_summary = result.get("benchmark_concurrent", {}).get("summary", {})
result["summary"] = {
    "sequential": {
        "latency_ms": bench_summary.get("latency_ms", {}),
        "nixl_xfer_ms": bench_summary.get("nixl_xfer_ms", {}),
        "ttft_ms": bench_summary.get("ttft_ms", {}),
        "tpot_ms": bench_summary.get("tpot_ms", {}),
        "throughput_req_per_sec": bench_summary.get("throughput_req_per_sec", 0),
        "throughput_tokens_per_sec": bench_summary.get("throughput_tokens_per_sec", 0),
    },
    "concurrent": {
        "latency_ms": conc_summary.get("latency_ms", {}),
        "nixl_xfer_ms": conc_summary.get("nixl_xfer_ms", {}),
        "ttft_ms": conc_summary.get("ttft_ms", {}),
        "tpot_ms": conc_summary.get("tpot_ms", {}),
        "throughput_req_per_sec": conc_summary.get("throughput_req_per_sec", 0),
        "throughput_tokens_per_sec": conc_summary.get("throughput_tokens_per_sec", 0),
    },
    "accuracy_percent": result.get("accuracy", {}).get("summary", {}).get("accuracy_percent", 0)
}

# =============================================================================
# Save to ONE file
# =============================================================================
with open(RESULT_FILE, "w") as f:
    json.dump(result, f, indent=2)

print("\n" + "="*60)
print("BENCHMARK COMPLETE")
print("="*60)
print(f"Result saved to: {RESULT_FILE}")
print("\n=== Quick Summary (at top of JSON) ===")
print(json.dumps(result["summary"], indent=2))
PYTHON_ALL

# =============================================================================
# nsys teardown: 벤치마크 완료 후 nsys 서버 종료 → 결과 파싱 → JSON에 병합
# =============================================================================
if [ "$NSYS_ENABLED" = "1" ]; then
    echo ""
    echo "=============================================="
    echo "nsys: Stopping server and parsing results"
    echo "=============================================="

    # Stop nsys/server (패턴을 구체적으로 지정하여 proxy 서버를 건드리지 않음)
    echo "[nsys] Stopping prefill server (nsys will save profile)..."
    kill $NSYS_PID 2>/dev/null
    wait $NSYS_PID 2>/dev/null
    sleep 2
    pkill -f "vllm serve.*--port 8100" 2>/dev/null || true
    sleep 1

    # Parse nsys results
    if [ -f "${NSYS_OUTPUT}.nsys-rep" ]; then
        echo "[nsys] Parsing nsys results..."

        nsys stats "${NSYS_OUTPUT}.nsys-rep" \
            --report cuda_gpu_trace \
            --format csv \
            --output /tmp/nsys_csv_${TIMESTAMP} \
            --force-overwrite=true \
            > /dev/null 2>&1

        python3 << NSYS_PYTHON
import json, csv, os, statistics

result_file = "${RESULT_FILE}"
csv_file = "/tmp/nsys_csv_${TIMESTAMP}_cuda_gpu_trace.csv"

nsys_result = {"status": "no_data"}

if os.path.exists(csv_file):
    dtoh_times = []
    htod_times = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f"  CSV columns: {headers}")
        rows = list(reader)
        print(f"  Total rows in CSV: {len(rows)}")
        # Show sample memcpy-related rows
        memcpy_rows = [r for r in rows if 'emcpy' in r.get('Name', '') or 'emcpy' in str(r)]
        print(f"  Rows containing 'memcpy': {len(memcpy_rows)}")
        if memcpy_rows:
            for r in memcpy_rows[:5]:
                print(f"    Name={r.get('Name','')}, Dur(ns)={r.get('Duration (ns)','')}, Bytes(MB)={r.get('Bytes (MB)','')}")
        else:
            # Show first 3 rows to understand format
            for r in rows[:3]:
                print(f"    Sample: {dict(r)}")

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('Name', '')
            # 컬럼명에 공백이 포함됨: 'Duration (ns)', 'Bytes (MB)'
            dur_ns = float(row.get('Duration (ns)', 0) or 0)
            dur_ms = dur_ns / 1_000_000
            bytes_mb = float(row.get('Bytes (MB)', 0) or 0)
            bytes_val = int(bytes_mb * 1024 * 1024)

            if 'Device-to-Host' in name or 'DtoH' in name:
                dtoh_times.append({"dur_ms": round(dur_ms, 4), "bytes": bytes_val})
            elif 'Host-to-Device' in name or 'HtoD' in name:
                htod_times.append({"dur_ms": round(dur_ms, 4), "bytes": bytes_val})

    def calc_stats(times_list):
        if not times_list:
            return {}
        durations = [t["dur_ms"] for t in times_list]
        sorted_d = sorted(durations)
        p50 = sorted_d[min(len(sorted_d)-1, int(len(sorted_d)*0.5))]
        p90 = sorted_d[min(len(sorted_d)-1, int(len(sorted_d)*0.9))]
        return {
            "count": len(times_list),
            "avg_ms": round(statistics.mean(durations), 4),
            "min_ms": round(min(durations), 4),
            "max_ms": round(max(durations), 4),
            "p50_ms": round(p50, 4),
            "p90_ms": round(p90, 4),
            "stddev_ms": round(statistics.stdev(durations), 4) if len(durations) > 1 else 0,
            "total_bytes": sum(t["bytes"] for t in times_list),
        }

    nsys_result = {
        "status": "success",
        "gpu_to_cpu_DtoH": calc_stats(dtoh_times),
        "cpu_to_gpu_HtoD": calc_stats(htod_times),
    }

    if dtoh_times:
        s = nsys_result["gpu_to_cpu_DtoH"]
        print(f"  GPU->CPU (DtoH): {s['count']} calls, avg={s['avg_ms']:.4f}ms, p50={s['p50_ms']:.4f}ms")
    if htod_times:
        s = nsys_result["cpu_to_gpu_HtoD"]
        print(f"  CPU->GPU (HtoD): {s['count']} calls, avg={s['avg_ms']:.4f}ms, p50={s['p50_ms']:.4f}ms")
    if not dtoh_times and not htod_times:
        print("  No memcpy operations captured")
else:
    print("  No CSV data found")

# Merge into benchmark result JSON
with open(result_file, 'r') as f:
    data = json.load(f)
data["nsys_memcpy_profiling"] = nsys_result
if nsys_result.get("gpu_to_cpu_DtoH"):
    data["summary"]["nsys_gpu_to_cpu_avg_ms"] = nsys_result["gpu_to_cpu_DtoH"].get("avg_ms", 0)
if nsys_result.get("cpu_to_gpu_HtoD"):
    data["summary"]["nsys_cpu_to_gpu_avg_ms"] = nsys_result["cpu_to_gpu_HtoD"].get("avg_ms", 0)
with open(result_file, 'w') as f:
    json.dump(data, f, indent=2)
print(f"  nsys results merged into: {result_file}")
NSYS_PYTHON

        rm -f /tmp/nsys_csv_${TIMESTAMP}*.csv
        rm -f "${NSYS_OUTPUT}.nsys-rep" "${NSYS_OUTPUT}.sqlite"
    else
        echo "  nsys profile not generated. Check /tmp/nsys_server_${TIMESTAMP}.log"
    fi

    echo "  NOTE: Prefill server is now stopped. Restart manually if needed."
fi

echo ""
echo "=============================================="
echo "Result saved to: $RESULT_FILE"
echo "=============================================="
