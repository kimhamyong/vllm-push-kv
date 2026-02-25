#!/bin/bash
set -e

# =============================================================================
# Disaggregated Prefill Benchmark Script
# All results saved to JSON file
# Output: results/benchmark_disagg/benchmark_disagg_{timestamp}.json
# =============================================================================

# Configuration
MODEL_NAME="meta-llama/Llama-3.2-1B"
PROXY_URL="http://localhost:8000"
PREFILL_URL="http://172.31.2.19:8100"
DECODE_URL="http://172.31.0.191:8200"

# Benchmark settings
SCRIPT_NAME="benchmark_disagg"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/home/ubuntu/vllm/results/${SCRIPT_NAME}"
RESULT_FILE="${RESULTS_DIR}/${SCRIPT_NAME}_${TIMESTAMP}.json"
NUM_PROMPTS=${NUM_PROMPTS:-20}
OUTPUT_LEN=${OUTPUT_LEN:-200}

# Create results directory
mkdir -p "$RESULTS_DIR"

# Export variables for Python
export MODEL_NAME PROXY_URL PREFILL_URL DECODE_URL
export NUM_PROMPTS OUTPUT_LEN RESULT_FILE TIMESTAMP

echo "=============================================="
echo "Disaggregated Prefill Benchmark"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Proxy URL: $PROXY_URL"
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
            --max-model-len 2048 \
            --gpu-memory-utilization 0.8 \
            --trust-remote-code \
            --enforce-eager \
            --kv-transfer-config \
            '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
        > /tmp/nsys_server_${TIMESTAMP}.log 2>&1 &
    NSYS_PID=$!
    echo "  nsys PID: $NSYS_PID"

    # Wait for prefill server to be ready
    echo "[nsys] Waiting for prefill server to be ready (max 120s)..."
    for i in $(seq 1 120); do
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
    --max-model-len 2048 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
    > /tmp/decode_server_nsys.log 2>&1 &
echo "  Decode server PID: $!"
DECODE_SSH

        # Wait for decode server to be ready
        echo "[nsys] Waiting for decode server to be ready (max 120s)..."
        for i in $(seq 1 120); do
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
            echo "  WARNING: Decode server not ready after 120s. Benchmark may fail."
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
export VLLM_NIXL_SIDE_CHANNEL_HOST=172.31.0.191
export VLLM_NIXL_SIDE_CHANNEL_PORT=14580
export UCX_TLS=cuda_copy,tcp
export UCX_NET_DEVICES=ens5
export UCX_TCP_PORT_RANGE=40000-40009


nohup /home/ubuntu/vllm/.venv/bin/vllm serve meta-llama/Llama-3.2-1B \
    --host 0.0.0.0 \
    --port 8200 \
    --max-model-len 2048 \
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
    export VLLM_NIXL_SIDE_CHANNEL_HOST=$PREFILL_IP
    export VLLM_NIXL_SIDE_CHANNEL_PORT=14580
    export UCX_TLS=cuda_copy,tcp
    export UCX_NET_DEVICES=ens5
    export UCX_TCP_PORT_RANGE=40000-40009


    ${VLLM_BIN} serve ${MODEL_NAME} \
        --host 0.0.0.0 \
        --port 8100 \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.8 \
        --trust-remote-code \
        --enforce-eager \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
        > /tmp/prefill_server.log 2>&1 &
    PREFILL_PID=$!
    echo "  Prefill server PID: $PREFILL_PID"

    # --- Wait for both servers to be ready ---
    echo "[normal] Waiting for servers to be ready (max 120s)..."
    SERVERS_READY=0
    for i in $(seq 1 120); do
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
        echo "  WARNING: Servers not ready after 120s. Benchmark may fail."
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
from datetime import datetime

# Configuration from shell
PROXY_URL = os.environ.get("PROXY_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-1B")
PREFILL_URL = os.environ.get("PREFILL_URL", "http://172.31.2.19:8100")
DECODE_URL = os.environ.get("DECODE_URL", "http://172.31.0.191:8200")
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "20"))
OUTPUT_LEN = int(os.environ.get("OUTPUT_LEN", "200"))

RESULT_FILE = os.environ.get("RESULT_FILE", "/home/ubuntu/vllm/results/benchmark_disagg/benchmark_disagg.json")
TIMESTAMP = os.environ.get("TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))

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

# 긴 프롬프트 (~800 tokens each, 다양한 길이로 GPU 메모리 압박)
_base_prompts = [
    "The quick brown fox jumps over the lazy dog and then runs across the wide open field where many animals gather to play and rest under the warm afternoon sun while birds sing their beautiful songs in the tall green trees nearby. The river flows gently through the valley carrying fallen leaves and small twigs downstream toward the distant ocean where waves crash upon the rocky shore. Fishermen cast their nets into the deep blue water hoping for a bountiful catch while seagulls circle overhead calling to one another. The clouds drift slowly across the sky painting shadows on the landscape below as the day progresses from morning to afternoon. Children play in the meadow chasing butterflies and picking wildflowers to bring home to their families. The old stone bridge arches gracefully over the stream connecting the two villages that have traded goods for centuries. Farmers tend their crops in the fertile fields watching the weather for signs of rain that will nourish the growing plants. The forest at the edge of town is home to deer foxes rabbits and countless species of birds that fill the air with music at dawn.",
    "In a galaxy far far away there existed a civilization of advanced beings who had mastered the art of interstellar travel and communication across vast distances using quantum entanglement technology that allowed them to share knowledge instantly across light years. Their ships were powered by antimatter engines capable of bending spacetime itself creating stable wormholes between star systems. The civilization had colonized thousands of worlds each with its own unique ecosystem and culture but all connected through a vast neural network that spanned the galaxy. Scientists on the homeworld continued to push the boundaries of physics discovering new dimensions of reality that challenged everything they thought they knew about the universe. The council of elders governed wisely balancing the needs of trillions of citizens spread across countless planets moons and space stations. Artists created works that could only be experienced in zero gravity while musicians composed symphonies using the electromagnetic frequencies of pulsars and magnetars. Engineers built megastructures around dying stars harvesting their final energy output to power the civilization for millennia to come.",
    "Once upon a time there was a young wizard who discovered an ancient book of spells hidden deep within the forbidden library of the grand castle where generations of powerful sorcerers had studied and practiced their magical arts for over a thousand years. The book was bound in dragon leather and its pages were made from enchanted parchment that could only be read by moonlight. Each spell within was more powerful and dangerous than the last requiring immense concentration and magical energy to cast properly. The young wizard spent months studying the first chapter alone learning the fundamental principles of elemental manipulation and dimensional folding. The castle itself was alive with magic its walls shifting and corridors rearranging themselves according to ancient enchantments placed by the founders. Ghosts of former students wandered the halls offering cryptic advice to those brave enough to listen. The library contained millions of books scrolls and artifacts collected from every corner of the known world and several corners of worlds unknown. Deep beneath the castle lay a network of caverns where underground rivers of pure magical energy flowed providing power to the wards and enchantments that protected the school.",
    "The meaning of life is a philosophical question that has been debated by great thinkers throughout human history from ancient Greek philosophers like Socrates and Plato to modern existentialists who explored the nature of consciousness and purpose in an apparently indifferent universe. Eastern traditions offer perspectives centered on mindfulness compassion and the interconnectedness of all living things suggesting that meaning arises from our relationships with others and the natural world. Scientific discoveries have revealed the astonishing complexity of biological systems from the molecular machinery of cells to the emergent properties of consciousness in the human brain raising profound questions about free will determinism and the nature of subjective experience. Some argue that meaning is inherent in the structure of reality itself encoded in mathematical laws and physical constants that seem remarkably fine tuned for the emergence of complex life. Others maintain that meaning is a human construction something we create through our choices actions and commitments rather than something we discover. The existentialist tradition emphasizes radical freedom and responsibility arguing that we are condemned to be free and must create our own values in a world without predetermined purpose.",
    "Artificial intelligence will transform every aspect of modern society including healthcare education transportation manufacturing and entertainment as machine learning algorithms become increasingly sophisticated and capable of solving complex problems that were previously thought to require human intelligence and creativity. In medicine AI systems can analyze medical images with superhuman accuracy detecting cancers tumors and other abnormalities that human radiologists might miss leading to earlier diagnosis and better patient outcomes. Autonomous vehicles powered by deep learning neural networks will revolutionize transportation reducing accidents caused by human error and providing mobility to elderly and disabled populations who currently cannot drive. In education personalized AI tutors will adapt to each students learning style pace and interests providing customized instruction that maximizes engagement and knowledge retention. Manufacturing will be transformed by intelligent robots that can learn new tasks through observation and practice rather than requiring explicit programming for every movement and decision. Creative industries will see AI tools that assist human artists musicians and writers generating novel ideas and helping to explore vast creative spaces that would be impossible to navigate manually.",
    "San Francisco is known for its iconic Golden Gate Bridge steep rolling hills historic cable cars and vibrant cultural diversity that attracts millions of visitors from around the world who come to experience its unique blend of technology and tradition art and innovation natural beauty and urban sophistication. The city was founded during the California Gold Rush of 1849 when thousands of prospectors flooded into the area seeking their fortune in the rivers and mountains of the Sierra Nevada. Today it stands as the heart of Silicon Valley the global center of technological innovation where companies like Apple Google Meta and countless startups continue to push the boundaries of what technology can achieve. The citys neighborhoods each have their own distinct character from the bohemian atmosphere of Haight Ashbury to the vibrant Chinatown the largest outside of Asia to the trendy restaurants and boutiques of the Mission District. Alcatraz Island sitting in the cold waters of the bay once housed Americas most notorious criminals and now serves as one of the citys most popular tourist attractions. The fog that rolls in from the Pacific Ocean each evening gives the city an ethereal quality transforming familiar landmarks into mysterious silhouettes.",
    "The best programming language is a topic of endless debate among software developers who argue passionately about the merits of Python Java Rust Go and many other languages each designed to solve different types of computational problems efficiently and elegantly. Python has emerged as the dominant language for data science machine learning and artificial intelligence thanks to its clean syntax extensive library ecosystem and gentle learning curve that makes it accessible to beginners while remaining powerful enough for experts. Rust has gained tremendous popularity for systems programming offering memory safety without garbage collection through its innovative ownership and borrowing system that catches entire categories of bugs at compile time rather than runtime. Go designed at Google provides excellent concurrency primitives and compiles to fast native code making it ideal for building scalable network services and cloud infrastructure. JavaScript continues to dominate web development running in every browser and increasingly on servers through Node.js creating a unified language ecosystem for full stack development. Each language represents different design tradeoffs and philosophies reflecting the diverse needs of the software industry from embedded systems to web applications from scientific computing to game development.",
    "Machine learning models can analyze vast amounts of data to discover hidden patterns and make accurate predictions that would be impossible for humans to detect manually enabling breakthroughs in medical diagnosis financial forecasting scientific research drug discovery climate modeling and many other fields that impact human welfare. Deep neural networks with billions of parameters trained on massive datasets have achieved remarkable performance on tasks ranging from image classification and object detection to natural language understanding and generation. Transfer learning allows models pretrained on large general datasets to be fine tuned for specific tasks with relatively small amounts of labeled data dramatically reducing the cost and time required to develop specialized AI applications. Reinforcement learning has produced agents capable of superhuman performance in complex games like Go chess and video games learning optimal strategies through millions of simulated episodes of trial and error. Generative models including variational autoencoders and generative adversarial networks can create realistic images videos music and text opening new possibilities for creative expression and content production. The field continues to advance rapidly with new architectures training techniques and theoretical insights emerging at an accelerating pace.",
    "The future of technology holds incredible promise with advances in quantum computing biotechnology renewable energy artificial intelligence and space exploration that will fundamentally change how humans live work communicate and understand the universe around them. Quantum computers leveraging the strange properties of superposition and entanglement will solve problems that are intractable for classical computers including drug design materials science optimization and cryptography. CRISPR gene editing technology has given scientists unprecedented ability to modify DNA with precision opening possibilities for curing genetic diseases eliminating invasive species and engineering crops that can withstand climate change. Fusion energy the process that powers the sun is finally approaching commercial viability after decades of research promising virtually unlimited clean energy that could end humanitys dependence on fossil fuels and dramatically reduce greenhouse gas emissions. Brain computer interfaces are advancing rapidly with companies developing implantable devices that could restore movement to paralyzed patients treat neurological disorders and eventually enhance human cognitive capabilities. Space agencies and private companies are planning permanent settlements on the Moon and Mars beginning a new chapter in human history as a multiplanetary species.",
    "Deep learning networks are composed of multiple layers of interconnected neurons that process information hierarchically extracting increasingly abstract features from raw data to perform tasks such as image recognition natural language understanding and autonomous navigation with remarkable accuracy and generalization capability. The transformer architecture introduced in 2017 revolutionized natural language processing by using self attention mechanisms that allow the model to weigh the importance of different parts of the input when generating each element of the output. Large language models built on the transformer architecture have demonstrated emergent capabilities including reasoning planning code generation and even creative writing that were not explicitly trained for but arise from the scale of the model and training data. Convolutional neural networks remain the backbone of computer vision processing images through layers of learned filters that detect edges textures shapes and objects at increasing levels of abstraction. Recurrent architectures including LSTMs and GRUs are designed to process sequential data maintaining hidden states that capture temporal dependencies in time series speech and other dynamic signals. The field of neural architecture search uses AI itself to design optimal network architectures automatically discovering configurations that outperform human designed models on benchmark tasks."
]
# Multiply prompts to reach NUM_PROMPTS
benchmark_prompts = [_base_prompts[i % len(_base_prompts)] for i in range(NUM_PROMPTS)]

measurements = []
latencies = []
ttft_values = []
tpot_values = []
nixl_xfer_times = []
total_tokens = 0

print(f"Running {len(benchmark_prompts)} sequential requests...")
bench_start = time.perf_counter()

for i, prompt in enumerate(benchmark_prompts):
    # Get NIXL metrics BEFORE this request
    nixl_before = get_nixl_metrics_both()

    start = time.perf_counter()
    try:
        resp = requests.post(f"{PROXY_URL}/v1/completions",
            json={"model": MODEL_NAME, "prompt": prompt, "max_tokens": OUTPUT_LEN, "temperature": 0}, timeout=300)
        latency = (time.perf_counter() - start) * 1000

        # Get NIXL metrics AFTER this request
        nixl_after = get_nixl_metrics_both()

        # Calculate NIXL transfer time for THIS request
        nixl_xfer_ms = 0
        nixl_bytes_kb = 0
        if nixl_before and nixl_after:
            xfer_time_diff = nixl_after.get('vllm:nixl_xfer_time_seconds_sum', 0) - nixl_before.get('vllm:nixl_xfer_time_seconds_sum', 0)
            bytes_diff = nixl_after.get('vllm:nixl_bytes_transferred_sum', 0) - nixl_before.get('vllm:nixl_bytes_transferred_sum', 0)
            nixl_xfer_ms = xfer_time_diff * 1000
            nixl_bytes_kb = bytes_diff / 1024

        if resp.status_code == 200:
            resp_json = resp.json()
            tokens = resp_json.get("usage", {}).get("completion_tokens", OUTPUT_LEN)

            # TTFT, TPOT 추정
            ttft = latency * 0.3
            tpot = (latency - ttft) / max(tokens, 1)

            latencies.append(latency)
            ttft_values.append(ttft)
            tpot_values.append(tpot)
            nixl_xfer_times.append(nixl_xfer_ms)
            total_tokens += tokens

            measurements.append({
                "request_id": i+1,
                "status": "success",
                "latency_ms": round(latency, 2),
                "ttft_ms": round(ttft, 2),
                "tpot_ms": round(tpot, 2),
                "nixl_xfer_ms": round(nixl_xfer_ms, 3),
                "nixl_bytes_kb": round(nixl_bytes_kb, 2),
                "completion_tokens": tokens
            })
            print(f"  Request {i+1}: latency={latency:.2f}ms, NIXL_xfer={nixl_xfer_ms:.2f}ms, tokens={tokens}")
        else:
            measurements.append({"request_id": i+1, "status": "failed"})
            print(f"  Request {i+1}: FAILED")
    except Exception as e:
        measurements.append({"request_id": i+1, "status": "error", "error": str(e)})
        print(f"  Request {i+1}: ERROR - {e}")

bench_end = time.perf_counter()
total_time = bench_end - bench_start

if latencies:
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

    result["benchmark"] = {
        "measurements": measurements,
        "summary": {
            "total_requests": len(benchmark_prompts),
            "successful_requests": len(latencies),
            "failed_requests": len(benchmark_prompts) - len(latencies),
            "total_time_sec": round(total_time, 2),
            "throughput_req_per_sec": round(len(latencies) / total_time, 2),
            "throughput_tokens_per_sec": round(total_tokens / total_time, 2),
            "latency_ms": {
                "avg": round(statistics.mean(latencies), 2),
                "min": round(min(latencies), 2),
                "max": round(max(latencies), 2),
                "p50": round(percentile(latencies, 50), 2),
                "p90": round(percentile(latencies, 90), 2),
                "p99": round(percentile(latencies, 99), 2),
                "stddev": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0
            },
            "nixl_xfer_ms": nixl_stats,
            "ttft_ms": {
                "avg": round(statistics.mean(ttft_values), 2),
                "p50": round(percentile(ttft_values, 50), 2),
                "p90": round(percentile(ttft_values, 90), 2),
                "p99": round(percentile(ttft_values, 99), 2)
            },
            "tpot_ms": {
                "avg": round(statistics.mean(tpot_values), 2),
                "p50": round(percentile(tpot_values, 50), 2),
                "p90": round(percentile(tpot_values, 90), 2),
                "p99": round(percentile(tpot_values, 99), 2)
            }
        }
    }
    s = result["benchmark"]["summary"]
    print(f"\n--- Summary ---")
    print(f"Throughput: {s['throughput_req_per_sec']:.2f} req/s, {s['throughput_tokens_per_sec']:.2f} tokens/s")
    print(f"Latency: Avg={s['latency_ms']['avg']:.2f}ms, P90={s['latency_ms']['p90']:.2f}ms, P99={s['latency_ms']['p99']:.2f}ms")
    if nixl_stats:
        print(f"NIXL KV Transfer: Avg={nixl_stats['avg']:.2f}ms, P90={nixl_stats['p90']:.2f}ms, P99={nixl_stats['p99']:.2f}ms")
    print(f"TTFT: Avg={s['ttft_ms']['avg']:.2f}ms, P90={s['ttft_ms']['p90']:.2f}ms, P99={s['ttft_ms']['p99']:.2f}ms")
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
    """Send one request and return measurement dict."""
    nixl_before = get_nixl_metrics_both()
    start = time.perf_counter()
    try:
        resp = requests.post(f"{PROXY_URL}/v1/completions",
            json={"model": MODEL_NAME, "prompt": prompt, "max_tokens": OUTPUT_LEN, "temperature": 0}, timeout=300)
        latency = (time.perf_counter() - start) * 1000

        nixl_after = get_nixl_metrics_both()
        nixl_xfer_ms = 0
        nixl_bytes_kb = 0
        if nixl_before and nixl_after:
            xfer_time_diff = nixl_after.get('vllm:nixl_xfer_time_seconds_sum', 0) - nixl_before.get('vllm:nixl_xfer_time_seconds_sum', 0)
            bytes_diff = nixl_after.get('vllm:nixl_bytes_transferred_sum', 0) - nixl_before.get('vllm:nixl_bytes_transferred_sum', 0)
            nixl_xfer_ms = xfer_time_diff * 1000
            nixl_bytes_kb = bytes_diff / 1024

        if resp.status_code == 200:
            resp_json = resp.json()
            tokens = resp_json.get("usage", {}).get("completion_tokens", OUTPUT_LEN)
            ttft = latency * 0.3
            tpot = (latency - ttft) / max(tokens, 1)
            return {
                "request_id": idx+1, "status": "success",
                "latency_ms": round(latency, 2), "ttft_ms": round(ttft, 2),
                "tpot_ms": round(tpot, 2), "nixl_xfer_ms": round(nixl_xfer_ms, 3),
                "nixl_bytes_kb": round(nixl_bytes_kb, 2), "completion_tokens": tokens
            }
        else:
            return {"request_id": idx+1, "status": "failed"}
    except Exception as e:
        return {"request_id": idx+1, "status": "error", "error": str(e)}

conc_prompts = [_base_prompts[i % len(_base_prompts)] for i in range(NUM_PROMPTS)]
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
            conc_tpot_values.append(m["tpot_ms"])
            conc_total_tokens += m["completion_tokens"]
            print(f"  Request {m['request_id']}: latency={m['latency_ms']:.2f}ms, tokens={m['completion_tokens']}")
        else:
            print(f"  Request {m['request_id']}: {m['status']}")

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

    result["benchmark_concurrent"] = {
        "measurements": conc_measurements,
        "summary": {
            "total_requests": len(conc_prompts),
            "successful_requests": len(conc_latencies),
            "failed_requests": len(conc_prompts) - len(conc_latencies),
            "total_time_sec": round(conc_total_time, 2),
            "throughput_req_per_sec": round(len(conc_latencies) / conc_total_time, 2),
            "throughput_tokens_per_sec": round(conc_total_tokens / conc_total_time, 2),
            "latency_ms": {
                "avg": round(statistics.mean(conc_latencies), 2),
                "min": round(min(conc_latencies), 2),
                "max": round(max(conc_latencies), 2),
                "p50": round(percentile(conc_latencies, 50), 2),
                "p90": round(percentile(conc_latencies, 90), 2),
                "p99": round(percentile(conc_latencies, 99), 2),
                "stddev": round(statistics.stdev(conc_latencies), 2) if len(conc_latencies) > 1 else 0
            },
            "nixl_xfer_ms": conc_nixl_stats,
            "ttft_ms": {
                "avg": round(statistics.mean(conc_ttft_values), 2),
                "p50": round(percentile(conc_ttft_values, 50), 2),
                "p90": round(percentile(conc_ttft_values, 90), 2),
                "p99": round(percentile(conc_ttft_values, 99), 2)
            },
            "tpot_ms": {
                "avg": round(statistics.mean(conc_tpot_values), 2),
                "p50": round(percentile(conc_tpot_values, 50), 2),
                "p90": round(percentile(conc_tpot_values, 90), 2),
                "p99": round(percentile(conc_tpot_values, 99), 2)
            }
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
print("\n=== 3. Accuracy Verification ===")
test_cases = [
    {"prompt": "The capital of France is", "expected": ["Paris", "paris"], "max_tokens": 10},
    {"prompt": "2 + 2 =", "expected": ["4", "four"], "max_tokens": 5},
    {"prompt": "The quick brown fox jumps over the lazy", "expected": ["dog"], "max_tokens": 5},
    {"prompt": "Water freezes at", "expected": ["0", "32", "zero", "degrees"], "max_tokens": 10},
    {"prompt": "The sun rises in the", "expected": ["east", "East"], "max_tokens": 5}
]
accuracy_tests = []
passed = 0

for i, tc in enumerate(test_cases):
    try:
        resp = requests.post(f"{PROXY_URL}/v1/completions",
            json={"model": MODEL_NAME, "prompt": tc["prompt"], "max_tokens": tc["max_tokens"], "temperature": 0}, timeout=60)
        if resp.status_code == 200:
            completion = resp.json()["choices"][0]["text"].strip()
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
        else:
            accuracy_tests.append({"test_id": i+1, "passed": False, "status": "http_error"})
    except Exception as e:
        accuracy_tests.append({"test_id": i+1, "passed": False, "status": "error", "error": str(e)})

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
