#!/usr/bin/env bash
# =============================================================================
# bench.sh — SwiftLM model performance benchmark
#
# Per-model measurements:
#   text / vlm  — load time, memory (idle + peak), TTFT, prefill tok/s,
#                 decode tok/s
#   ssd         — load time, memory idle, TTFT only (generation too slow)
#   embed       — load time, memory idle, single-request latency (p50/p95/p99),
#                 concurrent throughput at 1 / 4 / 8 / 16 / 32 requests
#
# Memory comes from the server's own /health endpoint (MLX-tracked unified
# memory), so it reflects GPU allocation, not RSS.
#
# TTFT is measured by streaming a request and timing the first SSE data chunk.
# Prefill tok/s: non-streaming long-prompt request with max_tokens=1.
# Decode tok/s:  completion_tokens / (total_time − TTFT) for a 300-token run.
#
# Output:
#   bench_results.md   — human-readable markdown tables
#   bench_results.json — machine-readable array of result objects
#
# Usage:
#   ./bench.sh                       # .build/release/SwiftLM, ~/models
#   ./bench.sh /path/to/binary       # override binary
#   MODELS_DIR=/path ./bench.sh      # override models directory
#
# Requires: curl, jq, python3
# =============================================================================
set -uo pipefail

BINARY="${1:-.build/release/SwiftLM}"
FILTER="${2:-}"   # optional: type (ssd|text|vlm|embed) or model name — run only matching models
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
BENCH_PORT=18002
BASE_URL="http://127.0.0.1:${BENCH_PORT}"
# When filtering, write to a separate file so the full results aren't overwritten
if [[ -n "$FILTER" ]]; then
    SAFE_FILTER="${FILTER//\//_}"
    OUT_MD="bench_results_${SAFE_FILTER}.md"
    OUT_JSON="bench_results_${SAFE_FILTER}.json"
else
    OUT_MD="bench_results.md"
    OUT_JSON="bench_results.json"
fi

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'; RED='\033[0;31m'

# ---------------------------------------------------------------------------
# Model table  (name | type | extra_flags | load_timeout_s)
# type: text | vlm | ssd | embed
# ---------------------------------------------------------------------------

# n_bench_runs (6th field) controls how many prefill+decode runs are averaged:
#   3 = small models  (<= ~10B)  — fast enough for 3 runs
#   2 = medium models (11–40B)   — 2 runs balances accuracy vs time
#   1 = large / SSD models       — single run; SSD decode is too slow for repeats
MODELS=(
    "Qwen3.5-2B-4bit|text||60|3"
    "Qwen3.5-4B-MLX-4bit|text||60|3"
    "Qwen3.5-9B-MLX-4bit|text||90|3"
    "Qwen3.6-35B-A3B|text||120|2"
    "Qwen3.6-35B-A3B-VLM-4bit|vlm|--vision|180|2"

    "Qwen3-VL-8B-Instruct-4bit|vlm|--vision|120|3"
    "Qwen3-Coder-Next-4bit|text||120|2"
    "DeepSeek-R1-Distill-Qwen-32B-4bit|text||150|2"
    "Qwen3.5-397B-A17B-4bit|ssd|--stream-experts|240|1"
    "Qwen3-Coder-480B-A35B-Instruct-4bit|ssd|--stream-experts|240|1"
    "olmOCR-2-7B-1025-MLX-6bit|vlm|--vision|90|2"
    "jina-embeddings-v5-text-small-retrieval-mlx|embed|--embed|60"
    "jina-embeddings-v5-text-nano-retrieval-mlx|embed|--embed|60"
    "Qwen3-Embedding-8B|embed|--embed|90"
)

# ---------------------------------------------------------------------------
# Fixed prompts
# ---------------------------------------------------------------------------

# Short prompt — minimal prefill, measures decode/TTFT
SHORT_PROMPT="Reply with only the word: PONG"

# Long prompt — ~500 tokens, used for prefill throughput measurement
LONG_PROMPT="Apple Silicon's unified memory architecture fundamentally changes how machine learning inference performs on Apple devices. Unlike traditional systems where CPU and GPU have separate memory pools requiring expensive data transfers, Apple Silicon allows the GPU, CPU, and Neural Engine to share the same high-bandwidth memory pool, eliminating the PCIe bottleneck that limits discrete GPU performance. The MLX framework exploits this architecture through lazy evaluation of computation graphs, automatic differentiation, and seamless CPU/GPU switching. Large language model inference is dominated by two phases: prefill, where the input prompt is processed and the key-value cache is built; and decode, where tokens are generated one at a time. Prefill throughput scales with model size and compute, while decode throughput is primarily limited by memory bandwidth — the cost of reading all model weights for each generated token. For quantised models at 4-bit precision, each parameter occupies half a byte, so a 7-billion-parameter model fits within 3.5 GB while a 27-billion-parameter model needs approximately 13 GB of unified memory. The Matryoshka representation learning technique allows embedding models to produce truncated embeddings at any sub-dimension while maintaining quality, enabling runtime tradeoffs between retrieval accuracy and storage cost. Dynamic batching in embedding servers accumulates concurrent requests and dispatches them as a single GPU forward pass, dramatically improving throughput under concurrent load. SSD streaming enables inference on models that exceed available unified memory by streaming expert weights from NVMe storage during the forward pass, trading memory capacity for storage bandwidth at the cost of significantly lower token throughput."

# Decode benchmark prompt — short (fast prefill), generates 300 tokens
DECODE_PROMPT="Explain in detail how transformer self-attention works, including the query, key, and value projections, scaled dot-product attention, and multi-head attention."

# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------
SERVER_PID=""
SERVER_LOG=""
PM_PID=""; PM_LOG=""
JSON_DIR=$(mktemp -d /tmp/swiftlm-bench-XXXXXX)

cleanup() {
    [[ -n "$SERVER_PID" ]] && { kill -9 "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; }
    [[ -n "$SERVER_LOG" ]] && rm -f "$SERVER_LOG"
    pm_stop 2>/dev/null || true
    rm -rf "$JSON_DIR"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
banner() { echo -e "\n${CYAN}${BOLD}━━━  $*  ━━━${NC}"; }
step()   { echo -e "  ${YELLOW}▸${NC} $*"; }
metric() { echo -e "  ${GREEN}◆${NC} $*"; }
warn()   { echo -e "  ${YELLOW}⚠${NC}  $*"; }
fail()   { echo -e "  ${RED}✗${NC}  $*"; }

# GPU/ANE utilisation via sudo powermetrics (requires sudoers entry — see darwin-configuration.nix)
pm_start() {
    PM_LOG=$(mktemp /tmp/bench-pm-XXXXXX)
    sudo /usr/bin/powermetrics --samplers cpu_power,gpu_power --sample-rate 200 \
        > "$PM_LOG" 2>/dev/null &
    PM_PID=$!
}

pm_stop() {
    [[ -n "$PM_PID" ]] || return
    sudo kill -TERM "$PM_PID" 2>/dev/null || true
    sleep 0.5
    wait "$PM_PID" 2>/dev/null || true
    PM_PID=""
}

pm_stats() {
    [[ -f "${PM_LOG:-}" ]] || { echo "- -"; return; }
    python3 - "$PM_LOG" <<'PYEOF'
import re, sys
try:
    txt = open(sys.argv[1]).read()
except Exception:
    print('- -'); sys.exit()
gpu_vals = [float(m) for m in re.findall(r'GPU HW active residency:\s+([\d.]+)%', txt)]
gpu = f'{sum(gpu_vals)/len(gpu_vals):.0f}' if gpu_vals else '-'
ane_vals = [float(m) for m in re.findall(r'ANE Power:\s+([\d.]+)\s*mW', txt)]
ane = f'{sum(ane_vals)/len(ane_vals):.0f}' if ane_vals else '-'
print(gpu, ane)
PYEOF
    rm -f "$PM_LOG"; PM_LOG=""
}

start_server() {
    local path="$1" type="$2" flags="$3"
    SERVER_LOG=$(mktemp /tmp/swiftlm-bench-log-XXXXXX)
    if [[ "$type" == "ssd" ]]; then
        SWIFTLM_TOP_K=6 "$BINARY" --model "$path" --port "$BENCH_PORT" $flags >"$SERVER_LOG" 2>&1 &
    elif [[ -n "$flags" ]]; then
        "$BINARY" --model "$path" --port "$BENCH_PORT" $flags >"$SERVER_LOG" 2>&1 &
    else
        "$BINARY" --model "$path" --port "$BENCH_PORT" >"$SERVER_LOG" 2>&1 &
    fi
    SERVER_PID=$!
}

kill_server() {
    [[ -n "$SERVER_PID" ]] && { kill -9 "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; }
    [[ -n "$SERVER_LOG" ]] && { rm -f "$SERVER_LOG"; SERVER_LOG=""; }
}

wait_for_ready() {
    local timeout="$1" elapsed=0
    while (( elapsed < timeout )); do
        curl -sf "$BASE_URL/health" >/dev/null 2>&1 && { echo "$elapsed"; return 0; }
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "died"; return 1; }
        sleep 1; (( elapsed++ ))
    done
    echo "timeout"; return 1
}

health_field() {
    # health_field <jq-path>  → value or "-"
    curl -sf "$BASE_URL/health" 2>/dev/null | jq -r "${1} // \"-\"" 2>/dev/null || echo "-"
}

# ---------------------------------------------------------------------------
# Benchmark primitives — all timing via python3 for sub-ms accuracy
# ---------------------------------------------------------------------------

# TTFT: stream a request, return ms to first token. Median of n_runs.
bench_ttft() {
    local prompt="$1" max_tok="${2:-50}" n="${3:-3}"
    python3 - "$BASE_URL" "$prompt" "$max_tok" "$n" <<'PYEOF'
import subprocess, time, json, sys, statistics

base_url = sys.argv[1]
prompt   = sys.argv[2]
max_tok  = int(sys.argv[3])
n_runs   = int(sys.argv[4])
url      = f"{base_url}/v1/chat/completions"
payload  = json.dumps({
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tok,
    "stream": True
}).encode()

timings = []
for _ in range(n_runs):
    cmd = ["curl", "-sf", "--no-buffer", "-N",
           "-X", "POST", url,
           "-H", "Content-Type: application/json",
           "-d", payload]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    found = False
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").strip()
        if line.startswith("data:") and "[DONE]" not in line and len(line) > 6:
            timings.append(int((time.perf_counter() - t0) * 1000))
            found = True
            break
    proc.kill(); proc.wait()
    if not found:
        timings.append(-1)

valid = [t for t in timings if t > 0]
print(int(statistics.median(valid)) if valid else -1)
PYEOF
}

# Prefill: non-streaming long prompt, max_tokens=1. Runs n_runs times, returns median.
# Output: "median_elapsed_ms prompt_tokens"
bench_prefill() {
    local prompt="$1" n_runs="${2:-1}"
    python3 - "$BASE_URL" "$prompt" "$n_runs" <<'PYEOF'
import urllib.request, json, time, sys, statistics

url    = sys.argv[1] + "/v1/chat/completions"
prompt = sys.argv[2]
n_runs = int(sys.argv[3])
body   = json.dumps({
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 1,
    "stream": False
}).encode()

timings = []; toks = 0
for _ in range(n_runs):
    req = urllib.request.Request(url, data=body,
          headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            data = json.loads(r.read())
            toks = data.get("usage", {}).get("prompt_tokens", 0)
            timings.append(elapsed_ms)
    except Exception:
        pass

if not timings:
    print(-1, 0)
else:
    print(int(statistics.median(timings)), toks)
PYEOF
}

# Decode: non-streaming, n_runs repetitions, returns median elapsed and last ctoks.
# Returns "median_elapsed_ms completion_tokens".
bench_decode_run() {
    local prompt="$1" max_tok="${2:-300}" n_runs="${3:-1}"
    python3 - "$BASE_URL" "$prompt" "$max_tok" "$n_runs" <<'PYEOF'
import urllib.request, json, time, sys, statistics

url      = sys.argv[1] + "/v1/chat/completions"
prompt   = sys.argv[2]
max_tok  = int(sys.argv[3])
n_runs   = int(sys.argv[4])
body = json.dumps({
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tok,
    "stream": False
}).encode()

timings = []; ctoks = 0
for _ in range(n_runs):
    req = urllib.request.Request(url, data=body,
          headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            data = json.loads(r.read())
            ctoks = data.get("usage", {}).get("completion_tokens", 0)
            timings.append(elapsed_ms)
    except Exception:
        pass

if not timings:
    print(-1, 0)
else:
    print(int(statistics.median(timings)), ctoks)
PYEOF
}

# Embed sequential latency: n runs, returns "p50 p95 p99" in ms.
bench_embed_latency() {
    local n="${1:-20}"
    python3 - "$BASE_URL" "$n" <<'PYEOF'
import urllib.request, json, time, sys

url = sys.argv[1] + "/v1/embeddings"
n   = int(sys.argv[2])
body = json.dumps({
    "input": "The quick brown fox jumps over the lazy dog.",
    "task": "retrieval.query"
}).encode()

timings = []
for _ in range(n):
    req = urllib.request.Request(url, data=body,
          headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        timings.append(int((time.perf_counter() - t0) * 1000))
    except Exception:
        pass

if not timings:
    print("-1 -1 -1"); sys.exit()

s = sorted(timings)
def pct(data, p):
    return data[min(int(len(data) * p / 100), len(data) - 1)]

print(pct(s, 50), pct(s, 95), pct(s, 99))
PYEOF
}

# Embed concurrent throughput: fire concurrency requests simultaneously,
# measure wall-clock, return req/s. Median of n_trials.
bench_embed_throughput() {
    local concurrency="$1" n_trials="${2:-3}"
    python3 - "$BASE_URL" "$concurrency" "$n_trials" <<'PYEOF'
import urllib.request, json, time, sys, threading, statistics

url         = sys.argv[1] + "/v1/embeddings"
concurrency = int(sys.argv[2])
n_trials    = int(sys.argv[3])
body = json.dumps({
    "input": "Concurrent throughput benchmark sentence for embedding model.",
    "task": "retrieval.query"
}).encode()

def fetch():
    req = urllib.request.Request(url, data=body,
          headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r: r.read()
    except Exception:
        pass

rates = []
for _ in range(n_trials):
    threads = [threading.Thread(target=fetch) for _ in range(concurrency)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed = time.perf_counter() - t0
    rates.append(concurrency / elapsed)

print(f"{statistics.median(rates):.1f}")
PYEOF
}

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
echo -e "${BOLD}=== SwiftLM Benchmark ===${NC}"
echo "  binary:     $BINARY"
echo "  models dir: $MODELS_DIR"
echo "  output:     $OUT_MD  /  $OUT_JSON"

[[ ! -f "$BINARY" ]]          && { echo -e "\n${RED}Error:${NC} binary not found at $BINARY"; exit 1; }
command -v jq      &>/dev/null || { echo -e "\n${RED}Error:${NC} jq required";     exit 1; }
command -v python3 &>/dev/null || { echo -e "\n${RED}Error:${NC} python3 required"; exit 1; }

# ---------------------------------------------------------------------------
# Dedup + split by type
# ---------------------------------------------------------------------------
declare -A _SEEN
LLM_ENTRIES=(); EMBED_ENTRIES=()
for entry in "${MODELS[@]}"; do
    name="${entry%%|*}"
    [[ -n "${_SEEN[$name]+x}" ]] && continue
    _SEEN[$name]=1
    IFS='|' read -r _ typ _ _ <<< "$entry"
    [[ "$typ" == "embed" ]] && EMBED_ENTRIES+=("$entry") || LLM_ENTRIES+=("$entry")
done

# ---------------------------------------------------------------------------
# Initialise output files
# ---------------------------------------------------------------------------
{
    echo "# SwiftLM Benchmark Results"
    echo ""
    echo "_$(date '+%Y-%m-%d %H:%M:%S') — binary: \`$(basename "$BINARY")\`_"
    echo ""
    echo "## Text / VLM / SSD Models"
    echo ""
    printf "| %-44s | %-4s | %8s | %13s | %13s | %9s | %9s | %10s | %15s | %13s |\n" \
        "Model" "Type" "Load (s)" "Mem idle (MB)" "Mem peak (MB)" "GPU avg%" "ANE (mW)" "TTFT (ms)" "Prefill (tok/s)" "Decode (tok/s)"
    printf "|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|\n" \
        "$(printf '%.0s-' {1..46})" "$(printf '%.0s-' {1..6})" \
        "$(printf '%.0s-' {1..10})" "$(printf '%.0s-' {1..15})" \
        "$(printf '%.0s-' {1..15})" "$(printf '%.0s-' {1..11})" \
        "$(printf '%.0s-' {1..11})" "$(printf '%.0s-' {1..12})" \
        "$(printf '%.0s-' {1..17})" "$(printf '%.0s-' {1..15})"
} > "$OUT_MD"

# ---------------------------------------------------------------------------
# Text / VLM / SSD benchmark loop
# ---------------------------------------------------------------------------
for entry in "${LLM_ENTRIES[@]}"; do
    IFS='|' read -r model_name loader_type extra_flags load_timeout n_bench_runs <<< "$entry"
    n_bench_runs="${n_bench_runs:-1}"   # default 1 if not provided
    model_path="$MODELS_DIR/$model_name"

    # Skip if a filter is active and this model doesn't match by type or name
    if [[ -n "$FILTER" ]]; then
        [[ "$FILTER" == "embed" ]] && continue                      # embed-only → skip all LLM
        [[ "$loader_type" == "$FILTER" ]] && : ||                   # type match
        [[ "$model_name"  == "$FILTER" ]] && : || continue          # name match
    fi

    banner "$model_name  [$loader_type]"

    if [[ ! -d "$model_path" ]]; then
        warn "not found in $MODELS_DIR — skipping"
        continue
    fi

    # ── Start ──────────────────────────────────────────────────────────────
    start_server "$model_path" "$loader_type" "$extra_flags"
    echo -n "  Loading (up to ${load_timeout}s) ... "
    load_start=$(date +%s)
    wait_result=$(wait_for_ready "$load_timeout")
    load_end=$(date +%s)

    if [[ "$wait_result" == "died" || "$wait_result" == "timeout" ]]; then
        fail "server failed ($wait_result)"; kill_server; continue
    fi
    load_s=$(( load_end - load_start ))
    echo "ready in ${load_s}s"

    # SSD safety gate
    if [[ "$loader_type" == "ssd" ]]; then
        ssd_ok=$(health_field '.partition.ssd_stream')
        if [[ "$ssd_ok" != "true" ]]; then
            fail "SSD stream not active — aborting (safety)"; kill_server; continue
        fi
        step "SSD streaming confirmed"
    fi

    # ── Memory idle ────────────────────────────────────────────────────────
    sleep 1
    mem_idle=$(health_field '.memory.active_mb')
    metric "Memory idle: ${mem_idle} MB"

    # ── Warmup ─────────────────────────────────────────────────────────────
    # One untimed short completion warms JIT / caches before measurements.
    step "Warmup (1 untimed completion) ..."
    bench_decode_run "$SHORT_PROMPT" 10 1 >/dev/null 2>&1 || true

    # ── TTFT + Prefill + Decode (GPU/ANE sampled across all three) ─────────
    gpu_pct="-"; ane_mw="-"
    pm_start

    ttft_runs=7
    [[ "$loader_type" == "ssd" ]] && ttft_runs=5   # SSD is slow; 5 is enough
    step "TTFT — short prompt (${ttft_runs} runs, streaming) ..."
    ttft_ms=$(bench_ttft "$SHORT_PROMPT" 50 "$ttft_runs")
    metric "TTFT: ${ttft_ms} ms  (median of ${ttft_runs} runs)"

    # ── Prefill + Decode ───────────────────────────────────────────────────
    # SSD models use fewer decode tokens so the run completes in ~1-2 minutes
    # instead of 10+. Prefill runs for all models (batch-processes the prompt
    # in one forward pass, so it's much faster than decode per token).
    mem_peak="-"; prefill_tps="-"; decode_tps="-"

    if [[ "$loader_type" == "ssd" ]]; then
        decode_tokens=150
    else
        decode_tokens=500
    fi

    # Prefill benchmark
    step "Prefill — long prompt, max_tokens=1 (${n_bench_runs} run(s), median) ..."
    prefill_result=$(bench_prefill "$LONG_PROMPT" "$n_bench_runs")
    prefill_ms=$(awk '{print $1}' <<< "$prefill_result")
    prompt_toks=$(awk '{print $2}' <<< "$prefill_result")

    if (( prefill_ms > 0 && prompt_toks > 0 )); then
        prefill_tps=$(python3 -c "print(f'{$prompt_toks / ($prefill_ms/1000):.0f}')")
        metric "Prefill: ${prefill_ms}ms  (${prompt_toks} tokens → ${prefill_tps} tok/s)"
    else
        prefill_tps="-"
        warn "Prefill measurement failed"
    fi

    # Decode benchmark
    step "Decode — ${decode_tokens} tokens (${n_bench_runs} run(s), median tok/s) ..."
    decode_result=$(bench_decode_run "$DECODE_PROMPT" "$decode_tokens" "$n_bench_runs")
    decode_elapsed=$(awk '{print $1}' <<< "$decode_result")
    decode_ctoks=$(awk '{print $2}' <<< "$decode_result")

    if (( decode_elapsed > 0 && decode_ctoks > 0 && ttft_ms > 0 )); then
        decode_tps=$(python3 -c "
elapsed = $decode_elapsed
ttft    = $ttft_ms
ctoks   = $decode_ctoks
decode_ms = max(elapsed - ttft, 1)
print(f'{ctoks / (decode_ms/1000):.1f}')
")
        metric "Decode: ${decode_tps} tok/s  (${decode_ctoks} tokens in ${decode_elapsed}ms)"
    else
        decode_tps="-"
        warn "Decode measurement failed"
    fi

    pm_stop
    read -r gpu_pct ane_mw <<< "$(pm_stats)"
    metric "GPU avg: ${gpu_pct}%  ANE avg: ${ane_mw} mW"

    # Peak memory after generation
    mem_peak=$(health_field '.memory.peak_mb')
    metric "Memory peak: ${mem_peak} MB"

    # ── VLM capability benchmarks ──────────────────────────────────────────
    # Qualitative scoring for vision models: chart analysis, OCR, spatial
    # reasoning — tuned to each model's documented strengths.
    # Benchmarks run here (not in test.sh) because they measure quality, not
    # correctness; results are informational, not a pass/fail gate.
    cap_score="-"
    if [[ "$loader_type" == "vlm" ]]; then
        step "VLM capability benchmarks ..."
        cap_raw=$(python3 tests/vision/run_model_tests.py "$model_name" "$BASE_URL" 2>&1 || true)
        echo "$cap_raw" | sed 's/^/    /'
        # Extract summary "N/M" from the final score line
        cap_score=$(echo "$cap_raw" | grep -oE '[0-9]+/[0-9]+' | tail -1)
        [[ -z "$cap_score" ]] && cap_score="-"
        metric "Capability score: ${cap_score}"
    fi

    # ── Write row ──────────────────────────────────────────────────────────
    printf "| %-44s | %-4s | %8s | %13s | %13s | %9s | %9s | %10s | %15s | %13s |\n" \
        "$model_name" "$loader_type" "$load_s" "$mem_idle" "$mem_peak" \
        "$gpu_pct" "$ane_mw" "$ttft_ms" "$prefill_tps" "$decode_tps" >> "$OUT_MD"

    # JSON
    python3 - "$JSON_DIR/${model_name}.json" \
        "$model_name" "$loader_type" "$load_s" "$mem_idle" "$mem_peak" \
        "$gpu_pct" "$ane_mw" \
        "$ttft_ms" "$prefill_tps" "$decode_tps" "$cap_score" <<'PYEOF'
import json, sys
out_path = sys.argv[1]
d = {
    "model":           sys.argv[2],
    "type":            sys.argv[3],
    "load_s":          int(sys.argv[4])   if sys.argv[4].lstrip('-').isdigit() else None,
    "mem_idle_mb":     int(sys.argv[5])   if sys.argv[5].lstrip('-').isdigit() else None,
    "mem_peak_mb":     int(sys.argv[6])   if sys.argv[6].lstrip('-').isdigit() else None,
    "gpu_avg_pct":     int(sys.argv[7])   if sys.argv[7].lstrip('-').isdigit() else None,
    "ane_avg_mw":      int(sys.argv[8])   if sys.argv[8].lstrip('-').isdigit() else None,
    "ttft_ms":         int(sys.argv[9])   if sys.argv[9].lstrip('-').isdigit() else None,
    "prefill_tps":     int(sys.argv[10])  if sys.argv[10].lstrip('-').isdigit() else None,
    "decode_tps":      float(sys.argv[11]) if sys.argv[11].replace('.','',1).lstrip('-').isdigit() else None,
    "capability_score": sys.argv[12] if sys.argv[12] != "-" else None,
}
json.dump(d, open(out_path, 'w'), indent=2)
PYEOF

    kill_server
done

# ---------------------------------------------------------------------------
# Embedding benchmark loop
# ---------------------------------------------------------------------------
{
    echo ""
    echo "## Embedding Models"
    echo ""
    printf "| %-50s | %8s | %13s | %9s | %9s | %12s | %12s | %12s | %9s | %9s | %9s | %10s | %10s | %10s |\n" \
        "Model" "Load (s)" "Mem idle (MB)" "GPU avg%" "ANE (mW)" "Lat p50 (ms)" "Lat p95 (ms)" "Lat p99 (ms)" \
        "@1 (req/s)" "@4 (req/s)" "@8 (req/s)" "@16 (req/s)" "@32 (req/s)" "@64 (req/s)"
    printf "|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|\n" \
        "$(printf '%.0s-' {1..52})" "$(printf '%.0s-' {1..10})" "$(printf '%.0s-' {1..15})" \
        "$(printf '%.0s-' {1..11})" "$(printf '%.0s-' {1..11})" \
        "$(printf '%.0s-' {1..14})" "$(printf '%.0s-' {1..14})" "$(printf '%.0s-' {1..14})" \
        "$(printf '%.0s-' {1..11})" "$(printf '%.0s-' {1..11})" "$(printf '%.0s-' {1..11})" \
        "$(printf '%.0s-' {1..12})" "$(printf '%.0s-' {1..12})" "$(printf '%.0s-' {1..12})"
} >> "$OUT_MD"

for entry in "${EMBED_ENTRIES[@]}"; do
    IFS='|' read -r model_name loader_type extra_flags load_timeout <<< "$entry"
    model_path="$MODELS_DIR/$model_name"

    # Skip embed loop when filtering for a non-embed type or specific model name
    if [[ -n "$FILTER" && "$FILTER" != "embed" && "$model_name" != "$FILTER" ]]; then
        continue
    fi

    banner "$model_name  [embed]"

    if [[ ! -d "$model_path" ]]; then
        warn "not found in $MODELS_DIR — skipping"; continue
    fi

    # ── Start ──────────────────────────────────────────────────────────────
    start_server "$model_path" "$loader_type" "$extra_flags"
    echo -n "  Loading (up to ${load_timeout}s) ... "
    load_start=$(date +%s)
    wait_result=$(wait_for_ready "$load_timeout")
    load_end=$(date +%s)

    if [[ "$wait_result" == "died" || "$wait_result" == "timeout" ]]; then
        fail "server failed ($wait_result)"; kill_server; continue
    fi
    load_s=$(( load_end - load_start ))
    echo "ready in ${load_s}s"

    sleep 1
    mem_idle=$(health_field '.memory.active_mb')
    metric "Memory idle: ${mem_idle} MB"

    # ── Sequential latency + throughput (GPU/ANE sampled across both) ─────
    gpu_pct="-"; ane_mw="-"
    pm_start
    step "Sequential latency (30 requests) ..."
    lat_result=$(bench_embed_latency 30)
    lat_p50=$(awk '{print $1}' <<< "$lat_result")
    lat_p95=$(awk '{print $2}' <<< "$lat_result")
    lat_p99=$(awk '{print $3}' <<< "$lat_result")
    metric "Latency — p50=${lat_p50}ms  p95=${lat_p95}ms  p99=${lat_p99}ms"

    # ── Throughput sweep ───────────────────────────────────────────────────
    step "Throughput sweep (1 / 4 / 8 / 16 / 32 / 64 concurrent, 3 trials each) ..."
    declare -A tput
    for c in 1 4 8 16 32 64; do
        rps=$(bench_embed_throughput "$c" 3)
        tput[$c]="$rps"
        metric "  concurrency=${c}: ${rps} req/s"
    done
    pm_stop
    read -r gpu_pct ane_mw <<< "$(pm_stats)"
    metric "GPU avg: ${gpu_pct}%  ANE avg: ${ane_mw} mW"

    # ── Write row ──────────────────────────────────────────────────────────
    printf "| %-50s | %8s | %13s | %9s | %9s | %12s | %12s | %12s | %9s | %9s | %9s | %10s | %10s | %10s |\n" \
        "$model_name" "$load_s" "$mem_idle" "$gpu_pct" "$ane_mw" \
        "$lat_p50" "$lat_p95" "$lat_p99" \
        "${tput[1]}" "${tput[4]}" "${tput[8]}" "${tput[16]}" "${tput[32]}" "${tput[64]}" \
        >> "$OUT_MD"

    python3 - "$JSON_DIR/${model_name}.json" \
        "$model_name" "$load_s" "$mem_idle" "$gpu_pct" "$ane_mw" \
        "$lat_p50" "$lat_p95" "$lat_p99" \
        "${tput[1]}" "${tput[4]}" "${tput[8]}" "${tput[16]}" "${tput[32]}" "${tput[64]}" <<'PYEOF'
import json, sys
f = float
out_path = sys.argv[1]
json.dump({
    "model":        sys.argv[2],
    "type":         "embed",
    "load_s":       int(sys.argv[3])   if sys.argv[3].lstrip('-').isdigit() else None,
    "mem_idle_mb":  int(sys.argv[4])   if sys.argv[4].lstrip('-').isdigit() else None,
    "gpu_avg_pct":  int(sys.argv[5])   if sys.argv[5].lstrip('-').isdigit() else None,
    "ane_avg_mw":   int(sys.argv[6])   if sys.argv[6].lstrip('-').isdigit() else None,
    "latency": {
        "p50_ms": int(sys.argv[7])    if sys.argv[7].lstrip('-').isdigit() else None,
        "p95_ms": int(sys.argv[8])    if sys.argv[8].lstrip('-').isdigit() else None,
        "p99_ms": int(sys.argv[9])    if sys.argv[9].lstrip('-').isdigit() else None,
    },
    "throughput": {
        "c1":  f(sys.argv[10]) if sys.argv[10].replace('.','',1).isdigit() else None,
        "c4":  f(sys.argv[11]) if sys.argv[11].replace('.','',1).isdigit() else None,
        "c8":  f(sys.argv[12]) if sys.argv[12].replace('.','',1).isdigit() else None,
        "c16": f(sys.argv[13]) if sys.argv[13].replace('.','',1).isdigit() else None,
        "c32": f(sys.argv[14]) if sys.argv[14].replace('.','',1).isdigit() else None,
        "c64": f(sys.argv[15]) if sys.argv[15].replace('.','',1).isdigit() else None,
    }
}, open(out_path, 'w'), indent=2)
PYEOF

    kill_server
done

# ---------------------------------------------------------------------------
# Footnotes
# ---------------------------------------------------------------------------
{
    echo ""
    echo "---"
    echo ""
    echo "**Notes**"
    echo ""
    echo "- **Memory**: MLX-tracked unified GPU/CPU memory from \`/health\` endpoint (more accurate than OS RSS for GPU allocations)."
    echo "- **GPU avg%**: mean GPU HW active residency during TTFT/prefill/decode or embed measurements, sampled at 200 ms intervals via \`sudo powermetrics\`."
    echo "- **ANE avg mW**: mean Apple Neural Engine power draw during inference measurements."
    echo "- **TTFT**: streaming request, median of 5 runs. Short prompt (~8 tokens)."
    echo "- **Prefill tok/s**: non-streaming long-prompt request (~500 tokens) with \`max_tokens=1\`; elapsed time ≈ prefill latency."
    echo "- **Decode tok/s**: \`completion_tokens ÷ (total_time − TTFT)\` for a 500-token generation (150 tokens for SSD models). Excludes prefill."
    echo "- **SSD models**: mem idle shows 0 (model not resident between requests); 150-token generation used due to low throughput."
    echo "- **Embed latency**: 30 sequential single-text requests, percentiles computed from sorted timings."
    echo "- **Embed throughput**: median of 3 concurrent-burst trials per concurrency level (1/4/8/16/32/64)."
    echo "- **Prefill / Decode**: median of n_bench_runs runs (3 for small models, 2 for medium, 1 for large/SSD)."
    echo "- **Warmup**: one untimed short completion runs before measurements to prime JIT and caches."
} >> "$OUT_MD"

# ---------------------------------------------------------------------------
# Combine JSON results
# ---------------------------------------------------------------------------
python3 - "$JSON_DIR" "$OUT_JSON" <<'PYEOF'
import json, os, glob, sys

json_dir, out_path = sys.argv[1], sys.argv[2]
files   = sorted(glob.glob(os.path.join(json_dir, "*.json")))
results = [json.load(open(f)) for f in files]
json.dump(results, open(out_path, 'w'), indent=2)
print(f"Wrote {len(results)} results to {out_path}")
PYEOF

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  Benchmark complete${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Markdown: $OUT_MD"
echo "  JSON:     $OUT_JSON"
echo ""
