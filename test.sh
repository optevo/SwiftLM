#!/usr/bin/env bash
# =============================================================================
# test.sh — SwiftLM model compatibility sweep
#
# Iterates every SwiftLM-compatible model in MODELS_DIR, starts the server,
# fires a minimal completion, and reports pass / fail / skip.
#
# SSD-streaming models (MoE >100B) are started with --stream-experts and
# SWIFTLM_TOP_K=6 so only the active expert slice (~6 GB) loads into RAM.
# The test VERIFIES ssd_stream=true in the health response before proceeding —
# loading a 224–270 GB model without that flag would blow all available RAM.
#
# Usage:
#   ./test.sh                              # uses .build/release/SwiftLM, ~/models
#   ./test.sh /path/to/SwiftLM            # override binary
#   ./test.sh /path/to/SwiftLM <filter>   # run only matching models (name or type)
#   MODELS_DIR=/alt/path ./test.sh        # override model directory
#
# Requires: curl, jq
# =============================================================================

set -uo pipefail   # -e intentionally omitted — per-model failures must not abort

BINARY="${1:-.build/release/SwiftLM}"
FILTER="${2:-}"   # optional: loader type (text|vlm|ssd|embed) or model dir name
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
TEST_PORT=18001
BASE_URL="http://127.0.0.1:${TEST_PORT}"
COMPLETION_TIMEOUT=60   # seconds to wait for a single completion response

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Model table
#
# Fields (colon-separated):
#   model_dir | loader_type | extra_flags | load_timeout_s
#
# loader_type:
#   text — standard text LLM   (no special flags)
#   vlm  — vision-language      (--vision)
#   ssd  — SSD-streaming MoE    (--stream-experts + SWIFTLM_TOP_K=6, REQUIRED)
#
# The script skips any model whose directory does not exist in MODELS_DIR,
# so adding future models here is safe — they are silently skipped until
# downloaded.
# ---------------------------------------------------------------------------
MODELS=(
    # ── Always-on: SLM trio ─────────────────────────────────────────────────
    "Qwen3.5-2B-4bit|text||60"
    "Qwen3.5-4B-MLX-4bit|text||60"
    "Qwen3.5-9B-MLX-4bit|text||90"

    # ── Always-on: large ────────────────────────────────────────────────────
    "Qwen3.6-35B-A3B|text||120"

    # ── Always-on: thinker (SSD — MUST use --stream-experts) ────────────────
    "Qwen3.5-397B-A17B-4bit|ssd|--stream-experts|240"

    # ── Always-on: vision router ─────────────────────────────────────────────
    "FastVLM-0.5B-bf16|vlm|--vision|60"

    # ── On-demand: coder ─────────────────────────────────────────────────────
    "Qwen3-Coder-Next-4bit|text||120"

    # ── On-demand: architect (SSD — MUST use --stream-experts) ──────────────
    "Qwen3-Coder-480B-A35B-Instruct-4bit|ssd|--stream-experts|240"

    # ── On-demand: vision extractors ─────────────────────────────────────────
    "olmOCR-2-7B-1025-MLX-6bit|vlm|--vision|90"
    "Qwen2.5-VL-3B-Instruct-6bit|vlm|--vision|90"

    # ── On-demand: fallback reasoning ────────────────────────────────────────
    "DeepSeek-R1-Distill-Qwen-32B-4bit|text||150"

    # ── On-demand: embedding server ──────────────────────────────────────────
    "jina-embeddings-v5-text-small-retrieval-mlx|embed|--embed|60"
    "jina-embeddings-v5-text-nano-retrieval-mlx|embed|--embed|60"
)

# Models explicitly NOT tested (not SwiftLM-compatible — different loaders)
NOT_SWIFTLM=(
    "parakeet-tdt-0.6b-v3"                          # mlx-audio (speech)
    "whisper-large-v3-turbo"                        # mlx-audio (speech-multi)
    "Voxtral-4B-TTS-2603-mlx-bf16"                  # mlx-audio (tts)
    "Kokoro-82M-bf16"                               # mlx-audio (tts-fast)
    "jina-reranker-v3-mlx"                          # rerank server (~/projects/rerank)
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
SERVER_PID=""
SERVER_LOG=""
PASSED=()
FAILED=()
SKIPPED=()
NOT_TESTED=()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
banner() { echo -e "\n${CYAN}${BOLD}━━━  $* ━━━${NC}"; }
info()   { echo -e "  ${YELLOW}→${NC} $*"; }
ok()     { echo -e "  ${GREEN}✅ PASS${NC}: $*"; }
fail()   { echo -e "  ${RED}❌ FAIL${NC}: $*"; }
skip()   { echo -e "  ${YELLOW}⏭  SKIP${NC}: $*"; }

kill_server() {
    if [ -n "$SERVER_PID" ]; then
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

show_log_tail() {
    if [ -n "$SERVER_LOG" ] && [ -f "$SERVER_LOG" ]; then
        local lines
        lines=$(wc -l < "$SERVER_LOG")
        if [ "$lines" -gt 0 ]; then
            echo "  Server log (last 15 lines):"
            tail -15 "$SERVER_LOG" | sed 's/^/    /'
        fi
    fi
}

cleanup() {
    kill_server
    [ -n "$SERVER_LOG" ] && rm -f "$SERVER_LOG"
}
trap cleanup EXIT

wait_for_ready() {
    local timeout="$1"
    local elapsed=0
    while [ "$elapsed" -lt "$timeout" ]; do
        if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
            echo "$elapsed"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "died"
            return 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "timeout"
    return 1
}

run_completion() {
    curl -sf \
        --max-time "$COMPLETION_TIMEOUT" \
        -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"Reply with the single word: PONG"}],"max_tokens":5}' \
        2>&1
}

run_embedding() {
    curl -sf \
        --max-time "$COMPLETION_TIMEOUT" \
        -X POST "$BASE_URL/v1/embeddings" \
        -H "Content-Type: application/json" \
        -d '{"input":"Hello world","task":"retrieval.query"}' \
        2>&1
}

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
echo -e "${BOLD}=== SwiftLM — model compatibility sweep ===${NC}"
echo "  binary:     $BINARY"
echo "  models dir: $MODELS_DIR"
echo "  test port:  $TEST_PORT"

if [ ! -f "$BINARY" ]; then
    echo ""
    echo -e "${RED}Error:${NC} binary not found at $BINARY"
    echo "  Run 'swift build -c release' first."
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo ""
    echo -e "${RED}Error:${NC} jq is required (nix: add jq to environment.systemPackages)"
    exit 1
fi

# Annotate the not-tested list
for m in "${NOT_SWIFTLM[@]}"; do
    NOT_TESTED+=("$m")
done

# ---------------------------------------------------------------------------
# Dedup model list (a model may appear in multiple role sections)
# ---------------------------------------------------------------------------
declare -A SEEN_MODELS
DEDUPED_MODELS=()
for entry in "${MODELS[@]}"; do
    model_name="${entry%%|*}"
    if [ -z "${SEEN_MODELS[$model_name]+set}" ]; then
        SEEN_MODELS[$model_name]=1
        DEDUPED_MODELS+=("$entry")
    fi
done

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for entry in "${DEDUPED_MODELS[@]}"; do
    IFS='|' read -r model_name loader_type extra_flags load_timeout <<< "$entry"
    model_path="$MODELS_DIR/$model_name"

    banner "$model_name  [$loader_type]"

    # ── Skip if filter active and this model doesn't match ─────────────────
    if [ -n "$FILTER" ]; then
        if [ "$loader_type" != "$FILTER" ] && [ "$model_name" != "$FILTER" ]; then
            continue
        fi
    fi

    # ── Skip if not downloaded ──────────────────────────────────────────────
    if [ ! -d "$model_path" ]; then
        skip "not found in $MODELS_DIR — not yet downloaded"
        SKIPPED+=("$model_name")
        continue
    fi

    # ── Build startup command ───────────────────────────────────────────────
    SERVER_LOG=$(mktemp /tmp/swiftlm-test-XXXXXX.log)

    if [ "$loader_type" = "ssd" ]; then
        info "SSD streaming mode — SWIFTLM_TOP_K=6 --stream-experts (RAM-safe)"
        SWIFTLM_TOP_K=6 "$BINARY" \
            --model "$model_path" \
            --port  "$TEST_PORT" \
            $extra_flags \
            >"$SERVER_LOG" 2>&1 &
    elif [ -n "$extra_flags" ]; then
        "$BINARY" \
            --model "$model_path" \
            --port  "$TEST_PORT" \
            $extra_flags \
            >"$SERVER_LOG" 2>&1 &
    else
        "$BINARY" \
            --model "$model_path" \
            --port  "$TEST_PORT" \
            >"$SERVER_LOG" 2>&1 &
    fi
    SERVER_PID=$!

    # ── Wait for /health ────────────────────────────────────────────────────
    echo -n "  Waiting for ready (up to ${load_timeout}s) ... "
    wait_result=$(wait_for_ready "$load_timeout")

    if [ "$wait_result" = "died" ]; then
        fail "server process died during startup"
        show_log_tail
        FAILED+=("$model_name — server died on startup")
        kill_server
        rm -f "$SERVER_LOG"; SERVER_LOG=""
        continue
    fi

    if [ "$wait_result" = "timeout" ]; then
        fail "timed out after ${load_timeout}s waiting for /health"
        show_log_tail
        FAILED+=("$model_name — startup timeout (${load_timeout}s)")
        kill_server
        rm -f "$SERVER_LOG"; SERVER_LOG=""
        continue
    fi

    echo "ready in ${wait_result}s"

    # ── SSD safety verification ─────────────────────────────────────────────
    # For SSD-streaming models, confirm the partition strategy before generating.
    # If ssd_stream is not true the model would be in RAM — dangerous on large MoE.
    if [ "$loader_type" = "ssd" ]; then
        health_json=$(curl -sf "$BASE_URL/health" 2>/dev/null || echo "{}")
        ssd_active=$(echo "$health_json" | jq -r '.partition.ssd_stream // false' 2>/dev/null)
        strategy=$(echo "$health_json"   | jq -r '.partition.strategy // "unknown"' 2>/dev/null)
        if [ "$ssd_active" != "true" ]; then
            fail "SSD streaming NOT active (strategy=$strategy, ssd_stream=$ssd_active)"
            echo "  This is a safety abort — loading this model fully into RAM is unsafe."
            show_log_tail
            FAILED+=("$model_name — ssd_stream not active (strategy=$strategy)")
            kill_server
            rm -f "$SERVER_LOG"; SERVER_LOG=""
            continue
        fi
        info "SSD streaming confirmed (strategy=$strategy)"
    fi

    # ── Smoke test — branch on loader type ─────────────────────────────────
    if [ "$loader_type" = "embed" ]; then
        # ── Embedding: single-request smoke test ────────────────────────────
        echo -n "  Embedding smoke test (single) ... "
        response=$(run_embedding)
        curl_exit=$?

        if [ $curl_exit -ne 0 ]; then
            fail "curl error (exit $curl_exit): $response"
            show_log_tail
            FAILED+=("$model_name — curl error $curl_exit")
            kill_server
            rm -f "$SERVER_LOG"; SERVER_LOG=""
            continue
        fi

        http_object=$(echo "$response" | jq -r '.object // ""' 2>/dev/null)
        emb_len=$(echo "$response"     | jq -r '.data[0].embedding | length' 2>/dev/null)

        if [ "$http_object" != "list" ]; then
            fail "unexpected response object: $http_object"
            echo "  Raw response: $response"
            show_log_tail
            FAILED+=("$model_name — bad response object: $http_object")
            kill_server
            rm -f "$SERVER_LOG"; SERVER_LOG=""
            continue
        fi

        if [ -z "$emb_len" ] || [ "$emb_len" -le 0 ] 2>/dev/null; then
            fail "empty embedding in response"
            echo "  Raw response: $(echo "$response" | head -c 200)"
            show_log_tail
            FAILED+=("$model_name — empty embedding")
            kill_server
            rm -f "$SERVER_LOG"; SERVER_LOG=""
            continue
        fi

        ok "dim=${emb_len}"

        # ── Embedding: concurrent batch test (validates dynamic batching) ───
        # Fire EMBED_CONCURRENCY requests simultaneously; all should land in
        # the same or consecutive batch windows and complete correctly.
        EMBED_CONCURRENCY=32
        echo -n "  Embedding batch test (${EMBED_CONCURRENCY} concurrent) ... "

        embed_pids=()
        embed_tmps=()
        for i in $(seq 1 "$EMBED_CONCURRENCY"); do
            tmp=$(mktemp /tmp/swiftlm-embed-XXXXXX)
            embed_tmps+=("$tmp")
            curl -sf --max-time 30 \
                -X POST "$BASE_URL/v1/embeddings" \
                -H "Content-Type: application/json" \
                -d "{\"input\":\"Batch throughput test sentence number $i\",\"task\":\"retrieval.query\"}" \
                -o "$tmp" &
            embed_pids+=($!)
        done

        embed_curl_failed=0
        for pid in "${embed_pids[@]}"; do
            wait "$pid" || embed_curl_failed=$((embed_curl_failed + 1))
        done

        embed_bad=0
        embed_dim=0
        for tmp in "${embed_tmps[@]}"; do
            obj=$(jq -r '.object // ""' "$tmp" 2>/dev/null)
            dim=$(jq -r '.data[0].embedding | length' "$tmp" 2>/dev/null)
            if [ "$obj" != "list" ] || ! [ "${dim:-0}" -gt 0 ] 2>/dev/null; then
                embed_bad=$((embed_bad + 1))
            else
                embed_dim=$dim
            fi
            rm -f "$tmp"
        done

        if [ "$embed_curl_failed" -gt 0 ]; then
            fail "${embed_curl_failed}/${EMBED_CONCURRENCY} concurrent requests failed (curl error)"
            show_log_tail
            FAILED+=("$model_name — batch test: ${embed_curl_failed} curl failures")
            kill_server
            rm -f "$SERVER_LOG"; SERVER_LOG=""
            continue
        fi

        if [ "$embed_bad" -gt 0 ]; then
            fail "${embed_bad}/${EMBED_CONCURRENCY} concurrent responses invalid"
            show_log_tail
            FAILED+=("$model_name — batch test: ${embed_bad} invalid responses")
            kill_server
            rm -f "$SERVER_LOG"; SERVER_LOG=""
            continue
        fi

        ok "${EMBED_CONCURRENCY}/${EMBED_CONCURRENCY} responses correct (dim=${embed_dim})"

        # ── Embedding: Matryoshka truncation test (dimensions=256) ──────────
        echo -n "  Embedding dim=256 test (Matryoshka) ... "
        mat_response=$(curl -sf --max-time "$COMPLETION_TIMEOUT" \
            -X POST "$BASE_URL/v1/embeddings" \
            -H "Content-Type: application/json" \
            -d '{"input":"Matryoshka truncation test","task":"retrieval.query","dimensions":256}' \
            2>&1)
        mat_exit=$?

        if [ $mat_exit -ne 0 ]; then
            fail "curl error (exit $mat_exit): $mat_response"
            show_log_tail
            FAILED+=("$model_name — dim=256 test: curl error $mat_exit")
            kill_server
            rm -f "$SERVER_LOG"; SERVER_LOG=""
            continue
        fi

        mat_obj=$(echo "$mat_response" | jq -r '.object // ""' 2>/dev/null)
        mat_dim=$(echo "$mat_response" | jq -r '.data[0].embedding | length' 2>/dev/null)

        if [ "$mat_obj" != "list" ]; then
            fail "unexpected response object: $mat_obj"
            echo "  Raw: $(echo "$mat_response" | head -c 200)"
            FAILED+=("$model_name — dim=256 test: bad object $mat_obj")
            kill_server
            rm -f "$SERVER_LOG"; SERVER_LOG=""
            continue
        fi

        if [ "${mat_dim:-0}" -ne 256 ] 2>/dev/null; then
            fail "expected dim=256, got dim=${mat_dim}"
            FAILED+=("$model_name — dim=256 test: got dim=${mat_dim}")
            kill_server
            rm -f "$SERVER_LOG"; SERVER_LOG=""
            continue
        fi

        ok "dim=${mat_dim} ✓"
        PASSED+=("$model_name")
    else
        # ── Chat completion smoke test ──────────────────────────────────────
        echo -n "  Completion smoke test ... "
        response=$(run_completion)
        curl_exit=$?

        if [ $curl_exit -ne 0 ]; then
            fail "curl error (exit $curl_exit): $response"
            show_log_tail
            FAILED+=("$model_name — curl error $curl_exit")
            kill_server
            rm -f "$SERVER_LOG"; SERVER_LOG=""
            continue
        fi

        http_object=$(echo "$response" | jq -r '.object // ""' 2>/dev/null)
        content=$(echo "$response"     | jq -r '.choices[0].message.content // ""' 2>/dev/null)

        if [ "$http_object" != "chat.completion" ]; then
            fail "unexpected response object: $http_object"
            echo "  Raw response: $response"
            show_log_tail
            FAILED+=("$model_name — bad response object: $http_object")
            kill_server
            rm -f "$SERVER_LOG"; SERVER_LOG=""
            continue
        fi

        if [ -z "$content" ]; then
            fail "empty content in completion response"
            echo "  Raw response: $response"
            show_log_tail
            FAILED+=("$model_name — empty content")
            kill_server
            rm -f "$SERVER_LOG"; SERVER_LOG=""
            continue
        fi

        # ── Token throughput from health ────────────────────────────────────
        health_after=$(curl -sf "$BASE_URL/health" 2>/dev/null || echo "{}")
        tok_s=$(echo "$health_after" | jq -r '.stats.avg_tokens_per_sec // "?"' 2>/dev/null)

        ok "response=$(echo "$content" | tr -d '\n' | head -c 60)  [${tok_s} tok/s]"
        PASSED+=("$model_name")
    fi

    # ── Tear down before next model ─────────────────────────────────────────
    kill_server
    rm -f "$SERVER_LOG"; SERVER_LOG=""
done

# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  REPORT${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo ""
echo -e "  ${GREEN}Passed  (${#PASSED[@]})${NC}"
for m in "${PASSED[@]}"; do echo "    ✅ $m"; done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo -e "  ${RED}Failed  (${#FAILED[@]})${NC}"
    for m in "${FAILED[@]}"; do echo "    ❌ $m"; done
fi

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo -e "  ${YELLOW}Skipped (${#SKIPPED[@]}) — not in $MODELS_DIR${NC}"
    for m in "${SKIPPED[@]}"; do echo "    ⏭  $m"; done
fi

if [ ${#NOT_TESTED[@]} -gt 0 ]; then
    echo ""
    echo "  Not tested — incompatible loader (not SwiftLM):"
    for m in "${NOT_TESTED[@]}"; do echo "    ○  $m"; done
fi

echo ""
total_tested=$(( ${#PASSED[@]} + ${#FAILED[@]} ))
echo -e "  Tested: ${total_tested}  |  Passed: ${#PASSED[@]}  |  Failed: ${#FAILED[@]}  |  Skipped: ${#SKIPPED[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    exit 1
fi

echo ""
echo -e "${GREEN}${BOLD}✅ All tested models passed.${NC}"
