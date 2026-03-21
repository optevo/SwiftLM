#!/bin/bash
# test-server.sh — Integration tests for mlx-server
#
# Usage:
#   ./tests/test-server.sh [binary_path] [port]
#
# Requires: curl, jq
# The script starts the server, runs tests, then kills it.

set -euo pipefail

BINARY="${1:-.build/release/mlx-server}"
PORT="${2:-15413}"
HOST="127.0.0.1"
MODEL="mlx-community/Qwen2.5-0.5B-Instruct-4bit"  # Smallest model for CI
URL="http://${HOST}:${PORT}"
PASS=0
FAIL=0
TOTAL=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${YELLOW}[test]${NC} $*"; }
pass() { PASS=$((PASS + 1)); TOTAL=$((TOTAL + 1)); echo -e "  ${GREEN}✅ PASS${NC}: $*"; }
fail() { FAIL=$((FAIL + 1)); TOTAL=$((TOTAL + 1)); echo -e "  ${RED}❌ FAIL${NC}: $*"; }

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        log "Stopping server (PID $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Check prerequisites ─────────────────────────────────────────────
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found at $BINARY"
    echo "Run 'swift build -c release' first."
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "Error: jq is required. Install with: brew install jq"
    exit 1
fi

# ── Start server ─────────────────────────────────────────────────────
log "Starting server: $BINARY --model $MODEL --port $PORT"
"$BINARY" --model "$MODEL" --port "$PORT" --host "$HOST" &
SERVER_PID=$!

# Wait for server to be ready (model download + load)
log "Waiting for server to be ready (this may take a while on first run)..."
MAX_WAIT=600  # 10 minutes for model download
for i in $(seq 1 "$MAX_WAIT"); do
    if curl -sf "$URL/health" >/dev/null 2>&1; then
        log "Server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: Server process died"
        exit 1
    fi
    sleep 1
done

if ! curl -sf "$URL/health" >/dev/null 2>&1; then
    echo "Error: Server did not become ready in ${MAX_WAIT}s"
    exit 1
fi

# ── Test 1: Health endpoint ──────────────────────────────────────────
log "Test 1: GET /health"
HEALTH=$(curl -sf "$URL/health")
if echo "$HEALTH" | jq -e '.status == "ok"' >/dev/null 2>&1; then
    pass "Health endpoint returns status=ok"
else
    fail "Health endpoint: $HEALTH"
fi

if echo "$HEALTH" | jq -e '.model' >/dev/null 2>&1; then
    pass "Health endpoint returns model name"
else
    fail "Health endpoint missing model field"
fi

# ── Test 2: Models list ──────────────────────────────────────────────
log "Test 2: GET /v1/models"
MODELS=$(curl -sf "$URL/v1/models")
if echo "$MODELS" | jq -e '.object == "list"' >/dev/null 2>&1; then
    pass "Models endpoint returns object=list"
else
    fail "Models endpoint: $MODELS"
fi

if echo "$MODELS" | jq -e '.data | length > 0' >/dev/null 2>&1; then
    pass "Models endpoint has at least one model"
else
    fail "Models endpoint has no models"
fi

# ── Test 3: Non-streaming chat completion ────────────────────────────
log "Test 3: POST /v1/chat/completions (non-streaming)"
COMPLETION=$(curl -sf -X POST "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":20,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}]}")

if echo "$COMPLETION" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
    CONTENT=$(echo "$COMPLETION" | jq -r '.choices[0].message.content')
    pass "Non-streaming: got response: \"$CONTENT\""
else
    fail "Non-streaming completion: $COMPLETION"
fi

if echo "$COMPLETION" | jq -e '.choices[0].finish_reason == "stop"' >/dev/null 2>&1; then
    pass "Non-streaming: finish_reason=stop"
else
    fail "Non-streaming: missing finish_reason"
fi

if echo "$COMPLETION" | jq -e '.id' >/dev/null 2>&1; then
    pass "Non-streaming: has completion ID"
else
    fail "Non-streaming: missing ID"
fi

# ── Test 4: Streaming chat completion ────────────────────────────────
log "Test 4: POST /v1/chat/completions (streaming)"
STREAM_OUTPUT=$(curl -sf -N -X POST "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"stream\":true,\"max_tokens\":20,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}" \
    --max-time 30 2>/dev/null || true)

if echo "$STREAM_OUTPUT" | grep -q "data: \[DONE\]"; then
    pass "Streaming: received [DONE] sentinel"
else
    fail "Streaming: missing [DONE] sentinel"
fi

CHUNK_COUNT=$(echo "$STREAM_OUTPUT" | grep -c "^data: {" || true)
if [ "$CHUNK_COUNT" -gt 0 ]; then
    pass "Streaming: received $CHUNK_COUNT data chunks"
else
    fail "Streaming: no data chunks received"
fi

FIRST_CHUNK=$(echo "$STREAM_OUTPUT" | grep "^data: {" | head -1 | sed 's/^data: //')
if echo "$FIRST_CHUNK" | jq -e '.object == "chat.completion.chunk"' >/dev/null 2>&1; then
    pass "Streaming: chunk has correct object type"
else
    fail "Streaming: chunk missing object type"
fi

# ── Test 5: System message handling ──────────────────────────────────
log "Test 5: System message"
SYSTEM_RESP=$(curl -sf -X POST "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":20,\"messages\":[{\"role\":\"system\",\"content\":\"You are a pirate.\"},{\"role\":\"user\",\"content\":\"Say hello.\"}]}")

if echo "$SYSTEM_RESP" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
    pass "System message: got response"
else
    fail "System message: $SYSTEM_RESP"
fi

# ── Test 6: Invalid request handling ─────────────────────────────────
log "Test 6: Error handling"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"invalid": true}')

if [ "$HTTP_CODE" -ge 400 ]; then
    pass "Invalid request returns HTTP $HTTP_CODE"
else
    fail "Invalid request returned HTTP $HTTP_CODE (expected 4xx/5xx)"
fi

# ── Results ──────────────────────────────────────────────────────────
echo ""
log "═══════════════════════════════════════"
log "Results: ${PASS} passed, ${FAIL} failed, ${TOTAL} total"
log "═══════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
