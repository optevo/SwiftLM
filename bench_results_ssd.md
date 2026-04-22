# SwiftLM Benchmark Results

_2026-04-22 01:57:02 — binary: `SwiftLM`_

## Text / VLM / SSD Models

| Model                                        | Type | Load (s) | Mem idle (MB) | Mem peak (MB) |  TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|----------------------------------------------|------|----------|---------------|---------------|------------|-----------------|---------------|
| Qwen3.5-397B-A17B-4bit                       | ssd  |        1 |             0 |         33322 |        798 |              30 |           6.3 |
| Qwen3-Coder-480B-A35B-Instruct-4bit          | ssd  |        1 |             0 |         48921 |       2147 |              12 |           2.1 |

## Embedding Models

| Model                                              | Load (s) | Mem idle (MB) | Lat p50 (ms) | Lat p95 (ms) | Lat p99 (ms) | @1 (req/s) | @4 (req/s) | @8 (req/s) | @16 (req/s) | @32 (req/s) |
|----------------------------------------------------|----------|---------------|--------------|--------------|--------------|-----------|-----------|-----------|------------|------------|

---

**Notes**

- **Memory**: MLX-tracked unified GPU/CPU memory from `/health` endpoint.
- **TTFT**: streaming request, median of 3 runs (5 runs for SSD models). Short prompt (~8 tokens).
- **Prefill tok/s**: non-streaming long-prompt request (~500 tokens) with `max_tokens=1`; elapsed time ≈ prefill latency.
- **Decode tok/s**: `completion_tokens ÷ (total_time − TTFT)` for a 300-token generation (150 tokens for SSD models). Excludes prefill.
- **SSD models**: mem idle shows 0 (model not resident between requests); 150-token generation used due to low throughput.
- **Embed latency**: 20 sequential single-text requests, percentiles computed from sorted timings.
- **Embed throughput**: median of 3 concurrent-burst trials per concurrency level.
