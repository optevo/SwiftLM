# SwiftLM Benchmark Results

_2026-04-22 10:13:22 — binary: `SwiftLM`_

## Text / VLM / SSD Models

| Model                                        | Type | Load (s) | Mem idle (MB) | Mem peak (MB) |  TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|----------------------------------------------|------|----------|---------------|---------------|------------|-----------------|---------------|
| DeepSeek-R1-Distill-Qwen-32B-4bit            | text |        2 |         17577 |         18167 |       1822 |             521 |          30.5 |

## Embedding Models

| Model                                              | Load (s) | Mem idle (MB) | Lat p50 (ms) | Lat p95 (ms) | Lat p99 (ms) | @1 (req/s) | @4 (req/s) | @8 (req/s) | @16 (req/s) | @32 (req/s) |
|----------------------------------------------------|----------|---------------|--------------|--------------|--------------|-----------|-----------|-----------|------------|------------|

---

**Notes**

- **Memory**: MLX-tracked unified GPU/CPU memory from `/health` endpoint.
- **TTFT**: streaming request, median of 5 runs. Short prompt (~8 tokens).
- **Prefill tok/s**: non-streaming long-prompt request (~500 tokens) with `max_tokens=1`; elapsed time ≈ prefill latency.
- **Decode tok/s**: `completion_tokens ÷ (total_time − TTFT)` for a 500-token generation (150 tokens for SSD models). Excludes prefill.
- **SSD models**: mem idle shows 0 (model not resident between requests); 150-token generation used due to low throughput.
- **Embed latency**: 20 sequential single-text requests, percentiles computed from sorted timings.
- **Embed throughput**: median of 3 concurrent-burst trials per concurrency level.
