# SwiftLM Benchmark Results

_2026-04-23 15:14:05 — binary: `SwiftLM`_

## Text / VLM / SSD Models

| Model                                        | Type | Load (s) | Mem idle (MB) | Mem peak (MB) |  GPU avg% |  ANE (mW) |  TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|----------------------------------------------|------|----------|---------------|---------------|-----------|-----------|------------|-----------------|---------------|
| Qwen3.6-27B-UD-MLX-4bit                      | text |        1 |         24098 |         25006 |        95 |         0 |        149 |             224 |          21.0 |

## Embedding Models

| Model                                              | Load (s) | Mem idle (MB) |  GPU avg% |  ANE (mW) | Lat p50 (ms) | Lat p95 (ms) | Lat p99 (ms) | @1 (req/s) | @4 (req/s) | @8 (req/s) | @16 (req/s) | @32 (req/s) | @64 (req/s) |
|----------------------------------------------------|----------|---------------|-----------|-----------|--------------|--------------|--------------|-----------|-----------|-----------|------------|------------|------------|

---

**Notes**

- **Memory**: MLX-tracked unified GPU/CPU memory from `/health` endpoint (more accurate than OS RSS for GPU allocations).
- **GPU avg%**: mean GPU HW active residency during TTFT/prefill/decode or embed measurements, sampled at 200 ms intervals via `sudo powermetrics`.
- **ANE avg mW**: mean Apple Neural Engine power draw during inference measurements.
- **TTFT**: streaming request, median of 5 runs. Short prompt (~8 tokens).
- **Prefill tok/s**: non-streaming long-prompt request (~500 tokens) with `max_tokens=1`; elapsed time ≈ prefill latency.
- **Decode tok/s**: `completion_tokens ÷ (total_time − TTFT)` for a 500-token generation (150 tokens for SSD models). Excludes prefill.
- **SSD models**: mem idle shows 0 (model not resident between requests); 150-token generation used due to low throughput.
- **Embed latency**: 30 sequential single-text requests, percentiles computed from sorted timings.
- **Embed throughput**: median of 3 concurrent-burst trials per concurrency level (1/4/8/16/32/64).
- **Prefill / Decode**: median of n_bench_runs runs (3 for small models, 2 for medium, 1 for large/SSD).
- **Warmup**: one untimed short completion runs before measurements to prime JIT and caches.
