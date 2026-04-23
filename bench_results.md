# SwiftLM Benchmark Results

_2026-04-22 21:41:40 — binary: `SwiftLM`_

## Text / VLM / SSD Models

| Model                                        | Type | Load (s) | Mem idle (MB) | Mem peak (MB) |  GPU avg% |  ANE (mW) |  TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|----------------------------------------------|------|----------|---------------|---------------|-----------|-----------|------------|-----------------|---------------|
| Qwen3.5-2B-4bit                              | text |        1 |          1010 |          1593 |        42 |         0 |         21 |           26750 |       15440.0 |
| Qwen3.5-4B-MLX-4bit                          | text |        1 |          2257 |          3110 |        99 |         0 |        134 |           16050 |         174.4 |
| Qwen3.5-9B-MLX-4bit                          | text |        1 |          4804 |          5620 |        97 |         0 |         71 |           11889 |         449.4 |
| Qwen3.6-35B-A3B                              | text |        2 |         19302 |         19693 |       100 |         0 |         95 |            1310 |          38.6 |
| Qwen3.6-35B-A3B-VLM-4bit                     | vlm  |        2 |         19023 |         19793 |        96 |         0 |         85 |             517 |         105.0 |
| Qwen3-VL-8B-Instruct-4bit                    | vlm  |        1 |          5494 |          6142 |        94 |         0 |         83 |            2113 |           0.4 |
| Qwen3-Coder-Next-4bit                        | text |        4 |         42766 |         43335 |        78 |         0 |         55 |            2348 |         135.1 |
| DeepSeek-R1-Distill-Qwen-32B-4bit            | text |        2 |         17577 |         18169 |       100 |         0 |        216 |             837 |          26.3 |
| Qwen3.5-397B-A17B-4bit                       | ssd  |        1 |             0 |         33340 |        82 |         0 |        662 |              31 |           6.2 |
| Qwen3-Coder-480B-A35B-Instruct-4bit          | ssd  |        1 |             0 |         48921 |        35 |         0 |       1438 |              11 |           1.8 |
| olmOCR-2-7B-1025-MLX-6bit                    | vlm  |        2 |          6593 |          7044 |       100 |         0 |         49 |            3527 |          41.3 |

## Embedding Models

| Model                                              | Load (s) | Mem idle (MB) |  GPU avg% |  ANE (mW) | Lat p50 (ms) | Lat p95 (ms) | Lat p99 (ms) | @1 (req/s) | @4 (req/s) | @8 (req/s) | @16 (req/s) | @32 (req/s) | @64 (req/s) |
|----------------------------------------------------|----------|---------------|-----------|-----------|--------------|--------------|--------------|-----------|-----------|-----------|------------|------------|------------|
| jina-embeddings-v5-text-small-retrieval-mlx        |        1 |             - |        52 |         0 |            6 |            7 |           33 |      82.4 |     199.3 |     395.8 |      705.8 |     1007.6 |      621.2 |
| jina-embeddings-v5-text-nano-retrieval-mlx         |        1 |             - |        29 |         0 |            4 |           10 |           28 |     103.9 |     306.7 |     452.1 |      721.1 |      847.9 |      907.6 |
| Qwen3-Embedding-8B                                 |        2 |             - |        92 |         0 |           52 |           54 |          103 |      18.6 |      34.9 |      59.2 |       98.8 |      120.7 |       94.6 |

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
