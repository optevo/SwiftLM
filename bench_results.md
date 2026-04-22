# SwiftLM Benchmark Results

_2026-04-22 00:57:44 — binary: `SwiftLM`_

## Text / VLM / SSD Models

| Model                                        | Type | Load (s) | Mem idle (MB) | Mem peak (MB) |  TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|----------------------------------------------|------|----------|---------------|---------------|------------|-----------------|---------------|
| Qwen3.5-2B-4bit                              | text |        1 |          1010 |          1593 |         16 |            5836 |         316.1 |
| Qwen3.5-4B-MLX-4bit                          | text |        1 |          2257 |          3109 |         28 |            3242 |         164.2 |
| Qwen3.5-9B-MLX-4bit                          | text |        1 |          4804 |          5619 |         83 |             549 |         102.0 |
| Qwen3.6-35B-A3B                              | text |        2 |         19302 |         19688 |        209 |             743 |          21.6 |
| Qwen3-Coder-Next-4bit                        | text |        2 |         42766 |         43335 |         76 |            1338 |          93.5 |
| DeepSeek-R1-Distill-Qwen-32B-4bit            | text |        2 |         17577 |         18167 |       1822 |             521 |          30.5 |

| Qwen3.5-397B-A17B-4bit                       | ssd  |        1 |             0 |         33322 |        798 |              30 |           6.3 |
| Qwen3-Coder-480B-A35B-Instruct-4bit          | ssd  |        1 |             0 |         48921 |       2147 |              12 |           2.1 |
| FastVLM-0.5B-bf16                            | vlm  |        1 |          1188 |          1611 |         39 |            6560 |         370.8 |
| olmOCR-2-7B-1025-MLX-6bit                    | vlm  |        1 |          6593 |          7043 |         49 |            2326 |          79.5 |
| Qwen2.5-VL-3B-Instruct-6bit                  | vlm  |        1 |          3074 |          3608 |         30 |            4205 |         139.6 |

## Embedding Models

| Model                                              | Load (s) | Mem idle (MB) | Lat p50 (ms) | Lat p95 (ms) | Lat p99 (ms) | @1 (req/s) | @4 (req/s) | @8 (req/s) | @16 (req/s) | @32 (req/s) |
|----------------------------------------------------|----------|---------------|--------------|--------------|--------------|-----------|-----------|-----------|------------|------------|
| jina-embeddings-v5-text-small-retrieval-mlx        |        1 |             - |            6 |           36 |           36 |     131.5 |     188.2 |     451.4 |      708.7 |     1012.3 |
| jina-embeddings-v5-text-nano-retrieval-mlx         |        1 |             - |            5 |           28 |           28 |      95.4 |     356.4 |     522.3 |      648.8 |      752.6 |

---

**Notes**

- **Memory**: MLX-tracked unified GPU/CPU memory from `/health` endpoint.
- **TTFT**: streaming request, median of 5 runs. Short prompt (~8 tokens).
- **Prefill tok/s**: non-streaming long-prompt request (~500 tokens) with `max_tokens=1`; elapsed time ≈ prefill latency.
- **Decode tok/s**: `completion_tokens ÷ (total_time − TTFT)` for a 500-token generation (150 tokens for SSD models). Excludes prefill.
- **SSD models**: mem idle shows 0 (model not resident between requests); 150-token generation used due to low throughput.
- **Embed latency**: 20 sequential single-text requests, percentiles computed from sorted timings.
- **Embed throughput**: median of 3 concurrent-burst trials per concurrency level.
