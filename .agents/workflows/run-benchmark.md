---
description: Run the extreme context benchmark suite for SwiftLM profiling
---

# Run Extreme Context Benchmark

This workflow runs the full 12-test profiling matrix (4 configurations × 3 context depths) for a given model and captures performance + memory metrics.

## Prerequisites

// turbo
1. Ensure the SwiftLM binary is built:
```bash
swift build -c release
```

## Run the Benchmark

2. Kill any existing SwiftLM server:
```bash
killall SwiftLM 2>/dev/null; sleep 2
```

// turbo
3. Run the profiling suite with the target model:
```bash
python3 -u scripts/profiling/profile_runner.py \
  --model gemma-4-26b-a4b-it-4bit \
  --contexts "512,40000,100000" \
  --out ./profiling_results_$(hostname -s).md
```

The profiler will:
- Start SwiftLM with each configuration (`Dense/Vanilla`, `SSD Stream`, `TurboQuant`, `SSD + TurboQuant`)
- Send requests at each context depth (512, 40K, 100K tokens)
- Measure TTFT, TPS, Active RAM, and GPU Alloc (via `ioreg AGXAccelerator`)
- Save results to the output markdown file

## Expected Runtime

| Context Size | Approximate Time per Config |
|---|---|
| 512 | ~10 seconds |
| 40,000 | ~40 seconds |
| 100,000 | ~120 seconds |

**Total**: ~12 minutes for the full 12-test matrix (4 configs × 3 contexts)

## Customizing

- **Different model**: Replace `gemma-4-26b-a4b-it-4bit` with any MLX model ID  
- **Different contexts**: Change `--contexts` (comma-separated list of token counts)
- **Output file**: Change `--out` path

## After the Benchmark

4. Review the generated markdown file and check for any `FAILED / OOM` entries.

5. If contributing results back to the project, append the device section to `profiling_results.md`:
   - Add a `## <Device Name> — <RAM> GB Unified Memory` section
   - Include chip, RAM, macOS version, SwiftLM version
   - Submit a PR with title: `bench: <device-name> results for <model-id>`

## Troubleshooting

- **Server fails to start**: Check that the model exists at `~/.aegis-ai/models/mlx_models/mlx-community/<model-id>` or use a full path
- **Python crashes silently**: Run with `python3 -u` for unbuffered output
- **100K tests OOM on <32 GB**: Use `--contexts "512,40000"` to skip 100K
- **Stale results**: The profiler kills any existing SwiftLM process before each config run
