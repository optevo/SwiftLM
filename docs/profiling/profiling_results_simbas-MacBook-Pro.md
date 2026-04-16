### `mlx-community/Qwen3.6-35B-A3B-4bit` — Context & Memory Profile

Context depths tested: 512,40000,100000

| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (Physical) | GPU Memory Allocated |
|---|---|---|---|---|---|---|
| Dense/Vanilla | 512 | 4.01s | 32.10 tok/s | N/A | 18.9 GB | 33.6 GB |
| Dense/Vanilla | 40000 | 26.41s | 23.99 tok/s | N/A | 49.4 GB | 64.2 GB |
| Dense/Vanilla | 100000 | 151.76s | 18.64 tok/s | N/A | 49.3 GB | 63.9 GB |
| SSD Stream | 512 | 1.81s | 15.01 tok/s | N/A | 4.5 GB | 18.8 GB |
| SSD Stream | 40000 | 28.89s | 5.13 tok/s | N/A | 37.4 GB | 51.7 GB |
| SSD Stream | 100000 | 100.72s | 4.08 tok/s | N/A | 49.4 GB | 63.9 GB |
| TurboQuant | 512 | 0.44s | 33.14 tok/s | N/A | 18.9 GB | 33.3 GB |
| TurboQuant | 40000 | 20.90s | 2.54 tok/s | N/A | 22.7 GB | 37.0 GB |
| TurboQuant | 100000 | 60.30s | 4.73 tok/s | N/A | 27.7 GB | 42.0 GB |
| SSD + TurboQuant | 512 | 1.64s | 14.51 tok/s | N/A | 4.5 GB | 19.3 GB |
| SSD + TurboQuant | 40000 | 27.56s | 5.39 tok/s | N/A | 8.5 GB | 23.2 GB |
| SSD + TurboQuant | 100000 | 75.59s | 3.86 tok/s | N/A | 13.6 GB | 28.3 GB |
| SSD + 16-Worker Prefetch | 512 | 0.94s | 16.70 tok/s | N/A | 4.5 GB | 19.4 GB |
| SSD + 16-Worker Prefetch | 40000 | 28.88s | 5.17 tok/s | N/A | 37.4 GB | 51.9 GB |
| SSD + 16-Worker Prefetch | 100000 | 101.96s | 3.79 tok/s | N/A | 49.4 GB | 63.9 GB |

> **Active RAM (Physical)**: Real memory wired into RAM by macOS (capped by device RAM).
> **GPU Memory Allocated**: Total memory requested by the GPU — includes data swapped to SSD. This shows the TRUE memory demand and reveals TurboQuant compression benefits even when Active RAM is saturated.
