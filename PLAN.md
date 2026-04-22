# SwiftLM Plan

## P1 — Add new model tests

Tests and bench entries are needed for models that are downloaded but not yet covered:

- [x] Qwen3-Embedding-8B — added to bench.sh and test.sh; all tests pass (4096-dim, matryoshka 256, 32-concurrent batch)
- [x] Qwen3-VL-4B-Instruct-4bit — added to bench.sh and test.sh (vision smoke test, 224×224 PNG); passes
