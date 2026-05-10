# SwiftLM Plan

## P0 — Planned removal

SwiftLM (`~/projects/SwiftLM`) will be removed once Daystrom is stable and its SSD
streaming logic has been fully ported. The SSD streaming oracle is now also available
at `~/projects/oracles/SwiftLM` for Daystrom's Stage 9/9.1 reference work. cube, which
currently depends on SwiftLM, is also planned for removal at the same time.

## P1 — Add new model tests

Tests and bench entries are needed for models that are downloaded but not yet covered:

- [x] Qwen3-Embedding-8B — added to bench.sh and test.sh; all tests pass (4096-dim, matryoshka 256, 32-concurrent batch)
- [x] Qwen3-VL-4B-Instruct-4bit — added to bench.sh and test.sh (vision smoke test, 224×224 PNG); passes
