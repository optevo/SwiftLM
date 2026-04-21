# SwiftLM Plan

## P1 — Embedding endpoint

Add `/v1/embeddings` support to SwiftLM for native Swift + MLX embedding serving.

**Reference implementation:** `~/projects/jina-embed/server.py` (Python/FastAPI, verified working)

Key details established by the Python version:
- **Model:** Jina v5 text-small (Qwen3 backbone, decoder architecture)
- **Pooling:** last-token (not mean, not CLS)
- **Normalisation:** L2 after pooling
- **Matryoshka:** truncate to target dim, then re-normalise
- **Task prefixes:** `"retrieval.query"` → `"Query: "`, `"retrieval.passage"` → `"Document: "`, etc.
- **Default dimension:** 256 (configurable per-request via `dimensions` field)
- **Request/response:** OpenAI-compatible (`/v1/embeddings`, standard JSON schema)

**Steps:**
1. Find HTTP server entry point in SwiftLM — identify where to add the new route
2. Add `EmbeddingRequest` / `EmbeddingResponse` Codable structs
3. Add model load path for non-generative (embedding) models
4. Implement forward pass without generation loop
5. Last-token pooling + L2 normalisation + Matryoshka truncation
6. Verify output matches Python reference server (compare embedding vectors)

**Motivation:** eliminate Python/GIL overhead; expect 2-3x throughput improvement under concurrent load.
