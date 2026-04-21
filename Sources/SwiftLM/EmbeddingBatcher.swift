// EmbeddingBatcher — dynamic-batching actor for /v1/embeddings throughput.
//
// Design goals:
//   • Maximise GPU utilisation by accumulating concurrent requests into large
//     batches before dispatching a single forward pass.
//   • Collect jobs while the GPU is busy with the previous batch; flush
//     immediately once the GPU is free (pipeline approach).
//   • Flush early if total queued texts reach `maxBatchTexts`.
//   • Group jobs that share the same (task, dimensions) into one GPU call so
//     texts with the same prefix are processed together.
//   • While the GPU is busy with batch N the HTTP layer and actor are free to
//     accumulate batch N+1 — no artificial back-pressure.
//   • Thread-safe by construction: the actor serialises queue mutations; Swift
//     Concurrency suspends the actor at every `await`, allowing concurrent
//     `submit` calls to proceed while a batch is in-flight.
//
// Batching strategy (pipeline, no timer):
//   The first job that arrives immediately starts inference. While the GPU is
//   busy, subsequent jobs accumulate in `pending`. When inference completes,
//   all queued jobs form the next batch. This provides natural batching at
//   high concurrency without any fixed wait time — and avoids Task.sleep,
//   which causes "freed pointer was not the last allocation" crashes in MLX's
//   Metal allocator when the resumed thread has no prior Metal state.
//
// Usage:
//   let batcher = EmbeddingBatcher(container: embedContainer, maxBatchTexts: 64)
//   let embeddings = try await batcher.submit(texts: ["hello"], task: "retrieval.query", dimensions: nil)

import Foundation
import MLX
import MLXEmbedders

actor EmbeddingBatcher {

    // MARK: - Internal types

    private struct Job {
        let texts: [String]      // raw (no prefix applied yet)
        let task: String
        let dimensions: Int?
        let continuation: CheckedContinuation<[[Float]], any Error>
    }

    // MARK: - Configuration

    /// Flush immediately when this many texts are queued (prevents unbounded latency).
    let maxBatchTexts: Int

    // MARK: - State

    private let container: EmbedderModelContainer
    private var pending: [Job] = []
    private var processing = false  // is a pipeline loop currently running?

    // MARK: - Init

    init(container: EmbedderModelContainer, maxBatchTexts: Int = 64) {
        self.container = container
        self.maxBatchTexts = maxBatchTexts
    }

    // MARK: - Public API

    /// Enqueue texts for embedding and suspend until the batch they land in completes.
    ///
    /// Multiple concurrent callers all push into the same queue; the batcher
    /// dispatches them together as one GPU call, then resumes all continuations.
    func submit(texts: [String], task: String, dimensions: Int?) async throws -> [[Float]] {
        try await withCheckedThrowingContinuation { continuation in
            pending.append(Job(texts: texts, task: task, dimensions: dimensions,
                               continuation: continuation))
            // Start the pipeline loop if not already running.
            // The loop drains `pending` in waves; each wave runs while the GPU
            // is busy, so subsequent requests accumulate into the next batch.
            if !processing {
                processing = true
                Task { await self.processLoop() }
            }
        }
    }

    // MARK: - Private: pipeline loop

    /// Continuously drain `pending` until empty, then clear `processing`.
    ///
    /// While each batch is being processed (actor is suspended at `await flush`),
    /// new `submit` calls can add more jobs to `pending`. When the current batch
    /// completes the loop picks them all up as the next batch — providing natural
    /// dynamic batching without any fixed timer or Task.sleep.
    private func processLoop() async {
        while !pending.isEmpty {
            // Enforce the max-batch ceiling: if more than maxBatchTexts are queued,
            // take only enough to fill one batch; the rest stay in `pending` and
            // are processed in the next iteration.
            var batchJobs: [Job]
            var totalTexts = 0
            var cutoff = pending.count
            for (i, job) in pending.enumerated() {
                if totalTexts + job.texts.count > maxBatchTexts && i > 0 {
                    cutoff = i
                    break
                }
                totalTexts += job.texts.count
            }
            batchJobs = Array(pending[..<cutoff])
            pending = Array(pending[cutoff...])

            await flushBatch(batchJobs)
        }
        processing = false
    }

    // MARK: - Private: flush

    private func flushBatch(_ jobs: [Job]) async {
        // Group by (task, dimensions) so texts that share a prefix and output
        // dimension are processed in a single GPU call.
        var groupOrder: [String] = []
        var groups: [String: [Job]] = [:]
        for job in jobs {
            let key = "\(job.task)|\(job.dimensions.map(String.init) ?? "nil")"
            if groups[key] == nil { groupOrder.append(key) }
            groups[key, default: []].append(job)
        }

        for key in groupOrder {
            let group = groups[key]!
            do {
                let results = try await runGroup(group)
                for (job, result) in zip(group, results) {
                    job.continuation.resume(returning: result)
                }
            } catch {
                for job in group {
                    job.continuation.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Private: batch dispatch

    /// Build the flattened text list and slice boundaries for a group, then
    /// hand off to a global (non-isolated) inference function.
    ///
    /// The inference is intentionally called via a global function rather than
    /// directly from this actor method.  MLX's Metal eval() has thread-local
    /// state that behaves incorrectly when invoked from within an actor's
    /// executor context; a global async function runs on the cooperative thread
    /// pool without actor isolation, matching the context used by the original
    /// route-handler-direct code (which worked correctly).
    private func runGroup(_ jobs: [Job]) async throws -> [[[Float]]] {
        let task       = jobs[0].task
        let dimensions = jobs[0].dimensions
        let prefix     = embeddingTaskPrefix(task)

        // Flatten all texts and record per-job slice boundaries.
        var allTexts = [String]()
        var slices   = [(Int, Int)]()
        for job in jobs {
            let start = allTexts.count
            allTexts.append(contentsOf: job.texts.map { prefix + $0 })
            slices.append((start, allTexts.count))
        }

        // Access actor-isolated property before suspending, then delegate
        // the actual GPU call to a nonisolated global function.
        let containerRef = container
        let flatEmbeddings = try await embeddingGroupInference(
            texts: allTexts, dimensions: dimensions, container: containerRef
        )

        return slices.map { start, end in Array(flatEmbeddings[start..<end]) }
    }
}

// MARK: - Inference (nonisolated, global)

/// Tokenise `texts`, run a single forward pass, pool, normalise, and return [[Float]].
///
/// Kept as a global async function (no actor isolation) so that MLX's Metal
/// eval() runs on a plain cooperative-pool thread — the same context used by
/// the original route-handler-direct embedding code.
func embeddingGroupInference(
    texts: [String],
    dimensions: Int?,
    container: EmbedderModelContainer
) async throws -> [[Float]] {
    return try await container.perform { ctx in
        let tokenizer = ctx.tokenizer

        // Tokenise (with BOS/EOS special tokens as required by Qwen3)
        let tokenized = texts.map { tokenizer.encode(text: $0, addSpecialTokens: true) }
        let maxLen    = tokenized.map { $0.count }.max() ?? 0
        guard maxLen > 0 else { return [] }

        // Build right-padded input_ids and attention_mask.
        // Pad token ID = 0; mask = 0 for pads (matches jina-embed reference).
        var flatIds  = [Int]()
        var flatMask = [Int]()
        flatIds.reserveCapacity(tokenized.count * maxLen)
        flatMask.reserveCapacity(tokenized.count * maxLen)
        for tokens in tokenized {
            let padLen = maxLen - tokens.count
            flatIds.append(contentsOf: tokens)
            flatIds.append(contentsOf: repeatElement(0, count: padLen))
            flatMask.append(contentsOf: repeatElement(1, count: tokens.count))
            flatMask.append(contentsOf: repeatElement(0, count: padLen))
        }

        let batchSize  = tokenized.count
        let inputArray = MLXArray(flatIds,  [batchSize, maxLen])
        let maskArray  = MLXArray(flatMask, [batchSize, maxLen])

        // Forward pass — pass maskArray as attentionMask so bidirectional models
        // (e.g. EuroBERT) can prevent attending to padding positions.
        // Causal models (e.g. Qwen3) ignore the attentionMask parameter.
        let output = ctx.model(
            inputArray, positionIds: nil, tokenTypeIds: nil, attentionMask: maskArray)

        // Last-token pooling; defer normalisation until after Matryoshka truncation.
        var pooled = ctx.pooling(output, mask: maskArray, normalize: false)

        if let dim = dimensions {
            let safeDim = min(dim, pooled.shape[1])
            if safeDim > 0 { pooled = pooled[0..., 0..<safeDim] }
        }

        // L2 normalise — inlined to avoid l2Normalized() ambiguity between
        // the MLX module and MLXEmbedders, both of which define this extension.
        let norms = sqrt(sum(pooled * pooled, axis: -1, keepDims: true))
        pooled = pooled / maximum(norms, MLXArray(Float(1e-12)))
        pooled.eval()

        let embedDim = pooled.shape[1]
        let flat     = pooled.asArray(Float.self)
        return (0..<batchSize).map { i in
            Array(flat[(i * embedDim)..<((i + 1) * embedDim)])
        }
    }
}

// MARK: - Task prefix

/// Maps OpenAI-compatible task strings to the prefix expected by jina-v5 models.
/// Other embedding models that do not use task prefixes ignore this (prefix stays "").
func embeddingTaskPrefix(_ task: String) -> String {
    switch task {
    case "retrieval.query":
        return "Query: "
    case "retrieval.passage", "classification", "text-matching", "clustering":
        return "Document: "
    default:
        return ""
    }
}
