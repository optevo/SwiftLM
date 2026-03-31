// ModelDownloadManager.swift — HuggingFace cache inspection and model lifecycle
// Manages local model storage: downloaded status, disk size, deletion, persistence.

import Foundation
import Combine

/// Represents a locally downloaded model entry.
public struct DownloadedModel: Identifiable, Sendable {
    public let id: String           // HuggingFace model ID
    public let cacheDirectory: URL  // Local cache path
    public let sizeBytes: Int64     // Total bytes on disk
    public let modifiedDate: Date?  // Last access/modification date

    public var displaySize: String {
        let gb = Double(sizeBytes) / 1_073_741_824
        let mb = Double(sizeBytes) / 1_048_576
        if gb >= 1.0 { return String(format: "%.1f GB", gb) }
        return String(format: "%.0f MB", mb)
    }
}

/// Download progress for an in-flight download.
public struct ModelDownloadProgress: Sendable {
    public let modelId: String
    public let fractionCompleted: Double  // 0.0–1.0
    public let speedMBps: Double?         // nil if unknown

    public var speedString: String {
        guard let s = speedMBps else { return "" }
        return String(format: "%.1f MB/s", s)
    }
    public var percentString: String { "\(Int(fractionCompleted * 100))%" }
}

/// Manages the HuggingFace model cache for SwiftLM Chat.
/// Thread-safe: all mutations happen on MainActor.
@MainActor
public final class ModelDownloadManager: ObservableObject {

    // MARK: — Published state

    @Published public private(set) var downloadedModels: [DownloadedModel] = []
    @Published public private(set) var activeDownloads: [String: ModelDownloadProgress] = [:]
    @Published public private(set) var totalDiskUsageBytes: Int64 = 0

    // MARK: — Persistence

    private let lastModelKey = "swiftlm.lastLoadedModelId"
    public var lastLoadedModelId: String? {
        get { UserDefaults.standard.string(forKey: lastModelKey) }
        set { UserDefaults.standard.set(newValue, forKey: lastModelKey) }
    }

    // MARK: — HuggingFace cache paths

    /// Primary HF hub cache directory.
    public static var huggingFaceCacheRoot: URL {
        // Respect $HF_HUB_CACHE > $HF_HOME > default
        if let hfCache = ProcessInfo.processInfo.environment["HF_HUB_CACHE"] {
            return URL(fileURLWithPath: hfCache)
        }
        if let hfHome = ProcessInfo.processInfo.environment["HF_HOME"] {
            return URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }

    /// Convert a HuggingFace model ID to its cache directory name.
    /// e.g. "mlx-community/Qwen2.5-7B-Instruct-4bit" → "models--mlx-community--Qwen2.5-7B-Instruct-4bit"
    public static func cacheDirName(for modelId: String) -> String {
        "models--" + modelId.replacingOccurrences(of: "/", with: "--")
    }

    /// Returns the cache directory URL for a given model ID, or nil if not found.
    public static func cacheDirectory(for modelId: String) -> URL? {
        let dir = huggingFaceCacheRoot.appendingPathComponent(cacheDirName(for: modelId))
        return FileManager.default.fileExists(atPath: dir.path) ? dir : nil
    }

    // MARK: — Public API

    public init() {
        refresh()
    }

    /// Re-scan the HuggingFace cache and update downloaded model list.
    public func refresh() {
        let root = Self.huggingFaceCacheRoot
        guard FileManager.default.fileExists(atPath: root.path) else {
            downloadedModels = []
            totalDiskUsageBytes = 0
            return
        }

        var found: [DownloadedModel] = []
        let fm = FileManager.default

        // Enumerate all "models--*" directories
        guard let contents = try? fm.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else { return }

        for dir in contents {
            guard dir.lastPathComponent.hasPrefix("models--") else { continue }

            // Map directory name back to model ID
            let dirName = dir.lastPathComponent
            let modelId = dirName
                .replacingOccurrences(of: "^models--", with: "", options: .regularExpression)
                .replacingOccurrences(of: "--", with: "/")

            // Only include models in our catalog
            guard ModelCatalog.all.contains(where: { $0.id == modelId }) else { continue }

            let size = directorySize(at: dir)
            let modDate = (try? dir.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate

            found.append(DownloadedModel(
                id: modelId,
                cacheDirectory: dir,
                sizeBytes: size,
                modifiedDate: modDate
            ))
        }

        downloadedModels = found.sorted { ($0.modifiedDate ?? .distantPast) > ($1.modifiedDate ?? .distantPast) }
        totalDiskUsageBytes = found.reduce(0) { $0 + $1.sizeBytes }
    }

    /// Returns true if the model has been fully downloaded to local cache.
    public func isDownloaded(_ modelId: String) -> Bool {
        downloadedModels.contains(where: { $0.id == modelId })
    }

    /// Returns the downloaded model entry for a given ID, if available.
    public func downloadedModel(for modelId: String) -> DownloadedModel? {
        downloadedModels.first(where: { $0.id == modelId })
    }

    /// Delete a model from local cache, freeing disk space.
    public func delete(_ modelId: String) throws {
        guard let dir = Self.cacheDirectory(for: modelId) else { return }
        try FileManager.default.removeItem(at: dir)
        refresh()
    }

    /// Update active download progress (called by InferenceEngine during load).
    public func updateProgress(_ progress: ModelDownloadProgress) {
        activeDownloads[progress.modelId] = progress
    }

    /// Mark a download as complete.
    public func completeDownload(modelId: String) {
        activeDownloads.removeValue(forKey: modelId)
        refresh()
        lastLoadedModelId = modelId
    }

    /// Cancel an active download tracking entry.
    public func cancelDownload(modelId: String) {
        activeDownloads.removeValue(forKey: modelId)
    }

    // MARK: — Helpers

    private func directorySize(at url: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            total += Int64(size)
        }
        return total
    }
}
