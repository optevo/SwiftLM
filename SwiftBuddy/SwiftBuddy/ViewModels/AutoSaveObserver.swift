import Foundation

/// Feature 34: Auto-Save Hooks (every N messages)
/// Listens to chat sequence counts and triggers a flush callback to mine context without explicit user instruction.
public class AutoSaveObserver {
    private var messageCount: Int = 0
    private let triggerThreshold: Int
    
    public var onFlushRequired: (() -> Void)?
    
    public init(threshold: Int = 20) {
        self.triggerThreshold = threshold
    }
    
    public func recordMessage() {
        messageCount += 1
        if messageCount >= triggerThreshold {
            onFlushRequired?()
            messageCount = 0
        }
    }
    
    public func currentBufferCount() -> Int {
        return messageCount
    }
}
