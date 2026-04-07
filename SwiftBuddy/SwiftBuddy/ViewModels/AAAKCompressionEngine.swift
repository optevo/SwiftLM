import Foundation

/// AAAK Compression Dialect (Autonomous Aggressive Context Window Compression)
/// A heuristics-based local text processor that simulates extreme token reduction 
/// for use in context windows before LLM ingestion.
public struct AAAKCompressionEngine {
    
    /// Compresses a string by stripping stop words and replacing transitional grammar with mathematical symbols
    public static func compress(_ englishText: String) -> String {
        let stopWords: Set<String> = [
            "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in", "that", 
            "it", "for", "on", "as", "with", "at", "by", "this", "but", "they", "we"
        ]
        
        let words = englishText.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            
        var buffer: [String] = []
        
        for w in words {
            let lower = w.lowercased().trimmingCharacters(in: .punctuationCharacters)
            if stopWords.contains(lower) { continue }
            
            // Replace structural connections with AAAK syntax
            var packed = w
            packed = packed.replacingOccurrences(of: "because", with: "->")
            packed = packed.replacingOccurrences(of: "therefore", with: "=>")
            packed = packed.replacingOccurrences(of: "however", with: "||")
            
            buffer.append(packed)
        }
        
        // Add framing delimiters
        return "AAAK<< " + buffer.joined(separator: " ") + " >>"
    }
}
