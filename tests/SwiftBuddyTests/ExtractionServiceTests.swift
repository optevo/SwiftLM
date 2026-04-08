import XCTest
@testable import SwiftBuddy
@testable import MLXInferenceCore

final class ExtractionServiceTests: XCTestCase {
    
    // We cannot instantiate ExtractionService directly if it heavily relies on @MainActor Engine.
    // Instead, we will extract the cleanJSON core logic mechanically into a testable function
    // or test it via its internal components using Swift mirrors if needed.
    // For now, we replicate the pure Regex extraction logic to mathematically verify its safety bounds.
    
    func testCleanJSON_withPerfectJSON() {
        let input = """
        {
            "extractions": [{"test": "value"}]
        }
        """
        
        let output = JSONSanitizer.cleanJSON(input)
        XCTAssertTrue(output.starts(with: "{"))
        XCTAssertTrue(output.hasSuffix("}"))
        XCTAssertEqual(output, input)
    }
    
    func testCleanJSON_withHallucinatedPreamble() {
        let input = """
        Here is the JSON you requested master:
        ```json
        {
            "extractions": [{"test": "value"}]
        }
        ```
        And a quick reminder to eat your vegetables!
        """
        
        let output = JSONSanitizer.cleanJSON(input)
        let expected = """
        {
            "extractions": [{"test": "value"}]
        }
        """
        XCTAssertEqual(output, expected)
    }
    
    func testCleanJSON_withInternalNestedBraces() {
        let input = """
        {
            "extractions": [
                { "key": "{value}" },
                { "key2": "value2" }
            ]
        }
        """
        
        let output = JSONSanitizer.cleanJSON(input)
        XCTAssertEqual(output, input) // Internal braces should NOT truncate early
    }
    
    func testFeature4_HandleMissingClosingBracket() {
        let input = """
        {
            "extractions": [{"room": "bedroom"}
        """
        // Abrupty truncated text simulation
        let output = JSONSanitizer.cleanJSON(input)
        XCTAssertTrue(output.hasSuffix("}"))
        XCTAssertTrue(output.hasPrefix("{"))
    }
    
    func testFeature5_HandleArrayRootGracefully() {
        let input = """
        [
            {"room": "living_room", "fact": "is cold"}
        ]
        """
        // Array roots get automatically shifted under the static payload mapping scheme "extractions": []
        let output = JSONSanitizer.cleanJSON(input)
        XCTAssertTrue(output.contains("\"extractions\": ["))
    }
    
    func testFeature6_HandleEmptyWhitespace() {
        let empty = "     \n   "
        let output = JSONSanitizer.cleanJSON(empty)
        XCTAssertEqual(output, "{}")
    }
    
    func testFeature7_ConcatenateFragmentedDecodes() {
        let hallucinatedFragments = """
        { "room": "A", "fact": "apple" }
        { "room": "B", "fact": "banana" }
        { "room": "C", "fact": "cherry" }
        """
        let output = JSONSanitizer.cleanJSON(hallucinatedFragments)
        XCTAssertTrue(output.starts(with: "{ \"extractions\": ["))
        XCTAssertTrue(output.contains("},{"))
        XCTAssertTrue(output.hasSuffix("] }"))
    }
    
    func testFeature8_PreserveUnicodeAndEmoji() {
        let input = """
        { "extractions": [{"fact": "Simba likes 🍍 and ✨"}] }
        """
        let output = JSONSanitizer.cleanJSON(input)
        XCTAssertTrue(output.contains("🍍")) // Ensure truncation didn't break UTF bounds
        
        let data = output.data(using: .utf8)
        XCTAssertNotNil(data)
    }
    
    func testFeature9_StripANSIEscapeSequences() {
        // Red color code + reset simulating terminal paste anomalies
        let input = "\u{001B}[31m{\u{001B}[0m \"extractions\": [] \u{001B}[32m}\u{001B}[0m"
        let output = JSONSanitizer.cleanJSON(input)
        
        XCTAssertEqual(output, "{ \"extractions\": [] }")
        XCTAssertFalse(output.contains("\u{001B}"))
    }
}
