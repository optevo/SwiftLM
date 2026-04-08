# Memory Handling — Feature Registry

## Scope
The `ExtractionService.cleanJSON()` function must reliably extract valid JSON from arbitrary LLM text output. Models routinely hallucinate markdown fences, conversational preambles, trailing commentary, partial structures, and nested formatting. This harness validates every known edge case.

## Features

| # | Feature | Status | Test Function | Last Verified |
|---|---------|--------|---------------|---------------|
| 1 | Extract JSON from hallucinated preamble + markdown fence | ✅ PASS | `testCleanJSON_withHallucinatedPreamble` | 2026-04-07 |
| 2 | Preserve nested internal braces without truncation | ✅ PASS | `testCleanJSON_withInternalNestedBraces` | 2026-04-07 |
| 3 | Pass through already-clean JSON unchanged | ✅ PASS | `testCleanJSON_withPerfectJSON` | 2026-04-07 |
| 4 | Handle missing closing bracket gracefully (no crash) | ✅ PASS | `testFeature4_HandleMissingClosingBracket` | 2026-04-07 |
| 5 | Handle JSON array root `[...]` (not just dict `{...}`) | ✅ PASS | `testFeature5_HandleArrayRootGracefully` | 2026-04-07 |
| 6 | Handle completely empty / whitespace-only input | ✅ PASS | `testFeature6_HandleEmptyWhitespace` | 2026-04-07 |
| 7 | Handle multiple JSON objects concatenated in output | ✅ PASS | `testFeature7_ConcatenateFragmentedDecodes` | 2026-04-07 |
| 8 | Handle unicode and emoji inside JSON string values | ✅ PASS | `testFeature8_PreserveUnicodeAndEmoji` | 2026-04-07 |
| 9 | Strip ANSI escape codes from terminal-pasted LLM output | ✅ PASS | `testFeature9_StripANSIEscapeSequences` | 2026-04-07 |
