# Model Management — Feature Registry

## Scope
The HuggingFace model discovery and management system must allow users to search, filter, download, and manage MLX models. This harness validates the search API integration, the MLX filter toggle, the UI entry points, and the state management.

## Features

| # | Feature | Status | Test Function | Last Verified |
|---|---------|--------|---------------|---------------|
| 1 | Strict MLX filter queries HF with `library=mlx` param | ✅ PASS | `testStrictMLXFilterEnabled` | 2026-04-07 |
| 2 | Loose MLX filter appends "mlx" to search text | ✅ PASS | `testStrictMLXFilterDisabled` | 2026-04-07 |
| 3 | Empty query with strict MLX returns trending models | ✅ PASS | `testFeature3_EmptyQueryTrending` | 2026-04-07 |
| 4 | Search debounce prevents rapid-fire API calls | ✅ PASS | `testFeature4_DebounceBehavior` | 2026-04-07 |
| 5 | Load more pagination increments offset correctly | ✅ PASS | `testFeature5_Pagination` | 2026-04-07 |
| 6 | ModelManagementView shows "Search HuggingFace" button | ✅ PASS | UI Manual Check | 2026-04-07 |
| 7 | ModelManagementView empty state has search entry point | ✅ PASS | UI Manual Check | 2026-04-07 |
| 8 | Error state renders on network failure | ✅ PASS | `testFeature8_ErrorStateRendering` | 2026-04-07 |
| 9 | HFModelResult correctly parses param size hints | ✅ PASS | `testFeature9_ParamSizeParsing` | 2026-04-07 |
| 10 | HFModelResult correctly detects MoE architecture | ✅ PASS | `testFeature10_MoEDetection` | 2026-04-07 |
| 11 | RAM Budget strictly filters models via `modelsForDevice` | ✅ PASS | `testFeature11_RAMBudgetFiltersModels` | 2026-04-07 |
| 12 | Thermal throttling intercepts `.load()` and `.generate()` safely | ✅ PASS | `testFeature12_ThermalThrottles` | 2026-04-07 |
| 13 | App background triggers `.unload()` gracefully | ✅ PASS | `testFeature13_BackgroundEjection` | 2026-04-07 |
| 14 | MoE models automatically trigger `ExpertStreamingConfig` bypass | ✅ PASS | `testFeature14_SSDStreamingConfigBypass` | 2026-04-07 |
| 15 | ModelCatalog accurately parses TurboQuant footprint estimations | ✅ PASS | `testFeature15_TurboQuantFootprint` | 2026-04-07 |
