# Changelog

All notable changes to MiRAGE will be documented in this file.

## [1.2.0] - 2026-01-21

### Added
- New checkpoint/resume functionality (`src/mirage/utils/checkpoint.py`)
- LLM response caching for improved performance (`src/mirage/utils/llm_cache.py`)
- Multi-hop visualization tool (`src/mirage/utils/visualize_multihop.py`)
- Pipeline visualization tool (`src/mirage/utils/visualize_pipeline.py`)
- Interactive QA generation visualization (`assets/mirage_qa_gen.html`)

### Changed
- Major README.md improvements with better documentation
- Enhanced prompts in `src/mirage/core/prompts.py`
- Improved QA generator with better chunk handling
- Updated context retrieval logic
- Cleaned up emoji usage in query headers and section titles
- Better "no relevant chunks" messaging

### Fixed
- Various bug fixes and stability improvements
- Improved error handling across modules

## [1.0.6] - 2026-01-14

### Added
- Python API section in README for programmatic usage
- Detailed project structure documentation

### Changed
- Complete codebase restructuring to `src/mirage/` package
- Professional package layout with proper submodules
- All internal imports now use `mirage.*` package structure

### Fixed
- Fixed all imports to use correct package paths
- Updated version numbers across all files

## [1.0.5] - 2026-01-13

### Fixed
- CRITICAL: Fixed infinite retry loop in `call_llm_simple`
- CRITICAL: Fixed incorrect attempt decrement in VLM functions
- Reduced default MAX_DEPTH and MAX_BREADTH to prevent runaway API calls
- Added circuit breaker in context completion

### Changed
- Improved CLI documentation
- Fixed optional dependencies installation

## [1.0.0] - 2026-01-06

### Added
- Initial release
- Multi-hop context completion agent
- QA generation pipeline with verification
- Multimodal support (text, tables, figures, images)
- Multiple backend support (Gemini, OpenAI, Ollama)
- Evaluation metrics (RAGAS integration)
- Deduplication pipeline

