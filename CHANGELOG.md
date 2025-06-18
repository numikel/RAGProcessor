# Changelog

All notable changes to RAGProcessor project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2024-12-19

### Added
- New `add_indexes()` method for automatic payload field indexing to improve query performance
- Support for metadata-based filtering in `query()` method via `filter_values` parameter
- Enhanced chunk statistics logging with detailed performance monitoring

### Changed
- **BREAKING**: `_get_keywords()` method now returns `List[str]` instead of comma-separated string
- Improved keyword extraction format for better metadata consistency
- Enhanced logging output with more detailed chunk processing statistics

### Improved
- Query performance through automatic indexing of payload fields
- Metadata handling with list-based keyword storage
- Development experience with better debugging information

### Technical Details
- Added automatic indexing for fields: `file_name`, `version_id`, `data_type`, `keywords`, `tokens`, `headers`, `urls`, `images`, `indexes`
- Modified `query()` method signature to accept `filter_values: List[Dict[str, str]]` parameter
- Enhanced performance logging in document processing pipeline

## [1.0.0] - 2024-12-18

### Added
- Initial stable release of RAGProcessor
- Multi-format document processing (text, images, audio/video)
- Intelligent text chunking with token-aware splitting
- OpenAI integration for embeddings and multimodal processing
- Qdrant vector database integration
- Parallel document processing
- Comprehensive logging and performance monitoring
- Directory analysis and processing time estimation
- Support for 15+ file formats

### Features
- High-performance text splitting with caching (3-10x improvement)
- Image description using OpenAI Vision API
- Audio transcription using OpenAI Whisper API
- Semantic search with configurable parameters
- Colored console logging for better debugging
- Strict token limit enforcement with 5% tolerance
- Metadata extraction and storage

---

*For more details about each release, see the [README.md](README.md) file.* 