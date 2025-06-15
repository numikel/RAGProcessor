# 🔍 RAGProcessor

RAGProcessor is a high-performance document processing and semantic search pipeline. It automatically processes various document types (text, images, audio), creates optimized text chunks, generates embeddings using OpenAI models, and stores them in a Qdrant vector database for efficient semantic search.

## 🚀 Features

✅ **Multi-format Document Processing** - Supports text (.txt, .md, .json, .csv, .html, .xml), code (.py, .js, .css), images (.png, .jpg, .jpeg, .webp), and audio/video files (.mp3, .mp4, .wav, .webm)  
✅ **Intelligent Text Chunking** - Advanced token-aware text splitting with 3-10x performance optimization through caching  
✅ **Image Processing** - Automatic image description using OpenAI Vision API (gpt-4.1-mini)  
✅ **Audio Transcription** - Audio/video content extraction using OpenAI Whisper API  
✅ **Semantic Search** - High-quality embeddings with OpenAI text-embedding-3-large model  
✅ **Vector Storage** - Efficient storage and retrieval using Qdrant vector database  
✅ **Parallel Processing** - Multi-threaded document processing for optimal performance  
✅ **Directory Analysis** - Smart file discovery with processing time estimation  
✅ **Colored Logging** - Enhanced console output with color-coded log levels (warnings in yellow, errors in red)  
✅ **Performance Monitoring** - Detailed statistics and cache performance metrics  
✅ **Strict Token Limits** - Configurable chunk size enforcement with 5% tolerance  

## 🛠 Requirements

Install dependencies using Poetry (Python ≥ 3.10 recommended):

```bash
poetry install
```

Or with pip:
```bash
pip install qdrant-client openai python-dotenv tiktoken pydantic
```

Configure environment variables in `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here  # Optional for local Qdrant
```

You will need:
- OpenAI API access for embeddings and multimodal processing
- Qdrant instance (local or cloud) for vector storage

## 📺 How to use

### Basic Usage

```python
from ragprocessor import VectorStoreService

# Initialize the service
vector_store = VectorStoreService(
    collection_name="my_documents",
    chunk_size=500,
    embedding_model="text-embedding-3-large",
    verbose=True,
    strict_chunking=True
)

# Create collection
vector_store.create_collection()

# Process documents from directory
stats = vector_store.add_documents(
    documents_path="path/to/your/documents",
    batch_size=5,
    chunk_validation=True
)

# Semantic search
results = vector_store.query(
    query="What is the main topic discussed?",
    k=5,
    score_threshold=0.7
)

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['text'][:200]}...")
    print(f"File: {result['file_name']}")
```

### Directory Analysis

```python
# Analyze directory before processing
dir_info = vector_store.get_directory_info("path/to/documents")
print(f"Found {dir_info['total_files']} files ({dir_info['total_size_mb']:.2f} MB)")

# Estimate processing time
estimates = vector_store.estimate_processing_time("path/to/documents")
print(f"Estimated time: {estimates['estimated_processing_time_minutes']:.1f} minutes")
```

### Advanced Text Splitting

```python
from ragprocessor import TextSplitter

# Configure optimized text splitter
splitter = TextSplitter(
    model_name="text-embedding-3-large",
    max_chunk_size=500,
    enable_debug=False  # Enable for detailed logging
)

# Split text with performance optimization
chunks = splitter.split_text(text, strict_limit=True)

# Get chunking statistics
stats = splitter.get_chunking_stats(chunks)
print(f"Cache efficiency: {stats['cache_hit_rate']:.1f}%")
```

## 🔹 Project Structure

```
RAGProcessor/
├── src/ragprocessor/
│   ├── __init__.py                    # Package initialization
│   ├── vector_store_service.py        # Main service class
│   ├── text_splitter.py              # Optimized text chunking
│   └── document_service.py           # Multi-format document processing
├── prompts/
│   └── describe_image.md              # Image analysis prompt template
├── logs/                              # Application logs (timestamped)
├── pyproject.toml                     # Poetry configuration
├── README.md                          # This file
└── .env                              # Environment variables (not versioned)
```

## 📂 Output

The system provides:

- **Semantic Search Results** with relevance scores, text content, and metadata
- **Processing Statistics** including chunk counts, token counts, and performance metrics
- **Colored Console Logs** for better debugging experience
- **Detailed Analytics** on file types, sizes, and processing estimates

## 🎯 Performance Features

- **Token Caching** - 3-10x performance improvement through intelligent caching
- **Parallel Processing** - Multi-threaded document handling
- **Memory Optimization** - Efficient chunk size management with binary search algorithms
- **Smart Chunking** - Context-aware text splitting respecting sentence and paragraph boundaries
- **Background Processing** - Non-blocking operations for large document sets

## 📍 Supported File Types

| Category | Extensions | Processing Method |
|----------|------------|-------------------|
| **Text Documents** | .txt, .md, .json, .csv, .html, .xml, .rst | Direct text extraction |
| **Code Files** | .py, .js, .css, .es | Syntax-aware processing |
| **Images** | .png, .jpg, .jpeg, .webp | OpenAI Vision API description |
| **Audio/Video** | .mp3, .mp4, .wav, .webm, .m4a, .mpeg, .mpga | OpenAI Whisper transcription |

## 🚀 Planned Features & Roadmap

The following features are planned for future releases:

- **Support for PDF Files**  
  Add robust PDF parsing and chunking, including extraction of text, images, and metadata.  
  _Planned: Integration with libraries such as PyPDF2 or pdfplumber._

- **Support for Microsoft Word Documents (.docx)**  
  Enable processing and semantic search for Word files, including tables and embedded images.  
  _Planned: Use of python-docx for extraction._

- **Support for Other Binary Formats**  
  Expand to additional formats such as PowerPoint (.pptx), Excel (.xlsx), and OpenDocument files.

- **Advanced Metadata Extraction**  
  Extract and index document metadata (author, creation date, etc.) for improved search and filtering.

> _Feel free to suggest or remove any features from this roadmap according to your needs!_

## 👤 Author

Made with ❤️ by Michał Kamiński

## 🧾 License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute it as you wish.

---

*RAGProcessor v1.0 - High-Performance Document Processing & Semantic Search Pipeline*
