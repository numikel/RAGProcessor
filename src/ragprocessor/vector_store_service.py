from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI
from dotenv import load_dotenv
import os
import uuid
import mimetypes
from text_splitter import TextSplitter, TextChunk
from document_service import DocumentService
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime

load_dotenv()


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[37m',      # White
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Get the original formatted message
        formatted = super().format(record)
        
        # Add color based on log level
        level_name = record.levelname
        if level_name in self.COLORS:
            color = self.COLORS[level_name]
            reset = self.COLORS['RESET']
            # Color the entire line
            formatted = f"{color}{formatted}{reset}"
        
        return formatted


class VectorStoreService:
    """Service for managing vector storage operations using Qdrant and OpenAI.

    This service handles document processing, embedding generation, and vector storage
    operations using Qdrant as the vector database and OpenAI for embeddings and text processing.

    Args:
        collection_name (str): Name of the Qdrant collection
        chunk_size (int, optional): Maximum size of text chunks. Defaults to 500.
        embedding_model (str, optional): OpenAI embedding model name. Defaults to "text-embedding-3-large".
        embedding_dimension (int, optional): Dimension of embedding vectors. Defaults to 3072.
        qdrant_url (str, optional): Qdrant server URL. Defaults to QDRANT_URL env variable.
        qdrant_api_key (str, optional): Qdrant API key. Defaults to QDRANT_API_KEY env variable.
        openai_api_key (str, optional): OpenAI API key. Defaults to OPENAI_API_KEY env variable.
        verbose (bool, optional): Enable verbose logging. Defaults to False.
        enable_debug (bool, optional): Enable debug logging for text splitter (impacts performance). Defaults to False.
        strict_chunking (bool, optional): Enforce strict token limits with 5% tolerance. Defaults to True.
        
    Raises:
        ValueError: If required parameters are missing or invalid
        ConnectionError: If connection to Qdrant or OpenAI fails
    """

    def __init__(
        self,
        collection_name: str,
        chunk_size: int = 500,
        embedding_model: str = "text-embedding-3-large",
        embedding_dimension: int = 3072,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        verbose: bool = False,
        enable_debug: bool = False,
        strict_chunking: bool = True
    ):
        """Initialize VectorStoreService with optimized text splitting.
        
        Args:
            collection_name: Name of the Qdrant collection
            chunk_size: Maximum size of text chunks in tokens
            embedding_model: OpenAI embedding model name
            embedding_dimension: Dimension of embedding vectors
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key  
            openai_api_key: OpenAI API key
            verbose: Enable verbose logging
            enable_debug: Enable debug logging for text splitter (impacts performance)
            strict_chunking: Enforce strict token limits with 5% tolerance
        """
        # Validate required parameters
        if not collection_name or not collection_name.strip():
            raise ValueError("Collection name cannot be empty")
        
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
            
        if embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")

        # Get credentials from environment if not provided
        qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not qdrant_url:
            raise ValueError("QDRANT_URL must be provided either as parameter or environment variable")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be provided either as parameter or environment variable")

        self.logger = self._setup_logger(log_to_file=verbose)
        self.verbose = verbose
        self.strict_chunking = strict_chunking
        
        try:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
            # Test connection
            self.client.get_collections()
            self.logger.info("Successfully connected to Qdrant")
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")
        
        try:
            self.openai_client = OpenAI(api_key=openai_api_key)
            # Test connection with a simple request
            self.openai_client.models.list()
            self.logger.info("Successfully connected to OpenAI")
        except Exception as e:
            self.logger.error(f"Failed to connect to OpenAI: {e}")
            raise ConnectionError(f"Failed to connect to OpenAI: {e}")
            
        self.collection_name = collection_name.strip()
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        
        # Initialize optimized TextSplitter with performance enhancements
        self.text_splitter = TextSplitter(
            model_name=self.embedding_model,
            max_chunk_size=chunk_size,
            log_to_file=verbose,
            enable_debug=enable_debug  # Performance optimization
        )
        self.document_service = DocumentService()
        
        self.logger.info(f"VectorStoreService initialized with chunk_size={chunk_size}, strict_chunking={strict_chunking}, debug={enable_debug}")

    def create_collection(self) -> None:
        """Create a new collection in Qdrant if it doesn't exist.
        
        Raises:
            Exception: If collection creation fails
        """
        try:
            self.logger.info(f"Creating collection {self.collection_name}")
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Collection {self.collection_name} created successfully")
                self.add_indexes()
            else:
                self.logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            self.logger.error(f"Failed to create collection {self.collection_name}: {e}")
            raise

    def _embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents using OpenAI API.

        Args:
            documents (List[str]): List of text documents to embed

        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            ValueError: If documents list is empty
            Exception: If embedding generation fails
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
            
        # Filter out empty documents
        valid_documents = [doc for doc in documents if doc and doc.strip()]
        if not valid_documents:
            raise ValueError("No valid documents to embed")
            
        try:
            self.logger.info(f"Embedding {len(valid_documents)} documents")
            embeddings = self.openai_client.embeddings.create(
                input=valid_documents,
                model=self.embedding_model
            )
            return [embedding.embedding for embedding in embeddings.data]
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise

    def _get_summary(self, content: str) -> str:
        """Generate a concise summary of the content using OpenAI API.

        Args:
            content (str): Text content to summarize

        Returns:
            str: Generated summary (max 100 words)
            
        Raises:
            ValueError: If content is empty
            Exception: If summary generation fails
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
            
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",  # Updated to more recent model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes content for a vector database. "
                        "Summarize the content in a way that is useful for the vector database. "
                        "The summary should be short and to the point. No longer than 100 words."
                    },
                    {"role": "user", "content": content}
                ],
                max_tokens=200,
                temperature=0.2
            )
            summary = response.choices[0].message.content
            return summary if summary else "No summary available"
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")
            return "Summary generation failed"

    def _get_keywords(self, content: str) -> list:
        """Extract keywords from the content using OpenAI API.

        Args:
            content (str): Text content to extract keywords from

        Returns:
            list: Comma-separated keywords (max 10 words)
            
        Raises:
            ValueError: If content is empty
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
            
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",  # Updated to more recent model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts keywords from content for a vector database. "
                                  "Extract the keywords in a way that is useful for the vector database. "
                                  "Return only keywords separated by commas. Maximum 10 single words."
                    },
                    {"role": "user", "content": content}
                ],
                max_tokens=250,
                temperature=0.2
            )
            keywords = response.choices[0].message.content
            keywords = keywords.split(",")
            keywords = [keyword.strip() for keyword in keywords]
            return keywords if keywords else []
        except Exception as e:
            self.logger.error(f"Failed to extract keywords: {e}")
            return []

    def _get_mime_type(self, file_path: str) -> str:
        """Determine the MIME type of a file.

        Args:
            file_path (str): Path to the file

        Returns:
            str: MIME type of the file, defaults to "application/octet-stream" if unknown
        """          
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type if mime_type else "application/octet-stream"

    def _prepare_qdrant_point(
        self,
        chunk: TextChunk,
        point_id: str,
        file_path: str,
        version_id: str
    ) -> PointStruct:
        """Prepare a data point for insertion into Qdrant vector database.

        Args:
            chunk (TextChunk): Text chunk to be stored
            point_id (str): Unique identifier for the point
            file_path (str): Path to the source file
            version_id (str): Version identifier of the document

        Returns:
            PointStruct: Prepared point structure for Qdrant
            
        Raises:
            ValueError: If required parameters are missing
            Exception: If point preparation fails
        """
        if not chunk or not chunk.text:
            raise ValueError("Chunk and chunk text cannot be empty")
        if not point_id:
            raise ValueError("Point ID cannot be empty")
        if not file_path:
            raise ValueError("File path cannot be empty")
            
        try:
            metadata = {
                "file_name": os.path.basename(file_path),
                "version_id": version_id or None,
                "data_type": self._get_mime_type(file_path),
                "summary": self._get_summary(chunk.text),
                "keywords": self._get_keywords(chunk.text),
                "tokens": getattr(chunk.metadata, 'tokens', 0),
                "headers": getattr(chunk.metadata, 'headers', []),
                "urls": getattr(chunk.metadata, 'urls', []),
                "images": getattr(chunk.metadata, 'images', []),
                "indexes": getattr(chunk.metadata, 'indexes', [])
            }

            point = PointStruct(
                id=point_id,
                vector=self._embed_documents([chunk.text])[0],
                payload={
                    "text": chunk.text,
                    **metadata
                }
            )
            return point
        except Exception as e:
            self.logger.error(f"Failed to prepare Qdrant point: {e}")
            raise

    def _split_document(self, document_path: str) -> Tuple[List[TextChunk], str]:
        """Split a document into chunks using optimized text splitter.

        Args:
            document_path (str): Path to the document

        Returns:
            Tuple[List[TextChunk], str]: List of text chunks and version ID
            
        Raises:
            ValueError: If document path is invalid
            Exception: If document processing fails
        """
        if not document_path or not document_path.strip():
            raise ValueError("Document path cannot be empty")
            
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document not found: {document_path}")
            
        try:
            self.logger.info(f"Splitting document {document_path}")
            content, version_id = self._process_document(document_path)
            
            if not content:
                if document_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.logger.info(f"Image document {document_path} processed but returned no text content (this is normal for failed image processing)")
                else:
                    self.logger.warning(f"Text document {document_path} has no content")
                return [], version_id
            
            # Use optimized text splitter with strict limit enforcement
            chunks = self.text_splitter.split_text(
                content, 
                strict_limit=self.strict_chunking
            )
            
            # Log chunking statistics if verbose
            if self.verbose and chunks:
                stats = self.text_splitter.get_chunking_stats(chunks)
                self.logger.info(f"Document {document_path} split statistics:")
                self.logger.info(f"Total chunks: {stats['total_chunks']}")
                self.logger.info(f"Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
                self.logger.info(f"Token range: {stats['min_tokens']}-{stats['max_tokens']}")
                self.logger.info(f"Chunks over limit: {stats['chunks_over_limit']}")
                if stats['chunks_over_limit'] > 0:
                    efficiency = (1 - stats['chunks_over_limit'] / stats['total_chunks']) * 100
                    self.logger.warning(f"  - Chunking efficiency: {efficiency:.1f}%")
            
            self.logger.info(f"Document {document_path} split into {len(chunks)} chunks")
            return chunks, version_id
            
        except Exception as e:
            self.logger.error(f"Failed to split document {document_path}: {e}")
            raise

    def _process_document(self, document_path: str) -> Tuple[str, str]:
        """Process a document and return its content as text.

        Args:
            document_path (str): Path to the document

        Returns:
            Tuple[str, str]: Document content and version ID
            
        Raises:
            Exception: If document processing fails
        """
        try:
            return self.document_service.process_document(document_path)
        except Exception as e:
            self.logger.error(f"Failed to process document {document_path}: {e}")
            raise

    def add_documents(self, documents_path: Union[str, List[str]], batch_size: int = 10, chunk_validation: bool = True, file_extensions: List[str] = None) -> Dict[str, Any]:
        """Add documents to the vector store with automatic file discovery.

        Args:
            documents_path (Union[str, List[str]]): Path to directory containing documents OR list of file paths
            batch_size (int): Number of documents to process in parallel
            chunk_validation (bool): Whether to validate chunk token limits
            file_extensions (List[str]): File extensions to include (e.g., ['.txt', '.md', '.pdf']).
                                       If None, includes common text formats.
            
        Returns:
            Dict[str, Any]: Processing statistics and results
            
        Raises:
            ValueError: If path is invalid or no files found
            Exception: If document addition fails
        """
        if file_extensions is None:
            file_extensions = self.get_supported_extensions()
        
        # Convert to list of file paths
        if isinstance(documents_path, str):
            documents_paths = self._discover_files(documents_path, file_extensions)
            self.logger.info(f"Discovered {len(documents_paths)} files in directory: {documents_path}")
        else:
            documents_paths = documents_path
            self.logger.info(f"Processing {len(documents_paths)} provided file paths")
        
        if not documents_paths:
            raise ValueError("No documents found to process")
            
        # Filter out invalid paths
        valid_paths = [path for path in documents_paths if path and os.path.exists(path)]
        invalid_paths = [path for path in documents_paths if not os.path.exists(path)]
        
        if invalid_paths:
            self.logger.warning(f"Found {len(invalid_paths)} invalid file paths: {invalid_paths[:5]}...")
            
        if not valid_paths:
            raise ValueError("No valid document paths found")
            
        self.logger.info(f"Adding {len(valid_paths)} documents to collection {self.collection_name}")
        
        # Processing statistics
        start_time = time.time()
        total_chunks = 0
        total_tokens = 0
        oversized_chunks = 0
        processing_errors = []
        
        def process_document(document_path: str) -> Dict[str, Any]:
            """Process a single document and return statistics."""
            try:
                file_name = os.path.basename(document_path)
                chunks, version_id = self._split_document(document_path)
                
                if not chunks:
                    if document_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        self.logger.info(f"Image document {document_path} generated no text chunks (image processing may have failed)")
                    else:
                        self.logger.warning(f"Document {document_path} generated no text chunks")
                    return {"chunks": 0, "tokens": 0, "oversized": 0, "error": None}
                    
                points = []
                doc_tokens = 0
                doc_oversized = 0
                
                for chunk in chunks:
                    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_name}_{chunk.text[:50]}"))
                    point = self._prepare_qdrant_point(
                        chunk,
                        chunk_id,
                        document_path,
                        version_id
                    )
                    points.append(point)
                    doc_tokens += chunk.metadata.tokens
                    
                    # Check for oversized chunks if validation is enabled
                    if chunk_validation and chunk.metadata.tokens > self.text_splitter.max_chunk_size:
                        doc_oversized += 1
                
                # Batch insert points
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                self.logger.info(f"Document {document_path} processed: {len(chunks)} chunks, {doc_tokens} tokens")
                return {
                    "chunks": len(chunks), 
                    "tokens": doc_tokens, 
                    "oversized": doc_oversized,
                    "error": None
                }
                
            except Exception as e:
                error_msg = f"Failed to process document {document_path}: {e}"
                self.logger.error(error_msg)
                return {"chunks": 0, "tokens": 0, "oversized": 0, "error": error_msg}

        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(process_document, path): path for path in valid_paths}
            
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    total_chunks += result["chunks"]
                    total_tokens += result["tokens"]
                    oversized_chunks += result["oversized"]
                    
                    if result["error"]:
                        processing_errors.append(result["error"])
                        
                except Exception as e:
                    error_msg = f"Failed to process document {path}: {e}"
                    self.logger.error(error_msg)
                    processing_errors.append(error_msg)
                    
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Compile final statistics
        stats = {
            "input_path": documents_path,
            "files_discovered": len(documents_paths) if isinstance(documents_path, str) else len(documents_paths),
            "files_valid": len(valid_paths),
            "files_invalid": len(invalid_paths),
            "documents_processed": len(valid_paths),
            "documents_failed": len(processing_errors),
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "oversized_chunks": oversized_chunks,
            "processing_time_seconds": processing_time,
            "chunks_per_second": total_chunks / processing_time if processing_time > 0 else 0,
            "tokens_per_second": total_tokens / processing_time if processing_time > 0 else 0,
            "errors": processing_errors
        }
        
        # Log final results
        self.logger.info(f"Document processing completed for collection {self.collection_name}")
        self.logger.info(f"  - Files discovered: {stats['files_discovered']}")
        self.logger.info(f"  - Files valid: {stats['files_valid']}")
        self.logger.info(f"  - Documents processed: {stats['documents_processed']}")
        self.logger.info(f"  - Total chunks created: {stats['total_chunks']}")
        self.logger.info(f"  - Total tokens processed: {stats['total_tokens']}")
        self.logger.info(f"  - Processing time: {stats['processing_time_seconds']:.2f}s")
        self.logger.info(f"  - Performance: {stats['chunks_per_second']:.1f} chunks/s, {stats['tokens_per_second']:.1f} tokens/s")
        
        if stats['oversized_chunks'] > 0:
            efficiency = (1 - stats['oversized_chunks'] / stats['total_chunks']) * 100
            self.logger.warning(f"  - Oversized chunks: {stats['oversized_chunks']} ({efficiency:.1f}% efficiency)")
        
        if stats['documents_failed'] > 0:
            self.logger.error(f"  - Failed documents: {stats['documents_failed']}")
            
        return stats

    def _discover_files(self, directory_path: str, file_extensions: List[str]) -> List[str]:
        """Discover files in a directory with specified extensions.
        
        Args:
            directory_path (str): Path to directory to search
            file_extensions (List[str]): File extensions to include
            
        Returns:
            List[str]: List of discovered file paths
            
        Raises:
            ValueError: If directory doesn't exist or is not a directory
        """
        if file_extensions is None:
            file_extensions = self.get_supported_extensions()
            
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory does not exist: {directory_path}")
            
        if not os.path.isdir(directory_path):
            # If it's a single file, return it as a list
            if os.path.isfile(directory_path):
                return [directory_path]
            else:
                raise ValueError(f"Path is not a directory or file: {directory_path}")
        
        discovered_files = []
        
        # Walk through directory recursively
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in file_extensions:
                    discovered_files.append(file_path)
                    
        # Sort files for consistent processing order
        discovered_files.sort()
        
        self.logger.info(f"File discovery in {directory_path}:")
        self.logger.info(f"  - Extensions searched: {file_extensions}")
        self.logger.info(f"  - Files found: {len(discovered_files)}")
        
        if self.verbose and discovered_files:
            self.logger.info(f"  - Sample files: {discovered_files[:5]}")
            if len(discovered_files) > 5:
                self.logger.info(f"  - ... and {len(discovered_files) - 5} more")
                
        return discovered_files

    def add_documents_from_directory(self, directory_path: str, **kwargs) -> Dict[str, Any]:
        """Convenience method to add documents from a directory.
        
        Args:
            directory_path (str): Path to directory containing documents
            **kwargs: Additional arguments passed to add_documents
            
        Returns:
            Dict[str, Any]: Processing statistics and results
        """
        return self.add_documents(directory_path, **kwargs)

    def add_indexes(self) -> None:
        """Add indexes to the collection.
        
        Raises:
            Exception: If index addition fails
        """
        try:
            payload_mapping = {
                "file_name": "keyword",
                "version_id": "keyword",
                "data_type": "keyword",
                "keywords": "keyword",
                "tokens": "integer",
                "headers": "keyword",
                "urls": "keyword",
                "images": "keyword",
                "indexes": "integer"
            }
            for field in payload_mapping.keys():
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field, 
                    field_schema=payload_mapping[field]
                )
            self.logger.info(f"Indexes added to collection {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to add indexes to collection {self.collection_name}: {e}")
            raise

    def query(self, query: str, filter_values: list[dict] = None, k: int = 10, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search the vector store for similar documents.

        Args:
            query (str): Search query
            filter (dict, optional): Filter query by metadata. Defaults to None.
            k (int, optional): Number of results to return. Defaults to 10.
            score_threshold (float, optional): Minimum similarity score. Defaults to 0.0.

        Returns:
            List[Dict[str, Any]]: List of search results with scores and metadata
            
        Raises:
            ValueError: If query is empty or k is invalid
            Exception: If search fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        if k <= 0:
            raise ValueError("k must be positive")
        
        if filter_values:
            filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=filter_value["key"],
                        match=models.MatchValue(value=filter_value["value"]),
                    ) for filter_value in filter_values
                ]
            )
        else:
            filter = None
            
        try:
            self.logger.info(f"Querying the collection {self.collection_name} with the query: {query[:100]}...")
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=self._embed_documents([query])[0],
                limit=k,
                with_payload=True,  
                score_threshold=score_threshold,
                query_filter=filter
            ).points
            
            # Format results for better usability
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'score': result.score,
                    'text': result.payload.get('text', ''),
                    'file_name': result.payload.get('file_name', ''),
                    'summary': result.payload.get('summary', ''),
                    'keywords': result.payload.get('keywords', ''),
                    'metadata': {
                        'version_id': result.payload.get('version_id', ''),
                        'data_type': result.payload.get('data_type', ''),
                        'tokens': result.payload.get('tokens', 0),
                        'headers': result.payload.get('headers', []),
                        'urls': result.payload.get('urls', []),
                        'images': result.payload.get('images', []),
                        'indexes': result.payload.get('indexes', [])
                    }
                })
            
            self.logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
        except Exception as e:
            self.logger.error(f"Failed to query collection {self.collection_name}: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete the vector store collection.
        
        Raises:
            Exception: If collection deletion fails
        """
        try:
            self.logger.info(f"Deleting collection {self.collection_name}")
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
                self.logger.info(f"Collection {self.collection_name} deleted successfully")
            else:
                self.logger.info(f"Collection {self.collection_name} does not exist")
        except Exception as e:
            self.logger.error(f"Failed to delete collection {self.collection_name}: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.
        
        Returns:
            Dict[str, Any]: Collection information including size and status
        """
        try:
            if not self.client.collection_exists(self.collection_name):
                return {"exists": False, "message": "Collection does not exist"}
                
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "exists": True,
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.value
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {"exists": False, "error": str(e)}

    def _setup_logger(self, log_to_file: bool = False) -> logging.Logger:
        """
        Setup the logger for the VectorStoreService.

        Args:
            log_to_file (bool): If True, logs will also be written to a file.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Remove all handlers associated with the logger object (avoid duplicate logs)
        if logger.hasHandlers():
            logger.handlers.clear()

        # Colored formatter for console output
        colored_formatter = ColoredFormatter(
            fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Regular formatter for file output (no colors)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler with colors
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(colored_formatter)
        logger.addHandler(ch)

        # Optional file handler without colors
        if log_to_file:
            os.makedirs("logs", exist_ok=True)
            fh = logging.FileHandler(f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log", encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)

        return logger

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Get list of supported file extensions.
        
        Returns:
            List[str]: List of supported file extensions
        """
        # Supported file extensions for text, code, images, and audio
        return [
            '.txt', '.md', '.json', '.csv', '.html', '.xml', '.rst',  # text formats
            '.py', '.js', '.css', '.es',                              # code formats
            '.png', '.jpg', '.jpeg', '.webp',                        # image formats
            '.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm' # audio/video formats
        ]

    def get_directory_info(self, directory_path: str, file_extensions: List[str] = None) -> Dict[str, Any]:
        """Get information about files in a directory without processing them.
        
        Args:
            directory_path (str): Path to directory to analyze
            file_extensions (List[str]): File extensions to include
            
        Returns:
            Dict[str, Any]: Directory analysis information
        """
        if file_extensions is None:
            file_extensions = self.get_supported_extensions()
            
        # Initialize default return structure
        default_result = {
            "directory_path": directory_path,
            "total_files": 0,
            "total_size_bytes": 0,
            "total_size_mb": 0.0,
            "file_types": {},
            "sample_files": [],
            "extensions_searched": file_extensions,
            "error": None
        }
            
        try:
            files = self._discover_files(directory_path, file_extensions)
            
            # Analyze file sizes and types
            file_stats = {}
            total_size = 0
            
            for file_path in files:
                try:
                    size = os.path.getsize(file_path)
                    ext = os.path.splitext(file_path)[1].lower()
                    
                    if ext not in file_stats:
                        file_stats[ext] = {"count": 0, "total_size": 0}
                    
                    file_stats[ext]["count"] += 1
                    file_stats[ext]["total_size"] += size
                    total_size += size
                    
                except OSError as e:
                    self.logger.warning(f"Could not get size for file {file_path}: {e}")
                    continue
            
            return {
                "directory_path": directory_path,
                "total_files": len(files),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "file_types": file_stats,
                "sample_files": files[:10],  # First 10 files as sample
                "extensions_searched": file_extensions,
                "error": None
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing directory {directory_path}: {e}")
            default_result["error"] = str(e)
            return default_result

    def estimate_processing_time(self, directory_path: str, file_extensions: List[str] = None) -> Dict[str, Any]:
        """Estimate processing time for documents in a directory.
        
        Args:
            directory_path (str): Path to directory
            file_extensions (List[str]): File extensions to include
            
        Returns:
            Dict[str, Any]: Processing time estimates
        """
        # Default return structure
        default_result = {
            "directory_info": {"total_files": 0, "total_size_bytes": 0},
            "estimated_processing_time_seconds": 0.0,
            "estimated_processing_time_minutes": 0.0,
            "estimated_chunks": 0,
            "note": "Estimates are approximate and actual times may vary significantly",
            "error": None
        }
        
        try:
            dir_info = self.get_directory_info(directory_path, file_extensions)
            
            if dir_info.get("error"):
                default_result["error"] = dir_info["error"]
                default_result["directory_info"] = dir_info
                return default_result
            
            if dir_info["total_files"] == 0:
                default_result["directory_info"] = dir_info
                return default_result
            
            # Rough estimates based on file size and type
            # These are approximate and can vary significantly
            processing_rates = {
                ".txt": 50000,    # chars per second
                ".md": 40000,     # chars per second  
                ".json": 30000,   # chars per second
                ".csv": 60000,    # chars per second
                ".html": 35000,   # chars per second
                ".xml": 35000,    # chars per second
                ".rst": 40000,    # chars per second
                ".py": 45000,     # chars per second
                ".js": 45000,     # chars per second
                ".css": 50000,    # chars per second
                # Note: Images and audio files will be processed by OpenAI APIs
                # so processing times will be different and depend on API response times
                ".png": 1000,     # much slower due to API calls
                ".jpg": 1000,     # much slower due to API calls
                ".jpeg": 1000,    # much slower due to API calls
                ".webp": 1000,    # much slower due to API calls
                ".mp3": 500,      # very slow due to transcription API
                ".mp4": 300,      # very slow due to transcription API
                ".wav": 500,      # very slow due to transcription API
                ".webm": 300,     # very slow due to transcription API
            }
            
            estimated_time = 0
            
            for ext, stats in dir_info["file_types"].items():
                rate = processing_rates.get(ext, 30000)  # Default rate
                # Rough conversion: 1 byte ≈ 1 character for text files
                estimated_chars = stats["total_size"]
                estimated_time += estimated_chars / rate
            
            return {
                "directory_info": dir_info,
                "estimated_processing_time_seconds": estimated_time,
                "estimated_processing_time_minutes": estimated_time / 60,
                "estimated_chunks": dir_info["total_size_bytes"] // (self.text_splitter.max_chunk_size * 4),  # Rough estimate
                "note": "Estimates are approximate and actual times may vary significantly",
                "error": None
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating processing time for {directory_path}: {e}")
            default_result["error"] = str(e)
            return default_result