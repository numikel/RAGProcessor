import tiktoken
import logging
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
import math
import re
import time


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

class TextChunkIndexes(BaseModel):
    """Represents the start and end indexes of a text chunk."""
    start_index: int
    end_index: int

class TextChunkMetadata(BaseModel):
    """Metadata associated with a text chunk."""
    tokens: int
    headers: dict
    urls: list[str]
    images: list[str]
    indexes: TextChunkIndexes

class TextChunk(BaseModel):
    """A text chunk with its associated metadata."""
    text: str
    metadata: TextChunkMetadata

class TextSplitter:
    """
    A class for splitting text into chunks based on token limits.
    
    This class uses tiktoken to count tokens and splits text into manageable
    chunks while respecting text structure (like line breaks).
    """
    
    def __init__(self, model_name: str, max_chunk_size: int = 1000, log_to_file: bool = False, enable_debug: bool = False):
        """
        Initialize the TextSplitter.
        
        Args:
            model_name: The name of the model to use for tokenization
            max_chunk_size: Maximum number of tokens per chunk
            log_to_file: Whether to log to a file in addition to console
            enable_debug: Whether to enable debug logging (can slow down processing)
        """
        self.model_name = model_name
        self.max_chunk_size = max_chunk_size
        self.enable_debug = enable_debug
        self.logger = self._setup_logger(log_to_file)
        self.tokenizer = self._setup_tokenizer(model_name)
        
        # Performance optimization: cache for token counts
        self._token_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _setup_tokenizer(self, model_name: str):   
        """
        Set up the tokenizer for the specified model.
        
        Args:
            model_name: The name of the model
            
        Returns:
            The tokenizer encoding object
        """
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            self.logger.info(f"Tokenizer initialized for model: {model_name}")
        except KeyError:
            self.logger.warning(f"Model {model_name} not found. Using o200k_base encoding.")
            encoding = tiktoken.get_encoding("o200k_base")
        return encoding
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text with caching.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            The number of tokens in the text
        """
        if self.tokenizer is None:
            self.logger.error("Tokenizer not initialized. Please initialize the TextSplitter with a valid model name.")
            return 0
        
        # Use hash of text as cache key for performance
        text_hash = hash(text)
        if text_hash in self._token_cache:
            self._cache_hits += 1
            return self._token_cache[text_hash]
        
        self._cache_misses += 1
        formatted_content = self._format_content(text)
        tokens = self.tokenizer.encode(formatted_content)
        token_count = len(tokens)
        
        # Cache result (limit cache size to prevent memory issues)
        if len(self._token_cache) < 1000:
            self._token_cache[text_hash] = token_count
        
        return token_count

    def _count_tokens_fast(self, text: str, start: int, end: int) -> int:
        """
        Fast token counting for substrings without caching overhead.
        
        Args:
            text: The full text
            start: Start position
            end: End position
            
        Returns:
            Number of tokens in the substring
        """
        if start >= end or start < 0 or end > len(text):
            return 0
        return self.count_tokens(text[start:end])
    
    def _debug_log(self, message: str):
        """Log debug message only if debug is enabled."""
        if self.enable_debug:
            self.logger.debug(message)

    def split_text(self, text: str, max_chunk_size: int = None, strict_limit: bool = True) -> list[TextChunk]:
        """
        Split text into chunks based on token limits.
        
        Args:
            text: The text to split
            max_chunk_size: Maximum tokens per chunk (uses instance default if None)
            strict_limit: If True, enforces hard token limit with 5% tolerance
            
        Returns:
            List of TextChunk objects
        """
        if max_chunk_size is None:
            max_chunk_size = self.max_chunk_size
        if self.tokenizer is None:
            self.logger.error("Tokenizer not initialized. Please initialize the TextSplitter with a valid model name.")
            return []
        
        # Reset cache stats
        self._cache_hits = 0
        self._cache_misses = 0
        
        total_tokens = self.count_tokens(text)
        self.logger.info(f"Starting text splitting. Total tokens: {total_tokens}, Max chunk size: {max_chunk_size}")
        
        chunk_position = 0
        chunks = []
        oversized_chunks = 0
        max_chunk_tokens = 0
        
        # Pre-calculate approximate characters per token for faster estimation
        chars_per_token = len(text) / max(total_tokens, 1)
        
        while chunk_position < len(text):
            self._debug_log(f"Processing chunk starting at position: {chunk_position}")
            chunk, new_chunk_position = self._get_chunk_optimized(text, chunk_position, max_chunk_size, chars_per_token)
            
            # Safety check to prevent infinite loops
            if new_chunk_position <= chunk_position:
                self.logger.error(f"Chunk position not advancing: {chunk_position} -> {new_chunk_position}")
                break
            
            chunk_tokens = self.count_tokens(chunk)
            
            # Track statistics
            max_chunk_tokens = max(max_chunk_tokens, chunk_tokens)
            if strict_limit and chunk_tokens > max_chunk_size:
                oversized_chunks += 1
                self.logger.warning(f"Chunk {len(chunks)+1} exceeds limit: {chunk_tokens} > {max_chunk_size} tokens")
            
            headers, urls, images = self._extract_metadata(chunk)

            chunks.append(TextChunk(
                text=chunk, 
                metadata=TextChunkMetadata(
                    tokens=chunk_tokens, 
                    headers=headers, 
                    urls=urls, 
                    images=images, 
                    indexes=TextChunkIndexes(
                        start_index=chunk_position, 
                        end_index=new_chunk_position
                    )
                )
            ))
            
            chunk_position = new_chunk_position
            self._debug_log(f"Chunk created with {chunk_tokens} tokens. Next chunk position: {chunk_position}")
        
        # Log final statistics
        avg_tokens = sum(chunk.metadata.tokens for chunk in chunks) / len(chunks) if chunks else 0
        self.logger.info(f"Text splitting completed. Total chunks: {len(chunks)}, Average tokens per chunk: {avg_tokens:.1f}, Max tokens in chunk: {max_chunk_tokens}, Oversized chunks: {oversized_chunks}")
        
        if self._cache_hits + self._cache_misses > 0:
            hit_rate = (self._cache_hits/(self._cache_hits+self._cache_misses)*100)
            self.logger.info(f"Cache performance: {self._cache_hits} hits, {self._cache_misses} misses, {hit_rate:.1f}% hit rate")
        
        if oversized_chunks > 0 and strict_limit:
            self.logger.warning(f"{oversized_chunks} chunks exceeded the token limit. Consider using a smaller chunk size or review text structure.")
        
        return chunks

    def _get_chunk_optimized(self, text: str, chunk_position: int, max_chunk_size: int, chars_per_token: float) -> tuple[str, int]:
        """
        Optimized version of _get_chunk with fewer token counting operations.
        
        Args:
            text: The full text
            chunk_position: Starting position for the chunk
            max_chunk_size: Maximum tokens allowed in the chunk
            chars_per_token: Pre-calculated ratio for estimation
            
        Returns:
            Tuple of (chunk_text, new_position)
        """
        if chunk_position >= len(text):
            return ("", chunk_position)
        
        start = chunk_position
        
        # Better initial estimation using pre-calculated ratio
        estimated_chunk_length = int(max_chunk_size * chars_per_token * 0.85)  # 85% safety margin
        end = min(start + estimated_chunk_length, len(text))
        
        # Ensure minimum chunk size to prevent single-word chunks
        min_chunk_length = max(100, int(max_chunk_size * chars_per_token * 0.1))  # At least 10% of max size
        if end - start < min_chunk_length and start + min_chunk_length < len(text):
            end = min(start + min_chunk_length, len(text))
        
        # Ensure we have at least some text
        if end == start and start < len(text):
            end = min(start + min_chunk_length, len(text))
        
        # Quick binary search to find approximate boundary
        left, right = start + min_chunk_length, min(start + int(max_chunk_size * chars_per_token * 1.2), len(text))
        best_end = max(start + min_chunk_length, end)
        
        # Binary search for optimal size (max 5 iterations for speed)
        iterations = 0
        while left <= right and iterations < 5:
            mid = (left + right) // 2
            tokens = self._count_tokens_fast(text, start, mid)
            
            if tokens <= max_chunk_size:
                best_end = mid
                left = mid + 1
            else:
                right = mid - 1
            iterations += 1
        
        end = best_end
        
        # Adjust for text structure (simplified for performance)
        end = self._adjust_chunk_end_fast(text, start, end, max_chunk_size)
        
        return (text[start:end], end)

    def _adjust_chunk_end_fast(self, text: str, start: int, end: int, max_chunk_size: int) -> int:
        """
        Fast version of chunk end adjustment with minimal token counting.
        
        Args:
            text: The full text
            start: Start position of the chunk
            end: Current end position of the chunk
            max_chunk_size: Maximum allowed tokens
            
        Returns:
            Adjusted end position
        """
        if end >= len(text):
            return len(text)
        if start >= end:
            return end
        
        # Calculate minimum acceptable chunk size (15% of max, but at least 50 characters)
        min_chunk_chars = max(50, int((end - start) * 0.15))
        
        # Quick structure-based adjustments without excessive token counting
        # Priority order optimized for speed
        
        # 1. Try previous paragraph (most likely to be good boundary)
        paragraph_pos = text.rfind("\n\n", start, end)
        if paragraph_pos != -1 and paragraph_pos > start:
            # Only check tokens if it looks promising
            potential_end = paragraph_pos + 2
            if (potential_end - start) >= min_chunk_chars:  # More lenient minimum size check
                tokens = self._count_tokens_fast(text, start, potential_end)
                if tokens <= max_chunk_size:
                    return potential_end
        
        # 2. Try previous newline
        newline_pos = text.rfind("\n", start, end)
        if newline_pos != -1 and newline_pos > start:
            potential_end = newline_pos + 1
            if (potential_end - start) >= min_chunk_chars:
                tokens = self._count_tokens_fast(text, start, potential_end)
                if tokens <= max_chunk_size:
                    return potential_end
        
        # 3. Try sentence endings (simplified search)
        for ending in ['. ', '! ', '? ']:
            pos = text.rfind(ending, start, end)
            if pos != -1 and pos > start:
                potential_end = pos + len(ending)
                if (potential_end - start) >= min_chunk_chars:
                    tokens = self._count_tokens_fast(text, start, potential_end)
                    if tokens <= max_chunk_size:
                        return potential_end
        
        # 4. Try word boundary - always check if reasonable size
        space_pos = text.rfind(" ", start, end)
        if space_pos != -1 and space_pos > start:
            potential_end = space_pos + 1
            # More lenient check for word boundaries
            if (potential_end - start) >= min(min_chunk_chars, 20):  # At least 20 chars for word boundary
                tokens = self._count_tokens_fast(text, start, potential_end)
                if tokens <= max_chunk_size:
                    return potential_end
        
        # 5. Final safety check - if current chunk is too big, force smaller
        current_tokens = self._count_tokens_fast(text, start, end)
        if current_tokens > max_chunk_size * 1.05:  # 5% tolerance
            # Binary search for safe cut point
            safe_end = self._binary_search_safe_cut(text, start, end, max_chunk_size)
            return safe_end
        
        return end

    def _binary_search_safe_cut(self, text: str, start: int, end: int, max_tokens: int) -> int:
        """
        Binary search to find safe cut point that doesn't exceed token limit.
        
        Args:
            text: The full text
            start: Start position
            end: End position
            max_tokens: Maximum allowed tokens
            
        Returns:
            Safe end position
        """
        # Ensure minimum chunk size
        min_chunk_chars = max(50, int((end - start) * 0.1))  # At least 10% of original size or 50 chars
        min_end = min(start + min_chunk_chars, end)
        
        left, right = min_end, end
        best_end = min_end
        
        # Limit iterations for performance
        for _ in range(6):  # Max 6 iterations
            if left > right:
                break
                
            mid = (left + right) // 2
            
            # Find word boundary at or before mid
            word_boundary = text.rfind(" ", start, mid)
            if word_boundary == -1 or word_boundary < min_end:
                word_boundary = mid
            else:
                word_boundary += 1
            
            # Ensure we don't go below minimum
            word_boundary = max(word_boundary, min_end)
            
            tokens = self._count_tokens_fast(text, start, word_boundary)
            
            if tokens <= max_tokens:
                best_end = word_boundary
                left = mid + 1
            else:
                right = mid - 1
        
        return best_end

    def get_chunking_stats(self, chunks: list[TextChunk]) -> dict:
        """
        Get detailed statistics about the chunking results.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {"error": "No chunks provided"}
        
        token_counts = [chunk.metadata.tokens for chunk in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "median_tokens": sorted(token_counts)[len(token_counts) // 2],
            "chunks_over_limit": sum(1 for tokens in token_counts if tokens > self.max_chunk_size),
            "token_distribution": {
                "0-250": sum(1 for t in token_counts if t <= 250),
                "251-500": sum(1 for t in token_counts if 250 < t <= 500),
                "501-750": sum(1 for t in token_counts if 500 < t <= 750),
                "751-1000": sum(1 for t in token_counts if 750 < t <= 1000),
                "1000+": sum(1 for t in token_counts if t > 1000)
            },
            "chunk_size_limit": self.max_chunk_size
        }
        
        return stats
   
    def _extract_metadata(self, text: str) -> tuple[dict, list[str], list[str]]:
        """
        Extract metadata from the text chunk.
        
        Args:
            text: The text chunk to analyze
            
        Returns:
            Tuple of (headers, urls, images)
        """
        import re
        
        headers = {}
        urls = []
        images = []
        
        # Extract markdown headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        for match in re.finditer(header_pattern, text, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            if level not in headers:
                headers[level] = []
            headers[level].append(title)
        
        # Extract URLs (basic pattern)
        url_pattern = r'https?://[^\s\)]+|www\.[^\s\)]+'
        urls = re.findall(url_pattern, text)
        
        # Extract image references
        image_pattern = r'!\[.*?\]\((.*?)\)'
        images = re.findall(image_pattern, text)
        
        return headers, urls, images

    def _format_content(self, text: str) -> str:
        """
        Format content for token counting with chat format.
        
        Args:
            text: The text to format
            
        Returns:
            Formatted text string
        """
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant<|im_end|>"
    
    def _setup_logger(self, log_to_file: bool = False) -> logging.Logger:
        """
        Setup the logger for the TextSplitter.

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
            fh = logging.FileHandler("vector_store_service.log", encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)

        return logger
    
if __name__ == "__main__":
    import time
    
    # Performance test for TextSplitter
    print("=== TextSplitter Performance Test ===")
    
    # Create longer test content
    base_text = """
    Sektor C jest jednym z najbardziej strzeżonych obszarów fabryki, przeznaczonym do testowania nowoczesnej broni. To rozległe, wzmocnione pomieszczenie, wyposażone w zaawansowane systemy bezpieczeństwa, mające na celu ochronę przed przypadkowym wyciekiem materiałów niebezpiecznych lub niekontrolowanymi eksplozjami. Znajdują się tutaj specjalnie przygotowane stanowiska testowe, otoczone grubymi ścianami oraz osłonami przeciwodłamkowymi, które pozwalają na prowadzenie prób z bronią o dużej mocy. Sektor wyposażono w automatyczne systemy monitorujące, rejestrujące każdy test i analizujące dane na bieżąco. Wstęp do sektora jest mocno ograniczony, a dostęp mają jedynie uprawnieni technicy i inżynierowie, wyposażeni w specjalne identyfikatory oraz odzież ochronną. Prace w Sektorze C są ściśle tajne, a każdy test broni podlega natychmiastowej archiwizacji i raportowaniu do wyższych przełożonych."""
    
    print(f"Test text length: {len(base_text)} characters")
    
    # Test optimized version
    print("\n--- Testing Optimized TextSplitter ---")
    start_time = time.time()
    
    splitter = TextSplitter(model_name="gpt-4.1-mini", max_chunk_size=500, enable_debug=True)
    chunks = splitter.split_text(base_text, strict_limit=True)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Processing time: {processing_time:.3f} seconds")
    print(f"Number of chunks: {len(chunks)}")
    
    # Get and display statistics
    stats = splitter.get_chunking_stats(chunks)
    print(f"\n--- Statistics ---")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
    print(f"Min tokens: {stats['min_tokens']}")
    print(f"Max tokens: {stats['max_tokens']}")
    print(f"Chunks over limit: {stats['chunks_over_limit']}")
    
    # Show token distribution
    print(f"\nToken distribution:")
    for range_key, count in stats["token_distribution"].items():
        if count > 0:
            print(f"  {range_key} tokens: {count} chunks")
    
    # Calculate efficiency
    efficiency = (1 - stats['chunks_over_limit'] / stats['total_chunks']) * 100 if stats['total_chunks'] > 0 else 0
    print(f"\nEfficiency (chunks within limit): {efficiency:.1f}%")
    
    # Show first few chunks as examples
    print(f"\n--- Sample Chunks ---")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\nChunk {i} ({chunk.metadata.tokens} tokens):")
        preview = chunk.text.strip()[:100].replace('\n', ' ')
        print(f"  Text: {preview}...")
        
    print(f"\nPerformance: {len(chunks)/processing_time:.1f} chunks/second")
    print(f"Speed: {stats['total_tokens']/processing_time:.1f} tokens/second")