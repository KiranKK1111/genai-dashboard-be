"""
Dynamic File Processor - Intelligent chunking and vectorization of uploaded files.

Features:
- Dynamic chunk sizing based on content type
- Semantic chunking (splits at logical boundaries)
- Metadata preservation
- Support for multiple file types (PDF, TXT, CSV, JSON)
- Efficient memory usage
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ChunkingStrategy(Enum):
    """Different strategies for chunking content."""
    SEMANTIC = "semantic"      # Split at logical boundaries (paragraphs, sections)
    SENTENCE = "sentence"      # Split at sentence boundaries
    TOKEN = "token"            # Split by approximate token count
    HYBRID = "hybrid"          # Combine multiple strategies


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    content: str
    chunk_id: int
    file_name: str
    file_type: str
    start_position: int
    end_position: int
    metadata: Dict[str, Any]
    
    def get_char_count(self) -> int:
        """Get character count of chunk."""
        return len(self.content)
    
    def get_word_count(self) -> int:
        """Get approximate word count."""
        return len(self.content.split())
    
    def get_line_count(self) -> int:
        """Get line count."""
        return len(self.content.split('\n'))


class DynamicFileProcessor:
    """Intelligently processes and chunks document content."""
    
    def __init__(self):
        """Initialize processor with default settings."""
        # Dynamic chunk size based on content type (in characters)
        self.chunk_sizes = {
            "text": 1000,      # ~200 words
            "pdf": 1200,       # Slightly larger for PDFs
            "csv": 800,        # Smaller for structured data
            "json": 900,       # Medium for JSON
            "code": 600,       # Smaller for code snippets
        }
        
        # Overlap SIZE (in characters) to maintain context
        self.overlap_ratios = {
            "text": 0.2,       # 20% overlap
            "pdf": 0.15,       # 15% overlap
            "csv": 0.1,        # 10% overlap
            "json": 0.15,      # 15% overlap
            "code": 0.3,       # 30% overlap for code
        }
    
    async def process_file(
        self,
        file_content: str,
        file_name: str,
        file_type: str,
        strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    ) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """
        Process file and return chunks with metadata.
        
        Args:
            file_content: Raw file content
            file_name: Name of the file
            file_type: Type of file (text, pdf, csv, json)
            strategy: Chunking strategy to use
            
        Returns:
            Tuple of (chunks, metadata)
        """
        # Detect actual file type if not provided
        if not file_type:
            file_type = self._detect_file_type(file_content, file_name)
        
        # Choose chunking strategy
        if strategy == ChunkingStrategy.HYBRID:
            chunks = await self._hybrid_chunk(file_content, file_name, file_type)
        elif strategy == ChunkingStrategy.SEMANTIC:
            chunks = await self._semantic_chunk(file_content, file_name, file_type)
        elif strategy == ChunkingStrategy.SENTENCE:
            chunks = await self._sentence_chunk(file_content, file_name, file_type)
        else:  # TOKEN
            chunks = await self._token_chunk(file_content, file_name, file_type)
        
        # Generate metadata
        metadata = {
            "file_name": file_name,
            "file_type": file_type,
            "total_size_bytes": len(file_content.encode('utf-8')),
            "total_characters": len(file_content),
            "total_words": len(file_content.split()),
            "total_lines": len(file_content.split('\n')),
            "chunk_count": len(chunks),
            "strategy": strategy.value,
            "avg_chunk_size": len(file_content) // len(chunks) if chunks else 0,
        }
        
        return chunks, metadata
    
    async def _hybrid_chunk(
        self,
        content: str,
        file_name: str,
        file_type: str
    ) -> List[DocumentChunk]:
        """Hybrid chunking: semantic + fallback to fixed size."""
        # Try semantic chunking first
        semantic_chunks = await self._semantic_chunk(content, file_name, file_type)
        
        if semantic_chunks:
            return semantic_chunks
        
        # Fallback to token-based chunking
        return await self._token_chunk(content, file_name, file_type)
    
    async def _semantic_chunk(
        self,
        content: str,
        file_name: str,
        file_type: str
    ) -> List[DocumentChunk]:
        """
        Semantic chunking: split at logical boundaries.
        
        Strategies:
        - For markdown: split at headers (##, ###)
        - For text: split at paragraphs
        - For code: split at functions/classes
        - For lists: split at list items
        """
        chunks = []
        
        # For markdown/structured text
        if '\n##' in content or '\n###' in content:
            chunks = self._chunk_by_headers(content, file_name, file_type)
        # For paragraphs
        elif '\n\n' in content:
            chunks = self._chunk_by_paragraphs(content, file_name, file_type)
        # For lines/lists
        elif '\n' in content:
            chunks = self._chunk_by_lines(content, file_name, file_type)
        # Fallback: fixed size
        else:
            chunks = self._chunk_fixed_size(content, file_name, file_type)
        
        return chunks
    
    async def _sentence_chunk(
        self,
        content: str,
        file_name: str,
        file_type: str
    ) -> List[DocumentChunk]:
        """Chunk by sentences."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        chunks = []
        current_chunk = ""
        chunk_start = 0
        chunk_id = 0
        
        target_size = self.chunk_sizes.get(file_type, 1000)
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < target_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        chunk_id=chunk_id,
                        file_name=file_name,
                        file_type=file_type,
                        start_position=chunk_start,
                        end_position=chunk_start + len(current_chunk),
                        metadata={
                            "chunking_strategy": "sentence",
                            "sentence_count": len(current_chunk.split('. ')),
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                chunk_start += len(current_chunk)
                current_chunk = sentence + " "
        
        # Add remaining chunk
        if current_chunk:
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                chunk_id=chunk_id,
                file_name=file_name,
                file_type=file_type,
                start_position=chunk_start,
                end_position=chunk_start + len(current_chunk),
                metadata={
                    "chunking_strategy": "sentence",
                    "sentence_count": len(current_chunk.split('. ')),
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _token_chunk(
        self,
        content: str,
        file_name: str,
        file_type: str
    ) -> List[DocumentChunk]:
        """Chunk by approximate token count."""
        # Rough estimate: 1 token ≈ 4 characters
        target_chars = self.chunk_sizes.get(file_type, 1000)
        return self._chunk_fixed_size(content, file_name, file_type, target_size=target_chars)
    
    def _chunk_by_headers(
        self,
        content: str,
        file_name: str,
        file_type: str
    ) -> List[DocumentChunk]:
        """Chunk markdown by headers."""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = ""
        chunk_start = 0
        chunk_id = 0
        position = 0
        
        for i, line in enumerate(lines):
            # Check if this is a header (markdown)
            is_header = line.startswith('#')
            
            if is_header and current_chunk and chunk_id > 0:
                # Save previous chunk
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    chunk_id=chunk_id,
                    file_name=file_name,
                    file_type=file_type,
                    start_position=chunk_start,
                    end_position=position,
                    metadata={
                        "chunking_strategy": "headers",
                        "section": current_chunk.split('\n')[0][:50],
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
                chunk_start = position
                current_chunk = ""
            
            current_chunk += line + "\n"
            position += len(line) + 1
        
        # Add final chunk
        if current_chunk:
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                chunk_id=chunk_id,
                file_name=file_name,
                file_type=file_type,
                start_position=chunk_start,
                end_position=position,
                metadata={
                    "chunking_strategy": "headers",
                    "section": current_chunk.split('\n')[0][:50],
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_paragraphs(
        self,
        content: str,
        file_name: str,
        file_type: str
    ) -> List[DocumentChunk]:
        """Chunk by paragraphs (double newline)."""
        paragraphs = content.split('\n\n')
        chunks = []
        chunk_id = 0
        position = 0
        
        for para in paragraphs:
            if para.strip():
                chunk = DocumentChunk(
                    content=para.strip(),
                    chunk_id=chunk_id,
                    file_name=file_name,
                    file_type=file_type,
                    start_position=position,
                    end_position=position + len(para),
                    metadata={
                        "chunking_strategy": "paragraphs",
                        "word_count": len(para.split()),
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
                position += len(para) + 2
        
        return chunks
    
    def _chunk_by_lines(
        self,
        content: str,
        file_name: str,
        file_type: str
    ) -> List[DocumentChunk]:
        """Chunk by lines."""
        lines = content.split('\n')
        chunks = []
        chunk_id = 0
        position = 0
        current_chunk = ""
        target_size = self.chunk_sizes.get(file_type, 1000)
        
        for line in lines:
            if len(current_chunk) + len(line) < target_size:
                current_chunk += line + "\n"
            else:
                if current_chunk:
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        chunk_id=chunk_id,
                        file_name=file_name,
                        file_type=file_type,
                        start_position=position,
                        end_position=position + len(current_chunk),
                        metadata={
                            "chunking_strategy": "lines",
                            "line_count": len(current_chunk.split('\n')),
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    position += len(current_chunk)
                
                current_chunk = line + "\n"
        
        # Add remaining
        if current_chunk:
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                chunk_id=chunk_id,
                file_name=file_name,
                file_type=file_type,
                start_position=position,
                end_position=position + len(current_chunk),
                metadata={
                    "chunking_strategy": "lines",
                    "line_count": len(current_chunk.split('\n')),
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_fixed_size(
        self,
        content: str,
        file_name: str,
        file_type: str,
        target_size: int = None
    ) -> List[DocumentChunk]:
        """Chunk into fixed-size pieces with overlap."""
        target_size = target_size or self.chunk_sizes.get(file_type, 1000)
        overlap_ratio = self.overlap_ratios.get(file_type, 0.2)
        overlap_size = int(target_size * overlap_ratio)
        
        chunks = []
        chunk_id = 0
        position = 0
        
        while position < len(content):
            end_pos = min(position + target_size, len(content))
            chunk_content = content[position:end_pos]
            
            # Try to break at word boundary
            if end_pos < len(content):
                last_space = chunk_content.rfind(' ')
                if last_space > target_size * 0.7:  # If space is reasonably far
                    end_pos = position + last_space
                    chunk_content = content[position:end_pos]
            
            if chunk_content.strip():
                chunk = DocumentChunk(
                    content=chunk_content.strip(),
                    chunk_id=chunk_id,
                    file_name=file_name,
                    file_type=file_type,
                    start_position=position,
                    end_position=end_pos,
                    metadata={
                        "chunking_strategy": "fixed_size",
                        "overlap": overlap_ratio,
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move position forward (accounting for overlap)
            position = end_pos - overlap_size if chunk_id > 0 else end_pos
        
        return chunks
    
    def _detect_file_type(self, content: str, file_name: str) -> str:
        """Detect file type from content or name."""
        file_name_lower = file_name.lower()
        
        # Check by extension
        if file_name_lower.endswith('.pdf'):
            return "pdf"
        elif file_name_lower.endswith('.csv'):
            return "csv"
        elif file_name_lower.endswith('.json'):
            return "json"
        elif file_name_lower.endswith('.txt') or file_name_lower.endswith('.md'):
            return "text"
        
        # Check by content
        if content.strip().startswith('{') or content.strip().startswith('['):
            return "json"
        elif ',' in content and '\n' in content:
            return "csv"
        else:
            return "text"


async def create_file_processor() -> DynamicFileProcessor:
    """Factory function to create processor."""
    return DynamicFileProcessor()
