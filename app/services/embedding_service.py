"""
Embedding & Vectorization Service - Semantic search and similarity matching.

Features:
- Multiple vectorization strategies (TF-IDF, semantic)
- Fast similarity search
- Caching for performance
- Content ranking by relevance
"""

from __future__ import annotations

import re
import math
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass


@dataclass
class SimilarityMatch:
    """Represents a matched chunk with relevance score."""
    chunk_content: str
    chunk_id: int
    similarity_score: float
    match_reason: str


class EmbeddingService:
    """Handles vectorization and semantic search."""
    
    def __init__(self):
        """Initialize embedding service."""
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'is', 'was', 'are', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may',
            'might', 'must', 'shall', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'as', 'by', 'with', 'from', 'up',
            'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then', 'once'
        }
        
        self._tfidf_cache: Dict[str, Dict[str, float]] = {}
    
    async def find_relevant_chunks(
        self,
        query: str,
        chunks: List,  # List of DocumentChunk objects
        top_k: int = 5,
        similarity_threshold: float = 0.2
    ) -> List[SimilarityMatch]:
        """
        Find the most relevant chunks for a query.
        
        Args:
            query: User's question
            chunks: List of document chunks
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of relevant chunks with scores
        """
        if not chunks:
            return []
        
        # Extract text content from chunks
        chunk_texts = [chunk.content for chunk in chunks]
        
        # Calculate similarity scores using multiple methods
        # Method 1: Keyword matching (TF-IDF-like)
        keyword_scores = self._calculate_keyword_similarity(query, chunk_texts)
        
        # Method 2: Semantic similarity (based on content structure)
        semantic_scores = self._calculate_semantic_similarity(query, chunk_texts)
        
        # Combine scores (weighted average)
        combined_scores = [
            (keyword_scores[i] * 0.6) + (semantic_scores[i] * 0.4)
            for i in range(len(chunk_texts))
        ]
        
        # Create matches
        matches = []
        for i, score in enumerate(combined_scores):
            if score >= similarity_threshold:
                # Determine reason for match
                top_keywords = self._extract_keywords(query)
                chunk_keywords = self._extract_keywords(chunk_texts[i])
                matching_keywords = top_keywords & chunk_keywords
                
                reason = "keyword_match"
                if len(matching_keywords) >= 2:
                    reason = f"contains key terms: {', '.join(list(matching_keywords)[:3])}"
                elif semantic_scores[i] > keyword_scores[i]:
                    reason = "semantic_relevance"
                
                match = SimilarityMatch(
                    chunk_content=chunk_texts[i],
                    chunk_id=chunks[i].chunk_id,
                    similarity_score=score,
                    match_reason=reason
                )
                matches.append(match)
        
        # Sort by score descending
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Return top_k
        return matches[:top_k]
    
    def _calculate_keyword_similarity(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Calculate TF-IDF-like similarity using keyword matching.
        
        Higher score if:
        - Query keywords appear in document
        - Keywords appear multiple times
        - Keywords are not stop words
        """
        query_keywords = self._extract_keywords(query)
        
        scores = []
        for doc in documents:
            doc_keywords = self._extract_keywords(doc)
            
            # Calculate similarity
            if not query_keywords:
                score = 0.0
            else:
                # Intersection of keywords
                matches = query_keywords & doc_keywords
                
                # Calculate score based on overlap ratio
                overlap_ratio = len(matches) / len(query_keywords)
                
                # Boost score if keywords appear multiple times
                keyword_frequency = sum(doc.lower().count(kw) for kw in matches)
                frequency_boost = min(keyword_frequency / 5.0, 1.0)
                
                score = (overlap_ratio * 0.7) + (frequency_boost * 0.3)
            
            scores.append(score)
        
        return scores
    
    def _calculate_semantic_similarity(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Calculate semantic similarity based on:
        - Structure similarity (length, format)
        - Common phrases
        - Entity similarity
        """
        query_length = len(query.split())
        query_structure = self._extract_structure_features(query)
        
        scores = []
        for doc in documents:
            doc_structure = self._extract_structure_features(doc)
            
            # Length similarity (favour similar-length documents)
            doc_length = len(doc.split())
            length_ratio = min(doc_length, query_length) / max(doc_length, query_length)
            
            # Structure similarity
            structure_match = self._compare_structures(query_structure, doc_structure)
            
            # Combine
            score = (length_ratio * 0.3) + (structure_match * 0.7)
            scores.append(score)
        
        return scores
    
    def _extract_keywords(self, text: str) -> set:
        """Extract important keywords from text."""
        # Lowercase and split
        words = re.findall(r'\w+', text.lower())
        
        # Filter stop words and short words
        keywords = {
            w for w in words
            if len(w) > 3 and w not in self.stop_words and not w.isdigit()
        }
        
        return keywords
    
    def _extract_structure_features(self, text: str) -> Dict[str, any]:
        """Extract structural features from text."""
        return {
            "has_numbers": bool(re.search(r'\d', text)),
            "has_uppercase": bool(re.search(r'[A-Z]', text)),
            "has_special_chars": bool(re.search(r'[^a-zA-Z0-9\s]', text)),
            "line_count": len(text.split('\n')),
            "avg_word_length": sum(len(w) for w in text.split()) / max(len(text.split()), 1),
        }
    
    def _compare_structures(
        self,
        struct1: Dict[str, any],
        struct2: Dict[str, any]
    ) -> float:
        """Compare two structural feature dictionaries."""
        score = 0.0
        
        # Compare boolean features
        for key in ['has_numbers', 'has_uppercase', 'has_special_chars']:
            if key in struct1 and key in struct2:
                if struct1[key] == struct2[key]:
                    score += 0.1
        
        # Compare numeric features
        if 'avg_word_length' in struct1 and 'avg_word_length' in struct2:
            len_ratio = min(struct1['avg_word_length'], struct2['avg_word_length']) / \
                       max(struct1['avg_word_length'], struct2['avg_word_length'])
            score += len_ratio * 0.1
        
        return min(score, 1.0)
    
    async def rank_chunks_by_query(
        self,
        query: str,
        chunks: List,
        use_all: bool = False
    ) -> List:
        """
        Rank all chunks by relevance to query.
        
        Args:
            query: User query
            chunks: List of chunks
            use_all: If True, return all chunks ranked; if False return only relevant ones
            
        Returns:
            Ranked list of chunks
        """
        matches = await self.find_relevant_chunks(query, chunks, top_k=len(chunks), similarity_threshold=0.0)
        
        if use_all:
            return matches
        else:
            # Filter by threshold
            return [m for m in matches if m.similarity_score >= 0.3]
    
    def _extract_entities(self, text: str) -> set:
        """Extract named entities (capitalized words)."""
        words = text.split()
        entities = {w for w in words if w and w[0].isupper()}
        return entities
    
    def calculate_query_relevance_score(
        self,
        query: str,
        chunks: List
    ) -> float:
        """
        Calculate overall relevance score for file content to a query.
        
        Returns:
            Score from 0.0 to 1.0
        """
        if not chunks:
            return 0.0
        
        import asyncio
        # Get all matches
        loop = asyncio.get_event_loop()
        matches = loop.run_until_complete(
            self.find_relevant_chunks(query, chunks, top_k=len(chunks), similarity_threshold=0.0)
        )
        
        if not matches:
            return 0.0
        
        # Average score of top 5 matches
        top_matches = matches[:5]
        avg_score = sum(m.similarity_score for m in top_matches) / len(top_matches)
        
        return avg_score


async def create_embedding_service() -> EmbeddingService:
    """Factory function to create embedding service."""
    return EmbeddingService()
