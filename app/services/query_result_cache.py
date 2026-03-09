"""
Query Result Cache - Cache results with smart invalidation.

Significantly reduces database load for repeated queries.
Features:
- Query normalization (catch semantic duplicates)
- Automatic TTL expiration
- Table-based dependency tracking
- Cache hit metrics
"""

from __future__ import annotations

import logging
import re
from hashlib import md5
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryResultCache:
    """
    Cache query results with intelligent invalidation.
    """
    
    def __init__(self, ttl_seconds: int = 300, max_cache_items: int = 1000):
        """
        Initialize cache with TTL.
        
        Args:
            ttl_seconds: Default time-to-live for cached results
            max_cache_items: Maximum number of items in cache
        """
        self.ttl_seconds = ttl_seconds
        self.max_cache_items = max_cache_items
        self._cache: Dict[str, Dict] = {}  # query_hash → cache entry
        self._metrics = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "invalidations": 0,
        }
    
    def generate_query_hash(self, sql: str) -> str:
        """
        Generate deterministic hash for SQL query.
        Normalize SQL first to catch semantically identical queries.
        """
        
        # Normalize SQL
        normalized_sql = self._normalize_sql(sql)
        
        # Generate hash
        hash_obj = md5(normalized_sql.encode())
        return hash_obj.hexdigest()
    
    async def cache_result(
        self,
        sql: str,
        result: List[Dict],
        depends_on_tables: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """
        Cache query result with table dependencies.
        """
        
        if ttl_seconds is None:
            ttl_seconds = self.ttl_seconds
        
        query_hash = self.generate_query_hash(sql)
        
        # Extract table dependencies if not provided
        if depends_on_tables is None:
            depends_on_tables = self._extract_tables_from_sql(sql)
        
        cache_entry = {
            "result": result,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(seconds=ttl_seconds),
            "depends_on_tables": depends_on_tables or [],
            "sql": sql,
            "result_count": len(result),
        }
        
        # Check cache size
        if len(self._cache) >= self.max_cache_items:
            self._evict_oldest()
        
        self._cache[query_hash] = cache_entry
        self._metrics["stores"] += 1
        
        logger.info(
            f"[CACHE-STORE] Cached {len(result)} rows "
            f"(hash={query_hash[:8]}, tables={depends_on_tables}, ttl={ttl_seconds}s)"
        )
        
        return query_hash
    
    async def get_cached_result(
        self,
        sql: str,
    ) -> Optional[List[Dict]]:
        """
        Retrieve cached result if valid and not expired.
        """
        
        query_hash = self.generate_query_hash(sql)
        
        if query_hash not in self._cache:
            logger.debug(f"[CACHE-MISS] Hash not found: {query_hash[:8]}")
            self._metrics["misses"] += 1
            return None
        
        cache_entry = self._cache[query_hash]
        now = datetime.now()
        
        # Check expiration
        if now > cache_entry["expires_at"]:
            logger.info(f"[CACHE-EXPIRE] Result expired (hash={query_hash[:8]})")
            del self._cache[query_hash]
            self._metrics["misses"] += 1
            return None
        
        logger.info(
            f"[CACHE-HIT] Returning {cache_entry['result_count']} cached rows "
            f"(age={(now - cache_entry['created_at']).total_seconds():.1f}s)"
        )
        
        self._metrics["hits"] += 1
        
        return cache_entry["result"]
    
    async def invalidate_table(self, table_name: str) -> int:
        """
        Invalidate all cached results for a table.
        Called after INSERT/UPDATE/DELETE on table.
        """
        
        invalidated_count = 0
        hashes_to_remove = []
        
        for query_hash, entry in self._cache.items():
            if table_name.lower() in [t.lower() for t in entry.get("depends_on_tables", [])]:
                hashes_to_remove.append(query_hash)
                invalidated_count += 1
        
        for query_hash in hashes_to_remove:
            del self._cache[query_hash]
        
        self._metrics["invalidations"] += invalidated_count
        
        logger.info(
            f"[CACHE-INVALIDATE] Table '{table_name}' modified, "
            f"invalidated {invalidated_count} cached queries"
        )
        
        return invalidated_count
    
    async def get_cache_metrics(self) -> Dict:
        """Get cache performance metrics."""
        
        total_requests = self._metrics["hits"] + self._metrics["misses"]
        hit_rate = (
            self._metrics["hits"] / total_requests if total_requests > 0 else 0.0
        )
        
        return {
            "hits": self._metrics["hits"],
            "misses": self._metrics["misses"],
            "hit_rate": hit_rate,
            "stores": self._metrics["stores"],
            "invalidations": self._metrics["invalidations"],
            "cache_size_items": len(self._cache),
        }
    
    def _normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL to catch semantic duplicates.
        """
        
        # Remove extra whitespace
        normalized = " ".join(sql.split())
        
        # Normalize IN clauses with single value
        # WHERE col IN ('X') → WHERE col = 'X'
        normalized = re.sub(
            r"IN\s*\(\s*'([^']+)'\s*\)",
            r"= '\1'",
            normalized,
            flags=re.IGNORECASE
        )
        
        # Remove LIMIT clauses (results with/without limit are different)
        # normalized = re.sub(r"LIMIT\s+\d+", "", normalized, flags=re.IGNORECASE)
        
        # Normalize case (uppercase for keywords)
        keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY", "LIMIT"]
        for kw in keywords:
            normalized = re.sub(f"\\b{kw}\\b", kw, normalized, flags=re.IGNORECASE)
        
        return normalized.upper()
    
    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL."""
        
        # Pattern: FROM table_name, JOIN table_name
        pattern = r"(?:FROM|JOIN|INNER|LEFT|RIGHT|FULL|CROSS)\s+(\w+)"
        tables = re.findall(pattern, sql, re.IGNORECASE)
        
        # De-duplicate and lowercase
        return list(set(t.lower() for t in tables))
    
    def _evict_oldest(self) -> None:
        """Remove oldest cache entry."""
        
        if not self._cache:
            return
        
        # Find entry with oldest created_at
        oldest_hash = min(
            self._cache.keys(),
            key=lambda h: self._cache[h]["created_at"]
        )
        
        del self._cache[oldest_hash]
        logger.debug(f"[CACHE-EVICT] Removed oldest entry to stay within limit")
    
    async def clear_all(self) -> int:
        """Clear entire cache."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"[CACHE-CLEAR] Cleared all {count} cached items")
        return count


# Singleton instance
_query_cache: Optional[QueryResultCache] = None


async def get_query_result_cache(
    ttl_seconds: int = 300,
    max_items: int = 1000,
) -> QueryResultCache:
    """Get or create query result cache."""
    global _query_cache
    
    if _query_cache is None:
        _query_cache = QueryResultCache(ttl_seconds=ttl_seconds, max_cache_items=max_items)
    
    return _query_cache
