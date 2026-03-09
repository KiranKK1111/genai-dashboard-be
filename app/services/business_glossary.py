"""
BUSINESS GLOSSARY - Centralized business metadata and semantic mappings.

Provides:
- Business term definitions (learned from queries)
- Reusable metric formulas
- Persistent synonym learning
- LLM-driven term resolution (no hardcoded synonyms)

This bridges the gap between business language and database schema.
NO HARDCODED SYNONYMS - all semantic understanding comes from LLM.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..llm import call_llm

logger = logging.getLogger(__name__)


class DomainType(str, Enum):
    """Business domain types."""
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    ECOMMERCE = "ecommerce"
    MANUFACTURING = "manufacturing"
    GENERIC = "generic"


@dataclass
class MetricDefinition:
    """Business metric with calculation formula."""
    name: str
    definition: str
    formula: str  # SQL-like formula
    synonyms: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    unit: Optional[str] = None  # e.g., "USD", "count", "percentage"
    category: Optional[str] = None  # e.g., "revenue", "entity", "operational"
    created_at: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "definition": self.definition,
            "formula": self.formula,
            "synonyms": self.synonyms,
            "tables": self.tables,
            "columns": self.columns,
            "unit": self.unit,
            "category": self.category,
            "usage_count": self.usage_count,
        }


@dataclass
class TermMapping:
    """Mapping between business term and database elements."""
    term: str
    synonyms: Set[str] = field(default_factory=set)
    primary_table: Optional[str] = None
    primary_column: Optional[str] = None
    related_columns: Set[str] = field(default_factory=set)
    context: Optional[str] = None  # Context where this mapping applies
    confidence: float = 1.0
    learned: bool = False  # Whether this was learned from queries
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "synonyms": list(self.synonyms),
            "primary_table": self.primary_table,
            "primary_column": self.primary_column,
            "related_columns": list(self.related_columns),
            "context": self.context,
            "confidence": self.confidence,
            "learned": self.learned,
            "usage_count": self.usage_count,
        }


class BusinessGlossary:
    """
    Centralized business glossary for semantic mappings.
    
    Features:
    - Define business metrics with formulas
    - Learn term mappings from queries (no hardcoded synonyms)
    - LLM-driven term resolution
    - Context-aware term resolution
    
    NO HARDCODED DOMAIN SYNONYMS - all semantic matching is LLM-driven.
    """
    
    def __init__(self, domain: DomainType = DomainType.GENERIC):
        """Initialize glossary - no hardcoded terms loaded."""
        self.domain = domain
        self.metrics: Dict[str, MetricDefinition] = {}
        self.term_mappings: Dict[str, TermMapping] = {}
        
        # No hardcoded vocabulary loading - all mappings are learned
        logger.info(f"[GLOSSARY] Initialized for domain: {domain.value} (LLM-driven, no hardcoded synonyms)")
    
    async def resolve_term_llm(
        self,
        user_term: str,
        available_tables: List[str],
        available_columns: Dict[str, List[str]],
        context: Optional[str] = None
    ) -> Optional[TermMapping]:
        """
        Use LLM to resolve a business term to database elements.
        
        NO HARDCODED SYNONYMS - relies entirely on LLM semantic understanding.
        
        Args:
            user_term: Term from user query
            available_tables: List of available table names
            available_columns: Dict of table -> column names
            context: Query context for disambiguation
            
        Returns:
            TermMapping if LLM can resolve, None otherwise
        """
        # First check learned mappings
        user_term_lower = user_term.lower().strip()
        if user_term_lower in self.term_mappings:
            mapping = self.term_mappings[user_term_lower]
            mapping.usage_count += 1
            return mapping
        
        # Use LLM for semantic resolution
        tables_info = ", ".join(available_tables[:20])
        columns_sample = {t: cols[:10] for t, cols in list(available_columns.items())[:10]}
        
        prompt = f"""Resolve the business term "{user_term}" to database elements.

Available tables: {tables_info}
Sample columns: {json.dumps(columns_sample, indent=2)}
Domain: {self.domain.value}
Context: {context or 'general query'}

Rules:
- Match semantically to the most relevant table/column from the list
- Consider business domain context
- Return the most likely table and column

Return JSON:
{{"table": "table_name", "column": "column_name", "confidence": 0.9}}

If no good match, return: {{"table": null, "column": null, "confidence": 0.0}}"""

        try:
            response = await call_llm([
                {"role": "system", "content": "You are a business term resolver. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ], max_tokens=200, temperature=0.0)
            
            response_text = str(response).strip()
            if "{" in response_text:
                json_start = response_text.index("{")
                json_end = response_text.rindex("}") + 1
                result = json.loads(response_text[json_start:json_end])
                
                if result.get("table") and result.get("confidence", 0) > 0.5:
                    # Create a learned mapping for future use
                    mapping = TermMapping(
                        term=user_term_lower,
                        synonyms={user_term},
                        primary_table=result["table"],
                        primary_column=result.get("column"),
                        confidence=result.get("confidence", 0.7),
                        learned=True,
                        usage_count=1
                    )
                    # Don't persist by default - let learn_from_query handle that
                    return mapping
        except Exception as e:
            logger.warning(f"[GLOSSARY] LLM term resolution failed for '{user_term}': {e}")
        
        return None
    
    def define_metric(
        self,
        name: str,
        definition: str,
        formula: str,
        synonyms: Optional[List[str]] = None,
        tables: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        unit: Optional[str] = None,
        category: Optional[str] = None
    ) -> MetricDefinition:
        """
        Define a business metric with calculation formula.
        
        Args:
            name: Metric name (e.g., "revenue", "entity_count")
            definition: Human-readable definition
            formula: SQL-like formula (e.g., "SUM(sales_amount)")
            synonyms: Alternative names for this metric
            tables: Tables involved in calculation
            columns: Columns used in formula
            unit: Unit of measurement
            category: Metric category
            
        Returns:
            MetricDefinition object
        """
        metric = MetricDefinition(
            name=name,
            definition=definition,
            formula=formula,
            synonyms=synonyms or [],
            tables=tables or [],
            columns=columns or [],
            unit=unit,
            category=category,
        )
        
        self.metrics[name] = metric
        logger.info(f"[GLOSSARY] Defined metric: {name}")
        
        return metric
    
    def add_term_mapping(
        self,
        term: str,
        synonyms: Optional[Set[str]] = None,
        primary_table: Optional[str] = None,
        primary_column: Optional[str] = None,
        related_columns: Optional[Set[str]] = None,
        context: Optional[str] = None,
        confidence: float = 1.0
    ) -> TermMapping:
        """
        Add a term mapping to the glossary.
        
        Args:
            term: Business term
            synonyms: Alternative terms
            primary_table: Primary table for this term
            primary_column: Primary column for this term
            related_columns: Related columns
            context: Context where this mapping applies
            confidence: Confidence score (0-1)
            
        Returns:
            TermMapping object
        """
        mapping = TermMapping(
            term=term,
            synonyms=synonyms or set(),
            primary_table=primary_table,
            primary_column=primary_column,
            related_columns=related_columns or set(),
            context=context,
            confidence=confidence,
        )
        
        self.term_mappings[term] = mapping
        logger.info(f"[GLOSSARY] Added term mapping: {term}")
        
        return mapping
    
    def resolve_term(
        self,
        user_term: str,
        context: Optional[str] = None,
        tables_in_query: Optional[List[str]] = None
    ) -> Optional[TermMapping]:
        """
        Resolve a user's term to database elements.
        
        Args:
            user_term: Term from user query
            context: Query context for disambiguation
            tables_in_query: Tables mentioned in query
            
        Returns:
            TermMapping if found, None otherwise
        """
        user_term_lower = user_term.lower().strip()
        
        # Direct match
        if user_term_lower in self.term_mappings:
            mapping = self.term_mappings[user_term_lower]
            mapping.usage_count += 1
            return mapping
        
        # Synonym match
        for term, mapping in self.term_mappings.items():
            if user_term_lower in [s.lower() for s in mapping.synonyms]:
                mapping.usage_count += 1
                return mapping
        
        # Context-aware match (if tables provided)
        if tables_in_query:
            for term, mapping in self.term_mappings.items():
                if mapping.primary_table in tables_in_query:
                    if user_term_lower in term.lower() or any(user_term_lower in s.lower() for s in mapping.synonyms):
                        mapping.usage_count += 1
                        return mapping
        
        logger.debug(f"[GLOSSARY] No mapping found for term: {user_term}")
        return None
    
    def get_metric(self, metric_name: str) -> Optional[MetricDefinition]:
        """Get metric definition by name or synonym."""
        metric_name_lower = metric_name.lower()
        
        # Direct match
        if metric_name_lower in self.metrics:
            metric = self.metrics[metric_name_lower]
            metric.usage_count += 1
            return metric
        
        # Synonym match
        for name, metric in self.metrics.items():
            if metric_name_lower in [s.lower() for s in metric.synonyms]:
                metric.usage_count += 1
                return metric
        
        return None
    
    def learn_from_query(
        self,
        user_term: str,
        resolved_table: str,
        resolved_column: str,
        confidence: float = 0.8
    ):
        """
        Learn a new term mapping from successful query resolution.
        
        Args:
            user_term: Term used by user
            resolved_table: Table that was accessed
            resolved_column: Column that was accessed
            confidence: Confidence in this mapping
        """
        user_term_lower = user_term.lower().strip()
        
        # Check if mapping already exists
        if user_term_lower in self.term_mappings:
            # Update existing mapping
            mapping = self.term_mappings[user_term_lower]
            mapping.related_columns.add(resolved_column)
            mapping.usage_count += 1
            logger.info(f"[GLOSSARY] Updated mapping for: {user_term}")
        else:
            # Create new learned mapping
            mapping = TermMapping(
                term=user_term_lower,
                synonyms={user_term},
                primary_table=resolved_table,
                primary_column=resolved_column,
                confidence=confidence,
                learned=True,
                usage_count=1
            )
            self.term_mappings[user_term_lower] = mapping
            logger.info(f"[GLOSSARY] Learned new mapping: {user_term} → {resolved_table}.{resolved_column}")
    
    def export_glossary(self) -> Dict[str, Any]:
        """Export glossary as JSON-serializable dictionary."""
        return {
            "domain": self.domain.value,
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
            "term_mappings": {term: mapping.to_dict() for term, mapping in self.term_mappings.items()},
            "total_metrics": len(self.metrics),
            "total_mappings": len(self.term_mappings),
        }
    
    def get_most_used_terms(self, limit: int = 10) -> List[str]:
        """Get most frequently used terms."""
        sorted_terms = sorted(
            self.term_mappings.items(),
            key=lambda x: x[1].usage_count,
            reverse=True
        )
        return [term for term, _ in sorted_terms[:limit]]


# Global instance
_business_glossary: Optional[BusinessGlossary] = None


def get_business_glossary(domain: DomainType = DomainType.GENERIC) -> BusinessGlossary:
    """Get or create business glossary."""
    global _business_glossary
    if _business_glossary is None:
        _business_glossary = BusinessGlossary(domain=domain)
    return _business_glossary
