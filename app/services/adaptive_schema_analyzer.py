"""
Adaptive Schema Analyzer - Database-agnostic schema understanding.

This service analyzes database schema dynamically to:
- Detect column purposes (location, city, date, ID, etc.) WITHOUT hardcoding
- Learn from sample values what columns contain
- Match user queries to actual database structure
- Build intelligent LLM prompts that adapt to ANY schema

Works with ANY database - not hardcoded for specific tables/columns.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from .schema_discovery import SchemaCatalog, TableInfo, ColumnInfo

logger = logging.getLogger(__name__)


@dataclass
class ColumnPurpose:
    """Detected purpose of a column."""
    column_name: str
    purpose_type: str  # "location", "city", "date", "id", "name", "amount", "status", etc.
    confidence: float  # 0.0 to 1.0
    keywords_matched: List[str]
    sample_values: List[Any]
    reasoning: str


class AdaptiveSchemaAnalyzer:
    """
    Analyzes database schema to understand column purposes dynamically.
    
    Works with ANY database - discovers patterns instead of using hardcoded mappings.
    """
    
    # Generic patterns that work across databases
    COLUMN_PURPOSE_PATTERNS = {
        "location": {
            "keywords": ["state", "prov", "region", "region_code", "region_id", "area", "province", "state_code"],
            "description": "Geographic state/region/province column"
        },
        "city": {
            "keywords": ["city", "town", "locality", "municipality", "urban_center"],
            "description": "City/town/locality column"
        },
        "country": {
            "keywords": ["country", "nation", "country_code", "nation_code"],
            "description": "Country identifier column"
        },
        "district": {
            "keywords": ["district", "county", "shire", "taluk"],
            "description": "District/county administrative division"
        },
        "pincode": {
            "keywords": ["zip", "postal", "pincode", "postcode", "postal_code"],
            "description": "Postal/zip code column"
        },
        "date_created": {
            "keywords": ["created_at", "created_date", "creation_date", "date_created", "created"],
            "description": "Record creation timestamp"
        },
        "date_updated": {
            "keywords": ["updated_at", "updated_date", "modification_date", "last_updated"],
            "description": "Record update timestamp"
        },
        "date_range": {
            "keywords": ["from_date", "start_date", "end_date", "to_date", "date", "start", "end"],
            "description": "Date range field"
        },
        "identifier": {
            "keywords": ["id", "code", "_id", "identifier", "ref", "reference", "number"],
            "description": "Unique identifier or code"
        },
        "name": {
            "keywords": ["name", "title", "full_name", "first_name", "last_name"],
            "description": "Name or text identifier"
        },
        "amount": {
            "keywords": ["amount", "value", "price", "cost", "total", "sum", "balance"],
            "description": "Numeric amount/value"
        },
        "status": {
            "keywords": ["status", "state", "flag", "type", "category"],
            "description": "Status or category indicator"
        },
        "phone": {
            "keywords": ["phone", "mobile", "telephone", "contact_number"],
            "description": "Phone number"
        },
        "email": {
            "keywords": ["email", "mail", "e_mail", "address"],
            "description": "Email address"
        },
        "address": {
            "keywords": ["address", "street", "location", "line1", "line2", "road", "avenue"],
            "description": "Full or partial address"
        }
    }
    
    def __init__(self, schema_catalog: SchemaCatalog):
        self.schema_catalog = schema_catalog
    
    async def analyze_table(self, table_info: TableInfo) -> Dict[str, ColumnPurpose]:
        """Analyze all columns in a table to detect their purposes."""
        purposes = {}
        
        for col_name, col_info in table_info.columns.items():
            purpose = await self._detect_column_purpose(col_name, col_info)
            purposes[col_name] = purpose
        
        return purposes
    
    async def _detect_column_purpose(self, col_name: str, col_info: ColumnInfo) -> ColumnPurpose:
        """Detect purpose of a single column."""
        col_name_lower = col_name.lower()
        col_type_lower = col_info.data_type.lower()
        
        matched_keywords = []
        best_purpose = None
        best_confidence = 0.0
        
        # Check each pattern
        for purpose_type, pattern_info in self.COLUMN_PURPOSE_PATTERNS.items():
            keywords = pattern_info["keywords"]
            
            # Check for keyword matches in column name
            for keyword in keywords:
                if keyword in col_name_lower:
                    confidence = 0.7 + (len(keyword) / len(col_name_lower)) * 0.3  # Up to 1.0
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_purpose = purpose_type
                        matched_keywords = [keyword]
        
        # If no name match, try to infer from sample values
        if best_purpose is None and col_info.sample_values:
            best_purpose, confidence = await self._infer_from_samples(
                col_name, 
                col_info.sample_values
            )
            best_confidence = confidence
        
        # Fallback
        if best_purpose is None:
            best_purpose = "other"
            best_confidence = 0.0
            matched_keywords = []
        
        reasoning = f"Based on column name '{col_name}' and type '{col_info.data_type}'"
        if col_info.sample_values:
            reasoning += f" and {len(col_info.sample_values)} sample values"
        
        return ColumnPurpose(
            column_name=col_name,
            purpose_type=best_purpose,
            confidence=best_confidence,
            keywords_matched=matched_keywords,
            sample_values=col_info.sample_values[:5],  # Keep first 5
            reasoning=reasoning
        )
    
    async def _infer_from_samples(self, col_name: str, samples: List[Any]) -> tuple:
        """
        Infer column purpose from sample values using pattern-based detection.
        Uses column naming patterns and value characteristics instead of hardcoded lists.
        """
        
        # Convert samples to strings for analysis
        sample_strs = [str(s).lower() for s in samples if s is not None]
        if not sample_strs:
            return None, 0.0
        
        col_lower = col_name.lower()
        
        # Infer location type from column name patterns (not hardcoded values)
        if any(geo in col_lower for geo in ['state', 'province', 'region']):
            # 2-3 char uppercase abbreviations suggest state/province codes
            if any(2 <= len(s) <= 3 and s.isalpha() for s in sample_strs):
                return "location", 0.8
        
        # Infer city type from column name patterns
        if any(geo in col_lower for geo in ['city', 'town', 'municipality']):
            # Values that are proper nouns (capitalized text, no digits) suggest city names
            if any(s.replace(' ', '').isalpha() for s in sample_strs):
                return "city", 0.7
        
        # Check for phone-like patterns (10+ digit numbers)
        if any(len(s) >= 10 and s.replace('-', '').replace(' ', '').isdigit() for s in sample_strs):
            return "phone", 0.7
        
        # Check for zip/postal codes (5-6 digit patterns)
        if any(5 <= len(s) <= 6 and s.isdigit() for s in sample_strs):
            return "pincode", 0.6
        
        # Check for dates
        date_indicators = ["2020", "2021", "2022", "2023", "2024", "2025", "2026", "-"]
        if any(indicator in s for s in sample_strs for indicator in date_indicators):
            return "date_created", 0.6
        
        return None, 0.0
    
    async def match_user_intent_to_columns(self, 
                                          user_query: str,
                                          table_info: TableInfo,
                                          column_purposes: Dict[str, ColumnPurpose]) -> List[Dict[str, Any]]:
        """
        Match what user is asking for to actual database columns.
        
        Returns list of relevant columns with their purposes.
        """
        query_lower = user_query.lower()
        matched_columns = []
        
        # Detect intent from user query
        intent_keywords = {
            "location": ["location", "region", "state", "area", "province"],
            "city": ["city", "town", "locality"],
            "date": ["date", "when", "time", "year", "month"],
            "name": ["name", "called", "named"],
            "phone": ["phone", "contact", "mobile", "number"],
        }
        
        # Find what user is asking for
        user_intent = None
        for intent, keywords in intent_keywords.items():
            if any(kw in query_lower for kw in keywords):
                user_intent = intent
                break
        
        # If we detected an intent, find matching columns
        if user_intent:
            for col_name, purpose in column_purposes.items():
                # Match on purpose type
                if self._intent_matches_purpose(user_intent, purpose.purpose_type):
                    matched_columns.append({
                        "column": col_name,
                        "purpose": purpose.purpose_type,
                        "confidence": purpose.confidence,
                        "samples": purpose.sample_values
                    })
        
        # Also do direct term matching against column names and samples
        for col_name, col_info in table_info.columns.items():
            # Direct column name mention
            if col_name.lower() in query_lower:
                if not any(m["column"] == col_name for m in matched_columns):
                    matched_columns.append({
                        "column": col_name,
                        "purpose": column_purposes[col_name].purpose_type,
                        "confidence": 1.0,
                        "reason": "Direct column name match"
                    })
            
            # Check sample values
            samples = column_purposes[col_name].sample_values
            for sample in samples:
                if str(sample).lower() in query_lower:
                    if not any(m["column"] == col_name for m in matched_columns):
                        matched_columns.append({
                            "column": col_name,
                            "purpose": column_purposes[col_name].purpose_type,
                            "confidence": 0.9,
                            "reason": f"Sample value '{sample}' found in query"
                        })
        
        return matched_columns
    
    def _intent_matches_purpose(self, intent: str, purpose: str) -> bool:
        """Check if user intent matches a column purpose."""
        intent_purpose_map = {
            "location": ["location", "city", "district"],
            "city": ["city"],
            "date": ["date_created", "date_updated", "date_range"],
            "name": ["name"],
            "phone": ["phone"],
            "amount": ["amount"],
        }
        
        if intent in intent_purpose_map:
            return purpose in intent_purpose_map[intent]
        
        return False
    
    async def build_intelligent_llm_context(self, 
                                           table_name: str,
                                           user_query: str) -> str:
        """
        Build intelligent LLM context by:
        1. Discovering actual schema
        2. Detecting column purposes
        3. Finding relevant columns for user query
        4. Building focused context for LLM
        """
        # Get table info
        table_info = self.schema_catalog.database.tables.get(table_name)
        if not table_info:
            logger.warning(f"Table {table_name} not in schema catalog")
            return ""
        
        # Analyze column purposes
        column_purposes = await self.analyze_table(table_info)
        
        # Find relevant columns for this query
        relevant_columns = await self.match_user_intent_to_columns(
            user_query, 
            table_info, 
            column_purposes
        )
        
        # Build context
        context = f"Table: {table_name}\n"
        context += f"Total columns: {len(table_info.columns)}\n\n"
        
        if relevant_columns:
            context += "RELEVANT COLUMNS FOR THIS QUERY:\n"
            for col_info in relevant_columns:
                col_name = col_info["column"]
                purpose = col_info["purpose"]
                samples = col_info.get("samples", [])
                
                context += f"\n- {col_name} ({table_info.columns[col_name].data_type})\n"
                context += f"  Purpose: {purpose}\n"
                
                if samples:
                    sample_str = ", ".join(str(s) for s in samples[:3])
                    context += f"  Examples: {sample_str}\n"
        
        context += "\nALL AVAILABLE COLUMNS:\n"
        for col_name, col_info in table_info.columns.items():
            purpose = column_purposes[col_name]
            context += f"- {col_name} ({col_info.data_type})"
            
            if purpose.purpose_type != "other":
                context += f" [{purpose.purpose_type}]"
            
            # Show enum values for USER-DEFINED types
            if col_info.enum_values:
                enum_str = ", ".join(col_info.enum_values)
                context += f" ✓ Valid values: {enum_str}"
            elif col_info.sample_values:
                # Show actual sample values from database (not guessed values)
                sample_str = "', '".join(str(s) for s in col_info.sample_values)
                context += f" (Sample values in DB: '{sample_str}')"
            
            context += "\n"
        
        return context
