"""
Value Normalizer - Maps user values to database canonical forms.

Handles value normalization without hardcoding:
- AP -> Andhra Pradesh (abbreviation -> full form)
- CUST0000001 -> CUST0000001 (identity)
- "New Delhi" -> "ND" or "Delhi" (depending on schema)
- Date format conversion (01/12/2024 -> 2024-01-12)

Uses database sampling + pattern matching to find the right normalization.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class ValueType(Enum):
    """Type of value user provided."""
    ABBREVIATION = "abbreviation"      # AP for Andhra Pradesh
    FULL_NAME = "full_name"            # Andhra Pradesh
    NUMERIC_ID = "numeric_id"          # 12345
    TEXT_CODE = "text_code"            # CUST0000001
    DATE = "date"                      # 2024-01-12 or 01/12/2024
    PARTIAL_MATCH = "partial_match"    # "Cust" could match "customer"


@dataclass
class NormalizationResult:
    """Result of value normalization."""
    original_value: str
    normalized_value: str
    column_name: str
    table_name: str
    confidence: float
    normalization_type: str
    sql_predicate: str  # Ready-to-use SQL WHERE clause
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original_value,
            "normalized": self.normalized_value,
            "column": self.column_name,
            "table": self.table_name,
            "confidence": round(self.confidence, 2),
            "type": self.normalization_type,
            "predicate": self.sql_predicate,
        }


class ValueNormalizer:
    """
    Normalizes user-provided values to match database format.
    
    Supports:
    - Abbreviation expansion (AP -> Andhra Pradesh)
    - Case normalization
    - Date format conversion
    - Fuzzy matching for partial values
    - Multi-option expansion (one value, multiple DB possibilities)
    """
    
    # Common state abbreviations (India)
    STATE_ABBREVIATIONS = {
        "AP": ["Andhra Pradesh", "A.P.", "Andhra"],
        "TS": ["Telangana", "T.S.", "TS"],
        "TN": ["Tamil Nadu", "T.N.", "Tamil Nadu"],
        "KA": ["Karnataka", "K.A.", "Karnatka"],
        "KL": ["Kerala", "K.L."],
        "MH": ["Maharashtra", "M.H."],
        "DL": ["Delhi", "D.L.", "New Delhi"],
        "WB": ["West Bengal", "W.B."],
        "UP": ["Uttar Pradesh", "U.P."],
        "MP": ["Madhya Pradesh", "M.P."],
        "GJ": ["Gujarat", "G.J."],
        "RJ": ["Rajasthan", "Raj"],
        "PB": ["Punjab", "P.B."],
        "HR": ["Haryana", "H.R."],
        "UT": ["Uttarakhand", "U.T."],
        "JK": ["Jammu & Kashmir", "J&K"],
        "HP": ["Himachal Pradesh", "H.P."],
        "NL": ["Nagaland", "N.L."],
        "ML": ["Meghalaya", "M.L."],
        "MN": ["Manipur", "M.N."],
        "AS": ["Assam", "A.S."],
        "AR": ["Arunachal Pradesh", "A.R."],
        "TR": ["Tripura", "T.R."],
        "SK": ["Sikkim", "S.K."],
        "OD": ["Odisha", "Orissa", "O.D."],
        "JH": ["Jharkhand", "J.H."],
        "CG": ["Chhattisgarh", "C.G."],
        "GG": ["Goa", "G.G."],
        "PN": ["Puducherry", "Pondicherry", "P.N."],
        "LA": ["Ladakh", "L.A."],
        "CH": ["Chandigarh", "C.H."],
        "AN": ["Andaman & Nicobar", "A&N"],
    }
    
    # Common business abbreviations
    BUSINESS_ABBREVIATIONS = {
        "AP": ["Accounts Payable", "Andhra Pradesh"],  # Context dependent
        "AR": ["Accounts Receivable", "Arunachal Pradesh"],
        "ERP": ["Enterprise Resource Planning"],
        "CRM": ["Customer Relationship Management"],
        "KPI": ["Key Performance Indicator"],
    }
    
    # Date patterns to try
    DATE_PATTERNS = [
        (r'^(\d{4})-(\d{2})-(\d{2})$', 'iso'),           # 2024-01-12
        (r'^(\d{2})/(\d{2})/(\d{4})$', 'dd/mm/yyyy'),    # 12/01/2024
        (r'^(\d{2})-(\d{2})-(\d{4})$', 'dd-mm-yyyy'),    # 12-01-2024
        (r'^(\d{4})/(\d{2})/(\d{2})$', 'yyyy/mm/dd'),    # 2024/01/12
    ]
    
    def __init__(self, schema_name: str = "public"):
        self.schema_name = schema_name
        self._normalization_cache: Dict[str, str] = {}
    
    async def normalize_value(
        self,
        value: str,
        table_name: str,
        column_name: str,
        db: AsyncSession,
        column_profile: Optional[Dict] = None,
    ) -> NormalizationResult:
        """
        Normalize a user-provided value to match database format.
        
        Try strategies in order:
        1. Exact match (value exists as-is)
        2. Case-insensitive match
        3. Abbreviation expansion
        4. Partial/fuzzy match
        5. Pattern-based (date conversion, etc.)
        
        Args:
            value: User-provided value
            table_name: Target table
            column_name: Target column
            db: Database session
            column_profile: Optional profile of the column (sample values, patterns)
            
        Returns:
            NormalizationResult with normalized value and SQL predicate
        """
        
        try:
            # Strategy 1: Exact match
            exact_match = await self._try_exact_match(db, table_name, column_name, value)
            if exact_match:
                sql_pred = f"{column_name} = '{exact_match}'"
                return NormalizationResult(
                    original_value=value,
                    normalized_value=exact_match,
                    column_name=column_name,
                    table_name=table_name,
                    confidence=1.0,
                    normalization_type="exact_match",
                    sql_predicate=sql_pred,
                )
            
            # Strategy 2: Case-insensitive match
            ci_match = await self._try_case_insensitive_match(db, table_name, column_name, value)
            if ci_match:
                sql_pred = f"{column_name} ILIKE '{ci_match}'"
                return NormalizationResult(
                    original_value=value,
                    normalized_value=ci_match,
                    column_name=column_name,
                    table_name=table_name,
                    confidence=0.95,
                    normalization_type="case_insensitive",
                    sql_predicate=sql_pred,
                )
            
            # Strategy 3: Abbreviation expansion
            abbr_matches = self._try_abbreviation_expansion(value)
            if abbr_matches:
                for expanded in abbr_matches:
                    expanded_match = await self._try_case_insensitive_match(
                        db, table_name, column_name, expanded
                    )
                    if expanded_match:
                        sql_pred = f"{column_name} ILIKE '{expanded_match}'"
                        return NormalizationResult(
                            original_value=value,
                            normalized_value=expanded_match,
                            column_name=column_name,
                            table_name=table_name,
                            confidence=0.85,
                            normalization_type="abbreviation_expansion",
                            sql_predicate=sql_pred,
                        )
            
            # Strategy 4: Partial/fuzzy match
            partial_match = await self._try_partial_match(db, table_name, column_name, value)
            if partial_match:
                sql_pred = f"{column_name} ILIKE '%{value}%'"
                return NormalizationResult(
                    original_value=value,
                    normalized_value=partial_match,
                    column_name=column_name,
                    table_name=table_name,
                    confidence=0.7,
                    normalization_type="partial_match",
                    sql_predicate=sql_pred,
                )
            
            # Strategy 5: Date conversion
            date_sql = self._try_date_format_conversion(value)
            if date_sql:
                return NormalizationResult(
                    original_value=value,
                    normalized_value=date_sql,
                    column_name=column_name,
                    table_name=table_name,
                    confidence=0.8,
                    normalization_type="date_conversion",
                    sql_predicate=f"{column_name}::date = '{date_sql}'",
                )
            
            # Fallback: Use value as-is with ILIKE
            logger.warning(f"No normalization found for {value} in {table_name}.{column_name}")
            return NormalizationResult(
                original_value=value,
                normalized_value=value,
                column_name=column_name,
                table_name=table_name,
                confidence=0.5,
                normalization_type="fallback_as_is",
                sql_predicate=f"{column_name} ILIKE '{value}'",
            )
            
        except Exception as e:
            logger.error(f"Normalization error: {e}")
            # Fallback to simple match
            return NormalizationResult(
                original_value=value,
                normalized_value=value,
                column_name=column_name,
                table_name=table_name,
                confidence=0.3,
                normalization_type="error_fallback",
                sql_predicate=f"{column_name} ILIKE '{value}'",
            )
    
    async def _try_exact_match(
        self,
        db: AsyncSession,
        table_name: str,
        column_name: str,
        value: str,
    ) -> Optional[str]:
        """Check if value exists exactly as-is."""
        try:
            query = text(f"""
                SELECT CAST({column_name} AS TEXT)
                FROM {self.schema_name}.{table_name}
                WHERE {column_name} = :val
                LIMIT 1
            """)
            result = await db.execute(query, {"val": value})
            found = result.scalar()
            return found
        except Exception:
            return None
    
    async def _try_case_insensitive_match(
        self,
        db: AsyncSession,
        table_name: str,
        column_name: str,
        value: str,
    ) -> Optional[str]:
        """Check if value exists case-insensitively."""
        try:
            query = text(f"""
                SELECT CAST({column_name} AS TEXT)
                FROM {self.schema_name}.{table_name}
                WHERE LOWER({column_name}::text) = LOWER(:val)
                LIMIT 1
            """)
            result = await db.execute(query, {"val": value})
            found = result.scalar()
            return found
        except Exception:
            return None
    
    async def _try_partial_match(
        self,
        db: AsyncSession,
        table_name: str,
        column_name: str,
        value: str,
    ) -> Optional[str]:
        """Check if value exists as partial match."""
        try:
            query = text(f"""
                SELECT CAST({column_name} AS TEXT)
                FROM {self.schema_name}.{table_name}
                WHERE {column_name}::text ILIKE :pattern
                LIMIT 1
            """)
            result = await db.execute(query, {"pattern": f"%{value}%"})
            found = result.scalar()
            return found
        except Exception:
            return None
    
    def _try_abbreviation_expansion(self, value: str) -> List[str]:
        """Expand abbreviations to possible full forms."""
        upper_val = value.upper()
        
        # Try state abbreviations first
        if upper_val in self.STATE_ABBREVIATIONS:
            return self.STATE_ABBREVIATIONS[upper_val]
        
        # Try business abbreviations
        if upper_val in self.BUSINESS_ABBREVIATIONS:
            # For business terms, prefer the first non-abbreviation expansion
            expanded = self.BUSINESS_ABBREVIATIONS[upper_val]
            # If any is geographic, it might be low confidence
            return expanded
        
        return []
    
    def _try_date_format_conversion(self, value: str) -> Optional[str]:
        """Try to convert various date formats to ISO."""
        for pattern, date_fmt in self.DATE_PATTERNS:
            match = re.match(pattern, value)
            if not match:
                continue
            
            groups = match.groups()
            
            try:
                if date_fmt == 'iso':
                    # Already correct
                    return value
                
                elif date_fmt == 'dd/mm/yyyy':
                    day, month, year = groups
                    return f"{year}-{month}-{day}"
                
                elif date_fmt == 'dd-mm-yyyy':
                    day, month, year = groups
                    return f"{year}-{month}-{day}"
                
                elif date_fmt == 'yyyy/mm/dd':
                    year, month, day = groups
                    return f"{year}-{month}-{day}"
            
            except Exception:
                continue
        
        return None
    
    def clear_cache(self) -> None:
        """Clear normalization cache."""
        self._normalization_cache.clear()
