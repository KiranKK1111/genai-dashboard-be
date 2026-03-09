"""
Query Coverage Verifier - Stage 5 of Query Understanding Pipeline

Verifies that generated SQL includes ALL semantic concepts from the original query.
This catches the exact bug described: missing temporal filters in generated SQL.

Example verification:
Query: "How many male and female clients have a birthday in January?"
Expected concepts: [count, gender filter, temporal month filter]
Generated SQL: "SELECT COUNT(*) FROM customers WHERE gender IN ('M','F')"
Verification: ❌ FAIL - Missing temporal month filter for "January"

This prevents incomplete SQL from being executed.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VerificationResult(Enum):
    """SQL verification results"""
    PASS = "pass"
    FAIL_MISSING_CONCEPTS = "fail_missing_concepts"
    FAIL_INCOMPLETE_FILTERS = "fail_incomplete_filters" 
    FAIL_WRONG_AGGREGATION = "fail_wrong_aggregation"


@dataclass
class ConceptCoverage:
    """Coverage analysis for a specific concept"""
    concept: str
    expected: bool
    found: bool
    evidence: Optional[str] = None  # SQL fragment that satisfies this concept
    confidence: float = 1.0


@dataclass
class CoverageReport:
    """Complete coverage verification report"""
    overall_result: VerificationResult
    concept_coverage: List[ConceptCoverage]
    missing_concepts: List[str]
    completeness_score: float  # 0.0 - 1.0
    issues: List[str]
    recommendations: List[str]
    
    def is_complete(self) -> bool:
        """Check if SQL covers all required concepts"""
        return self.overall_result == VerificationResult.PASS


class QueryCoverageVerifier:
    """
    Verifies that generated SQL includes all semantic concepts from user query.
    
    This is the CRITICAL missing validation that would have caught the temporal bug:
    - User mentioned "birthday in January"
    - Generated SQL has no month filter 
    - Verifier detects missing temporal concept
    - System regenerates or asks for clarification
    """
    
    def __init__(self):
        # Concept detection patterns
        self.temporal_indicators = {
            "birthday", "birth", "born", "dob", "birthdate",
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "jan", "feb", "mar", "apr", "may", "jun", 
            "jul", "aug", "sep", "oct", "nov", "dec",
            "month", "year", "date"
        }
        
        self.aggregation_indicators = {
            "how many": "COUNT",
            "count": "COUNT", 
            "total": "COUNT",
            "sum": "SUM",
            "average": "AVG",
            "maximum": "MAX",
            "minimum": "MIN"
        }
        
        self.gender_indicators = {
            "male", "female", "men", "women", "man", "woman", "gender"
        }
        
        logger.info("[COVERAGE_VERIFIER] Initialized concept detection patterns")
    
    def verify_sql_coverage(self, user_query: str, generated_sql: str, 
                          semantic_intent: Optional[Dict[str, Any]] = None) -> CoverageReport:
        """
        Verify that generated SQL covers all concepts in user query.
        
        Args:
            user_query: Original user query
            generated_sql: Generated SQL to verify
            semantic_intent: Optional structured intent for detailed verification
            
        Returns:
            CoverageReport with detailed analysis
        """
        query_lower = user_query.lower()
        sql_lower = generated_sql.lower()
        
        coverage_results = []
        missing_concepts = []
        issues = []
        recommendations = []
        
        # Check temporal coverage (THE KEY BUG DETECTOR)
        temporal_coverage = self._verify_temporal_coverage(query_lower, sql_lower)
        coverage_results.append(temporal_coverage)
        if not temporal_coverage.found and temporal_coverage.expected:
            missing_concepts.append("temporal_filter")
            issues.append("Query mentions date/month concepts but SQL has no temporal filters")
            recommendations.append("Add month/date filter (e.g., EXTRACT(MONTH FROM dob) = 1)")
        
        # Check aggregation coverage
        agg_coverage = self._verify_aggregation_coverage(query_lower, sql_lower)
        coverage_results.append(agg_coverage)
        if not agg_coverage.found and agg_coverage.expected:
            missing_concepts.append("aggregation")
            issues.append("Query asks for count/sum but SQL has no aggregation")
            recommendations.append("Add appropriate aggregation function")
        
        # Check gender coverage 
        gender_coverage = self._verify_gender_coverage(query_lower, sql_lower)
        coverage_results.append(gender_coverage)
        if not gender_coverage.found and gender_coverage.expected:
            missing_concepts.append("gender_filter")
            issues.append("Query mentions gender but SQL has no gender filter")
            recommendations.append("Add gender filter (e.g., WHERE gender IN ('M', 'F'))")
        
        # Calculate completeness score
        expected_concepts = sum(1 for c in coverage_results if c.expected)
        found_concepts = sum(1 for c in coverage_results if c.expected and c.found)
        completeness_score = found_concepts / expected_concepts if expected_concepts > 0 else 1.0
        
        # Determine overall result
        if completeness_score >= 1.0:
            overall_result = VerificationResult.PASS
        elif missing_concepts:
            overall_result = VerificationResult.FAIL_MISSING_CONCEPTS
        else:
            overall_result = VerificationResult.FAIL_INCOMPLETE_FILTERS
            
        return CoverageReport(
            overall_result=overall_result,
            concept_coverage=coverage_results,
            missing_concepts=missing_concepts,
            completeness_score=completeness_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _verify_temporal_coverage(self, query: str, sql: str) -> ConceptCoverage:
        """
        Verify temporal concept coverage - THE CRITICAL CHECK.
        
        This would catch the bug where:
        Query: "birthday in January" 
        SQL: No month filter
        Result: FAIL - missing temporal concept
        """
        # Check if query contains temporal indicators
        has_temporal_query = any(indicator in query for indicator in self.temporal_indicators)
        
        if not has_temporal_query:
            return ConceptCoverage(
                concept="temporal",
                expected=False,
                found=False,
                evidence="No temporal concepts in query"
            )
        
        # Check if SQL contains temporal filters
        temporal_patterns = [
            r"extract\s*\(\s*month\s+from",      # EXTRACT(MONTH FROM col)
            r"extract\s*\(\s*year\s+from",       # EXTRACT(YEAR FROM col)  
            r"date_part\s*\(\s*['\"]month['\"]", # DATE_PART('month', col)
            r"strftime\s*\(\s*['\"]%m['\"]",     # STRFTIME('%m', col)
            r"month\s*\(",                       # MONTH(col)
            r"year\s*\(",                        # YEAR(col)
            r"between\s+.*\s+and\s+.*date",      # Date ranges
            r">=.*date.*and.*<=.*date"           # Date comparisons
        ]
        
        temporal_evidence = None
        for pattern in temporal_patterns:
            match = re.search(pattern, sql, re.IGNORECASE)
            if match:
                temporal_evidence = match.group(0)
                break
        
        has_temporal_sql = temporal_evidence is not None
        
        return ConceptCoverage(
            concept="temporal",
            expected=True,
            found=has_temporal_sql,
            evidence=temporal_evidence or "No temporal filters found in SQL",
            confidence=0.9
        )
    
    def _verify_aggregation_coverage(self, query: str, sql: str) -> ConceptCoverage:
        """Verify aggregation function coverage"""
        # Check if query asks for aggregation
        expected_agg = None
        for phrase, agg_func in self.aggregation_indicators.items():
            if phrase in query:
                expected_agg = agg_func
                break
                
        if not expected_agg:
            return ConceptCoverage(
                concept="aggregation",
                expected=False,
                found=False,
                evidence="No aggregation requested in query"
            )
        
        # Check if SQL contains expected aggregation
        agg_pattern = rf"\b{expected_agg}\s*\("
        match = re.search(agg_pattern, sql, re.IGNORECASE)
        
        return ConceptCoverage(
            concept="aggregation",
            expected=True,
            found=match is not None,
            evidence=match.group(0) if match else f"Missing {expected_agg}() function",
            confidence=0.95
        )
    
    def _verify_gender_coverage(self, query: str, sql: str) -> ConceptCoverage:
        """Verify gender filter coverage"""
        # Check if query mentions gender
        has_gender_query = any(indicator in query for indicator in self.gender_indicators)
        
        if not has_gender_query:
            return ConceptCoverage(
                concept="gender",
                expected=False,
                found=False,
                evidence="No gender concepts in query"
            )
        
        # Check if SQL has gender filter
        gender_patterns = [
            r"gender\s*=",
            r"gender\s+in\s*\(",
            r"sex\s*=",
            r"sex\s+in\s*\("
        ]
        
        gender_evidence = None
        for pattern in gender_patterns:
            match = re.search(pattern, sql, re.IGNORECASE)
            if match:
                gender_evidence = match.group(0)
                break
        
        return ConceptCoverage(
            concept="gender", 
            expected=True,
            found=gender_evidence is not None,
            evidence=gender_evidence or "No gender filter found in SQL",
            confidence=0.9
        )
    
    def suggest_sql_improvements(self, report: CoverageReport, user_query: str, 
                               current_sql: str) -> Optional[str]:
        """
        Suggest SQL improvements based on coverage analysis.
        
        Returns improved SQL or None if current SQL is adequate.
        """
        if report.is_complete():
            return None
            
        # Start with current SQL
        improved_sql = current_sql
        query_lower = user_query.lower()
        
        # Add missing temporal filter
        if "temporal_filter" in report.missing_concepts:
            # Detect month mentioned in query
            months = {
                "january": 1, "january": 1, "jan": 1,
                "february": 2, "feb": 2,
                "march": 3, "mar": 3,
                "april": 4, "apr": 4,
                "may": 5,
                "june": 6, "jun": 6,
                "july": 7, "jul": 7,
                "august": 8, "aug": 8,
                "september": 9, "sep": 9,
                "october": 10, "oct": 10, 
                "november": 11, "nov": 11,
                "december": 12, "dec": 12
            }
            
            for month_name, month_num in months.items():
                if month_name in query_lower:
                    # Add month filter
                    if "where" in improved_sql.lower():
                        improved_sql = improved_sql.replace(
                            " LIMIT", f" AND EXTRACT(MONTH FROM dob) = {month_num} LIMIT"
                        )
                    else:
                        improved_sql = improved_sql.replace(
                            " LIMIT", f" WHERE EXTRACT(MONTH FROM dob) = {month_num} LIMIT"
                        )
                    break
        
        return improved_sql if improved_sql != current_sql else None


def get_coverage_verifier() -> QueryCoverageVerifier:
    """Get singleton instance of coverage verifier"""
    global _coverage_verifier
    if '_coverage_verifier' not in globals():
        _coverage_verifier = QueryCoverageVerifier()
    return _coverage_verifier