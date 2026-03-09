"""
Evaluation Harness - Test suite for measuring system accuracy.

Tracks:
- Query parsing success rate
- SQL execution success rate
- Follow-up retention correctness
- Ambiguity handling accuracy
- Schema understanding
"""

from __future__ import annotations

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Test result status."""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    PARTIAL = "partial"


@dataclass
class TestCase:
    """A single test case."""
    id: str
    name: str
    description: str
    user_query: str
    expected_tables: List[str] = field(default_factory=list)
    expected_sql_pattern: Optional[str] = None  # Regex pattern for expected SQL
    expected_modifiers: List[str] = field(default_factory=list)  # ["DISTINCT", "WHERE", "LIMIT"]
    context_assumptions: Optional[Dict[str, Any]] = None
    should_clarify: bool = False  # Whether clarification should be requested
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "user_query": self.user_query,
            "expected_tables": self.expected_tables,
            "expected_sql_pattern": self.expected_sql_pattern,
            "expected_modifiers": self.expected_modifiers,
            "context_assumptions": self.context_assumptions,
            "should_clarify": self.should_clarify,
        }


@dataclass
class TestResult:
    """Result of a single test."""
    test_id: str
    test_name: str
    status: TestStatus
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Actual results
    generated_sql: Optional[str] = None
    detected_tables: Optional[List[str]] = None
    detected_modifiers: Optional[List[str]] = None
    
    # Validation results
    sql_valid: bool = False
    tables_match: bool = False
    modifiers_match: bool = False
    sql_pattern_match: bool = False
    
    # Error/notes
    error_message: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        status_val = self.status.value if hasattr(self.status, 'value') else str(self.status)
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "status": status_val,
            "timestamp": self.timestamp.isoformat(),
            "generated_sql": self.generated_sql,
            "detected_tables": self.detected_tables,
            "detected_modifiers": self.detected_modifiers,
            "sql_valid": self.sql_valid,
            "tables_match": self.tables_match,
            "modifiers_match": self.modifiers_match,
            "sql_pattern_match": self.sql_pattern_match,
            "error_message": self.error_message,
            "notes": self.notes,
        }


class EvaluationHarness:
    """
    Test harness for measuring system accuracy and improvements.
    """
    
    # Golden test cases for baseline metrics
    GOLDEN_TEST_CASES: List[TestCase] = [
        TestCase(
            id="t-001",
            name="Simple SELECT all",
            description="Basic query fetching all records from a table",
            user_query="show me all users",
            expected_tables=["users"],
            expected_sql_pattern=r"SELECT \* FROM.*users",
            expected_modifiers=[],  # No modifiers needed
        ),
        TestCase(
            id="t-002",
            name="Get unique values",
            description="Query for distinct values (should use DISTINCT)",
            user_query="get all unique categories",
            expected_tables=["categories"],
            expected_sql_pattern=r"SELECT DISTINCT",
            expected_modifiers=["DISTINCT"],
        ),
        TestCase(
            id="t-003",
            name="Filter query",
            description="Query with WHERE clause",
            user_query="show me all active items",
            expected_tables=["items"],
            expected_sql_pattern=r"WHERE.*=",
            expected_modifiers=["WHERE"],
        ),
        TestCase(
            id="t-004",
            name="Top N query",
            description="Query for top N records (should use LIMIT)",
            user_query="get top 10 users by balance",
            expected_tables=["users"],
            expected_sql_pattern=r"LIMIT 10",
            expected_modifiers=["LIMIT"],
        ),
        TestCase(
            id="t-005",
            name="Aggregation query",
            description="Query requesting count/sum/avg",
            user_query="how many active accounts are there",
            expected_tables=["accounts"],
            expected_sql_pattern=r"COUNT\(\*\)|COUNT\(.*\)",
            expected_modifiers=[],  # Aggregation might not need LIMIT
        ),
        TestCase(
            id="t-006",
            name="Join query",
            description="Query requiring join between tables",
            user_query="show users and their account balances",
            expected_tables=["users", "accounts"],
            expected_sql_pattern=r"JOIN",
            expected_modifiers=["JOIN"],
        ),
        TestCase(
            id="t-007",
            name="Ambiguous query",
            description="Query that should trigger clarification",
            user_query="show me AP data",
            expected_tables=[],  # Ambiguous
            should_clarify=True,
            expected_modifiers=[],
        ),
        TestCase(
            id="t-008",
            name="Multi-condition filter",
            description="Query with AND/OR conditions",
            user_query="show active items from accounts in New York",
            expected_tables=["items", "accounts"],
            expected_sql_pattern=r"WHERE",
            expected_modifiers=["WHERE"],
        ),
        TestCase(
            id="t-009",
            name="Ordering query",
            description="Query with ORDER BY",
            user_query="show users sorted by account balance desc",
            expected_tables=["users"],
            expected_sql_pattern=r"ORDER BY",
            expected_modifiers=["ORDER BY"],
        ),
        TestCase(
            id="t-010",
            name="Follow-up refinement",
            description="Follow-up that refines previous query",
            user_query="only show active ones",
            context_assumptions={"previous_table": "accounts", "previous_sql": "SELECT * FROM accounts"},
            expected_tables=["accounts"],
            expected_sql_pattern=r"WHERE.*active",
            expected_modifiers=["WHERE"],
        ),
    ]
    
    def __init__(self, test_cases: Optional[List[TestCase]] = None):
        """
        Initialize harness.
        
        Args:
            test_cases: Custom test cases (uses golden if None)
        """
        self.test_cases = test_cases or self.GOLDEN_TEST_CASES
        self.results: List[TestResult] = []
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a new test case."""
        self.test_cases.append(test_case)
    
    def run_test(
        self,
        test_case: TestCase,
        generated_sql: Optional[str] = None,
        detected_tables: Optional[List[str]] = None,
        detected_modifiers: Optional[List[str]] = None,
        sql_valid: bool = False,
        error_message: Optional[str] = None,
    ) -> TestResult:
        """
        Run a test case and record results.
        
        Args:
            test_case: Test case to run
            generated_sql: SQL generated by system
            detected_tables: Tables detected by system
            detected_modifiers: SQL modifiers detected
            sql_valid: Whether generated SQL isvalid
            error_message: Any error message
            
        Returns:
            TestResult with outcome
        """
        import re
        
        # Check tables match
        tables_match = False
        if test_case.expected_tables and detected_tables:
            expected_set = set(t.lower() for t in test_case.expected_tables)
            detected_set = set(t.lower() for t in detected_tables)
            tables_match = expected_set == detected_set
        
        # Check modifiers match
        modifiers_match = False
        if test_case.expected_modifiers and detected_modifiers:
            expected_mods = set(m.upper() for m in test_case.expected_modifiers)
            detected_mods = set(m.upper() for m in detected_modifiers)
            modifiers_match = expected_mods == detected_mods or detected_mods.issuperset(expected_mods)
        
        # Check SQL pattern match
        sql_pattern_match = False
        if generated_sql and test_case.expected_sql_pattern:
            sql_pattern_match = bool(re.search(test_case.expected_sql_pattern, generated_sql, re.IGNORECASE))
        
        # Determine overall status
        if error_message:
            status = TestStatus.FAIL
        elif test_case.should_clarify:
            status = TestStatus.PASS if detected_tables else TestStatus.FAIL
        elif sql_valid and sql_pattern_match:
            if tables_match and modifiers_match:
                status = TestStatus.PASS
            else:
                status = TestStatus.PARTIAL
        else:
            status = TestStatus.FAIL
        
        result = TestResult(
            test_id=test_case.id,
            test_name=test_case.name,
            status=status,
            generated_sql=generated_sql,
            detected_tables=detected_tables,
            detected_modifiers=detected_modifiers,
            sql_valid=sql_valid,
            tables_match=tables_match,
            modifiers_match=modifiers_match,
            sql_pattern_match=sql_pattern_match,
            error_message=error_message,
        )
        
        self.results.append(result)
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary metrics of all tests run."""
        if not self.results:
            return {"error": "No results to summarize"}
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == TestStatus.PASS)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAIL)
        partial = sum(1 for r in self.results if r.status == TestStatus.PARTIAL)
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "partial": partial,
            "success_rate": f"{(passed / total * 100):.1f}%" if total > 0 else "0%",
            "parse_success_rate": f"{(sum(1 for r in self.results if r.generated_sql) / total * 100):.1f}%" if total > 0 else "0%",
            "table_detection_rate": f"{(sum(1 for r in self.results if r.tables_match) / total * 100):.1f}%" if total > 0 else "0%",
            "modifier_accuracy": f"{(sum(1 for r in self.results if r.modifiers_match) / total * 100):.1f}%" if total > 0 else "0%",
        }
    
    def export_results(self, filepath: str) -> None:
        """Export results to JSON file."""
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": self.get_summary(),
            "results": [r.to_dict() for r in self.results],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported results to {filepath}")
    
    def print_report(self) -> None:
        """Print a text report of results."""
        print("\n" + "="*70)
        print("EVALUATION HARNESS REPORT")
        print("="*70)
        
        summary = self.get_summary()
        if "error" in summary:
            print(f"Error: {summary['error']}")
            return
        
        print(f"\nTotal Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Partial: {summary['partial']}")
        print(f"\nSuccess Rate: {summary['success_rate']}")
        print(f"Parse Success Rate: {summary['parse_success_rate']}")
        print(f"Table Detection Rate: {summary['table_detection_rate']}")
        print(f"Modifier Accuracy: {summary['modifier_accuracy']}")
        
        print("\n" + "-"*70)
        print("FAILED TESTS:")
        print("-"*70)
        
        for result in self.results:
            if result.status == TestStatus.FAIL:
                print(f"\n[FAIL] {result.test_name} ({result.test_id})")
                if result.error_message:
                    print(f"  Error: {result.error_message}")
                if not result.tables_match:
                    print(f"  Table mismatch")
                if not result.modifiers_match:
                    print(f"  Modifier mismatch")
                if not result.sql_pattern_match:
                    print(f"  SQL pattern mismatch")
        
        print("\n" + "="*70 + "\n")


# Convenience factory
def create_evaluation_harness(
    test_cases: Optional[List[TestCase]] = None
) -> EvaluationHarness:
    """Create a new evaluation harness."""
    return EvaluationHarness(test_cases=test_cases)
