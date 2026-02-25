"""
PRINCIPLE 8: Evaluation Harness (Test Suite + Metrics)
======================================================
Build a test set of 100-500 queries with:
- Expected SQL (or expected result checksum)
- Run generated SQL and compare results
- Measure: execution success, result correctness, table accuracy, predicate accuracy

Metrics that matter:
1. Execution Success Rate (% queries that run without error)
2. Result Correctness (% where result set matches expected)
3. Table Accuracy (% where used correct tables)
4. Predicate Accuracy (% where filters match expectations)

This is the fastest way to improve systematically.
"""

import logging
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class QueryMetric(str, Enum):
    """Measurable query characteristics."""
    EXECUTION_SUCCESS = "execution_success"
    RESULT_CORRECTNESS = "result_correctness"
    TABLE_ACCURACY = "table_accuracy"
    COLUMN_ACCURACY = "column_accuracy"
    PREDICATE_ACCURACY = "predicate_accuracy"
    PERFORMANCE = "performance"


@dataclass
class GroundedTestCase:
    """A single test case for grounding principles."""
    
    id: str
    user_question: str
    expected_sql: str
    expected_tables: List[str]
    expected_columns: Optional[List[str]] = None
    expected_predicates: Optional[Dict[str, str]] = None  # {column: value}
    expected_row_count: Optional[int] = None
    expected_result_checksum: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "user_question": self.user_question,
            "expected_sql": self.expected_sql,
            "expected_tables": self.expected_tables,
            "expected_columns": self.expected_columns,
            "expected_predicates": self.expected_predicates,
            "expected_row_count": self.expected_row_count,
            "expected_result_checksum": self.expected_result_checksum,
            "difficulty": self.difficulty
        }


@dataclass
class GroundedQueryEvaluation:
    """Results from evaluating a single query."""
    
    test_case_id: str
    user_question: str
    generated_sql: str
    expected_sql: str
    
    # Execution
    execution_success: bool
    execution_error: Optional[str] = None
    execution_time_ms: float = 0.0
    
    # Results
    result_rows: Optional[int] = None
    result_checksum: Optional[str] = None
    
    # Accuracy metrics
    table_accuracy: float = 0.0  # 0.0-1.0
    column_accuracy: float = 0.0
    predicate_accuracy: float = 0.0
    
    # Overall
    passed: bool = False
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "test_case_id": self.test_case_id,
            "user_question": self.user_question,
            "generated_sql": self.generated_sql,
            "expected_sql": self.expected_sql,
            "execution_success": self.execution_success,
            "execution_error": self.execution_error,
            "execution_time_ms": self.execution_time_ms,
            "result_rows": self.result_rows,
            "result_checksum": self.result_checksum,
            "table_accuracy": self.table_accuracy,
            "column_accuracy": self.column_accuracy,
            "predicate_accuracy": self.predicate_accuracy,
            "passed": self.passed,
            "notes": self.notes
        }


class GroundedQueryEvaluator:
    """Evaluates generated SQL against test cases."""

    def __init__(self, schema_grounding):
        self.schema = schema_grounding

    async def evaluate_query(
        self,
        test_case: GroundedTestCase,
        generated_sql: str,
        session: AsyncSession
    ) -> GroundedQueryEvaluation:
        """
        Evaluate a generated query.
        
        Returns:
            GroundedQueryEvaluation with all metrics
        """
        evaluation = GroundedQueryEvaluation(
            test_case_id=test_case.id,
            user_question=test_case.user_question,
            generated_sql=generated_sql,
            expected_sql=test_case.expected_sql
        )

        # 1. Try to execute
        try:
            # Add LIMIT for safety
            safe_query = generated_sql if "LIMIT" in generated_sql else generated_sql + "\nLIMIT 10000"
            result = await session.execute(text(safe_query))
            rows = result.fetchall()
            
            evaluation.execution_success = True
            evaluation.result_rows = len(rows)
            evaluation.execution_time_ms = 0.0  # TODO: measure timing
            
            # Compute result checksum
            if rows:
                result_str = json.dumps([dict(r) for r in rows], default=str, sort_keys=True)
                evaluation.result_checksum = hashlib.md5(result_str.encode()).hexdigest()

        except Exception as e:
            evaluation.execution_success = False
            evaluation.execution_error = str(e)
            return evaluation

        # 2. Table accuracy
        generated_tables = self._extract_tables(generated_sql)
        expected_tables_set = set(test_case.expected_tables)
        generated_tables_set = set(generated_tables)
        
        if generated_tables_set:
            table_intersection = generated_tables_set & expected_tables_set
            table_accuracy = len(table_intersection) / len(expected_tables_set) if expected_tables_set else 0.0
            evaluation.table_accuracy = table_accuracy

        # 3. Column accuracy (if expected_columns provided)
        if test_case.expected_columns:
            generated_columns = self._extract_columns(generated_sql)
            generated_cols_set = set(generated_columns)
            expected_cols_set = set(test_case.expected_columns)
            
            if generated_cols_set:
                col_intersection = generated_cols_set & expected_cols_set
                col_accuracy = len(col_intersection) / len(expected_cols_set) if expected_cols_set else 0.0
                evaluation.column_accuracy = col_accuracy

        # 4. Predicate accuracy (if expected_predicates provided)
        if test_case.expected_predicates:
            predicate_accuracy = self._evaluate_predicates(
                generated_sql,
                test_case.expected_predicates
            )
            evaluation.predicate_accuracy = predicate_accuracy

        # 5. Result correctness
        if test_case.expected_result_checksum and evaluation.result_checksum:
            result_matches = test_case.expected_result_checksum == evaluation.result_checksum
            if result_matches:
                evaluation.passed = True
                evaluation.notes = "Result matches expected output"
        elif test_case.expected_row_count and evaluation.result_rows is not None:
            # Fallback: compare row count
            row_count_matches = abs(test_case.expected_row_count - evaluation.result_rows) < 5
            if row_count_matches:
                evaluation.passed = True
                evaluation.notes = f"Row count matches ({evaluation.result_rows})"

        # Overall pass: success + decent accuracy
        if evaluation.execution_success and evaluation.table_accuracy >= 0.8:
            evaluation.passed = True

        return evaluation

    @staticmethod
    def _extract_tables(sql: str) -> List[str]:
        """Extract table names from SQL."""
        # Simple regex; could be improved with sqlglot
        pattern = r"(?:FROM|JOIN)\s+(?:genai\.)?(\w+)"
        matches = re.findall(pattern, sql, re.IGNORECASE)
        return matches

    @staticmethod
    def _extract_columns(sql: str) -> List[str]:
        """Extract column names from SELECT clause."""
        pattern = r"SELECT\s+(.+?)\s+FROM"
        match = re.search(pattern, sql, re.IGNORECASE | re.DOTALL)
        if match:
            cols_str = match.group(1)
            cols = [c.strip() for c in cols_str.split(",")]
            # Filter out aggregates and constants
            cols = [c.split(".")[-1].split(" ")[0] for c in cols if c]
            return cols
        return []

    @staticmethod
    def _evaluate_predicates(sql: str, expected_predicates: Dict[str, str]) -> float:
        """
        Check if WHERE clause contains expected filters.
        
        Returns:
            0.0-1.0 based on how many expected filters are present
        """
        where_match = re.search(r"WHERE\s+(.+?)(?:GROUP BY|ORDER BY|LIMIT|$)", sql, re.IGNORECASE)
        if not where_match:
            return 0.0

        where_clause = where_match.group(1).lower()
        found_count = 0

        for column, value in expected_predicates.items():
            # Check if both column and value appear in WHERE
            if column.lower() in where_clause and str(value).lower() in where_clause:
                found_count += 1

        return found_count / len(expected_predicates) if expected_predicates else 0.0


class GroundedTestSuite:
    """Collection of test cases for grounding principles."""

    def __init__(self):
        self.test_cases: Dict[str, GroundedTestCase] = {}
        self.evaluations: List[GroundedQueryEvaluation] = []

    def add_test_case(
        self,
        user_question: str,
        expected_sql: str,
        expected_tables: List[str],
        expected_columns: Optional[List[str]] = None,
        expected_predicates: Optional[Dict[str, str]] = None,
        difficulty: str = "medium"
    ) -> str:
        """Add a test case to the suite."""
        tc_id = f"tc_{len(self.test_cases):04d}"
        
        test_case = GroundedTestCase(
            id=tc_id,
            user_question=user_question,
            expected_sql=expected_sql,
            expected_tables=expected_tables,
            expected_columns=expected_columns,
            expected_predicates=expected_predicates,
            difficulty=difficulty
        )
        
        self.test_cases[tc_id] = test_case
        logger.info(f"[GROUNDED_TEST_SUITE] Added test case: {tc_id}")
        return tc_id

    def add_default_test_cases(self) -> None:
        """Add default test cases for banking schema."""
        
        self.add_test_case(
            user_question="Show me all credit card customers",
            expected_sql="""
                SELECT DISTINCT c.customer_id, c.customer_name
                FROM genai.customers c
                LEFT JOIN genai.cards ca ON c.customer_id = ca.customer_id
                WHERE ca.card_type = 'CREDIT'
            """,
            expected_tables=["customers", "cards"],
            expected_columns=["customer_id", "customer_name", "card_type"],
            expected_predicates={"card_type": "CREDIT"},
            difficulty="easy"
        )

        self.add_test_case(
            user_question="Find ATM withdrawals over $1000",
            expected_sql="""
                SELECT t.txn_id, t.amount, t.txn_type
                FROM genai.transactions t
                WHERE t.txn_type = 'ATM_WITHDRAWAL'
                  AND t.amount > 1000
            """,
            expected_tables=["transactions"],
            expected_columns=["txn_id", "amount", "txn_type"],
            expected_predicates={"txn_type": "ATM_WITHDRAWAL"},
            difficulty="easy"
        )

        self.add_test_case(
            user_question="Show customers with both credit and debit cards",
            expected_sql="""
                SELECT c.customer_id, c.customer_name
                FROM genai.customers c
                LEFT JOIN genai.cards ca ON c.customer_id = ca.customer_id
                GROUP BY c.customer_id, c.customer_name
                HAVING COUNT(DISTINCT ca.card_type) >= 2
            """,
            expected_tables=["customers", "cards"],
            expected_columns=["customer_id", "customer_name"],
            difficulty="hard"
        )

    async def run_all(
        self,
        evaluator: GroundedQueryEvaluator,
        sql_generator,  # Function that generates SQL for a question
        session: AsyncSession
    ) -> Dict[str, Any]:
        """
        Run all test cases and collect metrics.
        
        Returns:
            Summary report with metrics
        """
        self.evaluations = []
        
        for test_case in self.test_cases.values():
            # Generate SQL for this test case
            try:
                generated_sql = await sql_generator(test_case.user_question)
            except Exception as e:
                logger.error(f"[GROUNDED_TEST_SUITE] Failed to generate SQL for {test_case.id}: {e}")
                continue
            
            # Evaluate
            evaluation = await evaluator.evaluate_query(test_case, generated_sql, session)
            self.evaluations.append(evaluation)

        # Compute metrics
        return self._compute_metrics()

    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics."""
        if not self.evaluations:
            return {}

        total = len(self.evaluations)
        success = sum(1 for e in self.evaluations if e.execution_success)
        passed = sum(1 for e in self.evaluations if e.passed)
        
        avg_table_accuracy = sum(e.table_accuracy for e in self.evaluations) / total if total else 0.0
        avg_column_accuracy = sum(e.column_accuracy for e in self.evaluations) / total if total else 0.0
        avg_predicate_accuracy = sum(e.predicate_accuracy for e in self.evaluations) / total if total else 0.0

        return {
            "total_tests": total,
            "execution_success_rate": success / total if total else 0.0,
            "result_correctness_rate": passed / total if total else 0.0,
            "avg_table_accuracy": avg_table_accuracy,
            "avg_column_accuracy": avg_column_accuracy,
            "avg_predicate_accuracy": avg_predicate_accuracy,
            "detailed_results": [e.to_dict() for e in self.evaluations]
        }

    def print_report(self) -> None:
        """Print human-readable report."""
        metrics = self._compute_metrics()
        
        print("\n" + "=" * 70)
        print("GROUNDED QUERY EVALUATION REPORT")
        print("=" * 70)
        print(f"Total Tests: {metrics.get('total_tests', 0)}")
        print(f"Execution Success Rate: {metrics.get('execution_success_rate', 0):.1%}")
        print(f"Result Correctness Rate: {metrics.get('result_correctness_rate', 0):.1%}")
        print(f"Avg Table Accuracy: {metrics.get('avg_table_accuracy', 0):.1%}")
        print(f"Avg Column Accuracy: {metrics.get('avg_column_accuracy', 0):.1%}")
        print(f"Avg Predicate Accuracy: {metrics.get('avg_predicate_accuracy', 0):.1%}")
        print("=" * 70)

        # Detailed results
        for eval_result in self.evaluations[:10]:  # Show first 10
            status = "✓ PASS" if eval_result.passed else "✗ FAIL"
            print(f"\n{status} [{eval_result.test_case_id}] {eval_result.user_question[:50]}")
            if not eval_result.execution_success:
                print(f"   Error: {eval_result.execution_error}")
            else:
                print(f"   Tables: {eval_result.table_accuracy:.1%} | Columns: {eval_result.column_accuracy:.1%}")
