"""
ROUTING EVALUATION HARNESS (P3 - Testing & Validation)

Offline evaluation of routing decisions:
- Confusion matrix (tool selection accuracy)
- Adversarial prompts (prompt injection, edge cases)
- Multi-domain testing (hospital, education, HR, commerce)
- Regression detection

Run periodically to validate routing quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class EvaluationMetric(str, Enum):
    """Evaluation metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    CONFUSION_MATRIX = "confusion_matrix"


@dataclass
class GroundTruth:
    """Expected routing for a query."""
    query: str
    expected_tool: str  # "CHAT" | "ANALYZE_FILE" | "RUN_SQL"
    expected_request_type: str  # "NEW_QUERY" | "FOLLOW_UP"
    domain: str  # "hospital", "education", "commerce", "hr"
    reason: str  # Why this tool?
    tags: List[str] = field(default_factory=list)  # "ambiguous", "edge_case", "prompt_injection"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "expected_tool": self.expected_tool,
            "expected_request_type": self.expected_request_type,
            "domain": self.domain,
            "reason": self.reason,
            "tags": self.tags,
        }


@dataclass
class EvaluationResult:
    """Result of evaluating a single query."""
    query: str
    expected: GroundTruth
    predicted_tool: str
    predicted_request_type: str
    confidence: float
    tool_match: bool
    request_type_match: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def overall_match(self) -> bool:
        """Both tool and request_type must match."""
        return self.tool_match and self.request_type_match
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "expected": self.expected.to_dict(),
            "predicted_tool": self.predicted_tool,
            "predicted_request_type": self.predicted_request_type,
            "confidence": round(self.confidence, 4),
            "tool_match": self.tool_match,
            "request_type_match": self.request_type_match,
            "overall_match": self.overall_match,
        }


@dataclass
class EvaluationReport:
    """Summary of evaluation run."""
    timestamp: datetime
    test_count: int
    pass_count: int
    fail_count: int
    accuracy: float
    tool_precision: Dict[str, float] = field(default_factory=dict)
    tool_recall: Dict[str, float] = field(default_factory=dict)
    domain_accuracy: Dict[str, float] = field(default_factory=dict)
    failures: List[EvaluationResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "test_count": self.test_count,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "accuracy": round(self.accuracy, 4),
            "tool_precision": {k: round(v, 4) for k, v in self.tool_precision.items()},
            "tool_recall": {k: round(v, 4) for k, v in self.tool_recall.items()},
            "domain_accuracy": {k: round(v, 4) for k, v in self.domain_accuracy.items()},
            "failure_count": len(self.failures),
        }


class RoutingEvaluationHarness:
    """Evaluate routing quality offline."""
    
    def __init__(self, router):
        self.router = router
        self.test_cases: List[GroundTruth] = []
        logger.info("Evaluation Harness initialized")
    
    def register_test_case(self, ground_truth: GroundTruth) -> None:
        """Register a test case."""
        self.test_cases.append(ground_truth)
    
    def register_test_cases(self, cases: List[GroundTruth]) -> None:
        """Register multiple test cases."""
        self.test_cases.extend(cases)
    
    async def evaluate(
        self,
        session,
        schema_discovery_service,
        has_files: bool = False,
    ) -> EvaluationReport:
        """
        Evaluate router against all test cases.
        
        Returns:
            EvaluationReport with metrics
        """
        results = []
        
        for ground_truth in self.test_cases:
            # Call router
            decision = await self.router.route(
                user_query=ground_truth.query,
                session=session,
                schema_discovery_service=schema_discovery_service,
                current_request_has_files=has_files,
            )
            
            # Compare
            result = EvaluationResult(
                query=ground_truth.query,
                expected=ground_truth,
                predicted_tool=decision.tool.value,
                predicted_request_type=decision.request_type.value,
                confidence=decision.confidence,
                tool_match=decision.tool.value == ground_truth.expected_tool,
                request_type_match=decision.request_type.value == ground_truth.expected_request_type,
            )
            
            results.append(result)
            
            if not result.overall_match:
                logger.warning(f"Mismatch: '{ground_truth.query}' "
                             f"expected {ground_truth.expected_tool}/{ground_truth.expected_request_type} "
                             f"got {decision.tool.value}/{decision.request_type.value}")
        
        # Compute metrics
        report = self._compute_metrics(results)
        return report
    
    def _compute_metrics(self, results: List[EvaluationResult]) -> EvaluationReport:
        """Compute accuracy, precision, recall, etc."""
        
        # Accuracy
        pass_count = sum(1 for r in results if r.overall_match)
        accuracy = pass_count / len(results) if results else 0.0
        
        # Per-tool precision/recall
        tool_precision = self._compute_precision(results)
        tool_recall = self._compute_recall(results)
        
        # Per-domain accuracy
        domain_accuracy = self._compute_domain_accuracy(results)
        
        # Failures
        failures = [r for r in results if not r.overall_match]
        
        report = EvaluationReport(
            timestamp=datetime.utcnow(),
            test_count=len(results),
            pass_count=pass_count,
            fail_count=len(failures),
            accuracy=accuracy,
            tool_precision=tool_precision,
            tool_recall=tool_recall,
            domain_accuracy=domain_accuracy,
            failures=failures[:100],  # Keep first 100 failures
        )
        
        return report
    
    @staticmethod
    def _compute_precision(results: List[EvaluationResult]) -> Dict[str, float]:
        """Precision per tool: TP / (TP + FP)."""
        precision = {}
        
        tools = set(r.predicted_tool for r in results) | set(r.expected.expected_tool for r in results)
        
        for tool in tools:
            tp = sum(1 for r in results if r.predicted_tool == tool and r.tool_match)
            fp = sum(1 for r in results if r.predicted_tool == tool and not r.tool_match)
            
            if tp + fp > 0:
                precision[tool] = tp / (tp + fp)
        
        return precision
    
    @staticmethod
    def _compute_recall(results: List[EvaluationResult]) -> Dict[str, float]:
        """Recall per tool: TP / (TP + FN)."""
        recall = {}
        
        tools = set(r.expected.expected_tool for r in results) | set(r.predicted_tool for r in results)
        
        for tool in tools:
            tp = sum(1 for r in results if r.expected.expected_tool == tool and r.tool_match)
            fn = sum(1 for r in results if r.expected.expected_tool == tool and not r.tool_match)
            
            if tp + fn > 0:
                recall[tool] = tp / (tp + fn)
        
        return recall
    
    @staticmethod
    def _compute_domain_accuracy(results: List[EvaluationResult]) -> Dict[str, float]:
        """Accuracy per domain."""
        by_domain = defaultdict(list)
        
        for result in results:
            by_domain[result.expected.domain].append(result.overall_match)
        
        accuracy = {}
        for domain, matches in by_domain.items():
            if matches:
                accuracy[domain] = sum(matches) / len(matches)
        
        return accuracy


# Built-in test suite (comprehensive coverage)

HOSPITAL_DOMAIN_TESTS = [
    GroundTruth(
        query="Show me patient demographics",
        expected_tool="RUN_SQL",
        expected_request_type="NEW_QUERY",
        domain="hospital",
        reason="Selecting data from structured DB",
    ),
    GroundTruth(
        query="What are the top diagnoses?",
        expected_tool="RUN_SQL",
        expected_request_type="NEW_QUERY",
        domain="hospital",
        reason="Aggregation query on clinical data",
    ),
    GroundTruth(
        query="Upload this file [file.pdf]",
        expected_tool="ANALYZE_FILE",
        expected_request_type="NEW_QUERY",
        domain="hospital",
        reason="File uploaded in same turn",
        tags=["file_upload"],
    ),
    GroundTruth(
        query="Sort by length of stay descending",
        expected_tool="RUN_SQL",
        expected_request_type="FOLLOW_UP",
        domain="hospital",
        reason="Modifying previous SQL result",
        tags=["follow_up"],
    ),
    GroundTruth(
        query="Explain patient privacy regulations",
        expected_tool="CHAT",
        expected_request_type="NEW_QUERY",
        domain="hospital",
        reason="General knowledge question",
    ),
]

EDUCATION_DOMAIN_TESTS = [
    GroundTruth(
        query="List all courses and enrollment",
        expected_tool="RUN_SQL",
        expected_request_type="NEW_QUERY",
        domain="education",
        reason="Database query for structured data",
    ),
    GroundTruth(
        query="Which students are at risk of dropping out?",
        expected_tool="RUN_SQL",
        expected_request_type="NEW_QUERY",
        domain="education",
        reason="Analytics query from student records",
    ),
    GroundTruth(
        query="Filter by grade A or B",
        expected_tool="RUN_SQL",
        expected_request_type="FOLLOW_UP",
        domain="education",
        reason="Refining previous query results",
        tags=["follow_up"],
    ),
    GroundTruth(
        query="What's the history of accreditation?",
        expected_tool="CHAT",
        expected_request_type="NEW_QUERY",
        domain="education",
        reason="General knowledge, not in DB",
    ),
]

COMMERCE_DOMAIN_TESTS = [
    GroundTruth(
        query="High-value accounts by revenue",
        expected_tool="RUN_SQL",
        expected_request_type="NEW_QUERY",
        domain="commerce",
        reason="Database query for business analytics",
    ),
    GroundTruth(
        query="Add a filter for accounts from California",
        expected_tool="RUN_SQL",
        expected_request_type="FOLLOW_UP",
        domain="commerce",
        reason="Refining previous query",
        tags=["follow_up"],
    ),
]

HR_DOMAIN_TESTS = [
    GroundTruth(
        query="Salary distribution by department",
        expected_tool="RUN_SQL",
        expected_request_type="NEW_QUERY",
        domain="hr",
        reason="HR analytics query",
    ),
    GroundTruth(
        query="Show me headcount by location",
        expected_tool="RUN_SQL",
        expected_request_type="NEW_QUERY",
        domain="hr",
        reason="Organizational data",
    ),
]

# Adversarial tests (edge cases, injection attempts)
ADVERSARIAL_TESTS = [
    GroundTruth(
        query="'; DROP TABLE users; --",
        expected_tool="CHAT",
        expected_request_type="NEW_QUERY",
        domain="hospital",
        reason="SQL injection attempt (should be blocked)",
        tags=["prompt_injection"],
    ),
    GroundTruth(
        query="Tell me what database you're using",
        expected_tool="CHAT",
        expected_request_type="NEW_QUERY",
        domain="hospital",
        reason="Information gathering (not a real query)",
        tags=["security"],
    ),
    GroundTruth(
        query="I don't know what I want to ask",
        expected_tool="CHAT",
        expected_request_type="NEW_QUERY",
        domain="hospital",
        reason="Ambiguous request",
        tags=["ambiguous"],
    ),
]

# Compile all test cases
ALL_TEST_CASES = (
    HOSPITAL_DOMAIN_TESTS +
    EDUCATION_DOMAIN_TESTS +
    COMMERCE_DOMAIN_TESTS +
    HR_DOMAIN_TESTS +
    ADVERSARIAL_TESTS
)
