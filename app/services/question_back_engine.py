"""
Question-Back Engine - Generates contextual follow-up questions after query execution.

This module analyzes query results and generates intelligent next-step suggestions,
creating a ChatGPT-like conversational experience.

Approach:
    1. Deterministic rules (fast, reliable)
    2. Optional LLM enhancement (better wording)

Question Types:
    - Limit expansion (show all records?)
    - Filter suggestions (filter by X/Y/Z?)
    - Visualization offers (want a chart?)
    - Drill-down prompts (see details?)
    - Refinement options (group/summarize?)

Author: GitHub Copilot
Created: 2026-03-11
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    """Types of questions the engine can generate."""
    LIMIT_EXPANSION = "limit_expansion"
    FILTER_ADDITION = "filter_addition"
    VISUALIZATION_OFFER = "visualization_offer"
    DRILL_DOWN = "drill_down"
    ROLLUP_SUMMARY = "rollup_summary"
    EXPORT_OFFER = "export_offer"
    REFINEMENT = "refinement"
    COMPARISON = "comparison"
    CLARIFICATION = "clarification"


@dataclass
class SuggestedAction:
    """Actionable suggestion with metadata."""
    action_type: str  # expand_limit, add_filter, create_chart, etc.
    label: str  # UI button/link text
    description: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionBackResult:
    """Generated questions and actions for user."""
    questions: List[str] = field(default_factory=list)
    actions: List[SuggestedAction] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 1.0  # Certainty that these are relevant


class QuestionBackEngine:
    """
    Intelligent question generator for post-query suggestions.
    
    Zero hardcoding approach:
    - Question templates are configurable
    - Rules are data-driven
    - LLM enhancement is optional
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize question-back engine with configuration.
        
        Args:
            config: Optional configuration with templates, rules, thresholds
        """
        self.config = config or self._default_config()
        logger.info("[QUESTION-BACK] Engine initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration (tunable)."""
        return {
            # Thresholds
            "limit_expansion_threshold": 0.8,  # If limit hit and showing X% of total
            "filter_suggestion_threshold": 500,  # Suggest filters if results > N
            "visualization_row_min": 2,  # Need at least N rows for chart
            "visualization_row_max": 10000,  # Too many rows for chart
            
            # Question templates (dynamic, can be overridden)
            "templates": {
                "limit_expansion": [
                    "Would you like to see all {total_count} records instead of just {shown_count}?",
                    "I'm showing the first {shown_count} of {total_count} total records. Want to see all?",
                ],
                "filter_addition": [
                    "Would you like to filter the {entity} by {dimension_list}?",
                    "Do you want to narrow down the results by {dimension_list}?",
                ],
                "visualization_offer": [
                    "Would you like a visualization such as {chart_suggestions}?",
                    "Any specific chart you want? I can generate {chart_suggestions}.",
                ],
                "drill_down": [
                    "Would you like to see the detailed records behind these {metric_name}?",
                    "Want to drill down into the {top_category} data?",
                ],
                "export_offer": [
                    "Would you like to export these results to CSV or Excel?",
                    "Need to download this data?",
                ],
            },
            
            # Feature weights
            "max_questions": 3,  # Don't overwhelm user
            "use_llm_enhancement": False,  # Set True for better wording
        }
    
    async def generate_questions(
        self,
        query_context: Dict[str, Any],
        result_context: Dict[str, Any],
        schema_context: Optional[Dict[str, Any]] = None,
    ) -> QuestionBackResult:
        """
        Generate contextual questions based on query and results.
        
        Args:
            query_context: Info about the query (user_query, intent, entity, SQL, etc.)
            result_context: Info about results (row_count, columns, limit_applied, etc.)
            schema_context: Optional schema intelligence (dimensions, measures, etc.)
        
        Returns:
            QuestionBackResult with generated questions and actions
        """
        logger.info("[QUESTION-BACK] Analyzing query and results for suggestions...")

        # Derive intent if not explicitly set
        if "intent" not in query_context:
            sql = (query_context.get("sql") or "").upper()
            if result_context.get("is_aggregated") or any(k in sql for k in ("COUNT(", "SUM(", "AVG(", "GROUP BY")):
                query_context = {**query_context, "intent": "aggregate"}
            else:
                query_context = {**query_context, "intent": "list"}

        questions = []
        actions = []

        # Apply deterministic rules in priority order
        
        # RULE 1: Limit expansion question
        limit_question = self._check_limit_expansion(query_context, result_context)
        if limit_question:
            questions.append(limit_question["question"])
            if limit_question.get("action"):
                actions.append(limit_question["action"])
        
        # RULE 2: Filter suggestion question
        filter_question = self._check_filter_suggestion(
            query_context, result_context, schema_context
        )
        if filter_question:
            questions.append(filter_question["question"])
            if filter_question.get("action"):
                actions.append(filter_question["action"])
        
        # RULE 3: Visualization offer question
        viz_question = self._check_visualization_offer(
            query_context, result_context, schema_context
        )
        if viz_question:
            questions.append(viz_question["question"])
            if viz_question.get("action"):
                actions.append(viz_question["action"])
        
        # RULE 4: Drill-down question (for aggregates)
        drill_question = self._check_drill_down(query_context, result_context)
        if drill_question:
            questions.append(drill_question["question"])
            if drill_question.get("action"):
                actions.append(drill_question["action"])
        
        # RULE 5: Export offer (for large results)
        export_question = self._check_export_offer(result_context)
        if export_question:
            questions.append(export_question["question"])
            if export_question.get("action"):
                actions.append(export_question["action"])
        
        # Limit to max_questions
        questions = questions[: self.config["max_questions"]]
        
        # Optional: Enhance with LLM for better wording
        if self.config["use_llm_enhancement"] and questions:
            questions = await self._enhance_with_llm(questions, query_context)
        
        result = QuestionBackResult(
            questions=questions,
            actions=actions,
            reasoning=f"Generated {len(questions)} contextual suggestions",
            confidence=1.0,
        )
        
        logger.info(f"[QUESTION-BACK] Generated {len(questions)} questions, {len(actions)} actions")
        return result
    
    def _check_limit_expansion(
        self,
        query_context: Dict[str, Any],
        result_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Check if user should be asked to expand result limit."""
        limit_applied = result_context.get("limit_applied", False)
        shown_count = result_context.get("row_count", result_context.get("row_count_returned", 0))
        total_count_estimate = result_context.get("total_count_estimate", shown_count)
        
        # Rule: If LIMIT was applied and there are more records available
        if limit_applied and total_count_estimate > shown_count:
            threshold = self.config["limit_expansion_threshold"]
            ratio = shown_count / total_count_estimate if total_count_estimate > 0 else 1.0
            
            # Only ask if showing less than threshold% of total
            if ratio < threshold:
                template = self.config["templates"]["limit_expansion"][0]
                question = template.format(
                    total_count=total_count_estimate,
                    shown_count=shown_count
                )
                
                action = SuggestedAction(
                    action_type="expand_limit",
                    label=f"Show all {total_count_estimate} records",
                    parameters={"remove_limit": True}
                )
                
                return {"question": question, "action": action}
        
        return None
    
    def _check_filter_suggestion(
        self,
        query_context: Dict[str, Any],
        result_context: Dict[str, Any],
        schema_context: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Suggest filters if results are broad and unfiltered."""
        row_count = result_context.get("row_count", result_context.get("row_count_returned", 0))
        has_filters = query_context.get("has_filters", result_context.get("has_filters", False))
        intent = query_context.get("intent", "unknown")
        
        # Rule: For list queries with many results and no filters
        if (
            intent == "list"
            and not has_filters
            and row_count >= self.config["filter_suggestion_threshold"]
        ):
            # Get available dimensions from schema or result columns
            available_dimensions = []
            
            if schema_context and "filterable_columns" in schema_context:
                available_dimensions = schema_context["filterable_columns"][:3]
            elif "columns" in result_context:
                # Fallback: suggest from first few columns
                columns = result_context["columns"]
                # Heuristic: categorical columns are often filterable
                available_dimensions = [
                    col for col in columns[:5]
                    if not any(x in col.lower() for x in ["id", "key", "date", "time"])
                ][:3]
            
            if available_dimensions:
                entity = query_context.get("entity", "results")
                dimension_list = ", ".join(available_dimensions)
                
                template = self.config["templates"]["filter_addition"][0]
                question = template.format(
                    entity=entity,
                    dimension_list=dimension_list
                )
                
                action = SuggestedAction(
                    action_type="add_filter",
                    label="Add filters",
                    parameters={"suggested_columns": available_dimensions}
                )
                
                return {"question": question, "action": action}
        
        return None
    
    def _check_visualization_offer(
        self,
        query_context: Dict[str, Any],
        result_context: Dict[str, Any],
        schema_context: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Offer visualization if data is suitable."""
        row_count = result_context.get("row_count", result_context.get("row_count_returned", 0))
        intent = query_context.get("intent", "unknown")
        has_chart = result_context.get("visualization_generated", False)
        
        # Rule: Offer chart if not already generated and data is suitable
        if (
            not has_chart
            and self.config["visualization_row_min"] <= row_count <= self.config["visualization_row_max"]
            and intent in ["list", "aggregate", "count"]
        ):
            # Determine suitable chart types
            chart_suggestions = []
            
            if intent == "aggregate" or result_context.get("has_group_by", False):
                chart_suggestions.extend(["bar chart", "pie chart"])
            
            if result_context.get("has_numeric_columns", False):
                chart_suggestions.append("distribution histogram")
            
            if result_context.get("has_time_series", False):
                chart_suggestions.append("time series line chart")
            
            if not chart_suggestions:
                chart_suggestions = ["bar chart", "pie chart"]
            
            chart_text = " or ".join(chart_suggestions)
            
            template = self.config["templates"]["visualization_offer"][0]
            question = template.format(chart_suggestions=chart_text)
            
            action = SuggestedAction(
                action_type="create_visualization",
                label="Generate visualization",
                parameters={"suggested_types": chart_suggestions}
            )
            
            return {"question": question, "action": action}
        
        return None
    
    def _check_drill_down(
        self,
        query_context: Dict[str, Any],
        result_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Offer drill-down for aggregate results."""
        intent = query_context.get("intent", "unknown")
        is_aggregate = intent in ["aggregate", "count"]
        row_count = result_context.get("row_count", result_context.get("row_count_returned", 0))
        
        # Rule: For aggregates with few summary rows, offer to see details
        if is_aggregate and 2 <= row_count <= 20:
            metric_name = result_context.get("primary_metric", "values")
            
            template = self.config["templates"]["drill_down"][0]
            question = template.format(metric_name=metric_name)
            
            action = SuggestedAction(
                action_type="drill_down",
                label="Show detailed records",
                parameters={"from_aggregate": True}
            )
            
            return {"question": question, "action": action}
        
        return None
    
    def _check_export_offer(
        self,
        result_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Offer export for substantial result sets."""
        row_count = result_context.get("row_count", result_context.get("row_count_returned", 0))
        
        # Rule: Offer export if user has meaningful data
        if row_count >= 50:
            template = self.config["templates"]["export_offer"][0]
            question = template
            
            action = SuggestedAction(
                action_type="export_results",
                label="Export to CSV",
                parameters={"format": "csv"}
            )
            
            return {"question": question, "action": action}
        
        return None
    
    async def _enhance_with_llm(
        self,
        questions: List[str],
        query_context: Dict[str, Any],
    ) -> List[str]:
        """
        Optional: Use LLM to improve question wording.
        
        This is kept minimal to maintain speed - only rephrasing, not generation.
        """
        # Placeholder for LLM enhancement
        # In production, this would call LLM with a simple prompt:
        # "Rephrase these questions to be more natural: {questions}"
        logger.info("[QUESTION-BACK] LLM enhancement not implemented yet")
        return questions


# Singleton instance
_question_back_engine: Optional[QuestionBackEngine] = None


def get_question_back_engine(config: Optional[Dict[str, Any]] = None) -> QuestionBackEngine:
    """Get or create singleton question-back engine instance."""
    global _question_back_engine
    if _question_back_engine is None:
        _question_back_engine = QuestionBackEngine(config)
    return _question_back_engine
