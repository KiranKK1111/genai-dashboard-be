"""
UNIFIED ROUTER DECISION SCHEMA & FACTORY

Combines all routing logic (tool selection + follow-up detection + clarification)
into ONE authoritative decision structure.

Zero hardcoding: All rules derive from signals, config, and schema—never domain assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import json


class Tool(str, Enum):
    """Available tools."""
    CHAT = "CHAT"
    ANALYZE_FILE = "ANALYZE_FILE"
    RUN_SQL = "RUN_SQL"


class RequestType(str, Enum):
    """Is this a new query or a follow-up?"""
    NEW_QUERY = "NEW_QUERY"
    FOLLOW_UP = "FOLLOW_UP"


class FollowupTool(str, Enum):
    """Follow-up domain (only if request_type=FOLLOW_UP)."""
    CHAT_FOLLOW_UP = "CHAT_FOLLOW_UP"
    ANALYZE_FILE_FOLLOW_UP = "ANALYZE_FILE_FOLLOW_UP"
    RUN_SQL_FOLLOW_UP = "RUN_SQL_FOLLOW_UP"


class RunSQLFollowupSubtype(str, Enum):
    """Deterministic edits to prior SQL (e.g., add filter)."""
    ADD_FILTER = "ADD_FILTER"
    REMOVE_FILTER = "REMOVE_FILTER"
    CHANGE_GROUPING = "CHANGE_GROUPING"
    CHANGE_METRIC = "CHANGE_METRIC"
    SORT_OR_TOPK = "SORT_OR_TOPK"
    EXPAND_COLUMNS = "EXPAND_COLUMNS"
    DRILLDOWN = "DRILLDOWN"
    PAGINATION = "PAGINATION"
    SWITCH_ENTITY = "SWITCH_ENTITY"
    FIX_ERROR = "FIX_ERROR"


class AnalyzeFileFollowupSubtype(str, Enum):
    """Follow-up operations on file analysis."""
    ASK_MORE_DETAIL = "ASK_MORE_DETAIL"
    ASK_SUMMARY_DIFFERENT_STYLE = "ASK_SUMMARY_DIFFERENT_STYLE"
    ASK_SOURCE_CITATION = "ASK_SOURCE_CITATION"
    COMPARE_SECTIONS = "COMPARE_SECTIONS"
    EXTRACT_TABLE_ENTITIES = "EXTRACT_TABLE_ENTITIES"


class ChatFollowupSubtype(str, Enum):
    """Follow-up operations in conversation."""
    CLARIFY = "CLARIFY"
    CONTINUE = "CONTINUE"
    APPLY_PREVIOUS_ADVICE = "APPLY_PREVIOUS_ADVICE"
    REPHRASE = "REPHRASE"
    NEW_TOPIC_SAME_SESSION = "NEW_TOPIC_SAME_SESSION"


@dataclass
class ClarificationOption:
    """A single clarification option presented to user."""
    label: str
    value: str
    hint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "value": self.value,
            "hint": self.hint,
        }


@dataclass
class RouterDecision:
    """
    Single authoritative routing decision.
    
    This encodes:
    1. Tool selection (CHAT | ANALYZE_FILE | RUN_SQL)
    2. Request type (NEW_QUERY | FOLLOW_UP)
    3. If follow-up: follow-up tool domain
    4. If RUN_SQL_FOLLOW_UP: deterministic plan modification
    5. Confidence (0-1)
    6. Clarification needs (if confidence low or context missing)
    """
    
    # Core routing
    tool: Tool
    request_type: RequestType
    confidence: float  # 0-1
    
    # Follow-up (only if request_type=FOLLOW_UP)
    followup_tool: Optional[FollowupTool] = None
    followup_subtype: Optional[str] = None  # RUN_SQL | ANALYZE_FILE | CHAT subtype string
    
    # Clarification
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_options: Optional[List[ClarificationOption]] = None
    
    # Debug
    reasoning: str = ""
    signals_used: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool.value,
            "request_type": self.request_type.value,
            "followup_tool": self.followup_tool.value if self.followup_tool else None,
            "followup_subtype": self.followup_subtype,
            "confidence": self.confidence,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "clarification_options": [o.to_dict() for o in (self.clarification_options or [])],
            "reasoning": self.reasoning,
            "signals_used": self.signals_used,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RouterDecision:
        """Reconstruct from dict."""
        return cls(
            tool=Tool(d["tool"]),
            request_type=RequestType(d["request_type"]),
            followup_tool=FollowupTool(d["followup_tool"]) if d.get("followup_tool") else None,
            followup_subtype=d.get("followup_subtype"),
            confidence=d["confidence"],
            needs_clarification=d.get("needs_clarification", False),
            clarification_question=d.get("clarification_question"),
            clarification_options=[
                ClarificationOption(**o) for o in (d.get("clarification_options") or [])
            ],
            reasoning=d.get("reasoning", ""),
            signals_used=d.get("signals_used", {}),
        )


# JSON Schema for Structured Outputs (OpenAI strict mode)
ROUTER_DECISION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": ["CHAT", "ANALYZE_FILE", "RUN_SQL"],
        },
        "request_type": {
            "type": "string",
            "enum": ["NEW_QUERY", "FOLLOW_UP"],
        },
        "followup_tool": {
            "type": ["string", "null"],
            "enum": ["CHAT_FOLLOW_UP", "ANALYZE_FILE_FOLLOW_UP", "RUN_SQL_FOLLOW_UP", None],
        },
        "followup_subtype": {
            "type": ["string", "null"],
            "description": "Subtype if follow-up (e.g., ADD_FILTER for RUN_SQL_FOLLOW_UP)",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "needs_clarification": {
            "type": "boolean",
        },
        "clarification_question": {
            "type": ["string", "null"],
        },
        "clarification_options": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "string"},
                    "hint": {"type": ["string", "null"]},
                },
                "required": ["label", "value"],
            },
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation of the decision",
        },
    },
    "required": [
        "tool",
        "request_type",
        "confidence",
        "needs_clarification",
        "reasoning",
    ],
    "additionalProperties": False,
}
