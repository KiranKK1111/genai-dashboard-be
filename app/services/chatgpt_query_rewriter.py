"""
ChatGPT-Level Query Rewriter - Advanced Semantic Intelligence

Multi-stage query rewriting with contextual understanding,
intelligent context enhancement, and sophisticated reasoning.
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession

from .advanced_semantic_analyzer import (
    AdvancedSemanticAnalyzer, 
    SemanticContext, 
    IntentType, 
    ConversationState,
    get_advanced_semantic_analyzer
)
from .. import llm

logger = logging.getLogger(__name__)


@dataclass 
class QueryRewriteResult:
    """Enhanced query rewrite result with rich context"""
    
    # Core rewrite
    rewritten_query: str
    confidence: float
    
    # Context enhancement
    context_added: List[str]
    filters_inherited: Dict[str, Any]
    entities_resolved: Dict[str, str]
    
    # Reasoning
    reasoning_chain: List[str]
    semantic_analysis: SemanticContext
    
    # Execution hints
    suggested_approach: str  # "use_cache", "full_query", "modify_filters"
    expected_result_type: str  # "list", "count", "summary"


class ChatGPTLevelQueryRewriter:
    """
    Advanced Query Rewriter with ChatGPT-Level Intelligence
    
    Features:
    - Multi-stage semantic analysis
    - Contextual query enhancement  
    - Intelligent reference resolution
    - Dynamic reasoning chains
    - Conversation state management
    """
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        """Initialize with advanced semantic capabilities"""
        
        # Core semantic analyzer
        self.semantic_analyzer = get_advanced_semantic_analyzer()
        self.db_session = db_session
        
        # Conversation memory for state tracking
        self.conversation_states = {}
        
        logger.info("🧠 ChatGPT-Level Query Rewriter initialized")
    
    async def analyze_and_rewrite(
        self, 
        query: str, 
        conversation_history: List[Dict] = None,
        session_id: str = None
    ) -> QueryRewriteResult:
        """
        Primary method: Analyze query and rewrite with ChatGPT-level intelligence
        """
        
        logger.info(f"🔄 Analyzing query: '{query}'")
        
        # Stage 1: Advanced Semantic Analysis
        semantic_context = await self.semantic_analyzer.analyze_semantic_context(
            query, 
            conversation_history, 
            self.conversation_states.get(session_id, {})
        )
        
        # Stage 2: Intent-Based Rewriting Strategy
        rewrite_strategy = self._determine_rewrite_strategy(semantic_context)
        
        # Stage 3: Context-Enhanced Query Rewriting
        rewrite_result = await self._execute_rewriting_strategy(
            query, 
            semantic_context, 
            rewrite_strategy, 
            conversation_history
        )
        
        # Stage 4: Query Validation & Enhancement
        validated_result = await self._validate_and_enhance_query(rewrite_result, semantic_context)
        
        # Update conversation state
        if session_id:
            self._update_conversation_state(session_id, semantic_context, query)
        
        logger.info(f"✅ Query rewritten with confidence: {validated_result.confidence:.2f}")
        return validated_result
    
    def _determine_rewrite_strategy(self, context: SemanticContext) -> Dict[str, Any]:
        """Determine optimal rewriting strategy based on semantic analysis"""
        
        strategies = {
            IntentType.CONTINUATION: {
                "approach": "paginate_results",
                "preserve_filters": True,
                "modify_limits": True,
                "add_context": False
            },
            IntentType.DRILL_DOWN: {
                "approach": "expand_aggregation", 
                "preserve_filters": True,
                "modify_select": True,
                "add_context": True
            },
            IntentType.REFINEMENT: {
                "approach": "modify_filters",
                "preserve_base": True, 
                "add_filters": True,
                "add_context": True
            },
            IntentType.EXPANSION: {
                "approach": "add_columns",
                "preserve_query": True,
                "modify_select": True,
                "add_context": False
            },
            IntentType.TRANSFORMATION: {
                "approach": "restructure_output",
                "preserve_filters": True,
                "modify_grouping": True,
                "add_context": False
            },
            IntentType.COMPARISON: {
                "approach": "add_comparison_context", 
                "create_subqueries": True,
                "add_temporal_context": True,
                "add_context": True
            },
            IntentType.TEMPORAL_SHIFT: {
                "approach": "modify_temporal_filters",
                "preserve_structure": True,
                "modify_date_filters": True,
                "add_context": False
            },
            IntentType.NEW_QUESTION: {
                "approach": "fresh_analysis",
                "preserve_filters": False,
                "add_context": False,
                "full_processing": True
            }
        }
        
        return strategies.get(context.intent_type, strategies[IntentType.NEW_QUESTION])
    
    async def _execute_rewriting_strategy(
        self, 
        query: str, 
        context: SemanticContext, 
        strategy: Dict[str, Any],
        history: List[Dict]
    ) -> QueryRewriteResult:
        """Execute the determined rewriting strategy"""
        
        reasoning_chain = [f"🎯 Executing {strategy['approach']} strategy for {context.intent_type.value}"]
        
        # Get enhanced query from semantic analysis
        base_query = context.enhanced_query if context.enhanced_query else query
        
        # Apply strategy-specific enhancements
        if strategy["approach"] == "expand_aggregation":
            rewritten_query = await self._expand_aggregation_query(base_query, context, history)
            reasoning_chain.append("📊 Expanded aggregation to show individual records")
            
        elif strategy["approach"] == "modify_filters": 
            rewritten_query = await self._modify_filters_intelligently(base_query, context, history)
            reasoning_chain.append("🔍 Modified filters based on new criteria")
            
        elif strategy["approach"] == "add_comparison_context":
            rewritten_query = await self._add_comparison_context(base_query, context, history)
            reasoning_chain.append("⚖️ Added comparison context from conversation")
            
        elif strategy["approach"] == "restructure_output":
            rewritten_query = await self._restructure_query_output(base_query, context, history)
            reasoning_chain.append("🔄 Restructured output format based on request")
            
        else:
            # Default: Intelligent context enhancement
            rewritten_query = await self._enhance_with_context(base_query, context, history)
            reasoning_chain.append("🧠 Enhanced query with conversation context")
        
        # Determine expected result type
        result_type = self._predict_result_type(rewritten_query, context)
        
        return QueryRewriteResult(
            rewritten_query=rewritten_query,
            confidence=context.confidence,
            context_added=[context.enhanced_query] if context.enhanced_query != query else [],
            filters_inherited=context.inherited_filters,
            entities_resolved=context.entity_resolution,
            reasoning_chain=reasoning_chain,
            semantic_analysis=context,
            suggested_approach=strategy["approach"],
            expected_result_type=result_type
        )
    
    async def _expand_aggregation_query(self, query: str, context: SemanticContext, history: List[Dict]) -> str:
        """Expand aggregated query to show individual records (ChatGPT-style drill-down)"""
        
        # Get the most recent query that had aggregation
        previous_query_data = None
        for h in reversed(history[-5:]):
            if "COUNT" in h.get("sql_query", "").upper() or "SUM" in h.get("sql_query", "").upper():
                previous_query_data = h
                break
        
        if not previous_query_data:
            return query  # Can't expand without previous aggregation
        
        # Use LLM to intelligently expand the query
        expansion_prompt = f"""
The user previously ran an aggregated query and now wants to see the individual records.

PREVIOUS USER QUERY: "{previous_query_data.get('user_query', '')}"
PREVIOUS SQL: {previous_query_data.get('sql_query', '')}

CURRENT REQUEST: "{query}"

Transform the aggregated query to show individual records instead of counts/sums.
Keep all the same filters and conditions, but change SELECT to show actual rows.

If the user wants specific columns (like "list their names"), include those columns.
Otherwise, use SELECT * to show all details.

Return only the enhanced natural language query that will generate the correct SQL:
"""
        
        try:
            enhanced_query = await llm.call_llm(
                [{"role": "user", "content": expansion_prompt}],
                stream=False,
                max_tokens=300,
                temperature=0.1
            )
            
            return enhanced_query.strip()
            
        except Exception as e:
            logger.warning(f"LLM expansion failed: {e}")
            
            # Fallback: Simple expansion
            if "list" in query.lower() or "show" in query.lower():
                base_filters = self._extract_context_from_history(history)
                return f"{query} {base_filters}".strip()
            
            return query
    
    async def _modify_filters_intelligently(self, query: str, context: SemanticContext, history: List[Dict]) -> str:
        """Intelligently modify filters based on conversation context"""
        
        # Get previous query context
        previous_context = self._extract_context_from_history(history)
        
        modification_prompt = f"""
The user wants to refine their previous query with additional filters or modifications.

CONVERSATION CONTEXT: {previous_context}
NEW FILTER REQUEST: "{query}"

Combine the previous context with the new filter request to create a complete query.
The user wants to modify or add to their previous search criteria.

Return the complete enhanced query:
"""
        
        try:
            enhanced_query = await llm.call_llm(
                [{"role": "user", "content": modification_prompt}],
                stream=False,
                max_tokens=300,
                temperature=0.1
            )
            
            return enhanced_query.strip()
            
        except Exception as e:
            logger.warning(f"Filter modification failed: {e}")
            return f"{previous_context} AND {query}" if previous_context else query
    
    async def _add_comparison_context(self, query: str, context: SemanticContext, history: List[Dict]) -> str:
        """Add comparison context for temporal or segment comparisons"""
        
        comparison_prompt = f"""
The user wants to compare data across different time periods or segments.

CURRENT REQUEST: "{query}"
CONVERSATION HISTORY: {[h.get('user_query', '') for h in history[-3:]]}

Enhance the query to include appropriate comparison context.
For example:
- "compare with last year" → include both current and previous year data
- "vs competitors" → structure for competitive analysis
- "this month vs last month" → temporal comparison

Return the enhanced query with comparison context:
"""
        
        try:
            enhanced_query = await llm.call_llm(
                [{"role": "user", "content": comparison_prompt}],
                stream=False,
                max_tokens=300,
                temperature=0.1
            )
            
            return enhanced_query.strip()
            
        except Exception as e:
            logger.warning(f"Comparison context failed: {e}")
            return query
    
    async def _restructure_query_output(self, query: str, context: SemanticContext, history: List[Dict]) -> str:
        """Restructure query output format (grouping, sorting, etc.)"""
        
        restructure_prompt = f"""
The user wants to change how the data is presented or organized.

CURRENT REQUEST: "{query}"
PREVIOUS QUERIES: {[h.get('user_query', '') for h in history[-3:]]}

Restructure the query to change the output format while preserving the core data request.
For example:
- "group by department" → add grouping
- "sort by date" → add ordering  
- "show as summary" → change to aggregated view
- "break down by month" → add temporal grouping

Return the restructured query:
"""
        
        try:
            enhanced_query = await llm.call_llm(
                [{"role": "user", "content": restructure_prompt}],
                stream=False,
                max_tokens=300,
                temperature=0.1
            )
            
            return enhanced_query.strip()
            
        except Exception as e:
            logger.warning(f"Restructuring failed: {e}")
            return query
    
    async def _enhance_with_context(self, query: str, context: SemanticContext, history: List[Dict]) -> str:
        """Default: Enhance query with conversation context"""
        
        if not context.is_followup:
            return query  # No enhancement needed for new queries
        
        # Get relevant context from history
        context_info = self._extract_context_from_history(history)
        
        if context_info and len(context_info) > 10:  # Only add if substantial context
            enhanced = f"{query} (building on: {context_info})"
            return enhanced
        
        return query
    
    def _extract_context_from_history(self, history: List[Dict]) -> str:
        """Extract relevant context from conversation history"""
        
        if not history:
            return ""
        
        # Get the most recent query with meaningful results
        for h in reversed(history[-3:]):
            user_query = h.get("user_query", "")
            if user_query and len(user_query) > 10:
                # Extract key context elements
                context_elements = []
                
                # Extract entity references
                if re.search(r'\b(customer|client|product|transaction|account)s?\b', user_query, re.I):
                    context_elements.append("business entities")
                
                # Extract temporal references  
                if re.search(r'\b(january|february|march|2023|2024|last|this)\b', user_query, re.I):
                    context_elements.append("time period")
                
                # Extract filter conditions
                if re.search(r'\b(where|filter|only|exclude)\b', user_query, re.I):
                    context_elements.append("filter conditions")
                
                if context_elements:
                    return f"previous query about {', '.join(context_elements)}"
                else:
                    return user_query[:50] + "..." if len(user_query) > 50 else user_query
        
        return ""
    
    def _predict_result_type(self, query: str, context: SemanticContext) -> str:
        """Predict the expected result type"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["count", "how many", "total"]):
            return "count"
        elif any(word in query_lower for word in ["list", "show", "display", "get"]):
            return "list" 
        elif any(word in query_lower for word in ["average", "sum", "mean", "summary"]):
            return "summary"
        elif context.intent_type == IntentType.DRILL_DOWN:
            return "list"
        elif context.intent_type == IntentType.AGGREGATION:
            return "summary"
        else:
            return "list"  # Default assumption
    
    async def _validate_and_enhance_query(self, result: QueryRewriteResult, context: SemanticContext) -> QueryRewriteResult:
        """Final validation and enhancement of the rewritten query"""
        
        # Confidence adjustment based on strategy success
        if len(result.reasoning_chain) > 2:  # Multiple reasoning steps
            result.confidence = min(result.confidence + 0.1, 1.0)
        
        # Add final reasoning step
        result.reasoning_chain.append(f"🎯 Final query confidence: {result.confidence:.2f}")
        
        return result
    
    def _update_conversation_state(self, session_id: str, context: SemanticContext, query: str):
        """Update conversation state for future reference"""
        
        if session_id not in self.conversation_states:
            self.conversation_states[session_id] = {}
        
        self.conversation_states[session_id].update({
            "last_intent": context.intent_type,
            "last_query": query,
            "conversation_state": context.conversation_state,
            "context_continuity": context.context_continuity
        })


# Factory function for integration
def get_chatgpt_query_rewriter(db_session: Optional[AsyncSession] = None) -> ChatGPTLevelQueryRewriter:
    """Get ChatGPT-level query rewriter instance"""
    return ChatGPTLevelQueryRewriter(db_session)


# Backward compatibility alias
def get_followup_rewriter(db_session: Optional[AsyncSession] = None) -> ChatGPTLevelQueryRewriter:
    """Backward compatibility: Get advanced query rewriter"""
    return get_chatgpt_query_rewriter(db_session)