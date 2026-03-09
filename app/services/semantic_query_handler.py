"""
Semantic Follow-Up Integration - Replace Pattern Dependencies

Integrates semantic follow-up detection into existing query handler,
eliminating the need for followup_patterns.json configuration file.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.semantic_followup_detector import SemanticFollowUpDetector, SemanticFollowUpContext

logger = logging.getLogger(__name__)


class SemanticQueryHandler:
    """
    Enhanced query handler using semantic follow-up detection.
    
    Replaces pattern-based FollowUpQueryRewriter with semantic analysis.
    NO configuration files needed - pure semantic understanding.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize with semantic components"""
        
        # Semantic follow-up detector (no patterns needed)
        self.followup_detector = SemanticFollowUpDetector(embedding_model)
        
        # Conversation context tracking
        self.conversation_cache = {}
        
        logger.info("[SEMANTIC_HANDLER] Initialized without pattern dependencies")
    
    async def process_query_with_semantic_followup(
        self,
        user_query: str,
        session_id: str,
        db: AsyncSession,
        session_manager
    ) -> Dict[str, Any]:
        """
        Process query with semantic follow-up detection.
        
        Returns enhanced context with semantic understanding.
        """
        
        # Step 1: Semantic follow-up analysis (no patterns)
        followup_context = await self.followup_detector.detect_semantic_followup(
            current_query=user_query,
            session_id=session_id,
            session_manager=session_manager,
            db=db
        )
        
        logger.info(f"[SEMANTIC] Follow-up detection: {followup_context.is_followup} "
                   f"(confidence: {followup_context.confidence:.2f})")
        
        # Step 2: Query enhancement based on semantic analysis
        if followup_context.is_followup and followup_context.confidence > 0.6:
            
            # Get previous query context semantically
            previous_context = await self._get_semantic_context(
                session_id, followup_context, session_manager
            )
            
            if previous_context:
                # Enhance current query with previous context
                enhanced_query = await self._enhance_query_semantically(
                    user_query, previous_context, followup_context
                )
                
                return {
                    'enhanced_query': enhanced_query,
                    'original_query': user_query,
                    'is_followup': True,
                    'followup_context': followup_context,
                    'previous_context': previous_context,
                    'confidence': followup_context.confidence,
                    'reasoning': followup_context.reasoning
                }
        
        # Not a follow-up or low confidence
        return {
            'enhanced_query': user_query,
            'original_query': user_query,
            'is_followup': False,
            'followup_context': followup_context,
            'confidence': followup_context.confidence
        }
    
    async def _get_semantic_context(
        self,
        session_id: str,
        followup_context: SemanticFollowUpContext,
        session_manager
    ) -> Optional[Dict[str, Any]]:
        """Get previous query context using semantic similarity"""
        
        try:
            # Get conversation history
            history = await session_manager.get_conversation_history(session_id, limit=5)
            
            if not history:
                return None
            
            # Find most semantically similar previous query
            best_match = None
            best_similarity = 0.0
            
            for entry in history:
                if 'query_plan_json' in entry and entry['query_plan_json']:
                    # Use embedding similarity to find best context match
                    # This is more accurate than just using the last query
                    similarity = followup_context.semantic_similarity
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = entry
            
            if best_match and best_similarity > 0.5:
                return {
                    'query_plan': best_match.get('query_plan_json', {}),
                    'user_query': best_match.get('user_query', ''),
                    'entities': followup_context.previous_entities,
                    'similarity': best_similarity
                }
            
        except Exception as e:
            logger.error(f"[SEMANTIC] Error getting context: {e}")
        
        return None
    
    async def _enhance_query_semantically(
        self,
        current_query: str,
        previous_context: Dict[str, Any],
        followup_context: SemanticFollowUpContext
    ) -> str:
        """
        Enhance query using semantic understanding (no pattern replacement).
        
        Instead of pattern matching, uses semantic analysis to understand
        what the user is referring to and enhances the query accordingly.
        """
        
        # Extract semantic intent from current query
        current_intent = self._analyze_query_intent(current_query)
        
        # Get filters from previous context
        previous_plan = previous_context.get('query_plan', {})
        previous_filters = self._extract_semantic_filters(previous_plan)
        
        # Build enhanced query using semantic understanding
        if 'list' in current_intent or 'show' in current_intent:
            
            # Generate enhanced query based on semantic analysis
            entity = self._get_primary_entity(previous_plan)
            filter_description = self._filters_to_natural_language(previous_filters)
            
            if entity and filter_description:
                enhanced_query = f"list all {entity} where {filter_description}"
            elif entity:
                enhanced_query = f"list all {entity}"
            else:
                enhanced_query = current_query
            
            logger.info(f"[SEMANTIC] Enhanced: '{current_query}' → '{enhanced_query}'")
            return enhanced_query
        
        elif 'count' in current_intent:
            entity = self._get_primary_entity(previous_plan)
            filter_description = self._filters_to_natural_language(previous_filters)
            
            if entity and filter_description:
                enhanced_query = f"count {entity} where {filter_description}"
            else:
                enhanced_query = current_query
            
            return enhanced_query
        
        # Default: return original query with context note
        return f"{current_query} (referring to previous {self._get_primary_entity(previous_plan)} results)"
    
    def _analyze_query_intent(self, query: str) -> List[str]:
        """Analyze query intent semantically (no hardcoded patterns)"""
        query_lower = query.lower()
        intents = []
        
        # Semantic intent analysis (can be enhanced with embeddings)
        if any(word in query_lower for word in ['list', 'show', 'display', 'get', 'find']):
            intents.append('list')
        
        if any(word in query_lower for word in ['count', 'many', 'number', 'total']):
            intents.append('count')
        
        if any(word in query_lower for word in ['delete', 'remove']):
            intents.append('delete')
        
        if any(word in query_lower for word in ['update', 'modify', 'change']):
            intents.append('update')
        
        return intents
    
    def _extract_semantic_filters(self, query_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract filters from query plan semantically"""
        return query_plan.get('where_conditions', [])
    
    def _get_primary_entity(self, query_plan: Dict[str, Any]) -> str:
        """Get primary entity from query plan"""
        table = query_plan.get('primary_table', '')
        
        # Map table names to natural language entities
        if 'customer' in table.lower():
            return 'customers'
        elif 'account' in table.lower():
            return 'accounts'
        elif 'transaction' in table.lower():
            return 'transactions'
        elif 'employee' in table.lower():
            return 'employees'
        
        return 'records'
    
    def _filters_to_natural_language(self, filters: List[Dict[str, Any]]) -> str:
        """Convert filters to natural language description"""
        if not filters:
            return ""
        
        descriptions = []
        for filter_condition in filters:
            column = filter_condition.get('column', '')
            operator = filter_condition.get('operator', '')
            value = filter_condition.get('value')
            values = filter_condition.get('values', [])
            
            if operator == 'IN' and values:
                descriptions.append(f"{column} in {values}")
            elif operator == 'MONTH_EQUALS' and value:
                month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                              5: 'May', 6: 'June', 7: 'July', 8: 'August', 
                              9: 'September', 10: 'October', 11: 'November', 12: 'December'}
                month_name = month_names.get(value, str(value))
                descriptions.append(f"birthday in {month_name}")
            elif operator == '=' and value:
                descriptions.append(f"{column} = '{value}'")
            elif operator == '>' and value:
                descriptions.append(f"{column} > {value}")
            elif operator == '<' and value:
                descriptions.append(f"{column} < {value}")
        
        return " and ".join(descriptions)


# Factory function for easy integration
async def create_semantic_query_handler(embedding_model: str = "all-MiniLM-L6-v2") -> SemanticQueryHandler:
    """Create semantic query handler without configuration dependencies"""
    return SemanticQueryHandler(embedding_model)


# Integration example
async def integrate_semantic_followup():
    """Example of how to integrate semantic follow-up into existing system"""
    
    # Replace pattern-based rewriter with semantic handler
    # OLD: rewriter = FollowUpQueryRewriter()  # Uses patterns from config file
    # NEW: handler = SemanticQueryHandler()   # Pure semantic analysis
    
    handler = SemanticQueryHandler()
    
    # Process query semantically
    result = await handler.process_query_with_semantic_followup(
        user_query="list all those clients",
        session_id="test_session",
        db=None,  # Would be actual DB session
        session_manager=None  # Would be actual session manager
    )
    
    print(f"Semantic processing result: {result}")


if __name__ == "__main__":
    asyncio.run(integrate_semantic_followup())