"""
Semantic Tool Router - LLM-based routing without any hardcoded keywords.

Replaces keyword heuristics with pure learned semantic understanding.
Uses LLM to classify: RUN_SQL vs ANALYZE_FILES vs CHAT based on semantic intent.
Includes confidence-based clarification triggering.

DYNAMIC SEMANTIC MATCHING:
- Uses LLM to match user query terms to database tables semantically
- Handles synonyms and informal terminology automatically
- Zero hardcoded keyword mappings
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

from .. import llm

logger = logging.getLogger(__name__)


class ToolType(str, Enum):
    """Available tools for routing."""
    RUN_SQL = "RUN_SQL"
    ANALYZE_FILES = "ANALYZE_FILES"
    CHAT = "CHAT"


class FollowupDomain(str, Enum):
    """Domain for follow-ups."""
    SQL = "sql"
    FILE = "file"
    CHAT = "chat"


@dataclass
class SemanticRoutingResult:
    """Result from semantic tool routing."""
    tool: ToolType
    followup_domain: FollowupDomain  # Which domain does this relate to?
    confidence: float  # 0.0-1.0
    reasoning: str
    
    # Uncertainty-driven clarification (ChatGPT-style)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_options: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        tool_val = self.tool.value if hasattr(self.tool, 'value') else str(self.tool)
        followup_domain_val = self.followup_domain.value if hasattr(self.followup_domain, 'value') else str(self.followup_domain)
        return {
            "tool": tool_val,
            "followup_domain": followup_domain_val,
            "confidence": round(self.confidence, 2),
            "reasoning": self.reasoning,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "clarification_options": self.clarification_options or [],
        }


class SemanticToolRouter:
    """
    LLM-based tool router with pure semantic understanding.
    
    No hardcoded keywords, rules, or heuristics.
    All routing decisions come from LLM semantic reasoning.
    
    DYNAMIC SEMANTIC TABLE MATCHING:
    - Uses LLM to find semantic matches between user terms and table names
    - Handles synonyms and business terminology automatically
    """
    
    def __init__(self):
        self.confidence_threshold = 0.75  # Below this, ask for clarification
        self._table_names: List[str] = []
        self._table_embeddings: Dict[str, List[float]] = {}
        self._embedding_service = None
        self._semantic_similarity_threshold = 0.65  # Similarity threshold for table matching
    
    def set_schema_context(self, table_names: List[str]) -> None:
        """Set schema context with available table names for better routing decisions."""
        if table_names:
            self._table_names = table_names
            logger.info(f"[SEMANTIC ROUTER] Schema context set with {len(table_names)} tables: {', '.join(table_names[:10])}...")
    
    async def _get_embedding_service(self):
        """Lazy-load embedding service."""
        if self._embedding_service is None:
            try:
                from .embedding_service import EmbeddingService
                self._embedding_service = EmbeddingService()
            except Exception as e:
                logger.warning(f"[SEMANTIC ROUTER] Could not load EmbeddingService: {e}")
        return self._embedding_service
    
    async def _embed_text(self, text: str) -> Optional[List[float]]:
        """
        Embed text using the real API-based embeddings for true semantic similarity.
        Falls back to local embedding service if API is unavailable.
        """
        try:
            # Use the real embedding API for true semantic similarity
            embeddings = await llm.embed_text([text])
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
        except Exception as e:
            logger.debug(f"[SEMANTIC ROUTER] API embedding failed, using fallback: {e}")
        
        # Fallback to local embedding service
        try:
            service = await self._get_embedding_service()
            if service:
                return await service.embed(text)
        except Exception as e:
            logger.warning(f"[SEMANTIC ROUTER] All embedding methods failed: {e}")
        
        return None
    
    async def _embed_table_names(self) -> None:
        """Embed all table names for semantic matching. Called once per routing session."""
        if not self._table_names or self._table_embeddings:
            return  # Already embedded or no tables
        
        try:
            # Embed each table name with semantic expansion
            for table in self._table_names:
                # Convert table_name to natural language for better embedding
                # e.g., "user_orders" → "user orders"
                natural_name = table.replace('_', ' ').lower()
                embedding = await self._embed_text(natural_name)
                if embedding:
                    self._table_embeddings[table] = embedding
            
            logger.info(f"[SEMANTIC ROUTER] ✅ Embedded {len(self._table_embeddings)} table names for semantic matching")
        except Exception as e:
            logger.warning(f"[SEMANTIC ROUTER] Failed to embed table names: {e}")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def _find_semantic_table_matches(self, query: str) -> List[Tuple[str, float]]:
        """
        Find database tables that semantically match terms in the user query.
        
        PURE LLM APPROACH: Uses LLM to understand semantic relationships.
        NO HARDCODED SYNONYMS - LLM determines semantic equivalence dynamically.
        
        Returns:
            List of (table_name, similarity_score) tuples sorted by score descending
        """
        if not self._table_names:
            return []
        
        # Use LLM to find semantic matches between query and table names
        matches = await self._llm_find_table_matches(query, self._table_names)
        
        if matches:
            logger.info(f"[SEMANTIC ROUTER] 🎯 Semantic table matches for '{query[:50]}': {matches[:3]}")
        
        return matches
    
    async def _llm_find_table_matches(self, query: str, table_names: List[str]) -> List[Tuple[str, float]]:
        """
        Use LLM to semantically match user query to database tables.
        
        ZERO HARDCODING: LLM understands synonyms, business terms, context.
        """
        from .. import llm
        import json
        
        tables_str = ", ".join(table_names)
        
        match_prompt = f"""Determine if this query is asking for DATA from a database, and if so, which tables.

User Query: "{query}"

Available Tables: {tables_str}

FIRST: Is this a DATA query?
- A DATA query explicitly requests information, records, counts, lists from a database
- Social/conversational utterances (greetings, pleasantries, meta-questions about capabilities) are NOT data queries
- The query must have clear DATA RETRIEVAL intent, not just incidental word overlap with table names

If NOT a data query, respond with: []

If it IS a data query:
1. Identify which tables are semantically relevant to the data being requested
2. Consider that users may use synonyms or informal terms for database entities
3. Return confidence score 0.0-1.0 for each relevant table

Respond with ONLY valid JSON array:
[{{"table": "table_name", "confidence": 0.85, "reason": "brief reason"}}]

If no tables match OR not a data query, return: []"""
        
        try:
            response = await llm.call_llm(
                [{"role": "system", "content": "You are a semantic table matcher. Respond ONLY with valid JSON."},
                 {"role": "user", "content": match_prompt}],
                stream=False,
                max_tokens=300,
                temperature=0.0
            )
            
            # Extract JSON from response
            json_match = response.find('[')
            json_end = response.rfind(']') + 1
            if json_match >= 0 and json_end > json_match:
                json_str = response[json_match:json_end]
                results = json.loads(json_str)
                
                matches = []
                for item in results:
                    table = item.get('table', '')
                    confidence = float(item.get('confidence', 0.5))
                    if table in table_names and confidence > 0.3:
                        matches.append((table, confidence))
                
                # Sort by confidence descending
                matches.sort(key=lambda x: x[1], reverse=True)
                return matches
                
        except Exception as e:
            logger.warning(f"[SEMANTIC ROUTER] LLM table matching failed: {e}")
        
        # Minimal fallback: direct word matching only (no synonyms)
        import re
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        matches = []
        for table in table_names:
            table_lower = table.lower()
            table_singular = table_lower.rstrip('s')
            if table_lower in query_words or table_singular in query_words:
                matches.append((table, 0.7))
        
        return matches
    
    async def _check_semantic_intent_with_llm(self, query: str, semantic_matches: List[Tuple[str, float]] = None) -> Optional[SemanticRoutingResult]:
        """
        Use LLM to determine if query is conversational vs database query.
        
        ZERO HARDCODING: Pure semantic analysis without regex patterns.
        Now includes dynamic semantic table matching for better database query detection.
        
        Args:
            query: User's natural language query
            semantic_matches: Pre-computed semantic table matches (table_name, similarity_score)
        
        Returns SemanticRoutingResult if LLM can classify, None if unclear.
        """
        
        # Build schema context for the prompt
        schema_info = ""
        if self._table_names:
            schema_info = f"\nDATABASE TABLES AVAILABLE: {', '.join(self._table_names)}\n"
        
        # Add semantic match information if we found any
        semantic_match_info = ""
        if semantic_matches:
            match_details = [f"'{table}' (similarity: {score:.2f})" for table, score in semantic_matches[:3]]
            semantic_match_info = f"""
SEMANTIC ANALYSIS RESULT:
The user's query semantically matches these database tables: {', '.join(match_details)}
This strongly suggests the user is asking for DATABASE data, even if they didn't use exact table names.
"""
        
        # Quick LLM call to classify: CONVERSATIONAL vs DATABASE vs FILE
        classification_prompt = f"""Classify this user query into ONE category:

USER QUERY: "{query}"
{schema_info}{semantic_match_info}
CATEGORIES:
1. CONVERSATIONAL: Social utterances, greetings, meta-questions about capabilities, acknowledgments, general chat
   - No explicit data retrieval intent
   - Simple greetings are ALWAYS conversational regardless of semantic matches

2. DATABASE: Explicitly asking for data from a database
   - User is requesting information, records, or data
   - Contains clear data retrieval verbs (show, get, find, list, count, retrieve, fetch) with data entities
   - Must have clear DATA RETRIEVAL intent, not just word similarity to table names

3. FILE: Asking about uploaded files or documents
   - Mentions files, documents, uploads, or attached content

NOTE: Semantic table matches suggest POTENTIAL database relevance, but the query must 
actually be asking for data. Incidental word overlap with table names is not sufficient.

Respond with ONLY valid JSON:
{{
  "category": "CONVERSATIONAL" | "DATABASE" | "FILE",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}}"""
        
        try:
            response = await llm.call_llm(
                [{"role": "system", "content": "You are a query classifier. Respond ONLY with valid JSON."},
                 {"role": "user", "content": classification_prompt}],
                stream=False,
                max_tokens=150,
                temperature=0.0
            )
            
            # Parse response
            import json
            result = json.loads(response.strip().replace('```json', '').replace('```', '').strip())
            
            category = result.get('category', '').upper()
            confidence = float(result.get('confidence', 0.5))
            reasoning = result.get('reasoning', '')
            
            # Boost confidence if we had semantic matches
            if semantic_matches and category == 'DATABASE':
                # Strong semantic match → boost confidence
                best_match_score = semantic_matches[0][1] if semantic_matches else 0
                confidence = min(1.0, confidence + (best_match_score * 0.2))
                reasoning += f" (Boosted by semantic match: {semantic_matches[0][0]})"
            
            # Map to tool types
            if category == 'CONVERSATIONAL':
                return SemanticRoutingResult(
                    tool=ToolType.CHAT,
                    followup_domain=FollowupDomain.CHAT,
                    confidence=confidence,
                    reasoning=f"LLM classified as conversational: {reasoning}",
                )
            elif category == 'FILE':
                return SemanticRoutingResult(
                    tool=ToolType.ANALYZE_FILES,
                    followup_domain=FollowupDomain.FILE,
                    confidence=confidence,
                    reasoning=f"LLM classified as file operation: {reasoning}",
                )
            elif category == 'DATABASE':
                return SemanticRoutingResult(
                    tool=ToolType.RUN_SQL,
                    followup_domain=FollowupDomain.SQL,
                    confidence=confidence,
                    reasoning=f"LLM classified as database query: {reasoning}",
                )
            
        except Exception as e:
            logger.warning(f"[SEMANTIC ROUTER] LLM classification failed: {e}")
        
        return None
    
    async def route_query(
        self,
        user_query: str,
        files_in_session: bool = False,
        file_names: Optional[List[str]] = None,
        last_tool_used: Optional[str] = None,
        last_sql_context: Optional[Dict[str, Any]] = None,
        last_file_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[str] = None,
    ) -> SemanticRoutingResult:
        """
        Route user query to tool using pure semantic understanding.
        
        ZERO HARDCODING: All routing based on learned semantics.
        
        Args:
            user_query: The user's natural language query
            files_in_session: Whether files exist in session
            file_names: Names of uploaded files (if any)
            last_tool_used: Previous tool used ("RUN_SQL" or "ANALYZE_FILES")
            last_sql_context: Context from last SQL query (tables, filters, etc.)
            last_file_context: Context from last file analysis
            conversation_history: Prior conversation for context
            
        Returns:
            SemanticRoutingResult with tool, confidence, and clarification if needed
        """
        
        # ========================
        # STEP 0: DYNAMIC SEMANTIC TABLE MATCHING (Zero Hardcoding)
        # ========================
        # Find if user query semantically matches any database tables
        # LLM determines synonym relationships dynamically
        semantic_matches = await self._find_semantic_table_matches(user_query)
        
        # ========================
        # STEP 1: Use LLM for semantic classification FIRST
        # ========================
        # Even with semantic matches, we need to verify this is actually a data query
        # This prevents non-data queries from being incorrectly routed to SQL
        semantic_result = await self._check_semantic_intent_with_llm(user_query, semantic_matches)
        if semantic_result and semantic_result.confidence >= 0.7:
            tool_val = semantic_result.tool.value if hasattr(semantic_result.tool, 'value') else str(semantic_result.tool)
            logger.info(
                f"[SEMANTIC ROUTER] ✅ LLM CLASSIFICATION: {tool_val}, "
                f"Confidence: {semantic_result.confidence:.2f}"
            )
            return semantic_result
        
        # ========================
        # STEP 2: If LLM is uncertain, use semantic matches as fallback
        # ========================
        # Only route to SQL based on semantic matches if LLM didn't clearly say otherwise
        if semantic_matches:
            best_table, best_score = semantic_matches[0]
            if best_score >= 0.80:  # Higher threshold for fallback
                # High confidence semantic match → route to SQL
                logger.info(
                    f"[SEMANTIC ROUTER] 🎯 HIGH CONFIDENCE SEMANTIC MATCH (fallback): "
                    f"'{user_query[:40]}...' → table '{best_table}' (similarity: {best_score:.2f})"
                )
                return SemanticRoutingResult(
                    tool=ToolType.RUN_SQL,
                    followup_domain=FollowupDomain.SQL,
                    confidence=best_score,
                    reasoning=f"Query semantically matches database table '{best_table}' with {best_score:.0%} similarity",
                )
        
        # ========================
        # STEP 2: Use LLM for complex cases (context-aware)
        # ========================
        
        # Build context for LLM
        files_info = ""
        if files_in_session:
            files_info = f"\n\nFILES IN SESSION: {', '.join(file_names or ['(names unknown)'])}"
        
        last_context = ""
        if last_tool_used:
            if last_tool_used == "RUN_SQL" and last_sql_context:
                tables = last_sql_context.get("tables", [])
                filters = last_sql_context.get("filters", [])
                last_context = f"\n\nLAST QUERY: SQL against {tables} with filters: {filters}"
            elif last_tool_used == "ANALYZE_FILES" and last_file_context:
                files = last_file_context.get("files", [])
                last_context = f"\n\nLAST QUERY: File analysis of {files}"
        
        conversation_context = ""
        if conversation_history:
            conversation_context = f"\n\nCONVERSATION:\n{conversation_history}"
        
        # Include schema context and semantic matches
        schema_routing_info = ""
        if self._table_names:
            schema_routing_info = f"\nDATABASE TABLES AVAILABLE: {', '.join(self._table_names)}\n"
        
        semantic_match_info = ""
        if semantic_matches:
            match_details = [f"'{table}' ({score:.0%} match)" for table, score in semantic_matches[:3]]
            semantic_match_info = f"""
SEMANTIC ANALYSIS (ZERO HARDCODING - EMBEDDING-BASED):
Query terms semantically match these database tables: {', '.join(match_details)}
This strongly suggests a DATABASE query even if exact table names weren't used.
"""
        
        llm_prompt = f"""You are a semantic tool router. Your job is to determine which tool best serves the user's intent.
{schema_routing_info}{semantic_match_info}
CRITICAL RULES FOR ROUTING:
1. NEVER route based ONLY on context/history - analyze the CURRENT QUERY semantically
2. Short social utterances without clear data intent → CHAT
3. Greetings, acknowledgments, meta-questions → CHAT (always)
4. Explicit data retrieval verbs + data entities → RUN_SQL
5. File operations + file references → ANALYZE_FILES
6. Conceptual questions without data entity requests → CHAT
7. **IF SEMANTIC ANALYSIS FOUND TABLE MATCHES AND query has data retrieval intent → RUN_SQL**

AVAILABLE TOOLS:
1. RUN_SQL: Query the database for data retrieval, analysis, filtering, aggregation
2. ANALYZE_FILES: Process uploaded files (documents, spreadsheets, images, PDFs)
3. CHAT: Conversational response, explanations, discussions

YOUR INPUT FOR THIS DECISION:
{files_info}{last_context}{conversation_context}

CURRENT USER QUERY:
"{user_query}"

YOUR TASK:
Analyze ONLY the semantic intent of the CURRENT QUERY. Don't be biased by context.

SEMANTIC INTENT ANALYSIS:
1. What is the user trying to DO with this query?
   - Seeking data from database? → RUN_SQL
   - Processing/analyzing files? → ANALYZE_FILES
   - Having conversation? → CHAT
2. Is the query a greeting, acknowledgment, or short utterance? → CHAT
3. Does the query mention specific data entities matching database tables? → RUN_SQL
4. What's the PRIMARY SEMANTIC INTENT (ignore context bias)?

RESPOND WITH ONLY THIS JSON (no markdown, no explanation):
{{
  "tool": "RUN_SQL" | "ANALYZE_FILES" | "CHAT",
  "followup_domain": "sql" | "file" | "chat",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of routing decision based on CURRENT QUERY semantics",
  "needs_clarification": true/false,
  "clarification_question": "If uncertain, what should the user clarify?",
  "clarification_options": ["Option A", "Option B"]
}}"""

        messages = [
            {
                "role": "system",
                "content": "You are a semantic tool router. Respond ONLY with valid JSON, no other text."
            },
            {
                "role": "user",
                "content": llm_prompt
            }
        ]
        
        try:
            response = await llm.call_llm(messages, stream=False, max_tokens=300)
            result_dict = self._parse_routing_response(response)
            
            # Create result with confidence threshold check
            result = SemanticRoutingResult(
                tool=ToolType(result_dict["tool"]),
                followup_domain=FollowupDomain(result_dict.get("followup_domain", "chat")),
                confidence=result_dict.get("confidence", 0.5),
                reasoning=result_dict.get("reasoning", ""),
                needs_clarification=result_dict.get("needs_clarification", False),
                clarification_question=result_dict.get("clarification_question"),
                clarification_options=result_dict.get("clarification_options", []),
            )
            
            # Force clarification if confidence too low
            if result.confidence < self.confidence_threshold and not result.needs_clarification:
                result.needs_clarification = True
                tool_val = result.tool.value if hasattr(result.tool, 'value') else str(result.tool)
                result.clarification_question = f"I'm {result.confidence*100:.0f}% sure about routing to {tool_val}. Is that correct?"
                result.clarification_options = [tool_val, "Something else"]
            
            tool_val = result.tool.value if hasattr(result.tool, 'value') else str(result.tool)
            logger.info(
                f"[SEMANTIC ROUTER] LLM ROUTING: Tool: {tool_val}, Confidence: {result.confidence:.2f}, "
                f"Clarification: {result.needs_clarification}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[SEMANTIC ROUTER] Routing failed: {e}", exc_info=True)
            # Fallback to CHAT if routing fails
            return SemanticRoutingResult(
                tool=ToolType.CHAT,
                followup_domain=FollowupDomain.CHAT,
                confidence=0.5,
                reasoning=f"Routing error: {str(e)}, defaulting to CHAT",
                needs_clarification=True,
                clarification_question="I encountered an error. Could you rephrase your query?",
            )
    
    def _parse_routing_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM routing response JSON."""
        # Clean markdown if present
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
            response = response.strip()
        if response.endswith("```"):
            response = response[:-3].strip()
        
        return json.loads(response)
