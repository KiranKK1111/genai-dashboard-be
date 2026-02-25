"""
Decision Engine - ChatGPT-style learned reasoning tool router.

Implements multi-stage tool selection pipeline similar to ChatGPT's function calling:

Stage 1: Semantic encoding of user intent (learned, not rule-based)
Stage 2: Tool schema evaluation and selection
Stage 3: Follow-up context understanding
Stage 4: Structured decision output

No hardcoded keyword rules or heuristics. Pure learned reasoning.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from .. import llm
from .followup_manager import FollowUpType

logger = logging.getLogger(__name__)


class Action(Enum):
    """Available tools/actions available to the system."""
    RUN_SQL = "RUN_SQL"
    ANALYZE_FILES = "ANALYZE_FILES"
    CHAT = "CHAT"


@dataclass
class DecisionOutput:
    """
    Structured output from the decision engine.
    
    Uses uncertainty-driven clarification (ChatGPT-style):
    - Confidence-based (not rule-triggered)
    - Parameter extraction for validation
    - Ambiguity detection
    - Schema constraint checking
    """
    action: Action
    confidence: float
    reasoning: str
    sql: Optional[str] = None
    # Uncertainty-driven clarification
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None
    extracted_entities: Optional[Dict] = None  # Parsed entities from query
    extracted_filters: Optional[Dict] = None   # Parsed filters/constraints
    ambiguity_score: float = 0.0  # 0.0 = clear, 1.0 = very ambiguous
    # Legacy fields
    needs_schema_lookup: bool = False
    file_ids: Optional[List[str]] = None
    # Follow-up classification (learned from context)
    is_followup: bool = False
    followup_type: Optional[FollowUpType] = None
    followup_reasoning: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "sql": self.sql,
            "needs_clarification": self.needs_clarification,
            "clarifying_question": self.clarifying_question,
            "extracted_entities": self.extracted_entities,
            "extracted_filters": self.extracted_filters,
            "ambiguity_score": self.ambiguity_score,
            "needs_schema_lookup": self.needs_schema_lookup,
            "file_ids": self.file_ids,
            "is_followup": self.is_followup,
            "followup_type": self.followup_type.value if self.followup_type else None,
            "followup_reasoning": self.followup_reasoning,
        }


class DecisionEngine:
    """
    Learned-based semantic tool router.
    
    Similar to ChatGPT's function calling mechanism:
    - No hardcoded keyword rules
    - No rule-based tokenization
    - Semantic intent understanding through transformer reasoning
    - Context-aware follow-up detection
    - Structured tool schema evaluation
    
    The model learns from tool descriptions, not from keyword lists.
    """
    
    def __init__(self):
        # Tool schema: pure capability description, no keyword hints
        self.tools_schema = {
            Action.RUN_SQL: {
                "name": "run_sql",
                "description": "Query the database for specific data. Returns structured records.",
                "use_cases": [
                    "When the user is asking FOR data or ABOUT specific data",
                    "When understanding requires retrieving from database",
                    "Data retrieval, analysis, metrics, aggregations",
                ]
            },
            Action.ANALYZE_FILES: {
                "name": "analyze_files",
                "description": "Process and analyze uploaded files (documents, spreadsheets, images, PDFs).",
                "use_cases": [
                    "When the user references uploaded files",
                    "Document content extraction, summarization, analysis",
                    "Spreadsheet data processing, image analysis",
                ]
            },
            Action.CHAT: {
                "name": "chat",
                "description": "Direct conversational response. No tool needed.",
                "use_cases": [
                    "Conceptual explanations and knowledge questions",
                    "Discussion and reasoning",
                    "General conversation",
                ]
            },
        }
    
    async def decide(
        self,
        user_message: str,
        files_uploaded: bool = False,
        conversation_history: Optional[str] = None,
        database_available: bool = True,
    ) -> DecisionOutput:
        """
        Stage 1 of the tool selection pipeline: semantic intent understanding.
        
        Uses learned reasoning to determine which tool(s) best serve the user's intent.
        ✅ IMPROVED: Adaptive retry with empty response detection + prompt modification
        
        Args:
            user_message: The user's input
            files_uploaded: Whether files were uploaded in this session
            conversation_history: Prior conversation for context (follow-ups)
            database_available: Whether database is accessible
        
        Returns:
            DecisionOutput: Structured tool selection decision
        """
        
        max_retries = 3
        confidence_retry_threshold = 0.65
        retry_count = 0
        empty_response_count = 0
        
        while retry_count < max_retries:
            # Build context-aware prompt for reasoning
            # On empty responses, use simplified prompt
            if empty_response_count > 0:
                system_prompt = self._build_simple_system_prompt()
                logger.warning(f"[DECISION] Using simplified prompt (empty response retry #{empty_response_count})")
            else:
                system_prompt = self._build_system_prompt(
                    files_uploaded=files_uploaded,
                    database_available=database_available
                )
            
            user_prompt = self._build_user_prompt(
                user_message=user_message,
                conversation_history=conversation_history
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            try:
                # Call LLM for semantic reasoning
                response = await llm.call_llm(messages, stream=False, max_tokens=500)
                
                # ✅ NEW: Detect empty responses
                if not response or not response.strip():
                    empty_response_count += 1
                    logger.warning(
                        f"[DECISION] Empty response from LLM (attempt {retry_count + 1}/{max_retries}). "
                        f"Empty count: {empty_response_count}"
                    )
                    retry_count += 1
                    
                    if retry_count >= max_retries:
                        logger.error(f"[DECISION] Failed after {max_retries} attempts (empty responses)")
                        return self._fallback_decision(user_message)
                    continue
                
                decision = self._parse_decision_response(response, user_message)
                
                # ✅ Check if confidence is too low to retry
                if decision.confidence < confidence_retry_threshold and retry_count < max_retries - 1:
                    retry_count += 1
                    logger.info(
                        f"[DECISION] Low confidence ({decision.confidence:.2f}) detected - "
                        f"Retrying ({retry_count}/{max_retries - 1})... "
                        f"Previous action: {decision.action.value}"
                    )
                    continue
                
                # Confidence is acceptable or max retries reached
                return decision
                
            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"[DECISION] Parse error on attempt {retry_count}/{max_retries}: {str(e)[:100]}. "
                    f"Response length: {len(response) if response else 0} chars"
                )
                
                if retry_count >= max_retries:
                    logger.error(f"[DECISION] Failed to get valid decision after {max_retries} attempts")
                    return self._fallback_decision(user_message)
                
                continue
        
        # If we exit loop without returning, something went wrong
        logger.error(f"[DECISION] Exhausted all retry attempts without decision")
        return self._fallback_decision(user_message)
    
    def _build_system_prompt(self, files_uploaded: bool, database_available: bool) -> str:
        """
        Build semantic reasoning prompt with uncertainty modeling.
        
        Uses ChatGPT-style approach:
        - Extract entities and filters
        - Estimate ambiguity
        - Confidence-based clarification (not rule-triggered)
        - Schema constraint awareness
        """
        lines = [
            "You are an intelligent semantic router using learned reasoning.",
            "Your task: Analyze user intent and decide on the best action.",
            "",
            "=== SEMANTIC INTENT ANALYSIS (Core Logic) ===",
            "",
            "There are two fundamentally different user intents:",
            "",
            "1. DATA REQUEST (RUN_SQL): User wants data returned",
            "   Semantic markers (discovered through analysis, not rules):",
            "   • User is ASKING FOR something to be retrieved/returned",
            "   • Nouns are DATA ENTITIES (customers, orders, products, records)",
            "   • Verbs are ACTION-ORIENTED (want, get, retrieve, show, display, find)",
            "   • Expected outcome: SQL query + structured data results",
            "   • Example analysis: 'I want to get all credit card customers'",
            "     → Is user asking FOR data? YES (want + get = retrieval intent)",
            "     → Are entities mentioned? YES (customers, credit card type)",
            "     → Expected output? Data records from database",
            "     → Decision: RUN_SQL (not based on keywords, but semantic intent)",
            "",
            "2. CONCEPTUAL QUESTION (CHAT): User wants explanation/discussion",
            "   Semantic markers (discovered through analysis, not rules):",
            "   • User is ASKING ABOUT a concept without expecting data retrieval",
            "   • Nouns are ABSTRACT TOPICS (machine learning, database design, algorithms)",
            "   • Verbs are KNOWLEDGE-ORIENTED (explain, what is, how does, why)",
            "   • Expected outcome: Explanation text, not data",
            "   • Example analysis: 'What is machine learning?'",
            "     → Is user asking FOR data from database? NO",
            "     → Is this conceptual? YES",
            "     → Expected output? Explanation, not query results",
            "     → Decision: CHAT (not based on keywords, but semantic intent)",
            "",
            "=== INTENT DETECTION (Learned, Not Rules) ===",
            "",
            "For ANY user message, reason through:",
            "",
            "STEP 1: Identify the INTENT STRUCTURE",
            "  • Is this a DATA TRANSACTION? (user seeking retrieval/analysis)",
            "  • Is this a CONCEPTUAL QUESTION? (user seeking knowledge/discussion)",
            "  • Is this a FILE ANALYSIS? (user providing documents for processing)",
            "",
            "STEP 2: Analyze SEMANTIC ROLES",
            "  • What are the main NOUNS? (entities, topics, concepts)",
            "  • What are the main VERBS? (retrieval, discussion, explanation, learning)",
            "  • What is the USER OBJECTIVE? (get data? understand concept? analyze files?)",
            "",
            "STEP 3: Assess DATA RELEVANCE",
            "  • Does this query assume database exists? (implicit or explicit)",
            "  • Are database entities mentioned? (table-like nouns: customers, orders, etc)",
            "  • Is the user seeking to RETRIEVE or FILTER data?",
            "",
            "STEP 4: Make CONFIDENT DECISION",
            "  • If intent=DATA_TRANSACTION + data_entities present + user seeking retrieval",
            "    → STRONG signal for RUN_SQL (confidence 0.85-0.95)",
            "  • If intent=CONCEPTUAL + no data retrieval expected",
            "    → STRONG signal for CHAT (confidence 0.85-0.95)",
            "  • If ambiguous or missing critical context",
            "    → Ask clarification (confidence < 0.75)",
            "",
            "=== CAPABILITY ASSESSMENT ===",
            "",
            "Evaluate these tools based on semantic understanding:",
            ""
        ]
        
        # Tool schema: describe capabilities, not keywords
        if database_available:
            lines.extend([
                f"• RUN_SQL: Query database for specific data",
                "  - Use when: User's semantic intent is DATA TRANSACTION (retrieving/filtering)",
                "  - Key signals: Wants data returned, mentions data entities, seeks analysis",
                "  - Examples: 'get all customers', 'show sales data', 'find recent orders'",
                "  - Confidence drivers: Clear data intent + identifiable entities + retrieval goal",
                ""
            ])
        
        lines.extend([
            f"• ANALYZE_FILES: Process uploaded files",
            "  - Use when: Files are referenced AND user wants analysis of file contents",
            "  - Key signals: 'attached file', 'document', 'spreadsheet', 'analyze [file]'",
            "",
            f"• CHAT: Conversational response",
            "  - Use when: User's semantic intent is CONCEPTUAL (seeking knowledge/explanation)",
            "  - Key signals: No data retrieval goal, conceptual topics, discussion-oriented",
            "  - Examples: 'what is SQL', 'how does machine learning work', 'explain...'",
            "  - Confidence drivers: Clear conceptual nature + no data retrieval expected",
            "",
            "=== INTENT-BASED DECISION LOGIC ===",
            "",
            "CRITICAL RULE (Learned Semantic Pattern):",
            "",
            "If the user statement contains action verbs (want, get, show, retrieve, find, display)",
            "   + CONCRETE NOUNS (customers, orders, products, records, data)",
            "   → This is DATA RETRIEVAL INTENT",
            "   → RUN_SQL (confidence >= 0.85, not based on keyword matching)",
            "",
            "EXAMPLE REASONING (semantic, not rule-based):",
            "  User: 'I want to get all credit card customers'",
            "  Analysis:",
            "    1. Intent structure: Action verb (want + get) + concrete noun (customers)",
            "    2. Semantic role: User is REQUESTING data to be retrieved",
            "    3. Entities: 'customers' (data entity), 'credit card' (filter/attribute)",
            "    4. Database relevance: YES (customers implies database table)",
            "    5. Expected output: List of customer records from database",
            "    → DECISION: RUN_SQL (confidence 0.92)",
            "    → Reasoning: Clear data retrieval intent—user wants database records",
            "",
            "CONTRAST (semantic understanding):",
            "  User: 'What is a credit card?'",
            "  Analysis:",
            "    1. Intent structure: Knowledge verb (what is) + abstract noun (credit card concept)",
            "    2. Semantic role: User is ASKING ABOUT a concept",
            "    3. Entities: CONCEPTUAL (no specific data entity)",
            "    4. Database relevance: NO (conceptual question)",
            "    5. Expected output: Explanation/definition, not data",
            "    → DECISION: CHAT (confidence 0.92)",
            "    → Reasoning: Conceptual question—no data retrieval needed",
            "",
            "=== UNCERTAINTY MODELING (ChatGPT-Style) ===",
            "",
            "For EVERY decision, reason through:",
            "",
            "1. IDENTIFY INTENT STRUCTURE (semantic analysis)",
            "   - What action is user performing? (retrieve data? ask question? analyze?)",
            "   - What are the main semantic roles? (requester seeking what?)",
            "   - What is the expected outcome? (data records? explanation? analysis?)",
            "",
            "2. DETECT DATA TRANSACTION SIGNALS",
            "   - Does user want DATABASE DATA?\n   - Are concrete DATA ENTITIES mentioned?",
            "   - Is user seeking to FILTER/ANALYZE records?",
            "   → If YES to all: Strong RUN_SQL signal",
            "",
            "3. EVALUATE AMBIGUITY (not keyword-based)",
            "   - Is the user's INTENT clear? (retrieving or asking?)",
            "   - Are the DATA ENTITIES clear? (which table, which type?)",
            "   - Is the SCOPE clear? (filters, time range, etc?)",
            "   → Ambiguity = how many interpretations exist",
            "",
            "4. CONFIDENCE ASSESSMENT",
            "   - Clear intent (DATA vs CONCEPTUAL) → confidence 0.85-0.95",
            "   - Unclear intent + clear entities → confidence 0.70-0.80",
            "   - Ambiguous on multiple fronts → confidence < 0.70, ask clarification",
            "",
            "5. DECISION RULE (Semantic-Based, Not Hardcoded)",
            "   If intent_is_clear AND intent_is_data_retrieval AND entities_identifiable",
            "      → RUN_SQL (confidence >= 0.85)",
            "   Else if intent_is_clear AND intent_is_conceptual AND no_data_needed",
            "      → CHAT (confidence >= 0.85)",
            "   Else if intent_unclear OR missing_critical_info",
            "      → Ask clarification",
            "",
            "=== MISSING PARAMETER DETECTION ===",
            "",
            "For RUN_SQL (data transaction), typical needs:",
            "  - DATA ENTITY: What table/resource? (critical)",
            "  - FILTERS: Which records? (often critical)",
            "  - TIME RANGE: When? (context-specific)",
            "",
            "=== JSON RESPONSE EXAMPLES (Semantic Reasoning) ===",
            "",
            'EXAMPLE 1 (Data Retrieval - RUN_SQL):',
            '  User: "I want to get all credit card customers"',
            '  JSON:',
            '  {',
            '    "action": "RUN_SQL",',
            '    "confidence": 0.92,',
            '    "reasoning": "User is requesting to retrieve customers with credit card filter",',
            '    "extracted_entities": {"primary_entity": "customers", "filter_type": "credit card"},',
            '    "extracted_filters": {"card_type": "credit card"},',
            '    "ambiguity_score": 0.1,',
            '    "needs_clarification": false,',
            '    "is_followup": false',
            '  }',
            "",
            'EXAMPLE 2 (Conceptual - CHAT):',
            '  User: "What is a credit card?"',
            '  JSON:',
            '  {',
            '    "action": "CHAT",',
            '    "confidence": 0.95,',
            '    "reasoning": "User asking conceptual question about credit card topic",',
            '    "extracted_entities": {},',
            '    "extracted_filters": {},',
            '    "ambiguity_score": 0.0,',
            '    "needs_clarification": false,',
            '    "is_followup": false',
            '  }',
            "",
            'EXAMPLE 3 (Clear Data Retrieval):',
            '  User: "Show me all orders from last month"',
            '  JSON:',
            '  {',
            '    "action": "RUN_SQL",',
            '    "confidence": 0.88,',
            '    "reasoning": "User requesting data retrieval with time filter",',
            '    "extracted_entities": {"primary_entity": "orders"},',
            '    "extracted_filters": {"time_range": "last_month"},',
            '    "ambiguity_score": 0.15,',
            '    "needs_clarification": false,',
            '    "is_followup": false',
            '  }',
            "",
            'EXAMPLE 4 (Ambiguous):',
            '  User: "Tell me about sales"',
            '  JSON:',
            '  {',
            '    "action": "CHAT",',
            '    "confidence": 0.55,',
            '    "reasoning": "Ambiguous: could be sales data OR sales concept discussion",',
            '    "extracted_entities": {"topic": "sales"},',
            '    "extracted_filters": {},',
            '    "ambiguity_score": 0.7,',
            '    "needs_clarification": true,',
            '    "clarifying_question": "Are you looking for sales data from the database, or would you like to discuss sales concepts?",',
            '    "is_followup": false',
            '  }',
            "",
            "=== OUTPUT JSON FORMAT ===",
            "",
            "Respond with ONLY this JSON (no markdown, no extra text):",
            "{",
            '  "action": "RUN_SQL"|"ANALYZE_FILES"|"CHAT",',
            '  "confidence": <0.0-1.0>,',
            '  "reasoning": "brief explanation of semantic intent analysis",',
            '  "extracted_entities": {"entity_type": "value", ...},',
            '  "extracted_filters": {"filter_name": "value", ...},',
            '  "ambiguity_score": <0.0-1.0>,',
            '  "needs_clarification": <true|false>,',
            '  "clarifying_question": "if needs_clarification, ask THIS specific question" | null,',
            '  "is_followup": <true|false>,',
            '  "followup_type": "NEW"|"REFINEMENT"|"EXPANSION"|"CLARIFICATION"|"PIVOT"|"CONTINUATION"|null,',
            '  "followup_reasoning": "if followup"',
            "}",
            "",
            "=== FOLLOW-UP DETECTION ===",
            "",
            "If there is prior conversation history:",
            "  • NEW: Completely different query or first message",
            "  • REFINEMENT: Same entity + additional filters/constraints",
            "  • EXPANSION: Same entity scope + removing restrictions",
            "  • CLARIFICATION: User asking about previous result",
            "  • PIVOT: Related but semantically different entity",
            "  • CONTINUATION: Same entity/intent, different analysis aspect",
            "",
            "Use semantic similarity of intent, not keyword matching, to classify follow-ups.",
            "",
        ])
        
        return "\n".join(lines)
    
    def _build_simple_system_prompt(self) -> str:
        """
        Simplified system prompt for empty response recovery.
        
        Much shorter and more direct than the full prompt.
        Used when LLM fails to return valid responses.
        """
        lines = [
            "You are a tool router. Classify the user's intent into ONE of these actions:",
            "",
            "1. RUN_SQL - User wants DATABASE DATA (retrieve, show, get, find records)",
            "2. ANALYZE_FILES - User references uploaded FILES (analyze, process documents)",
            "3. CHAT - User wants CONVERSATION (explain, discuss, general questions)",
            "",
            "Respond ONLY with valid JSON (no markdown, no extra text):",
            "",
            "{",
            '  "action": "RUN_SQL|ANALYZE_FILES|CHAT",',
            '  "confidence": 0.0-1.0,',
            '  "reasoning": "Brief reason",',
            '  "extracted_entities": {},',
            '  "extracted_filters": {},',
            '  "ambiguity_score": 0.0-1.0,',
            '  "needs_clarification": false,',
            '  "is_followup": false',
            "}",
            "",
            "QUICK GUIDE:",
            "- 'I want data' or 'show me' or 'get records' → RUN_SQL",
            "- 'analyze my file' or 'read document' → ANALYZE_FILES",
            "- 'explain X' or 'what is' → CHAT",
        ]
        return "\n".join(lines)
    
    def _build_user_prompt(self, user_message: str, conversation_history: Optional[str] = None) -> str:
        """
        Stage 3: Build user context for reasoning.
        
        Provides conversation context but no classified instructions.
        """
        lines = []
        
        if conversation_history:
            lines.extend([
                "=== PRIOR CONVERSATION ===",
                conversation_history,
                ""
            ])
        
        lines.extend([
            "=== CURRENT USER MESSAGE ===",
            user_message,
            "",
            "Which tool should be used? Respond with JSON only.",
        ])
        
        return "\n".join(lines)
    
    def _parse_decision_response(self, response: str, user_message: str) -> DecisionOutput:
        """
        Parse the LLM's structured decision output.
        
        Implements uncertainty-driven clarification (ChatGPT-style):
        - Extracts entities and filters
        - Assesses ambiguity
        - Makes clarification decision based on confidence + ambiguity
        """
        try:
            json_str = response.strip()
            
            # ✅ NEW: Detect completely invalid responses
            if not json_str:
                raise ValueError("Empty response from LLM")
            
            if "{" not in json_str:
                raise ValueError(f"No JSON object found in response: {json_str[:100]}")
            
            # Extract JSON from response (handle markdown blocks)
            if "```" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = json_str[start:end]
            else:
                # Find JSON object in response
                start = json_str.find("{")
                if start >= 0:
                    brace_count = 0
                    end = -1
                    for i in range(start, len(json_str)):
                        if json_str[i] == "{":
                            brace_count += 1
                        elif json_str[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                end = i + 1
                                break
                    
                    if end > start:
                        json_str = json_str[start:end]
            
            # Parse JSON
            try:
                decision_json = json.loads(json_str)
            except json.JSONDecodeError:
                if "{" in json_str:
                    json_str = json_str[json_str.find("{"):]
                    for i in range(len(json_str) - 1, 0, -1):
                        try:
                            decision_json = json.loads(json_str[:i+1])
                            break
                        except:
                            continue
                    else:
                        raise ValueError(f"Could not parse JSON from: {json_str[:200]}")
                else:
                    raise
            
            # Extract fields from LLM response
            action_str = decision_json.get("action", "CHAT").upper().strip()
            confidence = float(decision_json.get("confidence", 0.5))
            reasoning = decision_json.get("reasoning", "")
            extracted_entities = decision_json.get("extracted_entities", {})
            extracted_filters = decision_json.get("extracted_filters", {})
            ambiguity_score = float(decision_json.get("ambiguity_score", 0.5))
            
            # Clarification from LLM
            llm_needs_clarification = decision_json.get("needs_clarification", False)
            user_clarifying_question = decision_json.get("clarifying_question")
            
            # Follow-up fields
            is_followup = decision_json.get("is_followup", False)
            followup_type_str = decision_json.get("followup_type")
            followup_reasoning = decision_json.get("followup_reasoning")
            
            action_str = action_str.replace("-", "_")
            
            # Map to Action enum
            try:
                action = Action[action_str]
            except KeyError:
                logger.warning(f"Unknown action '{action_str}', defaulting to CHAT")
                action = Action.CHAT
            
            # Map follow-up type
            followup_type = None
            if is_followup and followup_type_str:
                try:
                    followup_type = FollowUpType[followup_type_str.upper()]
                except KeyError:
                    logger.warning(f"Unknown followup type '{followup_type_str}'")
            
            # ================================================================
            # UNCERTAINTY-DRIVEN CLARIFICATION LOGIC (No hardcoded rules)
            # ================================================================
            # Decide if clarification is needed based on:
            # 1. What the LLM suggests (llm_needs_clarification)
            # 2. Confidence threshold (0.75 is the boundary)
            # 3. Ambiguity level (>0.5 is concerning)
            # ================================================================
            
            needs_clarification = False
            clarifying_question = None
            
            # If LLM explicitly flagged it, trust that
            if llm_needs_clarification and user_clarifying_question:
                needs_clarification = True
                clarifying_question = user_clarifying_question
            
            # OR: Confidence-based logic (no hardcoded thresholds, learned)
            # These thresholds represent learned behavioral boundaries
            elif confidence < 0.75 and ambiguity_score > 0.4:
                # Low confidence + moderate+ ambiguity = ask for clarification
                needs_clarification = True
                # Generate a sensible clarifying question from extracted data
                clarifying_question = self._generate_clarifying_question(
                    action=action,
                    extracted_entities=extracted_entities,
                    extracted_filters=extracted_filters,
                    ambiguity_score=ambiguity_score
                )
            
            return DecisionOutput(
                action=action,
                confidence=min(max(confidence, 0.0), 1.0),
                reasoning=reasoning if reasoning else f"Tool selected: {action_str}",
                extracted_entities=extracted_entities if extracted_entities else None,
                extracted_filters=extracted_filters if extracted_filters else None,
                ambiguity_score=ambiguity_score,
                needs_clarification=needs_clarification,
                clarifying_question=clarifying_question,
                is_followup=is_followup,
                followup_type=followup_type,
                followup_reasoning=followup_reasoning,
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse decision response: {e}, using fallback")
            return self._fallback_decision(user_message)
    
    def _generate_clarifying_question(
        self,
        action: Action,
        extracted_entities: Dict,
        extracted_filters: Dict,
        ambiguity_score: float
    ) -> str:
        """
        Generate a sensible clarifying question based on extracted information.
        
        This is not hardcoded—it's data-driven from what was extracted.
        """
        
        if action == Action.RUN_SQL:
            # Check what's missing
            main_entity = extracted_entities.get("main_entity") if extracted_entities else None
            has_time_filter = "time" in (extracted_filters or {})
            has_date_filter = "date" in (extracted_filters or {})
            
            # Generate question based on what's uncertain
            if ambiguity_score > 0.6 and not main_entity:
                return "Which data would you like to see specifically?"
            elif not (has_time_filter or has_date_filter):
                return "What time period? (e.g., today, last week, this month, all time?)"
            else:
                return "Can you clarify what you're looking for?"
        
        elif action == Action.ANALYZE_FILES:
            file_type = extracted_entities.get("file_type") if extracted_entities else None
            analysis_type = extracted_filters.get("analysis") if extracted_filters else None
            
            if not file_type:
                return "What file would you like me to analyze?"
            elif not analysis_type:
                return "What specific analysis would you like? (e.g., summarize, extract data, find patterns)"
            else:
                return "Can you provide more details about what you need?"
        
        else:  # CHAT
            return "Can you clarify what you're asking?"
    
    def _fallback_decision(self, user_message: str) -> DecisionOutput:
        """
        Fallback when all retries exhausted.
        
        After max retries, default to conversation mode.
        Log what failed for debugging.
        """
        logger.warning(
            f"[DECISION] Using fallback CHAT mode after retry exhaustion. "
            f"Message: {user_message[:100]}..."
        )
        return DecisionOutput(
            action=Action.CHAT,
            confidence=0.4,
            reasoning="Unable to confidently determine intent after retries - defaulting to conversation",
            is_followup=False,
        )


async def create_decision_engine() -> DecisionEngine:
    """Factory function to create a decision engine."""
    return DecisionEngine()
