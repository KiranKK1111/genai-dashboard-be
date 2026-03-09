"""
Dynamic response generation service with ChatGPT-like capabilities.

This module implements ChatGPT-like features:
- Streaming response generation
- Conversation state management
- Response refinement and regeneration
- Context-aware prompt adaptation
- Multi-turn conversation handling
- Response variation based on user feedback
- SQL-based data query generation
- File upload and analysis support
"""

from __future__ import annotations

import json
import re
from typing import AsyncGenerator, Optional, List, Dict, Any, Tuple

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from .. import llm, models, schemas
from ..config import settings
from ..helpers import current_timestamp, extract_assistant_message_text, format_conversation_context
from .sql_generator import generate_sql, generate_sql_with_analysis
from .query_executor import run_sql
from .file_handler import add_file, retrieve_relevant_chunks


def clean_llm_response(response_text: str) -> str:
    """Clean LLM response by removing instruction markers and conversational preambles.
    
    Handles:
    - Instruction markers: [INST0], [INST1], etc. (from LLM processing artifacts)
    - Conversational openings: "I see", "Thank you", "I appreciate", etc.
    - Extra whitespace
    - Preserves emojis for visual enhancement
    
    Args:
        response_text: Raw LLM response text
        
    Returns:
        Cleaned response text with enhanced formatting
    """
    if not response_text:
        return response_text
    
    # Remove instruction markers like [INST0], [INST1], [INST2], etc.
    cleaned = re.sub(r'\[INST\d+\]', '', response_text, flags=re.IGNORECASE)
    
    # Remove common conversational openings (but preserve content)
    # Pattern: Opening phrase at start, optionally followed by punctuation and space
    conversational_patterns = [
        r'^Ah,?\s+I\s+see[!.]\s*',  # "Ah, I see!" or "Ah I see."
        r'^Thank\s+you\s+for\s+[^.]*[.!]\s*',  # "Thank you for..." sentences
        r'^I\s+appreciate\s+[^.]*[.!]\s*',  # "I appreciate..." sentences
        r'^I\s+understand\s+[^.]*[.!]\s*',  # "I understand..." sentences
        r'^Unfortunately[,.]?\s+[^.]*[.!]\s*',  # "Unfortunately..." sentences
        r'^I\s+apologize[,.]?\s*',  # "I apologize" openings
        r'^Great[!.]?\s+',  # "Great!" openings
        r'^Sure[!.]?\s+',  # "Sure!" openings
        r'^Certainly[!.]?\s+',  # "Certainly!" openings
    ]
    
    for pattern in conversational_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()
    
    # Clean up multiple spaces (but preserve intentional paragraph breaks with emojis)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    
    # Enhance formatting: Apply beautification to remove text actions
    # This ensures clean, natural conversational responses
    cleaned = beautify_response(cleaned)
    
    return cleaned


def beautify_response(response_text: str) -> str:
    """Beautify response text by enhancing formatting and visual flow.
    
    Enhances:
    - Visual separation between sections
    - Removes roleplay text actions
    - Removes "Response #X:" prefixes
    - Removes emojis from start
    - Better spacing
    - Consistent formatting
    
    Args:
        response_text: Response text to beautify
        
    Returns:
        Beautified response text with enhanced formatting
    """
    if not response_text:
        return response_text
    
    beautified = response_text
    
    # Remove roleplay text actions (anything between asterisks that looks like an action)
    # Pattern matches *action* style text (e.g., *smiles*, *nods*)
    beautified = re.sub(r'\*[a-z\s]+\*', '', beautified, flags=re.IGNORECASE)
    
    # Remove "Response #X:" prefix that LLM might include
    beautified = re.sub(r'^Response\s*#\d+:\s*', '', beautified, flags=re.IGNORECASE)
    
    # Remove emojis and special characters from the very beginning of response
    beautified = re.sub(r'^[^\w\s]+', '', beautified)
    
    # Clean up spacing issues from removal
    beautified = re.sub(r'\s{2,}', ' ', beautified)  # Multiple spaces to single space
    beautified = re.sub(r'\s+([.,!?])', r'\1', beautified)  # Space before punctuation
    beautified = beautified.strip()
    
    return beautified


class ConversationState:
    """Manages conversation context and state for ChatGPT-like interactions."""
    
    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.message_history: List[models.Message] = []
        self.last_query_type: Optional[str] = None
        self.last_response_metadata: Optional[Dict] = None
        self.context_summary: Optional[str] = None
        self.conversation_tokens_used: int = 0
    
    def add_message(self, message: models.Message):
        """Add a message to conversation history."""
        self.message_history.append(message)
    
    def set_last_query_type(self, query_type: str):
        """Set the last query type for context awareness."""
        self.last_query_type = query_type
    
    def set_response_metadata(self, metadata: Dict):
        """Store response metadata for refinement and follow-ups."""
        self.last_response_metadata = metadata
    
    def update_context_summary(self, summary: Optional[str]):
        """Update the current conversation context summary."""
        self.context_summary = summary
    
    def get_recent_context(self, max_messages: int = 5) -> str:
        """Get formatted context from recent messages."""
        recent = self.message_history[-max_messages:] if self.message_history else []
        return format_conversation_context(recent)
    
    def should_refine_response(self, user_feedback: str) -> bool:
        """Determine if user feedback indicates need for response refinement."""
        refinement_keywords = [
            "different", "another", "more", "less", "explain", "simplify",
            "show me", "clarify", "rephrase", "different approach", "otherwise",
            "better", "improve", "change", "varied", "again", "try again"
        ]
        feedback_lower = user_feedback.lower()
        return any(keyword in feedback_lower for keyword in refinement_keywords)


class DynamicResponseGenerator:
    """Generate responses with ChatGPT-like capabilities."""
    
    def __init__(self):
        self.model_name = settings.ai_factory_model or "gpt-4o"
        self.token_budget = 4000  # Token budget for response generation
    
    async def generate_visualizations_for_sql_results(
        self,
        query: str,
        sql_results: List[Dict[str, Any]],
    ) -> List[schemas.Visualization]:
        """Generate visualization suggestions based on SQL query results.
        
        Args:
            query: Original user query
            sql_results: Results from SQL execution
        
        Returns:
            List of Visualization objects with chart suggestions
        """
        if not sql_results:
            return []
        
        # Analyze the data structure to suggest visualizations
        first_row = sql_results[0]
        columns = list(first_row.keys()) if first_row else []
        
        # Simple heuristic: detect numeric and categorical columns
        numeric_cols = []
        categorical_cols = []
        
        for col in columns[:10]:  # Limit to first 10 columns
            try:
                val = sql_results[0].get(col)
                if isinstance(val, (int, float)):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            except Exception:
                categorical_cols.append(col)
        
        visualizations = []
        
        # Default visualization based on data structure
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            # Bar chart: categories vs numeric
            bar_viz = schemas.Visualization(
                chart_id="bar_chart_1",
                type="bar",
                title=f"Distribution of {numeric_cols[0] or 'value'} by {categorical_cols[0] or 'category'}",
                data=sql_results[:50],  # Limit to first 50 rows for performance
                config=schemas.VisualizationConfig()
            )
            visualizations.append(bar_viz)
        elif len(numeric_cols) >= 2:
            # Line chart: multiple numeric columns over time/order
            line_viz = schemas.Visualization(
                chart_id="line_chart_1",
                type="line",
                title=f"Trend: {numeric_cols[0]} vs {numeric_cols[1]}",
                data=sql_results[:50],
                config=schemas.VisualizationConfig()
            )
            visualizations.append(line_viz)
        elif len(numeric_cols) == 1:
            # Pie chart for single numeric column with categories
            if len(categorical_cols) > 0:
                pie_viz = schemas.Visualization(
                    chart_id="pie_chart_1",
                    type="pie",
                    title=f"{numeric_cols[0]} by {categorical_cols[0]}",
                    data=sql_results[:20],  # Pie charts work better with fewer slices
                    config=schemas.VisualizationConfig()
                )
                visualizations.append(pie_viz)
            else:
                # Fallback: table visualization
                table_viz = schemas.Visualization(
                    chart_id="table_1",
                    type="table",
                    title="Query Results",
                    data=sql_results[:100],
                    config=schemas.VisualizationConfig()
                )
                visualizations.append(table_viz)
        else:
            # Pure categorical - table view
            table_viz = schemas.Visualization(
                chart_id="table_1",
                type="table",
                title="Query Results",
                data=sql_results[:100],
                config=schemas.VisualizationConfig()
            )
            visualizations.append(table_viz)
        
        return visualizations
    
    async def generate_response(
        self,
        query: str,
        query_type: str,
        db: AsyncSession,
        session_id: str,
        user_id: str,
        conversation_state: ConversationState,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a dynamic response based on query type and context.
        
        Args:
            query: User's natural language query
            query_type: Classification of the query (data_query, file_query, etc.)
            db: Database session
            session_id: Chat session ID
            user_id: User ID
            conversation_state: Current conversation state
            context_data: Additional context data (SQL results, file content, etc.)
        
        Returns:
            Generated response string
        """
        # Build dynamic system prompt based on context
        system_prompt = self._build_dynamic_system_prompt(
            query_type, conversation_state, context_data
        )
        
        # Build messages with conversation context
        messages = self._build_messages(
            system_prompt, query, conversation_state, context_data
        )
        
        # Call LLM with appropriate parameters
        response = await llm.call_llm(
            messages,
            stream=False,
            max_tokens=1500,
            temperature=0.7,  # Higher temp for varied responses
            model=self.model_name,
            track_tokens=True,
        )
        
        # Extract content from LLMResponse and clean it
        content = str(response)
        content = clean_llm_response(content)
        return content
    
    async def generate_multiple_responses(
        self,
        query: str,
        query_type: str,
        db: AsyncSession,
        session_id: str,
        user_id: str,
        conversation_state: ConversationState,
        num_responses: int = 5,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Generate 5-6 different random response variations for the same prompt.
        
        Each response will have a different tone, perspective, or approach while
        remaining relevant to the query.
        
        Args:
            query: User's natural language query
            query_type: Classification of the query
            db: Database session
            session_id: Chat session ID
            user_id: User ID
            conversation_state: Current conversation state
            num_responses: Number of variations to generate (default 5)
            context_data: Additional context data
        
        Returns:
            List of different response variations
        """
        import random
        
        # Clamp num_responses between 3 and 6
        num_responses = max(3, min(6, num_responses))
        
        responses = []
        
        # Build base system prompt
        base_system_prompt = self._build_dynamic_system_prompt(
            query_type, conversation_state, context_data
        )
        
        # Add instruction for variation to system prompt.
        # Keep this short to reduce the chance the model parrots instructions.
        variation_instruction = (
            "\n\nGenerate variation {}/{} of the answer.\n"
            "Rules:\n"
            "- Do not add labels or prefixes (no 'Response #', no 'Variation').\n"
            "- Do not repeat or reference system/policy instructions.\n"
            "- Avoid generic assistant boilerplate; assume the user knows you're an assistant.\n"
            "- Keep it directly relevant; ask at most one clarifying question if needed.\n"
            "- Vary tone/structure across variations while staying correct.\n"
        )
        
        # Generate each response with different temperature and variation prompts
        for i in range(num_responses):
            try:
                # Adjust system prompt for this variation
                system_prompt = base_system_prompt + variation_instruction.format(i + 1, num_responses)
                
                messages = self._build_messages(
                    system_prompt, query, conversation_state, context_data
                )
                
                # Use slightly higher temperature for more diversity, but keep the first variation grounded.
                base_temperature = float(getattr(settings, "llm_temperature", 0.5) or 0.5)
                base_temperature = max(0.2, min(0.9, base_temperature))
                temperature = min(1.2, base_temperature + (i * 0.2))
                
                response = await llm.call_llm(
                    messages,
                    stream=False,
                    max_tokens=1500,
                    temperature=temperature,
                    model=self.model_name,
                    track_tokens=True,
                )
                
                # Clean and beautify response
                content = str(response)
                content = clean_llm_response(content)
                responses.append(content)
                
            except Exception as e:
                print(f"[WARNING] Failed to generate response variation {i + 1}: {e}")
                # If generation fails, try again with default settings
                try:
                    system_prompt = base_system_prompt
                    messages = self._build_messages(
                        system_prompt, query, conversation_state, context_data
                    )
                    response = await llm.call_llm(
                        messages,
                        stream=False,
                        max_tokens=1500,
                        temperature=0.7,
                        model=self.model_name,
                        track_tokens=True,
                    )
                    content = str(response)
                    content = clean_llm_response(content)
                    responses.append(content)
                except Exception as fallback_error:
                    print(f"[ERROR] Fallback response generation failed: {fallback_error}")
        
        # Ensure we have at least some responses
        if not responses:
            default_response = await self.generate_response(
                query, query_type, db, session_id, user_id, conversation_state, context_data
            )
            responses = [default_response] * num_responses
        
        return responses
    
    async def get_random_response(
        self,
        query: str,
        query_type: str,
        db: AsyncSession,
        session_id: str,
        user_id: str,
        conversation_state: ConversationState,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate multiple responses and return a random one.
        
        Provides variety while maintaining single-response interface.
        
        Args:
            query: User's natural language query
            query_type: Classification of the query
            db: Database session
            session_id: Chat session ID
            user_id: User ID
            conversation_state: Current conversation state
            context_data: Additional context data
        
        Returns:
            Single random response from generated variations
        """
        import random
        
        responses = await self.generate_multiple_responses(
            query, query_type, db, session_id, user_id, 
            conversation_state, num_responses=5, context_data=context_data
        )
        
        return random.choice(responses) if responses else ""
    
    async def stream_response(
        self,
        query: str,
        query_type: str,
        db: AsyncSession,
        session_id: str,
        user_id: str,
        conversation_state: ConversationState,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a response token-by-token (ChatGPT-like).
        
        Args:
            query: User's natural language query
            query_type: Classification of the query
            db: Database session
            session_id: Chat session ID
            user_id: User ID
            conversation_state: Current conversation state
            context_data: Additional context data
        
        Yields:
            Response tokens as they're generated
        """
        system_prompt = self._build_dynamic_system_prompt(
            query_type, conversation_state, context_data
        )
        
        messages = self._build_messages(
            system_prompt, query, conversation_state, context_data
        )
        
        # Note: Streaming would require additional implementation in llm.py
        # For now, we generate full response and yield chunks
        response_text = await self.generate_response(
            query, query_type, db, session_id, user_id, conversation_state, context_data
        )
        
        # Simulate streaming by yielding chunks
        chunk_size = 50
        for i in range(0, len(response_text), chunk_size):
            yield response_text[i:i + chunk_size]
    
    async def refine_response(
        self,
        original_query: str,
        original_response: str,
        refinement_feedback: str,
        query_type: str,
        conversation_state: ConversationState,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Refine a response based on user feedback (ChatGPT-like refinement).
        
        Args:
            original_query: The original user query
            original_response: The original response to refine
            refinement_feedback: User's feedback on the response
            query_type: Type of the query
            conversation_state: Current conversation state
            context_data: Additional context data
        
        Returns:
            Refined response string
        """
        # Determine refinement strategy based on feedback
        refinement_strategy = self._determine_refinement_strategy(refinement_feedback)
        
        # Build refinement prompt
        system_prompt = (
            f"You are an expert assistant. The user asked: '{original_query}'\n"
            f"You provided this response:\n{original_response}\n\n"
            f"Now the user has feedback: '{refinement_feedback}'\n"
            f"Refinement strategy: {refinement_strategy}\n\n"
            "Generate an improved response that addresses the feedback while "
            "maintaining accuracy and clarity. Be concise but comprehensive."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original Query: {original_query}\n\nFeedback: {refinement_feedback}"},
        ]
        
        # Add conversation context for better refinement
        if conversation_state.message_history:
            context_msg = conversation_state.get_recent_context(max_messages=3)
            messages[1]["content"] = (
                f"Conversation Context:\n{context_msg}\n\n"
                f"Original Query: {original_query}\n\n"
                f"Feedback: {refinement_feedback}"
            )
        
        refined_response = await llm.call_llm(
            messages,
            stream=False,
            max_tokens=1500,
            temperature=0.8,
            model=self.model_name,
            track_tokens=True,
        )
        
        return str(refined_response)
    
    async def generate_followup_suggestions(
        self,
        original_query: str,
        response: str,
        query_type: str,
        conversation_state: ConversationState,
    ) -> List[str]:
        """Generate intelligent, contextual follow-up question suggestions (ChatGPT-like).
        
        Creates follow-up questions that are specific to the data/context provided,
        not generic. For data queries, suggests drill-downs and comparisons.
        For file queries, suggests deeper analysis paths.
        
        Args:
            original_query: The original user query
            response: The generated response
            query_type: Type of the query
            conversation_state: Current conversation state
        
        Returns:
            List of 3-5 highly intelligent follow-up questions
        """
        # Build context-aware suggestions based on query type
        suggested_questions = []
        
        try:
            if query_type == "data_query":
                # For data queries, suggest intelligent drill-downs
                suggested_questions = self._generate_data_query_followups(
                    original_query, response, conversation_state
                )
            elif query_type == "file_query":
                # For file queries, suggest deeper analysis
                suggested_questions = self._generate_file_query_followups(
                    original_query, response, conversation_state
                )
            elif query_type == "file_lookup":
                # For file lookups, suggest related investigations
                suggested_questions = self._generate_file_lookup_followups(
                    original_query, response, conversation_state
                )
            
            # If pattern-based suggestions are good, return them
            if len(suggested_questions) >= 3:
                return suggested_questions[:5]
        
        except Exception as e:
            print(f"Error generating pattern-based followups: {e}")
        
        # Fallback: Use LLM-based generation with enhanced prompt
        system_prompt = (
            "You are an expert data analyst who asks intelligent follow-up questions.\n\n"
            "Based on the user's question and the response provided, generate 3-4 SPECIFIC follow-up questions.\n"
            "- Questions should be concrete and actionable, not generic\n"
            "- Suggest drill-downs into the data (e.g., by time period, region, category)\n"
            "- For numbers, suggest comparisons or trends (growth, decline, comparison to previous period)\n"
            "- Each question should be a natural progression from the previous answer\n\n"
            "Format: Return ONLY a JSON array of strings (no other text): [\"q1\", \"q2\", \"q3\", \"q4\"]"
        )
        
        conversation_context = ""
        if conversation_state.message_history:
            context_msgs = conversation_state.message_history[-2:]
            for msg in context_msgs:
                if msg.query:
                    conversation_context += f"USER: {msg.query}\n"
                elif msg.response and isinstance(msg.response, dict):
                    msg_text = msg.response.get("message", "")
                    if msg_text:
                        conversation_context += f"ASSISTANT: {msg_text}\n"
        
        user_prompt = (
            f"Conversation:\n{conversation_context}\n\n"
            f"Latest User Question: {original_query}\n\n"
            f"Response Provided:\n{response[:500]}\n\n"
            f"Generate 3-4 intelligent follow-up questions that would help the user explore further."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            suggestions_text = await llm.call_llm(
                messages,
                stream=False,
                max_tokens=500,
                temperature=0.7,
                model=self.model_name,
                track_tokens=True,
            )
            
            suggestions_text = str(suggestions_text).strip()
            
            # Clean up response
            suggestions_text = re.sub(r'\[INST\d+\]', '', suggestions_text, flags=re.IGNORECASE)
            
            # Extract JSON array
            json_match = re.search(r'\[.*?\]', suggestions_text, re.DOTALL)
            if json_match:
                try:
                    suggestions = json.loads(json_match.group())
                    return [str(s).strip() for s in suggestions if s][:5]
                except json.JSONDecodeError:
                    # If JSON parsing fails, fall back to pattern-based
                    return suggested_questions[:5] if suggested_questions else []
            else:
                # Check if response contains lines that look like questions
                lines = [line.strip() for line in suggestions_text.split('\n') if line.strip()]
                questions = [l for l in lines if '?' in l]
                if questions:
                    return questions[:5]
                return suggested_questions[:5] if suggested_questions else []
        
        except Exception as e:
            print(f"Error in LLM-based followup generation: {e}")
            return suggested_questions[:5] if suggested_questions else []
    
    def _generate_data_query_followups(
        self,
        original_query: str,
        response: str,
        conversation_state: ConversationState,
    ) -> List[str]:
        """Generate follow-up questions for data queries."""
        suggestions = []
        query_lower = original_query.lower()
        response_lower = response.lower()
        
        try:
            # Dynamic ID code detection (any pattern like CUST123, INV456, ACC789, etc.)
            # ZERO HARDCODING: Works with any identifier pattern
            id_codes = re.findall(r'\b([A-Z]{3,6}\d+)\b', original_query, re.IGNORECASE)
            if id_codes:
                entity_code = id_codes[0].upper()
                entity_type = ''.join(c for c in entity_code if c.isalpha())  # Extract prefix
                
                suggestions.extend([
                    f"What is the complete history for {entity_code}?",
                    f"Show me more details about {entity_code}",
                    f"Are there any related records for {entity_code}?",
                    f"What is the current status of {entity_code}?",
                ])
            
            # Generic activity/record queries (works for any table)
            if any(word in query_lower for word in ['activity', 'record', 'entry', 'event', 'log']):
                suggestions.extend([
                    "Show me the breakdown by category - which categories have the most volume?",
                    "Let's see the value distribution - are there any outliers?",
                    "What's the daily record count trend for the last 30 days?",
                    "Filter for unusually large values - are there any patterns?",
                ])
            
            # Check for amount/value queries
            if any(word in query_lower for word in ['amount', 'total', 'sum', 'revenue', 'payment', 'value']):
                suggestions.extend([
                    "What's the average value per group?",
                    "Show the top 10 groups by total value.",
                    "What percentage of records are above the average value?",
                    "Let's compare this period to the previous period - what changed?",
                ])
            
            # Check for time-based queries
            if any(word in query_lower for word in ['last', 'today', 'this month', 'recent', 'date']):
                suggestions.extend([
                    "How does this compare to the same period last month?",
                    "Show me the week-over-week trend.",
                    "What time of day has the most activity?",
                    "Are there any interesting patterns in this data by day of week?",
                ])
            
            # Generic data drill-down suggestions
            if not suggestions:
                suggestions = [
                    "Can you break this down by region or location?",
                    "Show me the top performers in this dataset.",
                    "Are there any unusual patterns or anomalies?",
                    "How does this compare to the previous period?",
                ]
        
        except Exception:
            pass
        
        return suggestions[:4]
    
    def _generate_file_query_followups(
        self,
        original_query: str,
        response: str,
        conversation_state: ConversationState,
    ) -> List[str]:
        """Generate follow-up questions for file queries."""
        suggestions = []
        query_lower = original_query.lower()
        
        try:
            # Infer file type and suggest analysis paths
            if any(ext in query_lower for ext in ['.csv', '.xlsx', '.json', 'csv', 'excel', 'json']):
                suggestions.extend([
                    "What are the key statistics (mean, median, std dev) for the numeric columns?",
                    "Are there any missing values or data quality issues I should know about?",
                    "Can you identify any correlations between the columns?",
                    "What's the row count, and are there any duplicates?",
                ])
            
            # If analyzing data/numbers
            if 'data' in query_lower or 'analysis' in query_lower or 'analyze' in query_lower:
                suggestions.extend([
                    "Can you find the distribution of values for the key columns?",
                    "Are there any outliers or anomalies in this data?",
                    "What patterns or trends do you see in this data?",
                    "Can you create a summary of the most important findings?",
                ])
            
            # Generic file analysis suggestions
            if not suggestions:
                suggestions = [
                    "What are the main insights from this file?",
                    "Are there any notable trends or patterns?",
                    "Can you identify any potential issues or anomalies?",
                    "What's the overall structure and quality of this data?",
                ]
        
        except Exception:
            pass
        
        return suggestions[:4]
    
    def _generate_file_lookup_followups(
        self,
        original_query: str,
        response: str,
        conversation_state: ConversationState,
    ) -> List[str]:
        """Generate follow-up questions for file lookup queries."""
        suggestions = []
        
        try:
            suggestions = [
                "Can you provide more details about the specific records we discussed?",
                "How does this compare to other records in the same file?",
                "Are there any related data points I should be aware of?",
                "Can you drill down into the underlying details?",
            ]
        
        except Exception:
            pass
        
        return suggestions[:4]
    
    async def generate_response_with_sql(
        self,
        query: str,
        query_type: str,
        db: AsyncSession,
        session_id: str,
        user_id: str,
        conversation_state: ConversationState,
        conversation_history: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response for SQL-based data queries.
        
        Args:
            query: User's natural language query
            query_type: Classification of the query
            db: Database session
            session_id: Chat session ID
            user_id: User ID
            conversation_state: Current conversation state
            conversation_history: Formatted conversation history
        
        Returns:
            Tuple of (response_text, context_data with SQL results and visualizations)
        """
        context_data = {}
        
        try:
            # Try intelligent SQL generation first
            print(f"[INTELLIGENCE] Attempting step-by-step SQL generation...")
            sql, clarifying_question = await generate_sql_with_analysis(query, db, conversation_history)
            
            # If there's a clarifying question, return it in the response
            if clarifying_question:
                print(f"[INTELLIGENCE] Generated clarifying question: {clarifying_question}")
                context_data["clarifying_question"] = clarifying_question
                context_data["requires_confirmation"] = True

                cq = schemas.ClarificationQuestionValueInput(
                    question=clarifying_question,
                    input_type="string",
                )
                context_data["clarification_questions"] = [cq.model_dump(mode="json")]

                return clarifying_question, context_data
            
            # If no SQL generated by intelligent analyzer, fall back to LLM
            if not sql:
                print(f"[FALLBACK] Intelligent generation failed, using LLM...")
                sql = await generate_sql(query, db, conversation_history)
            
            context_data["generated_sql"] = sql
            
            # ✅ VALIDATE & ENFORCE SAFETY LIMIT BEFORE EXECUTION
            from .sql_safety_validator import SQLSafetyValidator
            validator = SQLSafetyValidator(allowed_schemas=None, max_rows=500)
            is_safe, validation_error, rewritten_sql = validator.validate_and_rewrite(sql)
            if not is_safe:
                print(f"❌ SQL VALIDATION FAILED: {validation_error}")
                raise ValueError(f"SQL validation failed: {validation_error}")
            sql = rewritten_sql  # Use rewritten SQL with safety LIMIT
            
            # Execute SQL
            rows = await run_sql(db, sql)
            context_data["sql_results"] = rows
            
            # Generate visualizations based on SQL results
            visualizations = await self.generate_visualizations_for_sql_results(query, rows)
            context_data["visualizations"] = visualizations
            
            # Build context for response generation
            sql_context = {
                "sql_results": rows[:10] if len(rows) > 10 else rows,  # Limit to 10 rows for context
                "row_count": len(rows),
                "sql_query": sql,
            }
            
            # Generate response with SQL results in context
            response_text = await self.generate_response(
                query=query,
                query_type=query_type,
                db=db,
                session_id=session_id,
                user_id=user_id,
                conversation_state=conversation_state,
                context_data=sql_context,
            )
            
            return response_text, context_data
            
        except Exception as e:
            error_msg = str(e)
            
            # Attempt to recover with error context
            if "undefined column" in error_msg.lower() or "does not exist" in error_msg.lower():
                context_data["error"] = "Could not find the requested columns or tables"
                fallback_response = (
                    "I encountered an issue accessing that data. "
                    "Please check the table or column names and try again."
                )
            else:
                context_data["error"] = error_msg[:200]
                fallback_response = f"Error processing query: {error_msg[:100]}"
            
            return fallback_response, context_data
    
    async def generate_response_with_files(
        self,
        query: str,
        query_type: str,
        db: AsyncSession,
        session_id: str,
        user_id: str,
        files: List[UploadFile],
        conversation_state: ConversationState,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response for file-based queries.
        
        Args:
            query: User's natural language query
            query_type: Classification of the query
            db: Database session
            session_id: Chat session ID
            user_id: User ID
            files: Uploaded files
            conversation_state: Current conversation state
        
        Returns:
            Tuple of (response_text, context_data with file content)
        """
        context_data = {}
        
        try:
            # Process and store files
            file_ids = []
            file_contents = []
            
            for file in files:
                try:
                    # Add file to database - returns UploadedFile model with content_text already extracted
                    uploaded_file_model = await add_file(db, session_id, file)
                    file_ids.append(uploaded_file_model.id)
                    
                    # Use the content_text that was already extracted by add_file
                    file_content_text = uploaded_file_model.content_text
                    
                    # Store with file info and already-extracted content
                    file_contents.append({
                        "filename": file.filename,
                        "content": file_content_text[:5000] if file_content_text else "",  # Limit to 5000 chars for context
                    })
                except Exception as e:
                    print(f"Error processing file {file.filename}: {str(e)}")
                    continue
            
            # Check if we successfully processed any files
            if not file_contents:
                context_data["file_ids"] = file_ids
                context_data["file_contents"] = []
                fallback_response = "Error: Could not read the contents of the uploaded file(s). The file may be empty or in an unsupported format."
                return fallback_response, context_data
            
            context_data["file_ids"] = file_ids
            context_data["file_contents"] = file_contents
            
            # Prepare file context - this will be given to the LLM
            file_context = {
                "file_content": "\n\n".join([
                    f"File: {f['filename']}\n---\n{f['content']}\n---" 
                    for f in file_contents
                ]) if file_contents else "No file content available",
                "file_count": len(file_contents),
            }
            
            # Generate response with file content in context
            response_text = await self.generate_response(
                query=query,
                query_type=query_type,
                db=db,
                session_id=session_id,
                user_id=user_id,
                conversation_state=conversation_state,
                context_data=file_context,
            )
            
            return response_text, context_data
            
        except Exception as e:
            error_msg = str(e)
            context_data["error"] = error_msg[:200]
            fallback_response = f"Error processing files: {error_msg[:100]}"
            return fallback_response, context_data
    
    async def generate_response_with_context(
        self,
        query: str,
        query_type: str,
        db: AsyncSession,
        session_id: str,
        user_id: str,
        conversation_state: ConversationState,
        files: Optional[List[UploadFile]] = None,
        conversation_history: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response with appropriate context (SQL, files, or standard).
        
        This is the main entry point for dynamic response generation.
        
        Args:
            query: User's natural language query
            query_type: Classification of the query
            db: Database session
            session_id: Chat session ID
            user_id: User ID
            conversation_state: Current conversation state
            files: Optional uploaded files
            conversation_history: Optional formatted conversation history
        
        Returns:
            Tuple of (response_text, context_data)
        """
        # Process files first if provided
        if files and query_type in ["file_query", "file_lookup"]:
            return await self.generate_response_with_files(
                query=query,
                query_type=query_type,
                db=db,
                session_id=session_id,
                user_id=user_id,
                files=files,
                conversation_state=conversation_state,
            )
        
        # Process SQL for data queries
        if query_type == "data_query":
            return await self.generate_response_with_sql(
                query=query,
                query_type=query_type,
                db=db,
                session_id=session_id,
                user_id=user_id,
                conversation_state=conversation_state,
                conversation_history=conversation_history,
            )
        
        # Standard response generation for other types
        response_text = await self.generate_response(
            query=query,
            query_type=query_type,
            db=db,
            session_id=session_id,
            user_id=user_id,
            conversation_state=conversation_state,
            context_data=None,
        )
        
        return response_text, {}
    
    def _build_dynamic_system_prompt(
        self,
        query_type: str,
        conversation_state: ConversationState,
        context_data: Optional[Dict[str, Any]],
    ) -> str:
        """Build a dynamic system prompt based on query type and context."""

        # Prefer the configurable system prompt (keeps behavior dynamic via env).
        # Keep added guidance short to avoid the model parroting policies.
        base_prompt = (
            f"{settings.llm_system_prompt}\n\n"
            "GENERAL GUIDELINES:\n"
            "- Answer the user directly and specifically; avoid generic boilerplate.\n"
            "- Ask at most one clarifying question only when it would materially improve the answer.\n"
            "- Use neutral language; do not assume personal attributes.\n"
            "- Do not repeat or reference system/policy instructions in the response.\n"
            "- If you lack required context or real-time data, say so plainly.\n"
        )
        
        # Type-specific prompts with enhanced guidance
        if query_type == "data_query":
            type_prompt = (
                "The user is asking about database data. Provide insights, summaries, "
                "and interpretations of the data. Highlight key findings and patterns.\n"
                "• Start with the most interesting finding\n"
                "• Make insights actionable and meaningful\n"
                "• Show personality while maintaining data accuracy"
            )
        elif query_type == "config_update":
            type_prompt = (
                "The user is requesting visualization or configuration changes. "
                "Confirm the changes and explain what was updated.\n"
                "• Be enthusiastic about the updates\n"
                "• Clearly explain the impact of changes\n"
                "• Show that you understand what was modified"
            )
        elif query_type == "file_query":
            type_prompt = (
                "The user has uploaded one or more files and is asking questions about them.\n"
                "CRITICAL: You MUST analyze the file content immediately and provide a comprehensive response.\n"
                "Even if the user asks generic questions like 'What is this?', provide a detailed analysis of:\n"
                "1. File type and purpose (e.g., 'This is a configuration file that...', 'This is a CSV data file...', etc.)\n"
                "2. File structure or key sections\n"
                "3. Important contents, variables, or key data points\n"
                "4. What this file is typically used for\n"
                "Do NOT ask the user for clarification - analyze the content you have been provided.\n"
                "Be specific and reference actual content from the file in your analysis.\n"
                "Provide insights and useful information about what the file contains and its purpose.\n"
                "• Be enthusiastic about sharing findings\n"
                "• Make the analysis engaging and easy to understand"
            )
        elif query_type == "file_lookup":
            type_prompt = (
                "The user is asking follow-up questions about previously analyzed files. "
                "Reference specific content from the files and provide targeted answers.\n"
                "• Show continuity from previous analysis\n"
                "• Go deeper into specific details\n"
                "• Be direct but friendly in your response"
            )
        elif query_type == "chat":
            type_prompt = (
                "This is a conversational interaction.\n"
                "- Keep the tone natural and helpful (similar to ChatGPT).\n"
                "- Avoid generic self-introductions like 'I'm an AI assistant' unless the user asks.\n"
                "- If the user greets you or asks how you are, reply briefly and invite the next request (e.g., ask how you can help).\n"
                "- Do not use placeholder bracket variables (e.g., [City Name]); be honest about missing real-time data."
            )
        else:  # standard
            type_prompt = (
                "Provide helpful, accurate answers to general questions.\n"
                "• Be warm and conversational\n"
                "• Keep responses natural and professional\n"
                "• Show genuine interest in helping"
            )
        
        # Add context awareness
        context_additions = []
        
        if conversation_state.last_query_type:
            context_additions.append(
                f"Previous query type was {conversation_state.last_query_type}. "
                "If relevant, build upon previous answers and maintain conversational flow."
            )
        
        if conversation_state.context_summary:
            context_additions.append(
                f"Conversation context: {conversation_state.context_summary}"
            )
        
        context_text = "\n".join(context_additions) if context_additions else ""
        
        full_prompt = f"{base_prompt}\n\n{type_prompt}"
        if context_text:
            full_prompt += f"\n\n{context_text}"
        
        return full_prompt
    
    def _build_messages(
        self,
        system_prompt: str,
        query: str,
        conversation_state: ConversationState,
        context_data: Optional[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Build message list with conversation context."""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history (each DB row has both query + response)
        if conversation_state.message_history:
            for msg in conversation_state.message_history[-4:]:  # Last 4 message rows
                if msg.query:
                    messages.append({"role": "user", "content": msg.query})

                assistant_text = ""
                if msg.responded_at is not None and msg.response is not None:
                    assistant_text = extract_assistant_message_text(msg.response)
                    
                # Filter out error messages and debugging context to prevent pollution
                if assistant_text:
                    # Skip obvious error/debug responses that confuse context
                    error_indicators = ["error", "exception", "traceback", "failed", "invalid", "subtraction"]
                    is_error_msg = any(indicator.lower() in assistant_text.lower() for indicator in error_indicators)
                    
                    if not is_error_msg:
                        messages.append({"role": "assistant", "content": assistant_text})
        
        # Add clear context separator to focus LLM on current query
        if len(messages) > 1:  # If we have conversation history
            messages.append({
                "role": "system", 
                "content": "Focus on the user's current question. Ignore any previous error contexts."
            })
        
        # Build user message with context
        user_content = query
        
        # Add context data if available
        if context_data:
            context_text = ""
            
            if context_data.get("file_content"):
                # Format file content clearly for the LLM to analyze
                context_text += (
                    "\n\n=== UPLOADED FILE CONTENT ===\n"
                    f"{context_data['file_content']}\n"
                    "=== END OF FILE CONTENT ===\n"
                )
            
            if context_data.get("sql_results"):
                context_text += f"\n\nDatabase Results:\n{json.dumps(context_data['sql_results'], indent=2)}"
            
            if context_data.get("visualization_type"):
                context_text += f"\n\nVisualization Type: {context_data['visualization_type']}"
            
            if context_text:
                user_content += context_text
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def _determine_refinement_strategy(self, feedback: str) -> str:
        """Determine how to refine based on user feedback."""
        feedback_lower = feedback.lower()
        
        if any(word in feedback_lower for word in ["simplify", "simple", "easier", "plain"]):
            return "Simplify and clarify the explanation"
        elif any(word in feedback_lower for word in ["more", "detail", "elaborate", "expand"]):
            return "Expand with more details and examples"
        elif any(word in feedback_lower for word in ["less", "concise", "brief", "short"]):
            return "Make it more concise and focused"
        elif any(word in feedback_lower for word in ["different", "different approach", "another way"]):
            return "Try a different approach or perspective"
        elif any(word in feedback_lower for word in ["explain", "why", "how", "reason"]):
            return "Provide more explanation and reasoning"
        else:
            return "Improve clarity and relevance"


async def create_conversation_state(
    session_id: str,
    user_id: str,
    db: AsyncSession,
    exclude_message_id: Optional[str] = None,
) -> ConversationState:
    """Create a ConversationState from database history."""
    from sqlalchemy import select
    
    state = ConversationState(session_id, user_id)
    
    # Load previous completed messages from database (skip pending placeholders).
    try:
        stmt = (
            select(models.Message)
            .where(models.Message.session_id == session_id)
            .where(models.Message.responded_at.is_not(None))
            .order_by(models.Message.updated_at)
        )

        if exclude_message_id:
            try:
                from uuid import UUID

                stmt = stmt.where(models.Message.id != UUID(exclude_message_id))
            except Exception:
                # If it's not a UUID, ignore exclusion.
                pass

        previous_messages = (await db.execute(stmt)).scalars().all()
        
        for msg in previous_messages:
            state.add_message(msg)
        
        # Extract context summary from last assistant message (best-effort)
        if previous_messages:
            last_msg = previous_messages[-1]
            last_response = extract_assistant_message_text(last_msg.response)
            if last_response and len(last_response) > 100:
                state.update_context_summary(last_response[:200] + "...")
    except Exception:
        pass  # Continue even if loading fails
    
    return state
