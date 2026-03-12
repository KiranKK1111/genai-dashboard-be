"""
Response Composer - Frames raw results into ChatGPT-like conversational responses.

Takes raw extraction/query results and transforms them into:
- Conversational titles and content blocks
- Structured followup suggestions
- Normalized intents
- Proper framing for different response modes
"""

from __future__ import annotations

import asyncio
import json
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import textwrap

# Timeouts for LLM calls (parallel execution means these don't stack)
_LLM_TIMEOUT_SECONDS = 20        # General narrative timeout
_FOLLOWUP_TIMEOUT_SECONDS = 15   # Follow-ups are lower priority

from app.services.dynamic_visualization_generator import DynamicVisualizationGenerator
from .. import llm

logger = logging.getLogger(__name__)


class ContentBlockType(Enum):
    """Types of content blocks in a response."""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    BULLETS = "bullets"
    NUMBERED = "numbered"
    CALLOUT = "callout"
    TABLE = "table"
    CODE = "code"


class CalloutVariant(Enum):
    """Callout block variants."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    NEXT = "next"
    ERROR = "error"


@dataclass
class ContentBlock:
    """Single content block in a response."""
    type: ContentBlockType
    text: Optional[str] = None
    items: Optional[List[str]] = None
    variant: Optional[str] = None
    headers: Optional[List[str]] = None  # For tables
    rows: Optional[List[List[str]]] = None  # For tables
    language: Optional[str] = None  # For code blocks
    emoji: Optional[str] = None  # Emoji for visual appeal
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        type_val = self.type.value if hasattr(self.type, 'value') else str(self.type)
        result = {"type": type_val}
        if self.text:
            result["text"] = self.text
        if self.items:
            result["items"] = self.items
        if self.variant:
            result["variant"] = self.variant
        if self.headers:
            result["headers"] = self.headers
        if self.rows:
            result["rows"] = self.rows
        if self.language:
            result["language"] = self.language
        if self.emoji:
            result["emoji"] = self.emoji
        return result


@dataclass
class FollowUp:
    """Follow-up suggestion for the user."""
    id: str
    text: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "text": self.text}


@dataclass
class ArtifactsMetadata:
    """Metadata about artifacts used in the response."""
    files_used: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    sql: Optional[str] = None
    files_analyzed: int = 0  # For file lookup responses
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_used": self.files_used,
            "citations": self.citations,
            "sql": self.sql,
            "files_analyzed": self.files_analyzed,
        }


@dataclass
class AssistantMessage:
    """Structured assistant message with content blocks."""
    role: str = "assistant"
    title: Optional[str] = None
    content: List[ContentBlock] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "title": self.title,
            "content": [block.to_dict() for block in self.content],
        }


class ResponseComposer:
    """
    Composes raw backend results into ChatGPT-like conversational responses.
    
    ZERO HARDCODING PRINCIPLE:
    - No hardcoded emojis or visual elements
    - LLM dynamically decides emoji usage based on context
    - All formatting is data-driven and adaptive
    """
    
    @staticmethod
    def compose_file_response(
        filename: str,
        summary: str,
        file_id: str,
        intent: str = "summarize_uploaded_file",
    ) -> tuple[AssistantMessage, ArtifactsMetadata, List[FollowUp]]:
        """
        Compose a response for file analysis.
        
        Args:
            filename: Name of the uploaded file
            summary: Extracted summary/content from file
            file_id: ID of the file in database
            intent: What the user was trying to do
            
        Returns:
            Tuple of (assistant_message, artifacts, followups)
        """
        # Generate title (no hardcoded emoji - let LLM decide)
        title = f"Summary of {filename}"
        
        # Parse summary for structure (look for headers/sections)
        blocks = ResponseComposer._parse_summary_into_blocks(filename, summary, mode="file")
        
        # Create assistant message
        assistant = AssistantMessage(
            title=title,
            content=blocks
        )
        
        # Create artifacts
        artifacts = ArtifactsMetadata(
            files_used=[
                {"file_id": file_id, "filename": filename}
            ]
        )
        
        # Generate contextual follow-ups
        followups = ResponseComposer._generate_file_followups(filename, intent)
        
        return assistant, artifacts, followups
    
    @staticmethod
    def compose_file_lookup_response(
        query: str,
        answer: str,
        file_count: int = 0,
    ) -> tuple[AssistantMessage, ArtifactsMetadata, List[FollowUp]]:
        """
        Compose a response for file follow-up queries (ChatGPT-like).
        
        Handles questions about previously uploaded files with proper context preservation.
        
        Args:
            query: The follow-up question about the file
            answer: The LLM-generated answer based on file context
            file_count: Number of files referenced in the answer
            
        Returns:
            Tuple of (assistant_message, artifacts, followups)
        """
        # Generate title (no hardcoded emoji - let LLM decide)
        title = "File Analysis Result"
        
        # Create structured response
        blocks = []
        
        # Add the answer as the main content (no hardcoded emoji)
        blocks.append(ContentBlock(
            type=ContentBlockType.PARAGRAPH,
            text=answer
        ))
        
        # Create assistant message
        assistant = AssistantMessage(
            title=title,
            content=blocks
        )
        
        # Create artifacts noting files were analyzed
        artifacts = ArtifactsMetadata(
            files_analyzed=file_count
        )
        
        # Generate follow-up questions about the files
        followups = ResponseComposer._generate_file_followups(
            "uploaded documents",
            "answer_file_question"
        )
        
        return assistant, artifacts, followups
    
    @staticmethod
    async def compose_sql_response_async(
        query: str,
        results: List[Dict[str, Any]],
        execution_time: float,
        intent: str = "run_sql",
        skip_llm_calls: bool = False,
        user_query: str = "",
    ) -> tuple[AssistantMessage, ArtifactsMetadata, List[FollowUp], Optional[Dict[str, Any]]]:
        """
        Compose a response for SQL query results with dynamic AI-powered visualizations.

        Args:
            query:          The SQL query that was executed
            results:        Result rows
            execution_time: Query execution time in seconds
            intent:         What the user was trying to do
            skip_llm_calls: Skip LLM for simple queries (performance)
            user_query:     Original natural-language question (used for narrative + follow-ups)

        Returns:
            Tuple of (assistant_message, artifacts, followups, visualizations)
        """
        row_count = len(results)

        # ── 1-5. Run LLM-powered steps concurrently for performance ──────────
        title = ResponseComposer._build_title(user_query, row_count)

        if user_query and not skip_llm_calls:
            # Run narrative, visualizations, and follow-ups in parallel
            narrative, visualizations, followups = await asyncio.gather(
                ResponseComposer._generate_narrative_async(
                    user_query=user_query, sql=query, results=results
                ),
                ResponseComposer._generate_visualizations_async(results, query, row_count),
                ResponseComposer.generate_dynamic_sql_followups(
                    results=results, user_query=user_query, sql_query=query
                ),
            )
        else:
            narrative = ResponseComposer._fallback_narrative(row_count, results)
            visualizations = ResponseComposer._generate_visualizations(results, query, row_count)
            followups = ResponseComposer._generate_sql_followups(row_count, intent, results, query)

        # Always ensure a table visualization is present in the viz payload
        visualizations = ResponseComposer._ensure_table_visualization(visualizations, results)

        # ── Content blocks ─────────────────────────────────────────────────────
        blocks: List[ContentBlock] = []
        blocks.append(ContentBlock(type=ContentBlockType.PARAGRAPH, text=narrative))

        if results:
            headers = list(results[0].keys())
            display_rows = [
                [str(row.get(h, "")) for h in headers]
                for row in results[:50]
            ]
            blocks.append(ContentBlock(
                type=ContentBlockType.TABLE,
                headers=headers,
                rows=display_rows,
            ))
            if row_count > 50:
                blocks.append(ContentBlock(
                    type=ContentBlockType.CALLOUT,
                    variant="info",
                    text=f"Showing 50 of {row_count:,} rows. Ask me to filter, sort, or export for the full set.",
                ))

        assistant = AssistantMessage(title=title, content=blocks)
        artifacts = ArtifactsMetadata(sql=query)

        return assistant, artifacts, followups, visualizations

    # ── Narrative helpers ──────────────────────────────────────────────────────

    @staticmethod
    async def _generate_narrative_async(
        user_query: str,
        sql: str,
        results: List[Dict[str, Any]],
    ) -> str:
        """Generate a ChatGPT-style analytical narrative for query results."""
        row_count = len(results)
        # Build a compact data summary for the LLM
        if row_count == 0:
            data_summary = "The query returned no rows."
        elif row_count == 1:
            data_summary = f"Result: {dict(list(results[0].items())[:8])}"
        else:
            cols = list(results[0].keys())
            sample = [dict(list(r.items())[:6]) for r in results[:3]]
            data_summary = (
                f"{row_count} rows, columns: {', '.join(cols[:10])}\n"
                f"First 3 rows: {sample}"
            )

        prompt = (
            f"A user asked: \"{user_query}\"\n"
            f"SQL executed: {sql}\n"
            f"Data: {data_summary}\n\n"
            "Write a clear, direct 1-3 sentence answer to the user's question based on this data. "
            "Be specific — mention key numbers, trends, or findings. "
            "Use **bold** for important numbers or values. "
            "Do NOT say 'I found X rows' or repeat the SQL. Just answer the question naturally."
        )
        try:
            response = await asyncio.wait_for(
                llm.call_llm(
                    [
                        {"role": "system", "content": (
                            "You are a data analyst. Answer questions about query results "
                            "concisely and clearly, like ChatGPT would. Respond in plain text "
                            "with markdown bold (**value**) for key numbers only."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                    max_tokens=150,
                    temperature=0.2,
                ),
                timeout=_LLM_TIMEOUT_SECONDS,
            )
            narrative = str(response).strip()
            if narrative:
                return narrative
        except Exception as e:
            logger.debug("[RESPONSE_COMPOSER] Narrative generation failed: %s", e)
        return ResponseComposer._fallback_narrative(row_count, results)

    @staticmethod
    def _fallback_narrative(row_count: int, results: List[Dict[str, Any]]) -> str:
        """Simple fallback narrative when LLM is unavailable."""
        if row_count == 0:
            return "No results were found for your query."
        if row_count == 1 and results:
            # For single-row results (COUNT, SUM, etc.) show the value directly
            row = results[0]
            pairs = ", ".join(f"**{k}**: {v}" for k, v in list(row.items())[:4])
            return f"Here is the result: {pairs}."
        return f"Found **{row_count:,}** records matching your query."

    @staticmethod
    def _build_title(user_query: str, row_count: int) -> str:
        """Build a meaningful title from the user's question."""
        if not user_query:
            return f"Query Results ({row_count:,} rows)"
        # Capitalise first letter, strip trailing punctuation
        title = user_query.strip().rstrip("?!.")
        title = title[0].upper() + title[1:] if title else "Query Results"
        return title

    @staticmethod
    def _ensure_table_visualization(
        visualizations: Optional[Dict[str, Any]],
        results: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Guarantee a table visualization is always present in the viz payload."""
        if not results:
            return visualizations

        table_viz = {
            "type": "table",
            "columns": list(results[0].keys()),
            "data": results[:500],          # cap rows for frontend safety
            "row_count": len(results),
            "is_default": True,
        }

        if visualizations is None:
            return {"charts": [], "table": table_viz, "default": "table"}

        if isinstance(visualizations, dict):
            if "table" not in visualizations:
                visualizations["table"] = table_viz
            if "default" not in visualizations:
                visualizations["default"] = "table"

        return visualizations
    
    @staticmethod
    def compose_sql_response(
        query: str,
        results: List[Dict[str, Any]],
        execution_time: float,
        intent: str = "run_sql",
    ) -> tuple[AssistantMessage, ArtifactsMetadata, List[FollowUp], Optional[Dict[str, Any]]]:
        """
        DEPRECATED: Use compose_sql_response_async for AI-powered visualizations.
        
        Compose a response for SQL query results (sync version for backward compatibility).
        
        Args:
            query: The SQL query that was executed
            results: Result rows
            execution_time: Query execution time in seconds
            intent: What the user was trying to do
            
        Returns:
            Tuple of (assistant_message, artifacts, followups, visualization_dict)
        """
        # Generate natural intro (no hardcoded emojis)
        row_count = len(results)
        title = f"Query Results ({row_count} rows)"
        
        blocks = []
        
        # Intro paragraph (clean professional response)
        if row_count > 0:
            blocks.append(ContentBlock(
                type=ContentBlockType.PARAGRAPH,
                text=f"I found {row_count} rows matching your query (took {execution_time:.2f}s)."
            ))
        else:
            blocks.append(ContentBlock(
                type=ContentBlockType.PARAGRAPH,
                text="Your query returned no results."
            ))
        
        # Show first few rows as table if data exists
        if results and len(results) > 0:
            first_row = results[0]
            headers = list(first_row.keys())
            rows = []
            
            # Take first 10 rows for display
            for row in results[:10]:
                rows.append([str(row.get(h, "")) for h in headers])
            
            blocks.append(ContentBlock(
                type=ContentBlockType.TABLE,
                headers=headers,
                rows=rows
            ))
            
            if len(results) > 10:
                blocks.append(ContentBlock(
                    type=ContentBlockType.CALLOUT,
                    variant="info",
                    text=f"Showing first 10 of {len(results)} rows. Use filtering or export for the full result set."
                ))
        
        # Suggest next actions (no hardcoded emojis)
        blocks.append(ContentBlock(
            type=ContentBlockType.CALLOUT,
            variant="next",
            text="Would you like to create a visualization, export results to CSV, or refine the query?"
        ))
        
        assistant = AssistantMessage(
            title=title,
            content=blocks
        )
        
        artifacts = ArtifactsMetadata(sql=query)
        
        # Generate simple fallback visualizations (synchronous)
        visualizations = ResponseComposer._generate_visualizations(results, query, row_count)
        
        # Generate context-aware follow-ups
        followups = ResponseComposer._generate_sql_followups(row_count, intent, results, query)
        
        return assistant, artifacts, followups, visualizations
    
    @staticmethod
    async def compose_chat_response(
        user_query: str,
        answer: str,
        intent: str = "general_answer",
    ) -> tuple[AssistantMessage, ArtifactsMetadata, List[FollowUp]]:
        """
        Compose a response for general chat/conversation.
        
        Args:
            user_query: The user's question
            answer: The assistant's answer
            intent: What type of question this is
            
        Returns:
            Tuple of (assistant_message, artifacts, followups)
        """
        # For chat, keep it simple (no hardcoded emojis - let LLM decide)
        blocks = [
            ContentBlock(
                type=ContentBlockType.PARAGRAPH,
                text=answer
            )
        ]
        
        # Parse answer for structure (headings, bullets)
        if any(c in answer for c in ['•', '-', '*', '1.', '2.', '3.']):
            blocks = ResponseComposer._parse_summary_into_blocks(user_query, answer, mode="chat")
        
        assistant = AssistantMessage(
            title=None,  # No title for plain chat
            content=blocks
        )
        
        artifacts = ArtifactsMetadata()

        # No follow-up suggestions for conversational chat (user preference)
        followups: List[FollowUp] = []

        return assistant, artifacts, followups
    
    @staticmethod
    def _parse_summary_into_blocks(
        filename: str, 
        summary: str,
        mode: str = "file"
    ) -> List[ContentBlock]:
        """Parse summary text into structured content blocks (no hardcoded emojis)."""
        blocks = []
        
        # For file mode, add opening line (no hardcoded emoji)
        if mode == "file":
            blocks.append(ContentBlock(
                type=ContentBlockType.PARAGRAPH,
                text=f"I read {filename}. Here's a structured summary:"
            ))
        
        # Try to detect sections (lines starting with ##, ###, or numbered)
        lines = summary.strip().split('\n')
        current_section = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                # Empty line - flush current section
                if current_section:
                    blocks.append(ContentBlock(
                        type=ContentBlockType.PARAGRAPH,
                        text='\n'.join(current_section)
                    ))
                    current_section = []
            elif line.startswith('## ') or line.startswith('# '):
                # Heading
                if current_section:
                    blocks.append(ContentBlock(
                        type=ContentBlockType.PARAGRAPH,
                        text='\n'.join(current_section)
                    ))
                    current_section = []
                
                heading_text = line.lstrip('#').strip()
                blocks.append(ContentBlock(
                    type=ContentBlockType.HEADING,
                    text=heading_text
                ))
            elif line.startswith('- ') or line.startswith('* '):
                # Bullet point
                if not blocks or blocks[-1].type != ContentBlockType.BULLETS:
                    if current_section:
                        blocks.append(ContentBlock(
                            type=ContentBlockType.PARAGRAPH,
                            text='\n'.join(current_section)
                        ))
                        current_section = []
                    
                    # Start new bullet list (no hardcoded emoji)
                    blocks.append(ContentBlock(
                        type=ContentBlockType.BULLETS,
                        items=[]
                    ))
                
                # Add to current bullet list
                if blocks and blocks[-1].type == ContentBlockType.BULLETS:
                    blocks[-1].items.append(line.lstrip('-* ').strip())
            elif line and line[0].isdigit() and ('. ' in line or ') ' in line):
                # Numbered item
                if not blocks or blocks[-1].type != ContentBlockType.NUMBERED:
                    if current_section:
                        blocks.append(ContentBlock(
                            type=ContentBlockType.PARAGRAPH,
                            text='\n'.join(current_section)
                        ))
                        current_section = []
                    
                    # Start new numbered list (no hardcoded emoji)
                    blocks.append(ContentBlock(
                        type=ContentBlockType.NUMBERED,
                        items=[]
                    ))
                
                # Remove numbering and add to list
                if blocks and blocks[-1].type == ContentBlockType.NUMBERED:
                    item_text = line
                    for sep in [') ', '. ']:
                        if sep in item_text:
                            item_text = item_text.split(sep, 1)[1]
                            break
                    blocks[-1].items.append(item_text.strip())
            else:
                # Regular text
                current_section.append(line)
        
        # Flush remaining (no hardcoded emoji)
        if current_section:
            blocks.append(ContentBlock(
                type=ContentBlockType.PARAGRAPH,
                text='\n'.join(current_section)
            ))
        
        return blocks
    
    @staticmethod
    async def _generate_visualizations_async(
        results: List[Dict[str, Any]],
        query: str,
        row_count: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate dynamic, AI-centric visualization configuration for query results.
        
        Uses LLM to intelligently analyze the query results and automatically generate:
        - Field schema with type detection and role hints (id, category, measure, time, geo, etc.)
        - Dynamic aggregators suitable for bar/line/pie charts
        - Interactive controls that adapt to the data
        - No hardcoding - everything generated based on actual data
        
        Args:
            results: Query result rows
            query: Original SQL query
            row_count: Total rows returned
            
        Returns:
            Single multi_viz visualization with dynamic schema and aggregators, or None if no results
        """
        if not results or row_count == 0:
            return None
        
        try:
            # Use new dynamic generator to create AI-powered visualization
            visualization = await DynamicVisualizationGenerator.generate_multi_viz(
                results=results
            )
            return visualization
        except Exception as e:
            logger.debug(f"WARNING: Dynamic visualization generation failed: {e}, using fallback")
            # Fallback to simple multi_viz if AI generation fails (no hardcoded emoji)
            return {
                "chart_id": "v1",
                "type": "multi_viz",
                "title": "Query Results",
                "subtitle": f"{row_count} rows returned",
                "description": "Multi-visualization panel - switch between different view types",
                "config": {
                    "primary_view": "table",
                    "available_views": ["table", "bar", "line", "pie"]
                },
                "show_raw_data": True,
                "exportable": True,
                "full_screen_enabled": True
            }
    
    @staticmethod
    def _generate_visualizations(
        results: List[Dict[str, Any]],
        query: str,
        row_count: int,
    ) -> Optional[Dict[str, Any]]:
        """
        DEPRECATED: Use _generate_visualizations_async instead.
        Kept for backward compatibility - returns simple multi_viz.
        """
        if not results or row_count == 0:
            return None
        
        # Return fallback simple multi-view visualization (no hardcoded emoji)
        return {
            "chart_id": "v1",
            "type": "multi_viz",
            "title": "Query Results",
            "subtitle": f"{row_count} rows returned",
            "description": "Multi-visualization panel - switch between different view types",
            "config": {
                "primary_view": "table",
                "available_views": ["table", "bar", "line", "pie"]
            },
            "show_raw_data": True,
            "exportable": True,
            "full_screen_enabled": True
        }
    
    @staticmethod
    def _generate_file_followups(filename: str, intent: str) -> List[FollowUp]:
        """Generate contextual follow-up questions for file responses (static fallback)."""
        return [
            FollowUp(
                id="fu_1",
                text=f"Extract key action items from {filename}"
            ),
            FollowUp(
                id="fu_2",
                text="Create a checklist or timeline based on this"
            ),
            FollowUp(
                id="fu_3",
                text="Compare this with another document"
            ),
        ]
    
    @staticmethod
    async def generate_dynamic_file_followups(
        filename: str,
        content_preview: str,
        intent: str = "summarize"
    ) -> List[FollowUp]:
        """
        Generate context-aware follow-up suggestions using LLM based on actual file content.
        
        Args:
            filename: Name of the file
            content_preview: First ~500 chars of file content for context
            intent: What the user was trying to do
            
        Returns:
            List of contextual follow-up suggestions
        """
        # Truncate content for LLM prompt
        content_snippet = content_preview[:500] + "..." if len(content_preview) > 500 else content_preview
        
        prompt = f"""Based on this uploaded file, generate 3 highly relevant follow-up questions a user might ask.

FILE: {filename}
CONTENT PREVIEW:
{content_snippet}

Generate follow-up questions that are:
1. Specific to THIS file's actual content (e.g., specific topics, entities, or data it contains)
2. Actionable (extract, analyze, compare, summarize specific parts)
3. Short (under 50 characters each)

Respond ONLY with JSON:
{{"followups": ["question1", "question2", "question3"]}}"""
        
        try:
            response = await asyncio.wait_for(
                llm.call_llm(
                    [{"role": "system", "content": "You generate context-aware follow-up questions. Respond ONLY with valid JSON."},
                     {"role": "user", "content": prompt}],
                    stream=False,
                    max_tokens=200,
                    temperature=0.3,
                ),
                timeout=_LLM_TIMEOUT_SECONDS,
            )

            # Parse response
            response_str = str(response).strip().replace('```json', '').replace('```', '').strip()
            json_match = re.search(r'\{[^{}]*"followups"[^{}]*\}', response_str, re.DOTALL)

            if json_match:
                result = json.loads(json_match.group(0))
                followup_texts = result.get('followups', [])[:3]

                return [
                    FollowUp(id=f"fu_{i+1}", text=text)
                    for i, text in enumerate(followup_texts) if text
                ]
        except asyncio.TimeoutError:
            logger.warning("[RESPONSE_COMPOSER] File followup LLM call timed out after %ds", _LLM_TIMEOUT_SECONDS)
        except json.JSONDecodeError as e:
            logger.warning("[RESPONSE_COMPOSER] JSON parse error in file followup generation: %s", e)
        except Exception as e:
            logger.warning("[RESPONSE_COMPOSER] Dynamic file followup generation failed: %s", e)

        # Fallback to static followups
        return ResponseComposer._generate_file_followups(filename, intent)
    
    @staticmethod
    async def generate_dynamic_sql_followups(
        results: List[Dict[str, Any]],
        user_query: str,
        sql_query: str = ""
    ) -> List[FollowUp]:
        """
        Generate context-aware follow-up suggestions for SQL/database results.
        
        Args:
            results: Query result rows
            user_query: Original user question
            sql_query: The SQL that was executed
            
        Returns:
            List of contextual follow-up suggestions
        """
        if not results:
            return [
                FollowUp(id="fu_1", text="Try different filters or conditions"),
                FollowUp(id="fu_2", text="Show me sample data from this table"),
                FollowUp(id="fu_3", text="What columns are available?"),
            ]
        
        # Get column names and sample data
        columns = list(results[0].keys()) if results else []
        sample_row = results[0] if results else {}
        row_count = len(results)
        
        # Create data summary for LLM
        data_summary = f"Columns: {', '.join(columns[:10])}"
        if sample_row:
            sample_preview = str(sample_row)[:200]
            data_summary += f"\nSample row: {sample_preview}"
        
        prompt = f"""Based on this database query result, generate 3 relevant follow-up questions.

USER QUERY: {user_query}
RESULT: {row_count} rows returned
{data_summary}

Generate follow-ups that:
1. Build on THIS specific result (drill down, filter, aggregate)
2. Explore related data (joins, comparisons)
3. Are short (under 50 chars each)

Respond ONLY with JSON:
{{"followups": ["question1", "question2", "question3"]}}"""
        
        try:
            response = await asyncio.wait_for(
                llm.call_llm(
                    [{"role": "system", "content": "You generate context-aware follow-up questions for data queries. Respond ONLY with valid JSON."},
                     {"role": "user", "content": prompt}],
                    stream=False,
                    max_tokens=200,
                    temperature=0.3,
                ),
                timeout=_FOLLOWUP_TIMEOUT_SECONDS,
            )

            response_str = str(response).strip().replace('```json', '').replace('```', '').strip()
            json_match = re.search(r'\{[^{}]*"followups"[^{}]*\}', response_str, re.DOTALL)

            if json_match:
                result = json.loads(json_match.group(0))
                followup_texts = result.get('followups', [])[:3]

                return [
                    FollowUp(id=f"fu_{i+1}", text=text)
                    for i, text in enumerate(followup_texts) if text
                ]
        except asyncio.TimeoutError:
            logger.warning("[RESPONSE_COMPOSER] SQL followup LLM call timed out after %ds", _FOLLOWUP_TIMEOUT_SECONDS)
        except json.JSONDecodeError as e:
            logger.warning("[RESPONSE_COMPOSER] JSON parse error in SQL followup generation: %s", e)
        except Exception as e:
            logger.warning("[RESPONSE_COMPOSER] Dynamic SQL followup generation failed: %s", e)

        # Fallback to static
        return ResponseComposer._generate_sql_followups(row_count, "run_sql", results, sql_query)
    
    @staticmethod
    def _generate_sql_followups(
        row_count: int,
        intent: str,
        results: Optional[List[Dict[str, Any]]] = None,
        query: Optional[str] = None,
    ) -> List[FollowUp]:
        """Generate context-aware follow-up questions for SQL responses."""
        followups = []
        
        if row_count == 0:
            # No results - suggest refinement
            followups.extend([
                FollowUp(
                    id="fu_1",
                    text="Try modifying the filters or conditions"
                ),
                FollowUp(
                    id="fu_2",
                    text="Show me a sample of this table"
                ),
                FollowUp(
                    id="fu_3",
                    text="What columns are available in this table?"
                ),
            ])
            return followups
        
        # Analyze query to suggest relevant follow-ups
        query_lower = (query or "").lower()
        
        # Extract context from query
        has_where = "where" in query_lower
        has_group = "group by" in query_lower
        has_order = "order by" in query_lower
        
        # Context-aware follow-ups based on query type (no hardcoded emojis)
        if row_count > 1000:
            # Large result set
            followups.append(FollowUp(
                id="fu_1",
                text=f"Filter by specific criteria (showing {row_count:,} rows)"
            ))
        elif row_count > 100:
            # Medium result set
            followups.append(FollowUp(
                id="fu_1",
                text=f"Create visualization of this data ({row_count} rows)"
            ))
        else:
            # Small result set
            followups.append(FollowUp(
                id="fu_1",
                text=f"Analyze this data ({row_count} rows)"
            ))
        
        # Add aggregation suggestion if not already grouped
        if not has_group and row_count > 50:
            followups.append(FollowUp(
                id="fu_2",
                text="Group by a category to see patterns"
            ))
        elif has_group:
            followups.append(FollowUp(
                id="fu_2",
                text="Sort or rank these grouped results"
            ))
        else:
            followups.append(FollowUp(
                id="fu_2",
                text="Export results to CSV or Excel"
            ))
        
        # Add refinement suggestion
        if has_where:
            followups.append(FollowUp(
                id="fu_3",
                text="Add more filters or conditions"
            ))
        else:
            followups.append(FollowUp(
                id="fu_3",
                text="Compare with different conditions"
            ))
        
        return followups
    
    @staticmethod
    async def _generate_chat_followups(user_query: str, intent: str, answer: str = "") -> List[FollowUp]:
        """Generate contextual follow-up questions for chat responses using LLM."""
        from app import llm
        
        # Use LLM to generate 3 contextual follow-ups based on the conversation
        prompt = f"""Based on this conversation, generate 3 relevant follow-up questions the user might ask next.

User asked: "{user_query}"

Assistant answered: "{answer[:500]}"

Generate 3 short, specific follow-up questions (not generic). Format as JSON array:
["question 1", "question 2", "question 3"]

Requirements:
- Make them specific to the topic discussed
- Keep each under 10 words
- Don't repeat the original question
- Focus on next logical steps or clarifications"""

        try:
            response = await llm.call_llm(
                [{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=200,
                temperature=0.7
            )
            
            response_text = str(response).strip()
            
            # Try to parse JSON array
            import json
            import re
            
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group(0))
                # Handle both array of strings and array of objects
                followups = []
                for i, q in enumerate(questions[:3]):
                    if isinstance(q, dict):
                        # LLM returned {"question": "..."} format
                        text = q.get('question') or q.get('text') or str(q)
                    else:
                        # LLM returned plain string
                        text = str(q)
                    followups.append(FollowUp(id=f"fu_{i+1}", text=text))
                return followups
        except Exception as e:
            logger.debug(f"[FOLLOWUPS] Dynamic generation failed: {e}, using fallback")
        
        # Fallback to generic questions if LLM fails
        return [
            FollowUp(id="fu_1", text="Tell me more about this"),
            FollowUp(id="fu_2", text="Give me an example"),
            FollowUp(id="fu_3", text="How does this compare to...?"),
        ]


# ============================================================================
# Clarification Response Builder
# ============================================================================

def build_clarification_lama_response(
    session_id: str,
    user_query: str,
    ambiguity_analysis: Any,  # AmbiguityAnalysis from clarification_engine
    message_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a LamaResponse dict for a clarification request.

    When the ClarificationEngine detects high ambiguity the route handler
    should return this response rather than running the full pipeline.

    The response:
    - Has mode="clarification"
    - Has routing.type="clarification"
    - Has NO visualizations (user must answer before we query)
    - Has NO followups (replaced by clarification block)
    - Includes a structured `clarification` block listing all questions

    Args:
        session_id: Current session ID
        user_query: The ambiguous user query
        ambiguity_analysis: AmbiguityAnalysis object from ClarificationEngine
        message_id: Optional pre-generated message UUID

    Returns:
        LamaResponse-compatible dict ready to return via ResponseWrapper
    """
    import time, uuid as _uuid

    _message_id = message_id or f"msg_{_uuid.uuid4().hex[:12]}"

    # Build human-friendly explanation
    reasoning = getattr(ambiguity_analysis, "reasoning", "")
    ambiguity_types = [
        a.value if hasattr(a, "value") else str(a)
        for a in getattr(ambiguity_analysis, "ambiguities", [])
    ]

    explanation = (
        "I need a little more information before I can answer precisely. "
        + (reasoning if reasoning else "The query could be interpreted in multiple ways.")
    )

    # Build the content blocks
    content_blocks = [
        {
            "type": "paragraph",
            "text": explanation,
        }
    ]

    # Serialize each clarification request as a structured question
    clarification_questions = []
    for req in getattr(ambiguity_analysis, "clarifications_needed", []):
        if hasattr(req, "to_dict"):
            clarification_questions.append(req.to_dict())
        elif isinstance(req, dict):
            clarification_questions.append(req)
        else:
            clarification_questions.append({"question": str(req)})

    return {
        "id": _message_id,
        "object": "chat.response",
        "created_at": int(time.time() * 1000),
        "session_id": session_id,
        "mode": "clarification",
        "assistant": {
            "role": "assistant",
            "title": "Could you clarify?",
            "content": content_blocks,
        },
        "artifacts": {
            "files_used": [],
            "citations": [],
            "sql": None,
        },
        "visualizations": None,
        "routing": {
            "type": "clarification",
            "intent": user_query,
            "confidence": float(getattr(ambiguity_analysis, "confidence", 0.5)),
        },
        "followups": [],
        "suggested_questions": [],
        "suggested_actions": [],
        "clarification": {
            "ambiguity_types": ambiguity_types,
            "questions": clarification_questions,
            "reasoning": reasoning,
            "can_proceed": bool(getattr(ambiguity_analysis, "can_proceed", True)),
        },
        "debug": {
            "clarification_triggered": True,
            "ambiguity_types": ambiguity_types,
        },
    }
