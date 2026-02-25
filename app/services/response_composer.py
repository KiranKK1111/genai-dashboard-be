"""
Response Composer - Frames raw results into ChatGPT-like conversational responses.

Takes raw extraction/query results and transforms them into:
- Conversational titles and content blocks
- Structured followup suggestions
- Normalized intents
- Proper framing for different response modes
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import textwrap

from app.services.dynamic_visualization_generator import DynamicVisualizationGenerator


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
        result = {"type": self.type.value}
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_used": self.files_used,
            "citations": self.citations,
            "sql": self.sql,
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
    """
    
    # Emoji mappings for visual appeal
    EMOJI_MAP = {
        "file": "📄",
        "summary": "📋",
        "goal": "🎯",
        "steps": "📍",
        "points": "💡",
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "next": "➡️",
        "sql": "🔍",
        "data": "📊",
        "table": "📑",
        "chart": "📈",
        "chat": "💬",
        "ai": "🤖",
        "export": "💾",
        "filter": "🔎",
        "config": "⚙️",
        "extract": "🎁",
        "checklist": "✓",
        "compare": "🔀",
    }
    
    @staticmethod
    def get_emoji(key: str, default: str = "•") -> str:
        """Get emoji for a key, with fallback."""
        return ResponseComposer.EMOJI_MAP.get(key.lower(), default)
    
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
        # Generate title
        title = f"{ResponseComposer.get_emoji('file')} Summary of {filename}"
        
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
    async def compose_sql_response_async(
        query: str,
        results: List[Dict[str, Any]],
        execution_time: float,
        intent: str = "run_sql",
    ) -> tuple[AssistantMessage, ArtifactsMetadata, List[FollowUp], Optional[Dict[str, Any]]]:
        """
        Compose a response for SQL query results with dynamic AI-powered visualizations.
        
        Uses LLM to generate intelligent visualization configurations based on the data.
        
        Args:
            query: The SQL query that was executed
            results: Result rows
            execution_time: Query execution time in seconds
            intent: What the user was trying to do
            
        Returns:
            Tuple of (assistant_message, artifacts, followups, visualization_dict_with_schema_and_aggregators)
        """
        # Generate natural intro
        row_count = len(results)
        title = f"{ResponseComposer.get_emoji('data')} Query Results ({row_count} rows)"
        
        blocks = []
        
        # Intro paragraph with emoji
        if row_count > 0:
            blocks.append(ContentBlock(
                type=ContentBlockType.PARAGRAPH,
                text=f"{ResponseComposer.get_emoji('success')} I found **{row_count} rows** matching your query (took {execution_time:.2f}s).",
                emoji=ResponseComposer.get_emoji('success')
            ))
        else:
            blocks.append(ContentBlock(
                type=ContentBlockType.PARAGRAPH,
                text=f"{ResponseComposer.get_emoji('warning')} Your query returned no results.",
                emoji=ResponseComposer.get_emoji('warning')
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
                rows=rows,
                emoji=ResponseComposer.get_emoji('table')
            ))
            
            if len(results) > 10:
                blocks.append(ContentBlock(
                    type=ContentBlockType.CALLOUT,
                    variant="info",
                    text=f"{ResponseComposer.get_emoji('info')} Showing first 10 of {len(results)} rows. Use filtering or export for the full result set.",
                    emoji=ResponseComposer.get_emoji('info')
                ))
        
        # Suggest next actions
        blocks.append(ContentBlock(
            type=ContentBlockType.CALLOUT,
            variant="next",
            text=f"{ResponseComposer.get_emoji('chart')} Would you like to create a visualization, export results to CSV, or refine the query?",
            emoji=ResponseComposer.get_emoji('chart')
        ))
        
        assistant = AssistantMessage(
            title=title,
            content=blocks
        )
        
        artifacts = ArtifactsMetadata(sql=query)
        
        # Generate DYNAMIC visualizations with AI-powered schema and aggregators
        visualizations = await ResponseComposer._generate_visualizations_async(results, query, row_count)
        
        # Generate context-aware follow-ups
        followups = ResponseComposer._generate_sql_followups(row_count, intent, results, query)
        
        return assistant, artifacts, followups, visualizations
    
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
        # Generate natural intro
        row_count = len(results)
        title = f"{ResponseComposer.get_emoji('data')} Query Results ({row_count} rows)"
        
        blocks = []
        
        # Intro paragraph with emoji
        if row_count > 0:
            blocks.append(ContentBlock(
                type=ContentBlockType.PARAGRAPH,
                text=f"{ResponseComposer.get_emoji('success')} I found **{row_count} rows** matching your query (took {execution_time:.2f}s).",
                emoji=ResponseComposer.get_emoji('success')
            ))
        else:
            blocks.append(ContentBlock(
                type=ContentBlockType.PARAGRAPH,
                text=f"{ResponseComposer.get_emoji('warning')} Your query returned no results.",
                emoji=ResponseComposer.get_emoji('warning')
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
                rows=rows,
                emoji=ResponseComposer.get_emoji('table')
            ))
            
            if len(results) > 10:
                blocks.append(ContentBlock(
                    type=ContentBlockType.CALLOUT,
                    variant="info",
                    text=f"{ResponseComposer.get_emoji('info')} Showing first 10 of {len(results)} rows. Use filtering or export for the full result set.",
                    emoji=ResponseComposer.get_emoji('info')
                ))
        
        # Suggest next actions
        blocks.append(ContentBlock(
            type=ContentBlockType.CALLOUT,
            variant="next",
            text=f"{ResponseComposer.get_emoji('chart')} Would you like to create a visualization, export results to CSV, or refine the query?",
            emoji=ResponseComposer.get_emoji('chart')
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
    def compose_chat_response(
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
        # For chat, keep it simple but add emojis
        blocks = [
            ContentBlock(
                type=ContentBlockType.PARAGRAPH,
                text=f"{ResponseComposer.get_emoji('ai')} {answer}",
                emoji=ResponseComposer.get_emoji('ai')
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
        
        followups = ResponseComposer._generate_chat_followups(user_query, intent)
        
        return assistant, artifacts, followups
    
    @staticmethod
    def _parse_summary_into_blocks(
        filename: str, 
        summary: str,
        mode: str = "file"
    ) -> List[ContentBlock]:
        """Parse summary text into structured content blocks with emoji support."""
        blocks = []
        
        # For file mode, add opening line
        if mode == "file":
            blocks.append(ContentBlock(
                type=ContentBlockType.PARAGRAPH,
                text=f"I read **{filename}**. Here's a structured summary:",
                emoji=ResponseComposer.get_emoji('file')
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
                        text='\n'.join(current_section),
                        emoji=ResponseComposer.get_emoji('info')
                    ))
                    current_section = []
            elif line.startswith('## ') or line.startswith('# '):
                # Heading
                if current_section:
                    blocks.append(ContentBlock(
                        type=ContentBlockType.PARAGRAPH,
                        text='\n'.join(current_section),
                        emoji=ResponseComposer.get_emoji('info')
                    ))
                    current_section = []
                
                heading_text = line.lstrip('#').strip()
                blocks.append(ContentBlock(
                    type=ContentBlockType.HEADING,
                    text=heading_text,
                    emoji=ResponseComposer.get_emoji('goal')
                ))
            elif line.startswith('- ') or line.startswith('* '):
                # Bullet point
                if not blocks or blocks[-1].type != ContentBlockType.BULLETS:
                    if current_section:
                        blocks.append(ContentBlock(
                            type=ContentBlockType.PARAGRAPH,
                            text='\n'.join(current_section),
                            emoji=ResponseComposer.get_emoji('info')
                        ))
                        current_section = []
                    
                    # Start new bullet list
                    blocks.append(ContentBlock(
                        type=ContentBlockType.BULLETS,
                        items=[],
                        emoji=ResponseComposer.get_emoji('points')
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
                            text='\n'.join(current_section),
                            emoji=ResponseComposer.get_emoji('info')
                        ))
                        current_section = []
                    
                    blocks.append(ContentBlock(
                        type=ContentBlockType.NUMBERED,
                        items=[],
                        emoji=ResponseComposer.get_emoji('steps')
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
        
        # Flush remaining
        if current_section:
            blocks.append(ContentBlock(
                type=ContentBlockType.PARAGRAPH,
                text='\n'.join(current_section),
                emoji=ResponseComposer.get_emoji('info')
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
            print(f"⚠️  Dynamic visualization generation failed: {e}, using fallback")
            # Fallback to simple multi_viz if AI generation fails
            return {
                "chart_id": "v1",
                "type": "multi_viz",
                "title": "Query Results",
                "subtitle": f"{row_count} rows returned",
                "description": "Multi-visualization panel - switch between different view types",
                "emoji": "🎨",
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
        
        # Return fallback simple multi-view visualization
        return {
            "chart_id": "v1",
            "type": "multi_viz",
            "title": "Query Results",
            "subtitle": f"{row_count} rows returned",
            "description": "Multi-visualization panel - switch between different view types",
            "emoji": "🎨",
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
        """Generate contextual follow-up questions for file responses."""
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
        
        # Context-aware follow-ups based on query type
        if row_count > 1000:
            # Large result set
            followups.append(FollowUp(
                id="fu_1",
                text=f"🔍 Filter by specific criteria (showing {row_count:,} rows)"
            ))
        elif row_count > 100:
            # Medium result set
            followups.append(FollowUp(
                id="fu_1",
                text=f"📊 Create visualization of this data ({row_count} rows)"
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
                text="📥 Export results to CSV or Excel"
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
    def _generate_chat_followups(user_query: str, intent: str) -> List[FollowUp]:
        """Generate contextual follow-up questions for chat responses."""
        return [
            FollowUp(
                id="fu_1",
                text="Tell me more about this"
            ),
            FollowUp(
                id="fu_2",
                text="Give me an example"
            ),
            FollowUp(
                id="fu_3",
                text="How does this compare to...?"
            ),
        ]
