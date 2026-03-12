"""
TOOL REGISTRY - Central registry for all AI tools

Manages available tools and provides unified execution interface.
Enables multi-tool planning and coordination.

Tools:
- SQL Executor
- File Reader
- Python Executor  
- Chart Generator
- Vector Search
- Knowledge Retrieval
- Memory Retrieval

Architecture:
    Master Orchestrator → Tool Registry → Tool Execution → Results
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class ToolType(str, Enum):
    """Available tool types."""
    SQL_EXECUTOR = "sql_executor"
    FILE_READER = "file_reader"
    PYTHON_EXECUTOR = "python_executor"
    CHART_GENERATOR = "chart_generator"
    VECTOR_SEARCH = "vector_search"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    MEMORY_RETRIEVAL = "memory_retrieval"
    CLARIFICATION = "clarification"


@dataclass
class ToolMetadata:
    """Metadata about a tool."""
    name: str
    tool_type: ToolType
    description: str
    parameters_schema: Dict[str, Any]
    capabilities: List[str]
    examples: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.tool_type.value,
            "description": self.description,
            "capabilities": self.capabilities,
            "examples": self.examples,
        }


class BaseTool(ABC):
    """Base class for all tools."""
    
    @abstractmethod
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Any:
        """Execute the tool with given parameters."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        """Get tool metadata."""
        pass


class SQLExecutorTool(BaseTool):
    """Tool for executing SQL queries."""
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute SQL query against database."""
        from .query_handler import build_data_query_response
        
        db_session = context.get("db_session")
        query = parameters.get("query") or context.get("user_query") or ""

        user_id = parameters.get("user_id") or context.get("user_id") or "orchestrator"
        session_id = parameters.get("session_id") or context.get("session_id") or "orchestrator"
        conversation_history = (
            parameters.get("conversation_history")
            or context.get("conversation_history")
            or ""
        )
        
        logger.info(f"[SQL EXECUTOR TOOL] Executing query: {query}")
        
        try:
            # Use existing query handler
            response_wrapper = await build_data_query_response(
                db=db_session,
                user_id=user_id,
                session_id=session_id,
                query=query,
                conversation_history=conversation_history,
            )
            
            # Extract results
            if response_wrapper.success and hasattr(response_wrapper.response, 'datasets'):
                datasets = response_wrapper.response.datasets
                if datasets and len(datasets) > 0:
                    return {
                        "success": True,
                        "rows": datasets[0].get("data", []),
                        "columns": datasets[0].get("columns", []),
                        "sql": response_wrapper.response.metadata.get("sql", ""),
                        "row_count": len(datasets[0].get("data", [])),
                    }
            
            return {
                "success": False,
                "error": "No data returned",
                "rows": [],
            }
        
        except Exception as e:
            logger.error(f"[SQL EXECUTOR TOOL] Error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "rows": [],
            }
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="SQL Executor",
            tool_type=ToolType.SQL_EXECUTOR,
            description="Queries database using natural language or SQL",
            parameters_schema={
                "query": {"type": "string", "required": True, "description": "Natural language query or SQL"},
                "conversation_history": {"type": "string", "required": False},
            },
            capabilities=[
                "Query any database table",
                "Automatic JOIN discovery",
                "Aggregations and filters",
                "Self-correcting SQL generation",
            ],
            examples=[
                "Show total sales by region",
                "Find top 10 users by revenue",
                "Count records in last month",
            ],
        )


class FileReaderTool(BaseTool):
    """Tool for reading and parsing files."""
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Read and parse file content."""
        from .file_handler import retrieve_relevant_chunks
        
        db_session = context.get("db_session")
        query = parameters.get("query") or context.get("user_query") or ""
        session_id = parameters.get("session_id") or context.get("session_id") or "orchestrator"
        
        logger.info(f"[FILE READER TOOL] Reading files for query: {query}")
        
        try:
            # Retrieve file chunks
            chunks = await retrieve_relevant_chunks(
                db=db_session,
                session_id=session_id,
                query=query,
                top_k=5,
            )
            
            if chunks:
                # Combine chunks into structured data
                file_data = []
                for chunk in chunks:
                    file_data.append({
                        "content": chunk.text,  # FileChunk has 'text', not 'content'
                        "file_id": str(chunk.file_id),
                        "chunk_id": str(chunk.id),  # FileChunk.id, not chunk_id
                        "chunk_index": chunk.chunk_index,
                    })
                
                return {
                    "success": True,
                    "files": file_data,
                    "chunk_count": len(chunks),
                }
            else:
                return {
                    "success": False,
                    "error": "No relevant file content found",
                    "files": [],
                }
        
        except Exception as e:
            logger.error(f"[FILE READER TOOL] Error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "files": [],
            }
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="File Reader",
            tool_type=ToolType.FILE_READER,
            description="Reads and extracts information from uploaded files",
            parameters_schema={
                "query": {"type": "string", "required": True, "description": "What to extract from files"},
                "session_id": {"type": "string", "required": True},
            },
            capabilities=[
                "Read PDF, CSV, Excel, TXT files",
                "Semantic chunk retrieval",
                "Table extraction",
                "Text extraction",
            ],
            examples=[
                "Extract sales data from uploaded CSV",
                "Read user list from Excel file",
                "Summarize uploaded document",
            ],
        )


class PythonExecutorTool(BaseTool):
    """Tool for executing Python code safely."""
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute Python code in sandbox."""
        from .python_sandbox import execute_python_code
        
        code = parameters.get("code", "")
        data_context = parameters.get("data_context", {})
        
        logger.info(f"[PYTHON EXECUTOR TOOL] Executing Python code ({len(code)} chars)...")
        
        try:
            # Execute in sandbox
            result = await execute_python_code(
                code=code,
                data_context=data_context,
                timeout=30,
            )
            
            return {
                "success": True,
                "result": result.get("result"),
                "output": result.get("output", ""),
                "execution_time": result.get("execution_time", 0),
            }
        
        except Exception as e:
            logger.error(f"[PYTHON EXECUTOR TOOL] Error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "result": None,
            }
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="Python Executor",
            tool_type=ToolType.PYTHON_EXECUTOR,
            description="Executes Python code for calculations and data processing",
            parameters_schema={
                "code": {"type": "string", "required": True, "description": "Python code to execute"},
                "data_context": {"type": "object", "required": False, "description": "Data variables"},
            },
            capabilities=[
                "Pandas data manipulation",
                "Statistical calculations",
                "Data transformations",
                "Comparisons and joins",
            ],
            examples=[
                "Calculate average of dataset",
                "Join two dataframes",
                "Compute correlation",
            ],
        )


class ChartGeneratorTool(BaseTool):
    """Tool for generating visualizations."""
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate chart from data."""
        from .dynamic_visualization_generator import generate_visualization_spec
        
        data = parameters.get("data", [])
        chart_type = parameters.get("chart_type", "auto")
        
        logger.info(f"[CHART GENERATOR TOOL] Generating {chart_type} chart...")
        
        try:
            chart_spec = await generate_visualization_spec(
                data=data,
                chart_type=chart_type,
                title=parameters.get("title", "Chart"),
            )
            
            return {
                "success": True,
                "chart_spec": chart_spec,
                "chart_type": chart_spec.get("chart_type", chart_type),
            }
        
        except Exception as e:
            logger.error(f"[CHART GENERATOR TOOL] Error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "chart_spec": None,
            }
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="Chart Generator",
            tool_type=ToolType.CHART_GENERATOR,
            description="Generates visualizations from data",
            parameters_schema={
                "data": {"type": "array", "required": True, "description": "Data to visualize"},
                "chart_type": {"type": "string", "required": False, "description": "Chart type (auto, bar, line, pie, etc)"},
            },
            capabilities=[
                "Auto-detect best chart type",
                "Bar, line, pie, scatter charts",
                "Time series visualization",
                "Comparison charts",
            ],
            examples=[
                "Create bar chart of sales by region",
                "Plot trend over time",
                "Generate pie chart of distribution",
            ],
        )


class VectorSearchTool(BaseTool):
    """Tool for semantic vector search."""
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform vector search."""
        from .rag_context_retriever import RAGContextRetriever
        
        db_session = context.get("db_session")
        query = parameters.get("query") or context.get("user_query") or ""
        session_id = parameters.get("session_id") or context.get("session_id") or "orchestrator"
        
        logger.info(f"[VECTOR SEARCH TOOL] Searching: {query}")
        
        try:
            retriever = RAGContextRetriever(db_session=db_session)
            context_result = await retriever.retrieve_context_for_followup(
                session_id=session_id,
                current_query=query,
            )
            
            if context_result:
                return {
                    "success": True,
                    "context": context_result.context_text,
                    "similarity": context_result.similarity_score,
                    "previous_query": context_result.previous_query,
                }
            else:
                return {
                    "success": False,
                    "error": "No relevant context found",
                    "context": "",
                }
        
        except Exception as e:
            logger.error(f"[VECTOR SEARCH TOOL] Error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "context": "",
            }
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="Vector Search",
            tool_type=ToolType.VECTOR_SEARCH,
            description="Semantic search across previous queries and context",
            parameters_schema={
                "query": {"type": "string", "required": True, "description": "Search query"},
                "session_id": {"type": "string", "required": True},
            },
            capabilities=[
                "Semantic similarity search",
                "Previous query retrieval",
                "Context-aware search",
            ],
            examples=[
                "Find similar previous queries",
                "Recall related context",
            ],
        )


class ToolRegistry:
    """
    Central registry for all tools in the system.
    
    Provides:
    - Tool registration and discovery
    - Unified execution interface
    - Tool metadata and capabilities
    - Error handling and logging
    """
    
    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[ToolType, BaseTool] = {}
        self._register_default_tools()
        logger.info(f"[TOOL REGISTRY] Initialized with {len(self.tools)} tools")
    
    def _register_default_tools(self):
        """Register all default tools."""
        self.register_tool(SQLExecutorTool())
        self.register_tool(FileReaderTool())
        self.register_tool(PythonExecutorTool())
        self.register_tool(ChartGeneratorTool())
        self.register_tool(VectorSearchTool())
    
    def register_tool(self, tool: BaseTool):
        """Register a new tool."""
        metadata = tool.get_metadata()
        self.tools[metadata.tool_type] = tool
        logger.debug(f"[TOOL REGISTRY] Registered tool: {metadata.name}")
    
    def get_tool(self, tool_type: ToolType) -> Optional[BaseTool]:
        """Get tool by type."""
        return self.tools.get(tool_type)
    
    def get_all_tools(self) -> List[ToolMetadata]:
        """Get metadata for all registered tools."""
        return [tool.get_metadata() for tool in self.tools.values()]
    
    async def execute_tool(
        self,
        tool_type: ToolType,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Any:
        """Execute a tool with parameters."""
        tool = self.get_tool(tool_type)
        
        if not tool:
            logger.error(f"[TOOL REGISTRY] Tool not found: {tool_type}")
            return {
                "success": False,
                "error": f"Tool {tool_type} not registered",
            }
        
        logger.info(f"[TOOL REGISTRY] Executing tool: {tool_type.value}")
        
        try:
            result = await tool.execute(parameters, context)
            return result
        except Exception as e:
            logger.error(f"[TOOL REGISTRY] Tool execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }


# Global instance
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create tool registry instance."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry
