"""
AGENTIC QUERY HANDLER - Unified Multi-Tool Agent

Replaces pipeline-based routing with intelligent agent-based execution.

Capabilities:
1. Multi-tool planning (combine SQL + Files + Python + Charts)
2. Autonomous reasoning and verification
3. Self-correction loops
4. Clarification engine integration
5. Knowledge graph utilization

Architecture:
    User Query → Master Orchestrator → Task Planner → Tool Registry →
    Execution → Verification → Clarification (if needed) → Response

Examples Handled:
- "Compare sales in uploaded file with database sales"
  → Queries DB, reads file, compares with Python, generates chart
  
- "Show top products by region and plot trend"
  → Queries DB, generates aggregation, creates chart
  
- "Find records matching file list and join related tables"
  → Reads file, queries DB with IN clause, joins tables

This is the NEW unified entry point for all queries.
"""

from __future__ import annotations

import logging
import uuid
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import UploadFile

from .. import schemas
from ..helpers import current_timestamp

logger = logging.getLogger(__name__)


class AgenticQueryHandler:
    """
    Agentic Query Handler - Multi-tool intelligent agent.
    
    Unlike pipeline-based handlers, this uses:
    - Master Orchestrator for planning
    - Tool Registry for execution
    - Knowledge Graph for schema understanding
    - Clarification Engine for ambiguity resolution
    - Observability for monitoring
    
    Flow:
    1. Interpret task (what user wants)
    2. Check for ambiguity → clarify if needed
    3. Plan execution steps
    4. Execute tools in sequence
    5. Verify results
    6. Retry if errors
    7. Generate response
    """
    
    def __init__(self, db_session: AsyncSession):
        """Initialize agentic handler."""
        self.db_session = db_session
        logger.info("[AGENTIC HANDLER] Initialized")
    
    async def handle_query(
        self,
        user_query: str,
        user_id: str,
        session_id: str,
        conversation_history: str = "",
        uploaded_files: Optional[List[UploadFile]] = None,
    ) -> schemas.ResponseWrapper:
        """
        Handle query using agentic approach.
        
        Args:
            user_query: User's natural language query
            user_id: User ID
            session_id: Session ID
            conversation_history: Previous conversation
            uploaded_files: Optional uploaded files
        
        Returns:
            ResponseWrapper with results
        """
        # Generate trace ID
        trace_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[AGENTIC HANDLER] Starting query processing (trace_id={trace_id})")
        logger.info(f"Query: {user_query}")
        logger.info(f"{'='*80}\n")
        
        # Log to observability
        from .observability import log_query_start, log_query_complete, get_execution_tracer
        log_query_start(user_query, user_id, session_id, trace_id)
        
        tracer = get_execution_tracer()
        tracer.start_trace(trace_id, "agentic_query")
        
        try:
            # Step 1: Get Master Orchestrator
            from .master_orchestrator import get_master_orchestrator
            orchestrator = await get_master_orchestrator(self.db_session)
            tracer.add_step(trace_id, "orchestrator_initialized")

            # Step 1A: Persist uploaded files (so FileReaderTool can retrieve chunks)
            persisted_files: List[Dict[str, Any]] = []
            if uploaded_files:
                logger.info("\n[STEP 0] Persisting uploaded files")
                from .file_handler import add_file

                for f in uploaded_files:
                    try:
                        # Ensure file pointer is at start (may have been read by security scanner).
                        await f.seek(0)
                    except Exception:
                        pass

                    uploaded = await add_file(self.db_session, session_id, f)
                    persisted_files.append(
                        {
                            "file_id": str(uploaded.id),
                            "filename": uploaded.filename,
                            "size": uploaded.size,
                        }
                    )

                tracer.add_step(trace_id, "files_persisted", metadata={
                    "files_count": len(persisted_files),
                    "filenames": [pf.get("filename") for pf in persisted_files[:5]],
                })
            
            # Step 2: Interpret task
            logger.info(f"[STEP 1] Task Interpretation")
            task_intent = await orchestrator.interpret_task(
                user_query=user_query,
                conversation_history=conversation_history,
                files_available=bool(uploaded_files),
                database_available=True,
            )
            tracer.add_step(trace_id, "task_interpreted", metadata={
                "task_type": task_intent.task_type.value,
                "complexity": task_intent.complexity_score,
                "confidence": task_intent.confidence,
            })
            
            logger.info(f"  Task Type: {task_intent.task_type.value}")
            logger.info(f"  Complexity: {task_intent.complexity_score:.2f}")
            logger.info(f"  Confidence: {task_intent.confidence:.2f}")
            logger.info(f"  Requires: DB={task_intent.requires_database}, "
                       f"Files={task_intent.requires_files}, "
                       f"Compute={task_intent.requires_computation}, "
                       f"Viz={task_intent.requires_visualization}")
            
            # Step 3: Check for ambiguity
            logger.info(f"\n[STEP 2] Ambiguity Detection")
            from .clarification_engine import get_clarification_engine
            clarification_engine = await get_clarification_engine(self.db_session)
            
            ambiguity_analysis = await clarification_engine.analyze_query(
                user_query=user_query,
                conversation_history=conversation_history,
            )
            tracer.add_step(trace_id, "ambiguity_checked", metadata={
                "has_ambiguity": ambiguity_analysis.has_ambiguity,
                "confidence": ambiguity_analysis.confidence,
            })
            
            if ambiguity_analysis.has_ambiguity and not ambiguity_analysis.can_proceed:
                logger.info(f"  ⚠️  Ambiguity detected (confidence={ambiguity_analysis.confidence:.2f})")
                logger.info(f"  Clarifications needed: {len(ambiguity_analysis.clarifications_needed)}")
                
                # Return clarification request
                duration_ms = (time.time() - start_time) * 1000
                log_query_complete(trace_id, duration_ms, user_id, session_id, success=True, metadata={
                    "needs_clarification": True,
                })
                tracer.end_trace(trace_id, success=True)
                
                return self._build_clarification_response(
                    user_query=user_query,
                    ambiguity_analysis=ambiguity_analysis,
                )
            else:
                logger.info(f"  ✓ No blocking ambiguity (confidence={ambiguity_analysis.confidence:.2f})")
            
            # Step 4: Build Knowledge Graph (for SQL queries)
            if task_intent.requires_database:
                logger.info(f"\n[STEP 3] Knowledge Graph Integration")
                from .knowledge_graph import get_knowledge_graph
                kg = await get_knowledge_graph(self.db_session, rebuild=False)
                tracer.add_step(trace_id, "knowledge_graph_loaded", metadata={
                    "entities": len(kg.entities),
                    "relationships": len(kg.relationships),
                })
                logger.info(f"  Knowledge Graph: {len(kg.entities)} entities, {len(kg.relationships)} relationships")
            
            # Step 5: Plan execution
            logger.info(f"\n[STEP 4] Execution Planning")

            # Fetch tool registry early so planning uses the actual available tools (no hardcoded list).
            from .tool_registry import get_tool_registry
            tool_registry = get_tool_registry()
            available_tools = tool_registry.get_all_tools()

            execution_plan = await orchestrator.plan_execution(
                task_intent=task_intent,
                user_query=user_query,
                available_tools=available_tools,
            )
            tracer.add_step(trace_id, "execution_planned", metadata={
                "steps": len(execution_plan.steps),
                "reasoning": execution_plan.reasoning[:100],
            })
            
            logger.info(f"  Plan created: {len(execution_plan.steps)} steps")
            logger.info(f"  Reasoning: {execution_plan.reasoning}")
            for step in execution_plan.steps:
                logger.info(f"    {step.step_number}. {step.tool.value}: {step.description}")
            
            # Step 6: Execute plan
            logger.info(f"\n[STEP 5] Tool Execution")
            execution_result = await orchestrator.execute_plan(
                plan=execution_plan,
                tool_registry=tool_registry,
                execution_context={
                    "user_id": user_id,
                    "session_id": session_id,
                    "conversation_history": conversation_history,
                    "user_query": user_query,
                    "files": persisted_files,
                },
            )
            tracer.add_step(trace_id, "execution_completed", metadata={
                "status": execution_result["status"],
                "steps_completed": execution_result["steps_completed"],
                "steps_failed": execution_result["steps_failed"],
            })
            
            logger.info(f"  Execution {execution_result['status']}: "
                       f"{execution_result['steps_completed']}/{len(execution_plan.steps)} steps completed")
            
            # Step 7: Build response
            logger.info(f"\n[STEP 6] Response Generation")
            response = await self._build_execution_response(
                user_query=user_query,
                task_intent=task_intent,
                execution_result=execution_result,
                execution_plan=execution_plan,
            )
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"\n[COMPLETE] Query processed in {duration_ms:.0f}ms (trace_id={trace_id})")
            logger.info(f"{'='*80}\n")
            
            # Log completion
            log_query_complete(
                trace_id=trace_id,
                duration_ms=duration_ms,
                user_id=user_id,
                session_id=session_id,
                success=execution_result['status'] == 'completed',
                metadata={
                    "task_type": task_intent.task_type.value,
                    "steps": len(execution_plan.steps),
                },
            )
            tracer.end_trace(trace_id, success=True)
            
            return response
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"[AGENTIC HANDLER] Error: {e}", exc_info=True)
            
            log_query_complete(
                trace_id=trace_id,
                duration_ms=duration_ms,
                user_id=user_id,
                session_id=session_id,
                success=False,
                metadata={"error": str(e)},
            )
            tracer.end_trace(trace_id, success=False, error=str(e))
            
            # Return error response
            error_response = schemas.StandardResponse(
                intent=user_query,
                confidence=0.0,
                message=f"I encountered an error processing your request: {str(e)}",
                related_queries=[],
                metadata={"error": True, "trace_id": trace_id},
            )
            
            return schemas.ResponseWrapper(
                success=False,
                response=error_response,
                timestamp=current_timestamp(),
                original_query=user_query,
            )
    
    def _build_clarification_response(
        self,
        user_query: str,
        ambiguity_analysis: Any,
    ) -> schemas.ResponseWrapper:
        """Build response requesting clarification."""
        # Get first clarification needed
        clarification = ambiguity_analysis.clarifications_needed[0] if ambiguity_analysis.clarifications_needed else None
        
        if not clarification:
            raise ValueError("No clarification available")
        
        # Format options
        options_text = []
        for i, opt in enumerate(clarification.options, 1):
            prefix = "→" if opt.is_recommended else " "
            desc = f" - {opt.description}" if opt.description else ""
            options_text.append(f"{prefix} {i}. {opt.label}{desc}")
        
        message = f"""I need clarification to answer your question accurately.

**{clarification.question}**

{chr(10).join(options_text)}

{clarification.context if clarification.context else ''}

Please select an option or provide more details."""

        response = schemas.StandardResponse(
            intent="clarification",
            confidence=ambiguity_analysis.confidence,
            message=message,
            related_queries=[],
            metadata={
                "needs_clarification": True,
                "clarification": clarification.to_dict(),
                "ambiguity_analysis": ambiguity_analysis.to_dict(),
            },
        )
        
        return schemas.ResponseWrapper(
            success=True,
            response=response,
            timestamp=current_timestamp(),
            original_query=user_query,
        )
    
    async def _build_execution_response(
        self,
        user_query: str,
        task_intent: Any,
        execution_result: Dict[str, Any],
        execution_plan: Any,
    ) -> schemas.ResponseWrapper:
        """Build response from execution results."""
        if execution_result['status'] == 'failed':
            # Generate error response
            failed_steps = [s for s in execution_plan.steps if s.status.value == 'failed']
            error_message = f"I encountered {len(failed_steps)} error(s) while processing your request:\n"
            
            for step in failed_steps:
                error_message += f"\n- Step {step.step_number} ({step.tool.value}): {step.error}"
            
            response = schemas.StandardResponse(
                intent=task_intent.task_type.value,
                confidence=0.3,
                message=error_message,
                related_queries=[],
                metadata={
                    "execution_failed": True,
                    "steps": execution_result['steps_details'],
                },
            )
            
            return schemas.ResponseWrapper(
                success=False,
                response=response,
                timestamp=current_timestamp(),
                original_query=user_query,
            )
        
        # Extract results from executed steps
        final_result = execution_result['results'].get(f"step_{len(execution_plan.steps)}", {})
        
        # Check if SQL execution
        if final_result.get('sql'):
            # SQL query response
            datasets = []
            if final_result.get('rows'):
                datasets.append({
                    "data": final_result['rows'],
                    "columns": final_result.get('columns', []),
                })
            
            response = schemas.DataQueryResponse(
                intent="data_query",
                confidence=task_intent.confidence,
                message=f"Found {final_result.get('row_count', 0)} results",
                datasets=datasets,
                metadata={
                    "sql": final_result.get('sql', ''),
                    "execution_plan": {
                        "steps": [s.to_dict() for s in execution_plan.steps],
                        "reasoning": execution_plan.reasoning,
                    },
                },
                suggested_visualizations=[],
            )
            
            return schemas.ResponseWrapper(
                success=True,
                response=response,
                timestamp=current_timestamp(),
                original_query=user_query,
            )
        
        # Generic response  
        response = schemas.StandardResponse(
            intent=task_intent.task_type.value,
            confidence=task_intent.confidence,
            message=f"Executed {len(execution_plan.steps)} steps successfully.",
            related_queries=[],
            metadata={
                "execution_result": execution_result,
                "steps": [s.to_dict() for s in execution_plan.steps],
            },
        )
        
        return schemas.ResponseWrapper(
            success=True,
            response=response,
            timestamp=current_timestamp(),
            original_query=user_query,
        )


# Factory function
async def create_agentic_handler(db_session: AsyncSession) -> AgenticQueryHandler:
    """Create agentic query handler."""
    return AgenticQueryHandler(db_session)
