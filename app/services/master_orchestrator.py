"""
MASTER AI ORCHESTRATOR - Central Agentic Reasoning Engine

Transforms the system from pipeline-based to agent-based architecture.

Features:
- Autonomous task planning and decomposition
- Multi-tool coordination (SQL + Files + Charts + Python)
- Self-verification and correction loops
- Reasoning-based execution (not fixed pipelines)

Architecture:
    User Query → Task Interpreter → Planning Engine → Tool Selector → 
    Execution Controller → Result Verifier → Self Correction → Response

This is the "brain" of the agentic system.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import UploadFile

from .. import llm
from .tool_registry import ToolType

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks the orchestrator can handle."""
    DATA_QUERY = "data_query"           # Query database
    FILE_ANALYSIS = "file_analysis"     # Analyze uploaded files
    COMPARISON = "comparison"           # Compare DB + File data
    VISUALIZATION = "visualization"     # Generate charts
    COMPUTATION = "computation"         # Python calculations
    CONVERSATION = "conversation"       # Chat/explanation
    MULTI_STEP = "multi_step"          # Complex multi-step tasks


class ExecutionStatus(str, Enum):
    """Status of task execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass
class TaskIntent:
    """Interpreted intent from user query."""
    task_type: TaskType
    description: str
    entities: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    requires_database: bool = False
    requires_files: bool = False
    requires_computation: bool = False
    requires_visualization: bool = False
    complexity_score: float = 0.0  # 0-1, higher means more complex
    confidence: float = 1.0


@dataclass
class ExecutionStep:
    """Single step in execution plan."""
    step_number: int
    tool: ToolType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)  # Step dependencies
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "tool": self.tool.value,
            "description": self.description,
            "parameters": self.parameters,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "error": self.error,
            "retry_count": self.retry_count,
        }


@dataclass
class ExecutionPlan:
    """Complete execution plan with multiple steps."""
    task_id: str
    task_intent: TaskIntent
    steps: List[ExecutionStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    reasoning: str = ""  # Why this plan was chosen
    
    def get_pending_steps(self) -> List[ExecutionStep]:
        """Get steps ready to execute (dependencies met)."""
        pending = []
        for step in self.steps:
            if step.status != ExecutionStatus.PENDING:
                continue
            # Check if dependencies are completed
            deps_completed = all(
                self.steps[dep - 1].status == ExecutionStatus.COMPLETED
                for dep in step.depends_on
            )
            if deps_completed:
                pending.append(step)
        return pending
    
    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(s.status == ExecutionStatus.COMPLETED for s in self.steps)
    
    def has_failures(self) -> bool:
        """Check if any step failed."""
        return any(s.status == ExecutionStatus.FAILED for s in self.steps)


@dataclass
class VerificationResult:
    """Result of verifying execution output."""
    is_valid: bool
    confidence: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    needs_retry: bool = False
    needs_clarification: bool = False
    clarification_question: Optional[str] = None


class MasterOrchestrator:
    """
    Master AI Orchestrator - Central reasoning engine for agentic system.
    
    Capabilities:
    1. Task Interpretation - Understand complex user intents
    2. Planning - Break tasks into executable steps
    3. Tool Selection - Choose appropriate tools
    4. Execution Control - Coordinate tool execution
    5. Verification - Validate results
    6. Self-Correction - Retry and fix errors automatically
    
    Example:
        User: "Compare sales in uploaded file with database and show chart"
        
        Plan:
        1. Query database for sales data
        2. Read and parse uploaded file
        3. Join/compare datasets using Python
        4. Generate comparison chart
        5. Generate insights
    """
    
    def __init__(self, db_session: AsyncSession):
        """Initialize the master orchestrator."""
        self.db_session = db_session
        self.execution_history: List[ExecutionPlan] = []
        logger.info("[MASTER ORCHESTRATOR] Initialized")
    
    async def interpret_task(
        self,
        user_query: str,
        conversation_history: str = "",
        files_available: bool = False,
        database_available: bool = True,
    ) -> TaskIntent:
        """
        Interpret user query and extract task intent.
        
        Uses LLM to understand:
        - What type of task (single vs multi-step)
        - What resources are needed (DB, files, computation)
        - Complexity level
        - Entities, metrics, filters
        
        Args:
            user_query: User's natural language query
            conversation_history: Previous conversation context
            files_available: Whether files are uploaded
            database_available: Whether database is connected
        
        Returns:
            TaskIntent with interpreted information
        """
        logger.info(f"[TASK INTERPRETER] Analyzing query: {user_query}")
        
        prompt = f"""You are an AI task interpreter. Analyze the user query and extract structured intent.

User Query: {user_query}

Context:
- Files available: {files_available}
- Database available: {database_available}
- Previous conversation: {conversation_history[:500] if conversation_history else "None"}

Analyze the query and respond in JSON format:
{{
    "task_type": "data_query|file_analysis|comparison|visualization|computation|conversation|multi_step",
    "description": "Clear description of what user wants",
    "entities": ["list", "of", "entities"],
    "metrics": ["list", "of", "metrics"],
    "filters": {{"key": "value"}},
    "requires_database": true/false,
    "requires_files": true/false,
    "requires_computation": true/false,
    "requires_visualization": true/false,
    "complexity_score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "Why you chose this interpretation"
}}

Examples:
- "Show sales by region" → data_query, requires_database=true
- "Compare uploaded CSV with database" → comparison, requires_database=true, requires_files=true
- "Calculate average from file and plot" → multi_step, requires_files=true, requires_computation=true, requires_visualization=true

Respond ONLY with valid JSON."""

        try:
            response = await llm.call_llm(prompt, temperature=0.1, json_mode=True)
            import json
            intent_data = json.loads(response)
            
            task_intent = TaskIntent(
                task_type=TaskType(intent_data.get("task_type", "conversation")),
                description=intent_data.get("description", ""),
                entities=intent_data.get("entities", []),
                metrics=intent_data.get("metrics", []),
                filters=intent_data.get("filters", {}),
                requires_database=intent_data.get("requires_database", False),
                requires_files=intent_data.get("requires_files", False),
                requires_computation=intent_data.get("requires_computation", False),
                requires_visualization=intent_data.get("requires_visualization", False),
                complexity_score=intent_data.get("complexity_score", 0.5),
                confidence=intent_data.get("confidence", 1.0),
            )
            
            logger.info(f"[TASK INTERPRETER] Task type: {task_intent.task_type.value}, "
                       f"Complexity: {task_intent.complexity_score:.2f}")
            
            return task_intent
            
        except Exception as e:
            logger.error(f"[TASK INTERPRETER] Failed to interpret task: {e}")
            # Fallback to simple conversation
            return TaskIntent(
                task_type=TaskType.CONVERSATION,
                description=user_query,
                confidence=0.3,
            )
    
    async def plan_execution(
        self,
        task_intent: TaskIntent,
        user_query: str,
        available_tools: Optional[List[Any]] = None,
    ) -> ExecutionPlan:
        """
        Generate execution plan with multiple steps.
        
        Uses LLM reasoning to:
        - Break complex tasks into steps
        - Select appropriate tools for each step
        - Define dependencies between steps
        - Order steps optimally
        
        Args:
            task_intent: Interpreted task intent
            user_query: Original user query
        
        Returns:
            ExecutionPlan with ordered steps
        """
        logger.info(f"[PLANNING ENGINE] Planning execution for: {task_intent.task_type.value}")
        
        # Generate task ID
        import uuid
        task_id = str(uuid.uuid4())[:8]
        
        default_tools_text = """- sql_executor: Query database
- file_reader: Read and parse files
- python_executor: Run Python code for calculations
- chart_generator: Generate visualizations
- vector_search: Semantic search
- knowledge_retrieval: Retrieve from knowledge base
- memory_retrieval: Recall previous context"""

        tools_text = default_tools_text
        if available_tools:
            try:
                tool_lines: List[str] = []
                for t in available_tools:
                    # Accept ToolMetadata-like objects or plain dicts.
                    tool_id = None
                    name = None
                    description = None
                    parameters_schema = None

                    if hasattr(t, "tool_type"):
                        tool_id = getattr(getattr(t, "tool_type"), "value", None) or str(getattr(t, "tool_type"))
                        name = getattr(t, "name", None)
                        description = getattr(t, "description", None)
                        parameters_schema = getattr(t, "parameters_schema", None)
                    elif isinstance(t, dict):
                        tool_id = t.get("type") or t.get("tool") or t.get("id")
                        name = t.get("name")
                        description = t.get("description")
                        parameters_schema = t.get("parameters_schema")

                    if not tool_id:
                        continue

                    params_summary = ""
                    if isinstance(parameters_schema, dict) and parameters_schema:
                        parts = []
                        for p_name, p_spec in list(parameters_schema.items())[:8]:
                            p_type = None
                            p_required = None
                            if isinstance(p_spec, dict):
                                p_type = p_spec.get("type")
                                p_required = p_spec.get("required")
                            req = ", required" if p_required else ""
                            parts.append(f"{p_name} ({p_type or 'any'}{req})")
                        params_summary = f" | params: {', '.join(parts)}"

                    display_name = f"{name} - " if name else ""
                    tool_lines.append(f"- {tool_id}: {display_name}{description or ''}{params_summary}".strip())

                if tool_lines:
                    tools_text = "\n".join(tool_lines)
            except Exception:
                tools_text = default_tools_text

        prompt = f"""You are an AI task planner. Create an execution plan to accomplish the user's goal.

User Query: {user_query}

Task Intent:
- Type: {task_intent.task_type.value}
- Description: {task_intent.description}
- Requires database: {task_intent.requires_database}
- Requires files: {task_intent.requires_files}
- Requires computation: {task_intent.requires_computation}
- Requires visualization: {task_intent.requires_visualization}

Available Tools:
{tools_text}

IMPORTANT:
- The "tool" field MUST be one of the exact tool identifiers listed above.
- Each step should include a "parameters" object. For tools that accept a "query", include it.

Create a step-by-step plan. Respond in JSON format:
{{
    "reasoning": "Why this plan will work",
    "steps": [
        {{
            "step_number": 1,
            "tool": "sql_executor",
            "description": "What this step does",
            "parameters": {{"key": "value"}},
            "depends_on": []
        }},
        ...
    ]
}}

Examples:
- Simple query: Single step with sql_executor
- File + DB comparison: 
  1. sql_executor (get DB data)
  2. file_reader (get file data)
  3. python_executor (compare/join data)
  4. chart_generator (visualize)

Respond ONLY with valid JSON."""

        try:
            response = await llm.call_llm(prompt, temperature=0.2, json_mode=True)
            import json
            plan_data = json.loads(response)
            
            # Create execution steps
            steps = []
            for step_data in plan_data.get("steps", []):
                step = ExecutionStep(
                    step_number=step_data.get("step_number", len(steps) + 1),
                    tool=ToolType(step_data.get("tool", "sql_executor")),
                    description=step_data.get("description", ""),
                    parameters=step_data.get("parameters", {}),
                    depends_on=step_data.get("depends_on", []),
                )
                steps.append(step)
            
            plan = ExecutionPlan(
                task_id=task_id,
                task_intent=task_intent,
                steps=steps,
                reasoning=plan_data.get("reasoning", ""),
            )
            
            logger.info(f"[PLANNING ENGINE] Created plan with {len(steps)} steps")
            for step in steps:
                logger.debug(f"  Step {step.step_number}: {step.tool.value} - {step.description}")
            
            return plan
            
        except Exception as e:
            logger.error(f"[PLANNING ENGINE] Failed to create plan: {e}")
            # Fallback: Create simple single-step plan
            fallback_tool = ToolType.SQL_EXECUTOR if task_intent.requires_database else ToolType.KNOWLEDGE_RETRIEVAL
            plan = ExecutionPlan(
                task_id=task_id,
                task_intent=task_intent,
                steps=[ExecutionStep(
                    step_number=1,
                    tool=fallback_tool,
                    description="Execute user query",
                    parameters={"query": user_query},
                )],
                reasoning="Fallback simple plan due to planning error",
            )
            return plan
    
    async def verify_result(
        self,
        step: ExecutionStep,
        result: Any,
    ) -> VerificationResult:
        """
        Verify execution result for correctness.
        
        Checks:
        - Result is not None/empty
        - No obvious errors
        - Result matches expected output
        - Data quality checks
        
        Args:
            step: Executed step
            result: Result from execution
        
        Returns:
            VerificationResult with validation status
        """
        logger.info(f"[RESULT VERIFIER] Verifying step {step.step_number} ({step.tool.value})")
        
        issues = []
        suggestions = []
        
        # Basic checks
        if result is None:
            issues.append("Result is None")
            suggestions.append("Check if tool executed successfully")
            return VerificationResult(
                is_valid=False,
                confidence=0.0,
                issues=issues,
                suggestions=suggestions,
                needs_retry=True,
            )
        
        # Check for error indicators
        if isinstance(result, dict):
            if result.get("error") or result.get("success") is False:
                issues.append(f"Execution error: {result.get('error', 'Unknown')}")
                suggestions.append("Retry with corrected parameters")
                return VerificationResult(
                    is_valid=False,
                    confidence=0.2,
                    issues=issues,
                    suggestions=suggestions,
                    needs_retry=True,
                )
        
        # Tool-specific validation
        if step.tool == ToolType.SQL_EXECUTOR:
            if isinstance(result, dict) and result.get("rows") is not None:
                row_count = len(result.get("rows", []))
                if row_count == 0:
                    issues.append("Query returned no results")
                    suggestions.append("Check filters and table data")
                    confidence = 0.5
                else:
                    confidence = 1.0
                    logger.info(f"[RESULT VERIFIER] SQL returned {row_count} rows")
            else:
                confidence = 0.7
        else:
            confidence = 0.8
        
        return VerificationResult(
            is_valid=len(issues) == 0,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions,
            needs_retry=False,
        )
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        tool_registry: Any,  # Will be ToolRegistry instance
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the plan step by step with verification and retry logic.
        
        Features:
        - Sequential execution with dependency checking
        - Automatic retry on failures (max 2 retries)
        - Result verification after each step
        - Self-correction loop
        
        Args:
            plan: ExecutionPlan to execute
            tool_registry: ToolRegistry with available tools
        
        Returns:
            Dict with execution results
        """
        logger.info(f"[EXECUTION CONTROLLER] Starting execution of plan {plan.task_id}")
        
        max_retries = 2
        results = {}
        
        while not plan.is_complete() and not plan.has_failures():
            pending_steps = plan.get_pending_steps()
            
            if not pending_steps:
                if not plan.is_complete():
                    logger.error("[EXECUTION CONTROLLER] No pending steps but plan not complete")
                    break
                break
            
            for step in pending_steps:
                step.status = ExecutionStatus.IN_PROGRESS
                logger.info(f"[EXECUTION CONTROLLER] Executing step {step.step_number}: {step.description}")
                
                try:
                    effective_parameters = dict(step.parameters or {})
                    ctx = dict(execution_context or {})

                    # Best-effort injection of common context into parameters.
                    if "query" not in effective_parameters and ctx.get("user_query"):
                        effective_parameters["query"] = ctx["user_query"]
                    for key in ("user_id", "session_id", "conversation_history"):
                        if key not in effective_parameters and ctx.get(key) is not None:
                            effective_parameters[key] = ctx[key]

                    # Execute tool
                    tool_result = await tool_registry.execute_tool(
                        tool_type=step.tool,
                        parameters=effective_parameters,
                        context={
                            "db_session": self.db_session,
                            "previous_results": results,
                            **ctx,
                        }
                    )
                    
                    # Verify result
                    verification = await self.verify_result(step, tool_result)
                    
                    if verification.is_valid:
                        step.status = ExecutionStatus.COMPLETED
                        step.result = tool_result
                        results[f"step_{step.step_number}"] = tool_result
                        logger.info(f"[EXECUTION CONTROLLER] ✓ Step {step.step_number} completed successfully")
                    else:
                        # Retry logic
                        if verification.needs_retry and step.retry_count < max_retries:
                            step.retry_count += 1
                            step.status = ExecutionStatus.PENDING
                            logger.warning(f"[EXECUTION CONTROLLER] ⟳ Retrying step {step.step_number} "
                                         f"(attempt {step.retry_count + 1}/{max_retries + 1})")
                            logger.warning(f"  Issues: {', '.join(verification.issues)}")
                        else:
                            step.status = ExecutionStatus.FAILED
                            step.error = "; ".join(verification.issues)
                            logger.error(f"[EXECUTION CONTROLLER] ✗ Step {step.step_number} failed: {step.error}")
                
                except Exception as e:
                    logger.error(f"[EXECUTION CONTROLLER] ✗ Step {step.step_number} error: {e}", exc_info=True)
                    
                    if step.retry_count < max_retries:
                        step.retry_count += 1
                        step.status = ExecutionStatus.PENDING
                        logger.warning(f"[EXECUTION CONTROLLER] ⟳ Retrying after exception "
                                     f"(attempt {step.retry_count + 1}/{max_retries + 1})")
                    else:
                        step.status = ExecutionStatus.FAILED
                        step.error = str(e)
        
        # Compile final results
        execution_summary = {
            "task_id": plan.task_id,
            "status": "completed" if plan.is_complete() else "failed",
            "steps_completed": len([s for s in plan.steps if s.status == ExecutionStatus.COMPLETED]),
            "steps_failed": len([s for s in plan.steps if s.status == ExecutionStatus.FAILED]),
            "results": results,
            "steps_details": [s.to_dict() for s in plan.steps],
        }
        
        logger.info(f"[EXECUTION CONTROLLER] Execution finished: {execution_summary['status']}")
        
        return execution_summary


async def get_master_orchestrator(db_session: AsyncSession) -> MasterOrchestrator:
    """Create a master orchestrator bound to the provided DB session.

    IMPORTANT: Do not cache this globally.
    FastAPI provides a request-scoped AsyncSession; caching a session-bound
    orchestrator would risk cross-request session leakage and concurrency bugs.
    """
    return MasterOrchestrator(db_session)
