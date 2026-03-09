"""
PROMPT BUILDER - Build optimized prompts for LLM.

Provides:
- Different prompt strategies
- Context optimization
- Few-shot examples
- Structured output prompts
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PromptType(str, Enum):
    """Types of prompts."""
    SQL_GENERATION = "sql_generation"
    CHAT_RESPONSE = "chat_response"
    FILE_ANALYSIS = "file_analysis"
    CLARIFICATION = "clarification"


class PromptStrategy(str, Enum):
    """Prompt engineering strategies."""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STRUCTURED_OUTPUT = "structured_output"


@dataclass
class PromptContext:
    """Context for prompt building."""
    user_query: str
    schema_info: Optional[str] = None
    conversation_history: Optional[str] = None
    examples: Optional[List[str]] = None
    constraints: Optional[List[str]] = None


@dataclass
class BuiltPrompt:
    """Built prompt ready for LLM."""
    prompt: str
    system_message: Optional[str] = None
    strategy_used: PromptStrategy = PromptStrategy.ZERO_SHOT
    token_count: int = 0


class PromptBuilder:
    """Builds optimized prompts for different tasks."""
    
    def __init__(self, default_strategy: PromptStrategy = PromptStrategy.FEW_SHOT):
        """Initialize prompt builder."""
        self.default_strategy = default_strategy
    
    async def build_prompt(
        self,
        prompt_type: PromptType,
        context: PromptContext,
        strategy: Optional[PromptStrategy] = None,
    ) -> BuiltPrompt:
        """
        Build optimized prompt for given type and context.
        
        Args:
            prompt_type: Type of prompt to build
            context: Context for prompt
            strategy: Optional prompt strategy override
            
        Returns:
            BuiltPrompt ready for LLM
        """
        strategy = strategy or self.default_strategy
        
        if prompt_type == PromptType.SQL_GENERATION:
            return await self._build_sql_prompt(context, strategy)
        elif prompt_type == PromptType.CHAT_RESPONSE:
            return await self._build_chat_prompt(context, strategy)
        elif prompt_type == PromptType.FILE_ANALYSIS:
            return await self._build_file_analysis_prompt(context, strategy)
        else:
            return await self._build_generic_prompt(context, strategy)
    
    async def _build_sql_prompt(
        self,
        context: PromptContext,
        strategy: PromptStrategy
    ) -> BuiltPrompt:
        """Build SQL generation prompt."""
        prompt_parts = []
        
        # Add user query
        prompt_parts.append(f"User query: {context.user_query}")
        
        # Add schema if available
        if context.schema_info:
            prompt_parts.append(f"\nDatabase schema:\n{context.schema_info}")
        
        # Add examples for few-shot
        if strategy == PromptStrategy.FEW_SHOT and context.examples:
            prompt_parts.append("\nExamples:")
            for example in context.examples:
                prompt_parts.append(f"  {example}")
        
        # Add constraints
        if context.constraints:
            prompt_parts.append("\nConstraints:")
            for constraint in context.constraints:
                prompt_parts.append(f"  - {constraint}")
        
        prompt = "\n".join(prompt_parts)
        
        system_message = "You are a SQL expert. Generate valid SQL queries based on user requests."
        
        return BuiltPrompt(
            prompt=prompt,
            system_message=system_message,
            strategy_used=strategy,
            token_count=len(prompt.split()) * 2  # Rough estimate
        )
    
    async def _build_chat_prompt(
        self,
        context: PromptContext,
        strategy: PromptStrategy
    ) -> BuiltPrompt:
        """Build chat response prompt."""
        prompt = context.user_query
        system_message = "You are a helpful assistant."
        
        return BuiltPrompt(
            prompt=prompt,
            system_message=system_message,
            strategy_used=strategy,
            token_count=len(prompt.split()) * 2
        )
    
    async def _build_file_analysis_prompt(
        self,
        context: PromptContext,
        strategy: PromptStrategy
    ) -> BuiltPrompt:
        """Build file analysis prompt."""
        prompt = f"Analyze the following query about uploaded files:\n{context.user_query}"
        system_message = "You are a data analysis expert."
        
        return BuiltPrompt(
            prompt=prompt,
            system_message=system_message,
            strategy_used=strategy,
            token_count=len(prompt.split()) * 2
        )
    
    async def _build_generic_prompt(
        self,
        context: PromptContext,
        strategy: PromptStrategy
    ) -> BuiltPrompt:
        """Build generic prompt."""
        return BuiltPrompt(
            prompt=context.user_query,
            system_message="You are a helpful assistant.",
            strategy_used=strategy,
            token_count=len(context.user_query.split()) * 2
        )


# Global instance
_prompt_builder: Optional[PromptBuilder] = None


def get_prompt_builder() -> PromptBuilder:
    """Get or create prompt builder."""
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder()
    return _prompt_builder
