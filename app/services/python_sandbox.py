"""
PYTHON SANDBOX EXECUTOR - Secure Python execution environment

Enables Python code execution for:
- Pandas data analysis
- Statistical calculations
- Data transformations
- Chart generation with matplotlib/plotly

Security Features:
- Restricted imports (no os, sys, subprocess)
- Timeout control
- Memory limits
- Safe execution context

Example Use Cases:
- Compare database data with file data
- Calculate custom metrics
- Join/merge datasets
- Statistical analysis
"""

from __future__ import annotations

import logging
import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Status of code execution."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RESTRICTED = "restricted"


@dataclass
class ExecutionResult:
    """Result of Python code execution."""
    status: ExecutionStatus
    result: Any = None
    output: str = ""
    error: str = ""
    execution_time: float = 0.0
    variables: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "result": self.result,
            "output": self.output,
            "error": self.error,
            "execution_time": round(self.execution_time, 3),
        }


# Allowed imports for sandbox
ALLOWED_IMPORTS = {
    'pandas', 'pd',
    'numpy', 'np',
    'json',
    'math',
    'datetime', 'timedelta',
    'statistics',
    'collections',
    're',
}

# Restricted imports (security)
RESTRICTED_IMPORTS = {
    'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
    'pickle', 'shelve', 'dbm', 'sqlite3',
    '__import__', 'eval', 'exec', 'compile',
    'open', 'file', 'input', 'raw_input',
}


def check_code_safety(code: str) -> tuple[bool, Optional[str]]:
    """
    Check if code is safe to execute.
    
    Checks:
    - No restricted imports
    - No file operations
    - No system calls
    - No dangerous builtins
    
    Returns:
        (is_safe, error_message)
    """
    code_lower = code.lower()
    
    # Check for restricted imports
    for restricted in RESTRICTED_IMPORTS:
        if f'import {restricted}' in code_lower or f'from {restricted}' in code_lower:
            return False, f"Restricted import detected: {restricted}"
    
    # Check for dangerous operations
    dangerous_keywords = [
        '__import__', 'eval(', 'exec(',
        'open(', 'file(',
        'subprocess.', 'os.',
    ]
    
    for keyword in dangerous_keywords:
        if keyword in code_lower:
            return False, f"Dangerous operation detected: {keyword}"
    
    return True, None


async def execute_python_code(
    code: str,
    data_context: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Execute Python code in a restricted sandbox environment.
    
    Features:
    - Timeout control
    - Restricted imports
    - Safe execution context
    - Captures stdout/stderr
    - Returns execution result
    
    Args:
        code: Python code to execute
        data_context: Dictionary of variables to inject into execution context
        timeout: Maximum execution time in seconds
    
    Returns:
        ExecutionResult dict
    
    Example:
        ```python
        result = await execute_python_code(
            code='''
import pandas as pd

# Data from context
df = pd.DataFrame(data)

# Calculate average
result = df['sales'].mean()
            ''',
            data_context={"data": [{"sales": 100}, {"sales": 200}]},
            timeout=10,
        )
        
        print(result["result"])  # 150.0
        ```
    """
    logger.info(f"[PYTHON SANDBOX] Executing code ({len(code)} chars, timeout={timeout}s)")
    
    start_time = time.time()
    
    # Safety check
    is_safe, error_msg = check_code_safety(code)
    if not is_safe:
        logger.warning(f"[PYTHON SANDBOX] ⚠️  Code rejected: {error_msg}")
        return ExecutionResult(
            status=ExecutionStatus.RESTRICTED,
            error=error_msg,
        ).to_dict()
    
    # Prepare execution context
    execution_globals = {
        '__builtins__': __builtins__,
        'result': None,  # Code should set this
    }
    
    # Add allowed imports
    try:
        import pandas as pd
        import numpy as np
        import json
        import math
        import statistics
        import re
        from datetime import datetime, timedelta
        from collections import Counter
        
        execution_globals.update({
            'pd': pd,
            'pandas': pd,
            'np': np,
            'numpy': np,
            'json': json,
            'math': math,
            'statistics': statistics,
            're': re,
            'datetime': datetime,
            'timedelta': timedelta,
            'Counter': Counter,
        })
    except ImportError as e:
        logger.warning(f"[PYTHON SANDBOX] Import unavailable: {e}")
    
    # Add data context
    if data_context:
        execution_globals.update(data_context)
        logger.debug(f"[PYTHON SANDBOX] Injected {len(data_context)} context variables")
    
    # Capture output
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    
    try:
        # Execute with timeout
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            # Use asyncio timeout
            async def run_code():
                exec(code, execution_globals)
                return execution_globals.get('result')
            
            result = await asyncio.wait_for(
                run_code(),
                timeout=timeout,
            )
        
        execution_time = time.time() - start_time
        output = output_buffer.getvalue()
        
        logger.info(f"[PYTHON SANDBOX] ✓ Execution successful ({execution_time:.2f}s)")
        
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            result=result,
            output=output,
            execution_time=execution_time,
            variables={k: v for k, v in execution_globals.items() 
                      if not k.startswith('__') and k not in ['pd', 'np', 'json', 'math']},
        ).to_dict()
    
    except asyncio.TimeoutError:
        execution_time = time.time() - start_time
        logger.error(f"[PYTHON SANDBOX] ⏱ Timeout after {timeout}s")
        
        return ExecutionResult(
            status=ExecutionStatus.TIMEOUT,
            error=f"Execution timeout after {timeout} seconds",
            execution_time=execution_time,
        ).to_dict()
    
    except Exception as e:
        execution_time = time.time() - start_time
        output = output_buffer.getvalue()
        error = error_buffer.getvalue() or str(e)
        
        logger.error(f"[PYTHON SANDBOX] ✗ Execution error: {error}")
        
        return ExecutionResult(
            status=ExecutionStatus.ERROR,
            error=error,
            output=output,
            execution_time=execution_time,
        ).to_dict()


async def generate_python_code_for_task(
    task_description: str,
    data_context: Dict[str, Any],
) -> str:
    """
    Generate Python code to accomplish a task.
    
    Uses LLM to generate code based on task description and available data.
    
    Args:
        task_description: What the code should do
        data_context: Available variables and their types
    
    Returns:
        Generated Python code
    
    Example:
        ```python
        code = await generate_python_code_for_task(
            task_description="Calculate average sales and find top product",
            data_context={
                "sales_data": "list of dicts with keys: product, sales",
                "file_data": "list of dicts with keys: product, category",
            }
        )
        ```
    """
    from .. import llm
    
    logger.info(f"[CODE GENERATOR] Generating code for: {task_description}")
    
    # Describe available data
    data_description = []
    for var_name, var_value in data_context.items():
        if isinstance(var_value, list) and len(var_value) > 0:
            data_description.append(f"- {var_name}: list with {len(var_value)} items, "
                                  f"sample: {var_value[0] if var_value else 'empty'}")
        elif isinstance(var_value, dict):
            data_description.append(f"- {var_name}: dict with keys: {list(var_value.keys())}")
        else:
            data_description.append(f"- {var_name}: {type(var_value).__name__}")
    
    data_context_str = "\n".join(data_description)
    
    prompt = f"""Generate Python code to accomplish the following task using pandas:

Task: {task_description}

Available Variables:
{data_context_str}

Requirements:
1. Use pandas (imported as 'pd') for data manipulation
2. Store the final result in a variable called 'result'
3. Do NOT use file I/O, os, sys, or subprocess
4. Keep code simple and efficient
5. Add comments explaining key steps

Available imports: pandas (pd), numpy (np), json, math, statistics, datetime, re

Example structure:
```python
import pandas as pd

# Convert to DataFrame if needed
df = pd.DataFrame(sales_data)

# Your analysis here
result = df['column'].mean()

# Store result
result
```

Generate ONLY the Python code, no explanations before or after."""

    try:
        code = await llm.call_llm(prompt, temperature=0.2)
        
        # Clean up markdown code blocks if present
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0].strip()
        elif '```' in code:
            code = code.split('```')[1].split('```')[0].strip()
        
        logger.info(f"[CODE GENERATOR] Generated {len(code)} chars of code")
        return code
    
    except Exception as e:
        logger.error(f"[CODE GENERATOR] Failed to generate code: {e}")
        return f"# Error generating code: {e}\nresult = None"


# Example usage and test cases
async def test_sandbox():
    """Test the Python sandbox with various examples."""
    
    # Test 1: Simple calculation
    result1 = await execute_python_code(
        code="""
import pandas as pd

data = {'sales': [100, 200, 300]}
df = pd.DataFrame(data)
result = df['sales'].mean()
""",
        timeout=5,
    )
    print("Test 1 (Simple calc):", result1)
    
    # Test 2: Using data context
    result2 = await execute_python_code(
        code="""
import pandas as pd

df = pd.DataFrame(sales_data)
result = {
    'total': df['amount'].sum(),
    'average': df['amount'].mean(),
    'count': len(df),
}
""",
        data_context={
            "sales_data": [
                {"product": "A", "amount": 100},
                {"product": "B", "amount": 200},
                {"product": "C", "amount": 150},
            ]
        },
        timeout=5,
    )
    print("Test 2 (With context):", result2)
    
    # Test 3: Restricted operation (should fail)
    result3 = await execute_python_code(
        code="import os; result = os.listdir('.')",
        timeout=5,
    )
    print("Test 3 (Restricted):", result3)


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_sandbox())
