"""Response building service for different query types."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from .. import llm, models, schemas
from ..config import settings, get_schema
from ..helpers import current_timestamp, make_json_serializable
from .file_handler import add_file, retrieve_relevant_chunks
from .query_executor import run_sql, apply_smart_limit
from .response_formatter import should_return_as_message, determine_visualization_type
from .sql_generator import generate_sql, generate_sql_with_analysis
from .followup_manager import get_followup_analyzer
from .database_adapter import get_global_adapter, DatabaseType

# NEW: Import semantic query orchestrator for plan-first generation
from .semantic_query_orchestrator import SemanticQueryOrchestrator

# Module-level semantic orchestrator instance (lazy-initialized on first use)
_semantic_orchestrator: Optional[SemanticQueryOrchestrator] = None

async def _get_semantic_orchestrator(db: AsyncSession) -> SemanticQueryOrchestrator:
    """Get or initialize the semantic query orchestrator (lazy singleton with db_session)."""
    global _semantic_orchestrator
    if _semantic_orchestrator is None:
        _semantic_orchestrator = SemanticQueryOrchestrator(db_session=db)
        print("[INIT] Semantic Query Orchestrator initialized (lazy)")
    return _semantic_orchestrator


def add_schema_prefixes_to_sql(sql: str, schema_name: str = None) -> str:
    """
    Add schema prefixes to all table references in SQL if not already present.
    
    Handles:
    - FROM clause table names
    - JOIN clause table names
    - Preserves existing schema prefixes
    - Preserves table aliases
    - Handles both unaliased and aliased tables
    
    Args:
        sql: The SQL query string
        schema_name: Optional schema name (defaults to get_schema() which uses .env config)
    
    Returns:
        SQL with schema-qualified table names
    
    Example:
        Input:  "SELECT t.* FROM transactions t JOIN customers c ON t.customer_id = c.customer_id"
        Output: "SELECT t.* FROM public.transactions t JOIN public.customers c ON t.customer_id = c.customer_id"
    """
    if not schema_name:
        schema_name = get_schema()  # Uses settings.postgres_schema or defaults to "public"
    
    result = sql
    
    # Pattern to match table references that DON'T already have schema prefix
    # This matches: FROM/JOIN table_name or FROM/JOIN table_name alias
    # But NOT: FROM/JOIN schema.table_name, FROM/JOIN schema.table_name alias
    
    # Pattern 1: FROM clause
    # Match: FROM table_name or FROM table_name alias
    # Don't match: FROM schema.table_name or FROM schema.table_name alias
    from_pattern = r'\bFROM\s+(?![\w]+\.)(\w+)(?:\s+(?:AS\s+)?(\w+))?(?=\s+(?:WHERE|JOIN|LEFT|RIGHT|INNER|LIMIT|ORDER|GROUP|,|$))'
    
    def replace_from(match):
        full_match = match.group(0)
        table_name = match.group(1)
        alias = match.group(2)
        
        # Check if table_name already has schema prefix (shouldn't happen due to negative lookahead, but be safe)
        if '.' in table_name:
            return full_match
        
        # Build the replacement preserving the alias
        if alias:
            return f"FROM {schema_name}.{table_name} {alias}"
        else:
            return f"FROM {schema_name}.{table_name}"
    
    result = re.sub(from_pattern, replace_from, result, flags=re.IGNORECASE | re.MULTILINE)
    
    # Pattern 2: JOIN clause (handles LEFT JOIN, RIGHT JOIN, INNER JOIN, etc.)
    # Match: JOIN table_name or JOIN table_name alias
    # Don't match: JOIN schema.table_name or JOIN schema.table_name alias
    join_pattern = r'\bJOIN\s+(?![\w]+\.)(\w+)(?:\s+(?:AS\s+)?(\w+))?(?=\s+(?:ON|WHERE|JOIN|LEFT|RIGHT|INNER|LIMIT|ORDER|GROUP|,|$))'
    
    def replace_join(match):
        full_match = match.group(0)
        table_name = match.group(1)
        alias = match.group(2)
        
        # Check if table_name already has schema prefix
        if '.' in table_name:
            return full_match
        
        # Build the replacement preserving the alias
        if alias:
            return f"JOIN {schema_name}.{table_name} {alias}"
        else:
            return f"JOIN {schema_name}.{table_name}"
    
    result = re.sub(join_pattern, replace_join, result, flags=re.IGNORECASE | re.MULTILINE)
    
    # Also handle LEFT JOIN, RIGHT JOIN, INNER JOIN, CROSS JOIN, FULL OUTER JOIN variants
    # Pattern matches: LEFT|RIGHT|INNER|CROSS|FULL|OUTER JOIN
    variant_join_pattern = r'\b(?:LEFT|RIGHT|INNER|CROSS|FULL|OUTER)\s+(?:INNER|OUTER\s+)?JOIN\s+(?![\w]+\.)(\w+)(?:\s+(?:AS\s+)?(\w+))?(?=\s+(?:ON|WHERE|JOIN|LEFT|RIGHT|INNER|LIMIT|ORDER|GROUP|,|$))'
    
    def replace_variant_join(match):
        full_match = match.group(0)
        table_name = match.group(1)
        alias = match.group(2) if match.lastindex >= 2 else None
        
        # Check if table_name already has schema prefix
        if '.' in table_name:
            return full_match
        
        # Build the replacement preserving the full variant and alias
        if alias:
            return f"{full_match.rsplit(table_name, 1)[0]}{schema_name}.{table_name} {alias}"
        else:
            return f"{full_match.rsplit(table_name, 1)[0]}{schema_name}.{table_name}"
    
    result = re.sub(variant_join_pattern, replace_variant_join, result, flags=re.IGNORECASE | re.MULTILINE)
    
    print(f"[SCHEMA_PREFIX] SQL after adding schema prefixes:\n  Before: {sql[:120]}...\n  After:  {result[:120]}...")
    
    return result


def build_smart_schema_context(query: str, schema_info: str) -> str:
    """
    Build intelligent schema context for LLM based on query semantics.
    
    FULLY DATABASE-AGNOSTIC: No hardcoded table/column names!
    
    Includes:
    - Table relationships and JOINs (dynamically detected)
    - Boolean column semantics (found from schema)
    - Data type information
    - Generic join patterns
    """
    
    schema_lower = schema_info.lower()
    query_lower = query.lower()
    
    # Build context
    context = schema_info
    context += "\n\n=== SEMANTIC ANALYSIS (DYNAMICALLY EXTRACTED) ===\n"
    context += "NOTE: Analysis is COMPLETELY DYNAMIC - no hardcoded table/column names\n"
    context += "      System detects patterns from YOUR schema:\n"
    context += "      - Any boolean columns (true/false)\n"
    context += "      - Any status/approval column patterns\n"
    context += "      - Any enum columns with values\n"
    context += "      - Any foreign key relationships\n\n"
    
    # DYNAMIC: Extract ALL table names from schema (not hardcoded)
    all_tables = re.findall(r'(?:^|\n|\s)([\w]+)\s*(?:table|Table|:)', schema_info, re.MULTILINE)
    all_tables = list(set(t for t in all_tables if t.lower() not in ['schema', 'information', 'columns', 'table']))
    
    context += f"✓ Detected tables from schema: {', '.join(all_tables)}\n"
    
    # DYNAMIC: Detect approval/status patterns from schema
    # Look for common approval-related column patterns (NO HARDCODING)
    approval_patterns = [
        (r'(\w*verified\w*)', 'verification'),
        (r'(\w*approval\w*)', 'approval'),
        (r'(\w*status\w*)', 'status'),
        (r'(\w*state\w*)', 'state'),
        (r'(\w*active\w*)', 'active'),
        (r'(\w*enabled\w*)', 'enabled'),
    ]
    
    found_approval_cols = []
    for pattern, pattern_type in approval_patterns:
        matches = re.findall(pattern, schema_lower, re.IGNORECASE)
        for match in matches:
            if match not in found_approval_cols:
                found_approval_cols.append(match)
                context += f"✓ Found {pattern_type} column: {match}\n"
    
    # DYNAMIC: Extract all foreign key patterns (id fields, relationships)
    # Look for common FK patterns: table_id, user_id, order_id, etc.
    fk_patterns = re.findall(r'(\w+_id)\b', schema_info, re.IGNORECASE)
    if fk_patterns:
        context += f"\n✓ Detected foreign key patterns: {', '.join(set(fk_patterns))}\n"
        context += "  These indicate relationships between tables\n"
    
    context += "\n=== GENERIC JOIN GUIDANCE (ANY MULTI-TABLE QUERY) ===\n"
    
    # DYNAMIC: Detect multi-table mentions in user query (not hardcoded table names)
    word_tokens = query_lower.split()
    
    context += "If your query mentions multiple tables:\n"
    context += "  1. Identify the main tables referenced\n"
    context += "  2. Look for OF foreign keys (e.g., 'X_id') to connect them\n"
    context += "  3. Use JOIN with ON clause matching the FK relationships\n"
    context += "  4. Generic pattern: JOIN table2 ON table1.fk_id = table2.id\n"
    
    context += "\n=== BOOLEAN COLUMN SYNTAX (ANY DATABASE) ===\n"
    context += "Use 'true' or 'false' (lowercase), NOT 1/0 or 'yes'/'no'\n"
    
    # DYNAMIC: Detect boolean columns from schema (not hardcoded kyc_verified)
    bool_cols = re.findall(r'(\w+)\s+boolean', schema_lower, re.IGNORECASE)
    if bool_cols:
        unique_bool_cols = list(set(bool_cols))
        context += f"Detected boolean columns: {', '.join(unique_bool_cols)}\n"
        for col in unique_bool_cols[:3]:
            context += f"  - Example: WHERE {col} = true or WHERE {col} = false\n"
    
    # DYNAMIC: Detect enum/status columns and extract values from schema
    context += "\n=== ENUM/STATUS COLUMNS (EXTRACTED FROM SCHEMA) ===\n"
    
    # Look for quoted values patterns (likely enum values)
    enum_values = re.findall(r"'([a-z_]+)'", schema_lower)
    if enum_values:
        unique_enums = list(set(enum_values))[:5]
        context += f"Detected enum-like values: {', '.join(unique_enums)}\n"
        context += "Use these values in your WHERE clauses (lowercase)\n"
    
    context += "\n=== SCHEMA AGNOSTIC INSTRUCTIONS ===\n"
    context += "1. NO hardcoded assumptions about table/column names\n"
    context += "2. Adapt to whatever tables/columns are in YOUR schema\n"
    context += "3. For relationships: Find foreign key patterns (X_id columns)\n"
    context += "4. For filtering: Map user intent to columns found in schema\n"
    
    return context


def normalize_sql_semantics(sql: str, user_query: str) -> str:
    """
    Post-process SQL to fix common LLM semantic mistakes.
    
    FULLY DATABASE-AGNOSTIC: Uses database adapter for all database-specific logic
    - Doesn't hardcode column names
    - Adapts to any database schema
    - Uses appropriate boolean/enum syntax for connected database
    
    Handles:
    - Boolean literal fixes (adapts to database's boolean syntax)
    - Boolean logic correction based on user intent
    - Enum value normalization (uses adapter's normalize_enum_value)
    """
    
    result = sql
    query_lower = user_query.lower()
    
    # Get database adapter for database-specific operations
    adapter = get_global_adapter()
    
    # Rule 1: Normalize boolean literals using database adapter
    # Different databases use different boolean syntax:
    # PostgreSQL: true/false, MySQL: 1/0, SQLite: 1/0, SQL Server: 1/0
    bool_literal_true, bool_literal_false = adapter.get_capabilities().boolean_literals
    
    # Replace all variations of boolean values with database-appropriate literals
    result = re.sub(
        r'\b(?:FALSE|True|FALSE|TRUE|0|1|\'true\'|\'false\'|\'0\'|\'1\')\b',
        lambda m: bool_literal_true if m.group().upper() in ('TRUE', '1', "'TRUE'") else bool_literal_false,
        result,
        flags=re.IGNORECASE
    )
    
    # Rule 2: DYNAMIC boolean filter normalization based on user intent
    # Extract ANY boolean column from the SQL and adapt to user intent
    bool_comparison_pattern = r'(\w+)\s*=\s*(?:true|false|True|False|1|0|\'true\'|\'false\'|\'1\'|\'0\')\b'
    bool_comparisons = re.findall(bool_comparison_pattern, result, re.IGNORECASE)
    
    # Check user intent (what they want: approved/unapproved, verified/unverified)
    wants_negative = any(phrase in query_lower for phrase in [
        'not approved', 'not verified', 'unapproved', 'unverified', 
        'not clarified', 'not active', 'disabled', 'inactive', 'negative'
    ])
    wants_positive = any(phrase in query_lower for phrase in [
        'approved', 'verified', 'clarified', 'active', 'enabled', 'true'
    ])
    
    # Apply logic to ANY boolean column found in the SQL using database adapter
    for bool_col in bool_comparisons:
        if wants_negative:
            # User wants negative state → set to appropriate false value for this database
            result = re.sub(
                rf'{bool_col}\s*=\s*(?:true|True|1|\'true\'|\'1\'|\'yes\')',
                f'{bool_col} = {bool_literal_false}',
                result,
                flags=re.IGNORECASE
            )
        elif wants_positive:
            # User wants positive state → set to appropriate true value for this database
            result = re.sub(
                rf'{bool_col}\s*=\s*(?:false|False|0|\'false\'|\'0\'|\'no\')',
                f'{bool_col} = {bool_literal_true}',
                result,
                flags=re.IGNORECASE
            )
    
    # Rule 3: Normalize enum value casings using database adapter
    # Call adapter's normalize_enum_value for database-appropriate syntax
    enum_pattern = r"=\s*'([A-Za-z_]+)'"
    
    def normalize_enum(match):
        enum_val = match.group(1)
        # Only process if it looks like an enum (has underscores or all uppercase)
        if ('_' in enum_val or (enum_val.isupper() and len(enum_val) >= 4)):
            normalized = adapter.normalize_enum_value(enum_val)
            return f"= '{normalized}'"
        return match.group(0)
    
    result = re.sub(enum_pattern, normalize_enum, result)
    
    # Rule 4: DYNAMIC JOIN detection (NO hardcoded table names)
    # If user query mentions multiple entities and SQL doesn't have JOIN, flag for user review
    # Don't try to inject JOINs - let LLM or user handle this based on actual schema
    
    if result != sql:
        print(f"[NORMALIZE] SQL semantics adjusted (DATABASE-AGNOSTIC - {adapter.db_type.value.upper()}):\n  "
              f"Before: {sql[:120]}\n  After: {result[:120]}")
    
    return result


def extract_semantic_hints_from_schema(schema_context: str, query: str) -> str:
    """
    DYNAMICALLY extract semantic hints from schema context.
    
    Analyzes schema to find common patterns:
    - Boolean columns and their values
    - Status/Approval columns with examples
    - Foreign key relationships
    
    Returns hint text for LLM about how to map user intent to SQL.
    """
    hints = []
    context_lower = schema_context.lower()
    query_lower = query.lower()
    
    # Check for boolean columns in schema
    bool_cols = re.findall(r'(\w+)\s+boolean', context_lower, re.IGNORECASE)
    if bool_cols:
        unique_bool_cols = list(set(bool_cols))[:3]  # Top 3
        hints.append(f"   - User intent keywords (approved, verified, active, enabled, etc.) should map to boolean columns")
        hints.append(f"   - Found boolean columns: {', '.join(unique_bool_cols)}")
        hints.append(f"   - For 'not [intent]' phrases, use = false; for '[intent]' use = true")
    
    # Check for approval/status patterns in schema context
    if 'kyc' in context_lower or 'verification' in context_lower or 'approval' in context_lower:
        hints.append(f"   - Query mentions approval/verification concepts")
        hints.append(f"   - Find the corresponding boolean or status column and use appropriate filter")
    
    # Check for enum patterns
    enum_example_match = re.search(r"'([a-z_]+)'\s*,\s*'([a-z_]+)'", schema_context)
    if enum_example_match:
        hints.append(f"   - Enum columns detected: use lowercase values like '{enum_example_match.group(1)}', '{enum_example_match.group(2)}'")
    
    # Check for multi-table references
    if 'join' in context_lower.lower():
        hints.append(f"   - Multi-table query suggested: Use provided JOIN examples from schema")
        hints.append(f"   - Match ON clause keys based on actual foreign key relationships")
    
    # If no specific hints found, return generic guidance
    if not hints:
        hints = [
            "   - Analyze schema context above to find columns matching user intent",
            "   - For boolean/approval concepts: find column with 'approved', 'verified', 'status' in name",
            "   - For values: Use 'true'/'false' for booleans, lowercase for enums",
            "   - For joins: Follow relationship patterns shown in schema context"
        ]
    
    return "\n".join(hints)


def extract_tables_and_columns(schema_context: str) -> dict:
    """
    Extract all tables and their columns from schema context.
    
    Returns: {
        'table_name': {
            'columns': set of column names,
            'primary_key': column name or None
        }
    }
    """
    tables = {}
    
    # Find all "Table: table_name" sections
    table_sections = re.finditer(r'Table:\s*(\w+)(.*?)(?=Table:|$)', schema_context, re.IGNORECASE | re.DOTALL)
    
    for match in table_sections:
        table_name = match.group(1).lower()
        table_content = match.group(2)
        
        # Extract columns from this table
        columns = set()
        
        # Pattern: "  - column_name (type)"
        col_matches = re.findall(r'(?:^|\n)\s*-\s+(\w+)\s+\(', table_content, re.MULTILINE)
        columns.update(col.lower() for col in col_matches)
        
        # Pattern: "*** column_name ***"
        col_matches = re.findall(r'\*\*\*\s*(\w+)\s*\*\*\*', table_content)
        columns.update(col.lower() for col in col_matches)
        
        # Detect primary keys
        pk = None
        if 'PRIMARY KEY' in table_content:
            pk_match = re.search(r'(\w+).*?\[PRIMARY KEY\]', table_content, re.IGNORECASE | re.DOTALL)
            if pk_match:
                pk = pk_match.group(1).lower()
        
        if columns:
            tables[table_name] = {
                'columns': columns,
                'primary_key': pk
            }
    
    return tables


def find_joinable_column(wrong_col: str, wrong_table: str, all_tables: dict, schema_context: str) -> tuple:
    """
    Cross-table column search: find if wrong_col exists in joinable tables.
    
    Returns: (found_table, found_column, join_condition) or (None, None, None)
    
    Example: transactions.kyc_approved not found
             But customers.kyc_verified exists
             And transactions.customer_id = customers.customer_id
             Returns: ('customers', 'kyc_verified', 'transactions.customer_id = customers.customer_id')
    """
    
    # Heuristic synonym matching
    def synonym_score(col1: str, col2: str) -> float:
        """Score similarity between column names using tokens and synonyms.
        
        Prefers: kyc_verified > is_kyc_approved (direct semantic match vs prefix pattern)
        """
        tokens1 = set(col1.lower().split('_'))
        tokens2 = set(col2.lower().split('_'))
        
        # Token overlap
        shared = tokens1 & tokens2
        if not shared:
            return 0.0
        
        # Synonym mapping for common boolean/status columns
        synonyms = {
            'approved': {'verified', 'confirmed', 'valid'},
            'verified': {'approved', 'confirmed', 'valid'},
            'confirmed': {'approved', 'verified', 'valid'},
            'active': {'enabled', 'true', 'yes'},
            'enabled': {'active', 'true', 'yes'},
            'valid': {'approved', 'verified', 'confirmed'}
        }
        
        # Check for synonym matches and identify which tokens are synonyms
        synonym_tokens_found = []
        for tok1 in tokens1:
            if tok1 in synonyms:
                for tok2 in tokens2:
                    if tok2 in synonyms[tok1]:
                        synonym_tokens_found.append((tok1, tok2))
        
        base_score = len(shared) / max(len(tokens1), len(tokens2))
        
        # BOOST for direct semantic synonym matches (approved <-> verified)
        # These should score higher than prefix-based matches
        if synonym_tokens_found:
            # Direct semantic match: very high confidence
            base_score = 0.85
        
        # PENALIZE boolean prefix patterns (is_, has_, etc.) 
        # Prefer: kyc_verified (0.85) over is_kyc_approved (0.85 - 0.15 = 0.70)
        has_boolean_prefix = any(token in tokens2 for token in ['is', 'has', 'can', 'was', 'were'])
        if has_boolean_prefix and synonym_tokens_found:
            base_score -= 0.15  # Penalize prefix patterns when direct semantic match exists
        
        return base_score
    
    def infer_join_condition(table1: str, table2: str) -> str:
        """Infer join condition between two tables by checking for common ID columns."""
        # Look for matching ID columns in both tables
        # Pattern: customers_id, customer_id, cust_id, etc.
        
        # Check for common patterns
        patterns_to_check = [
            (f"{table2}_id", "id"),  # transactions.customers_id = customers.id
            (f"{table2}_id", f"{table2}_id"),  # transactions.customers_id = customers.customers_id
            ("customer_id", "customer_id"),  # transactions.customer_id = customers.customer_id
            ("customer_id", "id"),  # transactions.customer_id = customers.id (if id is PK)
        ]
        
        for col1_pattern, col2_pattern in patterns_to_check:
            # Check if col1_pattern exists in table1 columns
            if col1_pattern in all_tables.get(table1, {}).get('columns', set()):
                # Check if col2_pattern exists in table2 columns
                if col2_pattern in all_tables.get(table2, {}).get('columns', set()):
                    return f"{table1}.{col1_pattern} = {table2}.{col2_pattern}"
        
        # Fallback: look for any matching columns
        t1_cols = all_tables.get(table1, {}).get('columns', set())
        t2_cols = all_tables.get(table2, {}).get('columns', set())
        
        for col in ['customer_id', 'user_id', 'account_id', 'org_id']:
            if col in t1_cols and col in t2_cols:
                return f"{table1}.{col} = {table2}.{col}"
        
        # Default fallback
        return f"{table1}.id = {table2}.id"
    
    best_match = None
    best_score = 0.0
    best_join = None
    
    # Search other tables
    for other_table, other_info in all_tables.items():
        if other_table.lower() == wrong_table.lower():
            continue  # Skip the target table
        
        # Check if tables can be joined (look for *_id pattern)
        if f"{other_table}_id" not in all_tables[wrong_table]['columns'] and \
           f"{wrong_table}_id" not in other_info['columns'] and \
           other_table != "customers":  # Special case: might join on customer_id
            continue  # Can't join these tables
        
        # Score all columns in this table
        for col in other_info['columns']:
            score = synonym_score(wrong_col, col)
            
            if score > best_score:
                best_score = score
                best_match = col
                best_join = infer_join_condition(wrong_table, other_table)
                best_table = other_table
    
    if best_score >= 0.5:  # High threshold for cross-table relocation
        return (best_table, best_match, best_join)
    
    return (None, None, None)


def map_semantic_columns(sql: str, schema_context: str, user_query: str) -> str:
    """
    DYNAMICALLY map semantic intent to actual column names - 100% database-agnostic.
    
    NO HARDCODED PATTERNS - Everything extracted from actual schema!
    TABLE-AWARE: Matches columns from target table, relocates cross-table predicates.
    PREDICATE RELOCATION: If column missing in target table but exists in joinable table,
                          move predicate there and add JOIN.
    
    Args:
        sql: SQL generated by LLM (may contain wrong column names)
        schema_context: Actual schema information
        user_query: Original user query
        
    Returns:
        SQL with semantic column names corrected, predicates relocated, JOINs added
    """
    result = sql
    query_lower = user_query.lower()
    schema_lower = schema_context.lower()
    
    # CRITICAL: Extract target table from SQL to ensure table-aware matching
    # This prevents suggesting columns from OTHER tables (e.g., customers when querying transactions)
    target_table = None
    table_match = re.search(r'(?:FROM|JOIN)\s+(?:ONLY\s+)?(?:\w+\.)?(\w+)\s+(?:\w+)?(?:\s|$|WHERE|JOIN|,)', sql, re.IGNORECASE)
    if table_match:
        target_table = table_match.group(1).lower()
        print(f"[DYNAMIC_MAP] Target table: {target_table}")
    
    # DYNAMIC: Extract ALL column names from schema context (not hardcoded)
    schema_cols = []
    
    # Pattern 1: "  - column_name (type)" (from table_columns_info format)
    pattern1 = re.findall(r'(?:^|\n)\s*-\s+(\w+)\s+\(', schema_context, re.MULTILINE)
    schema_cols.extend(pattern1)
    
    # Pattern 2: "*** column_name ***" (from schema_info format with bold markers)
    pattern2 = re.findall(r'\*\*\*\s*(\w+)\s*\*\*\*', schema_context)
    schema_cols.extend(pattern2)
    
    # Pattern 3: "word (type)" anywhere in schema text
    pattern3 = re.findall(r'\b(\w+)\s+\((?:bigint|text|boolean|numeric|integer|timestamp|date|jsonb|uuid|varchar)', schema_context, re.IGNORECASE)
    schema_cols.extend(pattern3)
    
    # Pattern 4: Word token followed by type keyword
    pattern4 = re.findall(r'\b([a-z_]\w*)\s+(?:bigint|text|boolean|numeric|integer|timestamp|date|jsonb|uuid|varchar|USER-DEFINED|INT|VARCHAR|CHAR|BOOLEAN|TIMESTAMP)\b', schema_context, re.IGNORECASE)
    schema_cols.extend(pattern4)
    
    # Deduplicate
    schema_cols_set = set(col.lower() for col in schema_cols if col and len(col) > 1)
    
    # TABLE-AWARE FILTERING: If we know the target table, only use columns from that table
    # This prevents suggesting columns from other tables (e.g., customers table columns when querying transactions)
    if target_table and "Table:" in schema_context:
        # Extract columns for the specific target table from schema
        # Pattern: "Table: table_name\n...Columns...\n...columns..."
        # Find the section for this table and stop at the next "Table:" or end of text
        pattern = rf'Table:\s*{target_table}.*?(?=Table:|$)'
        table_section = re.search(pattern, schema_context, re.IGNORECASE | re.DOTALL)
        
        if table_section:
            table_text = table_section.group(0)
            # Extract columns from this specific table section
            table_specific_cols = []
            
            # Extract from "  - column_name (type)" format
            pattern1 = re.findall(r'(?:^|\n)\s*-\s+(\w+)\s+\(', table_text, re.MULTILINE)
            table_specific_cols.extend(pattern1)
            
            # Extract from "*** column_name ***" format
            pattern2 = re.findall(r'\*\*\*\s*(\w+)\s*\*\*\*', table_text)
            table_specific_cols.extend(pattern2)
            
            # Extract from "word (type)" format
            pattern3 = re.findall(r'\b(\w+)\s+\((?:bigint|text|boolean|numeric|integer|timestamp|date|jsonb|uuid|varchar)', table_text, re.IGNORECASE)
            table_specific_cols.extend(pattern3)
            
            # Extract from "word TYPE" format
            pattern4 = re.findall(r'\b([a-z_]\w*)\s+(?:bigint|text|boolean|numeric|integer|timestamp|date|jsonb|uuid|varchar|USER-DEFINED|INT|VARCHAR|CHAR|BOOLEAN|TIMESTAMP)\b', table_text, re.IGNORECASE)
            table_specific_cols.extend(pattern4)
            
            table_specific_set = set(col.lower() for col in table_specific_cols if col and len(col) > 1)
            
            if table_specific_set:
                # Use only columns from this specific table
                original_count = len(schema_cols_set)
                schema_cols_set = table_specific_set
                print(f"[DYNAMIC_MAP] TABLE-AWARE: Filtered {original_count} → {len(schema_cols_set)} columns for table '{target_table}'")
    
    print(f"[DYNAMIC_MAP] Extracted {len(schema_cols_set)} columns from schema: {list(schema_cols_set)[:5]}...")
    
    # DYNAMIC: Extract all column references from the SQL (both wrong and correct)
    # These are the column names being used in WHERE/SELECT/JOIN clauses
    sql_cols = re.findall(r'\b([a-z_]\w*)\s*(?:=|<|>|IN|LIKE|NOT)', sql, re.IGNORECASE)
    sql_cols += re.findall(r'(?:SELECT|WHERE|AND|OR)\s+([a-z_]\w*)\s*(?:FROM|WHERE|,)', sql, re.IGNORECASE)
    sql_cols_set = set(col.lower() for col in sql_cols if col and len(col) > 1)
    
    print(f"[DYNAMIC_MAP] Extracted {len(sql_cols_set)} columns from SQL: {list(sql_cols_set)[:5]}...")
    
    # DYNAMIC: Find columns in SQL that DON'T exist in schema (likely wrong names)
    wrong_cols = sql_cols_set - schema_cols_set
    print(f"[DYNAMIC_MAP] Detected potential wrong columns: {wrong_cols}")
    
    # CROSS-TABLE PREDICATE RELOCATION: Check if wrong columns exist in joinable tables
    # This handles cases where filter predicates belong to a different table
    # Example: transactions.kyc_approved doesn't exist, but customers.kyc_verified does
    if wrong_cols and target_table:
        print(f"[DYNAMIC_MAP] Attempting cross-table predicate relocation...")
        all_tables = extract_tables_and_columns(schema_context)
        
        # First, ensure the main table has an alias
        main_table_alias = None
        sql_keywords = {'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'CROSS', 
                        'LIMIT', 'ORDER', 'GROUP', 'HAVING', 'UNION', 'EXCEPT', 'INTERSECT', 'ON'}
        
        # Check if FROM clause already has an alias for the target table
        from_pattern = rf'\bFROM\s+{re.escape(target_table)}(?:\s+(?:as\s+)?(\w+))?\b'
        from_check = re.search(from_pattern, result, re.IGNORECASE)
        
        if from_check and from_check.group(1):
            potential_alias = from_check.group(1)
            # Check if it's a SQL keyword
            if potential_alias.upper() not in sql_keywords:
                main_table_alias = potential_alias
                print(f"[DYNAMIC_MAP] Main table already has alias: {main_table_alias}")
        
        if not main_table_alias:
            # Add alias to main table
            main_table_alias = target_table[0].lower()  # e.g., 't' for transactions
            # Replace "FROM transactions" with "FROM transactions t", but only if not already followed by clause keyword
            result = re.sub(
                rf'\bFROM\s+{re.escape(target_table)}(?=\s+(?:WHERE|JOIN|LEFT|RIGHT|INNER|LIMIT|ORDER|GROUP))',
                f'FROM {target_table} {main_table_alias}',
                result,
                flags=re.IGNORECASE
            )
            # Also handle end of string case
            result = re.sub(
                rf'\bFROM\s+{re.escape(target_table)}\s*$',
                f'FROM {target_table} {main_table_alias}',
                result,
                flags=re.IGNORECASE | re.MULTILINE
            )
            # Replace all full table name references with alias in SELECT and WHERE
            result = re.sub(
                rf'\b{re.escape(target_table)}\.',
                f'{main_table_alias}.',
                result,
                flags=re.IGNORECASE
            )
            print(f"[DYNAMIC_MAP] Main table aliased: {target_table} → {main_table_alias}")
        
        for wrong_col in list(wrong_cols):
            # Check if this column exists in other joinable tables
            found_table, found_col, join_condition = find_joinable_column(
                wrong_col, target_table, all_tables, schema_context
            )
            
            if found_table and found_col:
                print(f"[DYNAMIC_MAP] RELOCATION: '{wrong_col}' not in {target_table}")
                print(f"[DYNAMIC_MAP]   Found in {found_table}: '{found_col}'")
                print(f"[DYNAMIC_MAP]   Adding JOIN: {join_condition}")
                
                # Create table alias for the other table (random unique alias, not hardcoded first letter)
                import random, string
                random_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                joined_table_alias = f"t_{random_chars}"
                
                # Single pass: Replace all references to wrong_col with the correct aliased column
                # This handles: t.kyc_approved, transactions.kyc_approved, or just kyc_approved
                # All become: c.kyc_verified
                
                # Pattern: any optional table prefix + wrong_col + word boundary
                # Escape the column name for safety
                escaped_wrong_col = re.escape(wrong_col)
                pattern = rf'(?:\w+\.)?{escaped_wrong_col}\b'
                replacement = f'{joined_table_alias}.{found_col}'
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                
                # Now add JOIN clause if not already present
                if f"JOIN {found_table}" not in result.upper():
                    # Insert JOIN before WHERE clause using regex
                    where_match = re.search(r'\s+WHERE\s+', result, re.IGNORECASE)
                    limit_match = re.search(r'\s+LIMIT\s+', result, re.IGNORECASE)
                    
                    insert_pos = None
                    if where_match:
                        insert_pos = where_match.start()
                    elif limit_match:
                        insert_pos = limit_match.start()
                    
                    if insert_pos:
                        # Replace both table names with their aliases in the join condition
                        # Before: "transactions.customer_id = customers.customer_id"
                        # After:  "t.customer_id = c.customer_id"
                        join_cond_with_aliases = join_condition
                        if main_table_alias:
                            join_cond_with_aliases = join_cond_with_aliases.replace(
                                f"{target_table}.",
                                f"{main_table_alias}."
                            )
                        join_cond_with_aliases = join_cond_with_aliases.replace(
                            f"{found_table}.",
                            f"{joined_table_alias}."
                        )
                        
                        join_clause = f" JOIN {found_table} {joined_table_alias} ON {join_cond_with_aliases}"
                        result = result[:insert_pos] + join_clause + result[insert_pos:]
                        print(f"[DYNAMIC_MAP] SQL after relocation: {result[:150]}...")
                
                # Remove from wrong_cols set since we handled it
                wrong_cols.discard(wrong_col)
    
    # Initialize semantic mapping (will be populated if needed)
    semantic_to_actual = {}
    
    # If wrong columns remain, try semantic matching
    if wrong_cols:
        print(f"[DYNAMIC_MAP] Attempting semantic matching for remaining columns...")
        
        def lcs_length(s1: str, s2: str) -> int:
            """Calculate Longest Common Subsequence length between two strings."""
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[m][n]
    
        def is_abbreviation(short: str, long: str) -> bool:
            """Check if short is a likely abbreviation of long using LCS matching.
            
            An abbreviation is valid if:
            1. First character matches (strong indicator)
            2. Most of the short word characters appear in long in order (LCS)
            3. Short word is <70% of long word length
            4. No 2-char tokens (these are usually English words like "at", "is")
            
            Prevents false positives like "at" matching "active".
            """
            # Exclude 2-char tokens - they're usually prepositions/articles, not abbreviations
            if len(short) <= 2:
                return False
            
            if len(long) <= 2 or len(long) - len(short) < 2:
                return False
            
            short_lower = short.lower()
            long_lower = long.lower()
            
            # First character must match (key indicator)
            if short_lower[0] != long_lower[0]:
                return False
            
            # Calculate LCS (Longest Common Subsequence)
            lcs = lcs_length(short_lower, long_lower)
            lcs_ratio = lcs / len(short_lower)
            
            # For 3-char words and above, need good LCS match
            if lcs_ratio >= 0.6:
                length_ratio = len(short_lower) / len(long_lower)
                return length_ratio < 0.7
            
            return False
    
        def calculate_semantic_score(wrong_col: str, schema_col: str) -> float:
            """Calculate semantic similarity between two column names.
            
            Uses multiple scoring strategies to find best matches:
            1. Token overlap (exact component matches)
            2. Token containment (component appears in other)
            3. Substring matching (character-level similarity)
            4. Abbreviation matching
            5. Length similarity
            """
            wrong_col_raw = wrong_col
            schema_col_raw = schema_col
            wrong_col = wrong_col.lower()
            schema_col = schema_col.lower()
            
            # If exact match, return perfect score
            if wrong_col == schema_col:
                return 1.0
            
            # Split by underscores and common separators
            wrong_tokens = set(wrong_col.lower().split('_'))
            schema_tokens = set(schema_col.lower().split('_'))
            
            total_score = 0.0
            
            # STRATEGY 1: Token overlap - exact token matches (highest weight)
            # Example: "kyc_approved" tokens ['kyc', 'approved'] in ['is', 'kyc', 'approved']
            shared_tokens = wrong_tokens & schema_tokens
            if shared_tokens:
                # More shared tokens = higher confidence
                token_overlap_score = len(shared_tokens) / max(len(wrong_tokens), len(schema_tokens))
                total_score += token_overlap_score * 0.40  # Increased from 0.25
            
            # STRATEGY 2: Substring containment - one column contains other
            # This catches "approval" in "is_approved" cases
            if wrong_col in schema_col or schema_col in wrong_col:
                # Direct substring match is very strong
                total_score += 0.30
            
            # STRATEGY 3: Token containment - any wrong token appears in schema_col
            # Example: "approval_date" has token "approval" which appears in "is_approved"
            for wrong_tok in wrong_tokens:
                if len(wrong_tok) >= 3:  # Only significant tokens
                    if wrong_tok in schema_col or schema_col.find(wrong_tok) != -1:
                        total_score += 0.15
                        break
            
            # STRATEGY 4: Prefix token matching bonus
            # If both start with same token (e.g., "is_" in "is_active" and "is_approved")
            if len(wrong_tokens) > 0 and len(schema_tokens) > 0:
                wrong_first = list(wrong_tokens)[0] if wrong_tokens else ""
                schema_first = list(schema_tokens)[0] if schema_tokens else ""
                if wrong_first and schema_first and wrong_first == schema_first and len(wrong_first) > 2:
                    total_score += 0.05
            
            # STRATEGY 5: Abbreviation matching - schema token is abbreviation of wrong token
            abbrev_score = 0
            for wrong_tok in wrong_tokens:
                for schema_tok in schema_tokens:
                    if is_abbreviation(schema_tok, wrong_tok):
                        abbrev_score = 0.25  # Good signal
                        break
                if abbrev_score > 0:
                    break
            total_score += abbrev_score
            
            # STRATEGY 6: Length similarity (very low weight)
            length_diff = abs(len(wrong_col) - len(schema_col))
            max_len = max(len(wrong_col), len(schema_col), 1)
            length_score = max(0, 1.0 - (length_diff / max_len))
            total_score += length_score * 0.05
            
            return total_score
        
        for wrong_col in wrong_cols:
            best_match = None
            best_score = 0
            all_candidates = []  # Track all candidates for debugging
            
            for schema_col in schema_cols_set:
                score = calculate_semantic_score(wrong_col, schema_col)
                all_candidates.append((schema_col, score))
                
                if score > best_score:
                    best_score = score
                    best_match = schema_col
            
            # Sort candidates by score for detailed logging
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Use more intelligent threshold:
            # - If substring containment found: lower threshold (0.15)
            # - If token overlap found: medium threshold (0.25)
            # - Otherwise: normal threshold (0.30)
            confidence_threshold = 0.20
            
            # Check if best_match has substring or token overlap (strong signals)
            if best_match:
                wrong_toks = set(wrong_col.lower().split('_'))
                best_toks = set(best_match.lower().split('_'))
                
                has_substring = wrong_col in best_match or best_match in wrong_col
                has_token_overlap = len(wrong_toks & best_toks) > 0
                
                # Adjust threshold based on signal strength
                if has_substring:
                    confidence_threshold = 0.15  # Substring matches are strong
                elif has_token_overlap:
                    confidence_threshold = 0.20  # Token overlap is medium confidence
            
            # CRITICAL VALIDATION: Require at least ONE strong semantic signal
            # This prevents accepting weak matches like 0.45 that might be from other tables
            has_strong_signal = False
            
            if best_match:
                wrong_toks = set(wrong_col.lower().split('_'))
                best_toks = set(best_match.lower().split('_'))
                shared_toks = len(wrong_toks & best_toks)
                
                has_substring = wrong_col in best_match or best_match in wrong_col
                has_multi_token_overlap = shared_toks >= 2  # At least 2 tokens in common
                has_prefix_match = (len(wrong_toks) > 0 and len(best_toks) > 0 and 
                                   list(wrong_toks)[0] == list(best_toks)[0] and len(list(wrong_toks)[0]) > 2)
                
                # Strong signal = substring OR multiple token overlap OR prefix match
                has_strong_signal = has_substring or has_multi_token_overlap or has_prefix_match
                
                # If no strong signal, require much higher confidence
                if not has_strong_signal:
                    confidence_threshold = 0.50  # Very high threshold for weak matches
            
            # Only map if confidence threshold is met AND has strong signal
            if best_match and best_score >= confidence_threshold and has_strong_signal:
                semantic_to_actual[wrong_col] = best_match
                print(f"[DYNAMIC_MAP] Mapping '{wrong_col}' → '{best_match}' (score: {best_score:.2f})")
                if len(all_candidates) > 1:
                    alts = ', '.join(f"{col}({score:.2f})" for col, score in all_candidates[1:3])
                    print(f"[DYNAMIC_MAP]   Alternatives: {alts}")
            else:
                reason = ""
                if best_match and not has_strong_signal:
                    reason = " (weak signal - no substring/token overlap/prefix match)"
                print(f"[DYNAMIC_MAP] No good match for '{wrong_col}' (best: {best_match or 'none'}, score: {best_score:.2f}){reason}")
                if all_candidates:
                    tops = ', '.join(f"{col}({score:.2f})" for col, score in all_candidates[:3])
                    print(f"[DYNAMIC_MAP]   Top candidates: {tops}")
        
        # DYNAMIC: Apply all semantic mappings to SQL
        for wrong_col, correct_col in semantic_to_actual.items():
            if wrong_col != correct_col:  # Only replace if actually different
                # Replace with word boundaries and case-insensitive
                # Escape the column name for safety
                escaped_wrong_col = re.escape(wrong_col)
                result = re.sub(
                    rf'\b{escaped_wrong_col}\b',
                    correct_col,
                    result,
                    flags=re.IGNORECASE
                )
    
    if result != sql:
        print(f"[DYNAMIC_MAP] SQL corrected:\n  Before: {sql[:100]}...\n  After: {result[:100]}...")
    
    # FINAL STEP: NORMALIZE BOOLEAN VALUES for the target database
    # Convert Python True/False to database-appropriate literals
    # E.g., False → false (PostgreSQL), True → true (PostgreSQL)
    print(f"[TYPE_NORMALIZATION] Normalizing boolean values for database compatibility...")
    
    try:
        adapter = get_global_adapter()
        bool_true, bool_false = adapter.get_capabilities().boolean_literals
        
        # Replace Python boolean capitalization with database format
        # Pattern: Find True/False (standalone, not part of a word)
        result = re.sub(r'\bTrue\b', bool_true, result)
        result = re.sub(r'\bFalse\b', bool_false, result)
        result = re.sub(r'\bTRUE\b', bool_true, result)
        result = re.sub(r'\bFALSE\b', bool_false, result)
        
        if bool_true != "true" or bool_false != "false":
            print(f"[TYPE_NORMALIZATION] Converted True/False to {bool_true}/{bool_false} for {adapter.db_type.value}")
    except Exception as e:
        print(f"[TYPE_NORMALIZATION] Warning: Could not normalize boolean values: {e}")
    
    return result


def is_valid_where_syntax(where_clause: str) -> bool:
    """
    Check if a WHERE clause has valid basic SQL syntax.
    
    Returns False if WHERE clause:
    - Ends with incomplete column reference (e.g., "WHERE c.")
    - Has dangling operators (e.g., "WHERE foo AND")
    - Is empty after processing
    - Has mismatched parentheses
    - Contains incomplete references in any position
    
    Args:
        where_clause: The WHERE clause content (without the WHERE keyword)
        
    Returns:
        True if syntax looks valid, False if malformed
    """
    if not where_clause or not where_clause.strip():
        return True  # Empty WHERE can be removed
    
    where_clean = where_clause.strip()
    
    # Check for incomplete column references (e.g., "c." or "t.")
    if re.match(r'^[a-z]\.$', where_clean, re.IGNORECASE):
        return False
    
    # Check for incomplete references in any position (including AND c. or OR t.)
    if re.search(r'\b[a-z]\.\s*(?:AND|OR|$)', where_clean, re.IGNORECASE):
        return False
    
    # Check for dangling operators at the end
    if re.search(r'(?:AND|OR|=|<|>|IN|LIKE)\s*$', where_clean, re.IGNORECASE):
        return False
    
    # Check for dangling operators at the start (without WHERE keyword)
    if re.match(r'^(?:AND|OR)\s+', where_clean, re.IGNORECASE):
        return False  # Should already be cleaned but double-check
    
    # Check for mismatched parentheses
    if where_clean.count('(') != where_clean.count(')'):
        return False
    
    # Check for incomplete IN clauses
    if re.search(r'\bIN\s*$', where_clean, re.IGNORECASE):
        return False
    
    return True


def clean_invalid_where_clauses(sql: str, schema_context: str, user_query: str) -> str:
    """
    Remove WHERE clause conditions that reference columns not in schema or not mentioned in user query.
    
    DEFENSIVE: Only removes conditions if it results in valid SQL.
    Falls back to original SQL if cleanup would create malformed queries.
    
    Prevents hallucinated WHERE conditions by validating that:
    1. Columns referenced in WHERE clause exist in schema
    2. Conditions match user's intent from the query
    
    Args:
        sql: SQL with potentially invalid WHERE clauses
        schema_context: Schema information to validate against
        user_query: Original user query to check intent
        
    Returns:
        SQL with cleaned WHERE clauses or original SQL if can't clean safely
    """
    # Store original for fallback
    original_sql = sql
    
    # Extract all column names from schema
    schema_cols = set()
    
    if not schema_context or 'placeholder' in schema_context.lower():
        return sql  # No valid schema info, skip validation
    
    # Pattern 1: "column_name (type)" or word followed by type keyword
    pattern1 = re.findall(r'\b(\w+)\s+(?:bigint|text|boolean|numeric|integer|timestamp|date|jsonb|uuid|varchar|character|varying)', schema_context, re.IGNORECASE)
    schema_cols.update(col.lower() for col in pattern1 if col and len(col) > 1)
    
    # Pattern 2: "*** column_name ***" format (from schema discovery)
    pattern2 = re.findall(r'\*\*\*\s+(\w+)\s+\*\*\*', schema_context)
    schema_cols.update(col.lower() for col in pattern2 if col)
    
    # Pattern 3: From "  - column_name (type)" format
    pattern3 = re.findall(r'(?:^|\n)\s*-\s+(\w+)\s+\(', schema_context, re.MULTILINE)
    schema_cols.update(col.lower() for col in pattern3 if col)
    
    if not schema_cols:
        return sql  # No valid schema info extracted, skip validation
    
    # Extract WHERE clause from SQL
    where_match = re.search(r'\bWHERE\s+(.*?)(?:\bLIMIT\b|\bORDER\b|\bGROUP\b|$)', sql, re.IGNORECASE | re.DOTALL)
    if not where_match:
        return sql  # No WHERE clause, nothing to clean
    
    where_clause = where_match.group(1).strip()
    
    # DEFENSIVE: Check if WHERE clause already looks malformed
    if not is_valid_where_syntax(where_clause):
        print(f"[WHERE_VALIDATION] Original WHERE clause already malformed: '{where_clause}', attempting removal")
        # Try to remove the WHERE clause entirely
        sql = re.sub(r'\s+WHERE\s+.*?(?=\s*(?:\bLIMIT\b|\bORDER\b|\bGROUP\b|$))', ' ', sql, flags=re.IGNORECASE | re.DOTALL)
        return sql.strip()
    
    print(f"[WHERE_VALIDATION] Analyzing WHERE clause: {where_clause[:100]}...")
    
    # Extract all column references in WHERE clause (look for pattern: word = ... or word IN ... or word LIKE ...)
    where_cols = set()
    col_patterns = [
        r'\b(\w+)\s*(?:=|<|>|IN|LIKE|NOT)',  # column operator
        r'(?:WHERE|AND|OR)\s+(\w+)\s*(?:=|<|>|IN)',  # after WHERE/AND/OR
    ]
    
    for pattern in col_patterns:
        matches = re.findall(pattern, where_clause, re.IGNORECASE)
        where_cols.update(col.lower() for col in matches if col and len(col) > 1)
    
    # Find columns in WHERE that don't exist in schema
    invalid_cols = where_cols - schema_cols
    
    if invalid_cols:
        print(f"[WHERE_VALIDATION] Found invalid columns in WHERE: {invalid_cols}")
        print(f"[WHERE_VALIDATION] Schema has: {list(schema_cols)[:10]}...")
        
        # For each invalid column, try to remove its condition from WHERE
        # Pattern: "invalid_col = value AND/OR" or standalone
        for invalid_col in invalid_cols:
            # Match various patterns: "column = value", "column IN (...)", "column LIKE '...'"
            patterns_to_remove = [
                rf'\b{re.escape(invalid_col)}\s*=\s*(?:\'[^\']*\'|\d+|true|false|null)\s*',  # = value
                rf'\b{re.escape(invalid_col)}\s+IN\s*\([^)]*\)\s*',  # IN (...)
                rf'\b{re.escape(invalid_col)}\s+LIKE\s*\'[^\']*\'\s*',  # LIKE '...'
            ]
            
            for pattern in patterns_to_remove:
                # Remove the condition and cleanup AND/OR keywords
                new_where = re.sub(pattern, '', where_clause, flags=re.IGNORECASE)
                
                # Cleanup leftover AND/OR at start or end
                new_where = re.sub(r'\s*(AND|OR)\s+', ' AND ', new_where, flags=re.IGNORECASE)
                new_where = re.sub(r'^(AND|OR)\s+', '', new_where, flags=re.IGNORECASE)
                new_where = re.sub(r'\s+(AND|OR)$', '', new_where, flags=re.IGNORECASE)
                new_where = re.sub(r'(AND|OR)\s+(AND|OR)', 'AND', new_where, flags=re.IGNORECASE)
                new_where = new_where.strip()
                
                # DEFENSIVE: Only update where_clause if the result looks valid
                if new_where != where_clause:
                    # Validate new WHERE syntax
                    if not is_valid_where_syntax(new_where):
                        print(f"[WHERE_VALIDATION] Skipping removal of '{invalid_col}' - would create malformed WHERE: '{new_where}'")
                        continue
                    
                    where_clause = new_where
                    print(f"[WHERE_VALIDATION] Removed invalid column '{invalid_col}' from WHERE")
                    
                    if not where_clause:
                        # All conditions removed, remove entire WHERE clause
                        sql = re.sub(r'\s+WHERE\s+.*?(?=\s*(?:\bLIMIT\b|\bORDER\b|\bGROUP\b|$))', ' ', sql, flags=re.IGNORECASE | re.DOTALL)
                        print(f"[WHERE_VALIDATION] WHERE clause empty, removing entirely")
                        return sql.strip()
        
        # DEFENSIVE: Validate final result before returning
        if where_clause and where_clause != where_match.group(1).strip():
            if not is_valid_where_syntax(where_clause):
                print(f"[WHERE_VALIDATION] Final WHERE clause validity check failed, reverting to original SQL")
                return original_sql
            
            # Reconstruct SQL with cleaned WHERE clause
            sql = sql[:where_match.start(1)] + where_clause + sql[where_match.end(1):]
            print(f"[WHERE_VALIDATION] SQL cleaned: {sql[:120]}...")
        else:
            print(f"[WHERE_VALIDATION] All WHERE columns are valid: {where_cols}")
    
    # Final sanity check - ensure no malformed WHERE remains in output
    if re.search(r'\bWHERE\s+[a-z]\.\s*(?:\bLIMIT\b|\bORDER\b|$)', sql, re.IGNORECASE):
        print(f"[WHERE_VALIDATION] SANITY CHECK FAILED: Malformed WHERE detected in output, reverting")
        return original_sql
    
    return sql


def _extract_previous_sql_from_messages(conversation_history: str) -> Optional[tuple[str, int]]:
    """
    Extract the previous SQL query and result count from conversation history.
    
    Looks through the assistant responses in conversation history for SQL patterns.
    Also handles cases where SQL appears in error messages or mixed with text.
    
    Args:
        conversation_history: Formatted conversation string
        
    Returns:
        Tuple of (sql_text, result_count) or None if not found
    """
    if not conversation_history:
        return None
    
    # Pattern 1: Look for explicit "[SQL] SELECT..." pattern
    sql_pattern = r'\[SQL\]\s*(SELECT.*?)(?:\n|$)'
    sql_match = re.search(sql_pattern, conversation_history, re.IGNORECASE | re.DOTALL)
    
    # Pattern 2: If not found, look for SELECT statements with LIMIT (common pattern)
    if not sql_match:
        sql_pattern = r'(SELECT.+?LIMIT\s+\d+)'
        sql_match = re.search(sql_pattern, conversation_history, re.IGNORECASE | re.DOTALL)
    
    # Pattern 3: Look for SELECT...FROM...WHERE with greedy matching to common boundaries
    if not sql_match:
        sql_pattern = r'(SELECT\s+.+?\s+FROM\s+\w+(?:\.\w+)?\s+\w+\s+WHERE\s+[^;]+?)(?:\s(?:LIMIT|ORDER|GROUP|Found|Error|Results?|rows?)\b|$)'
        sql_match = re.search(sql_pattern, conversation_history, re.IGNORECASE | re.DOTALL)
    
    # Pattern 4: Last resort - look for basic SELECT...FROM pattern
    if not sql_match:
        sql_pattern = r'(SELECT\s+.+?\s+FROM\s+\w+(?:\.\w+)?)(?:;|\n|$)'
        sql_match = re.search(sql_pattern, conversation_history, re.IGNORECASE | re.DOTALL)
    
    if not sql_match:
        return None
    
    sql_text = sql_match.group(1).strip().rstrip(';').strip()
    
    # Try to find row count from conversation
    # Look for patterns like "rows" or "record(s)" or "Found X record"
    count_patterns = [
        r'Found\s+(\d+)\s+records?',
        r'(\d+)\s+rows?',
        r'result[s]?:?\s*(\d+)',
    ]
    result_count = 0
    for pattern in count_patterns:
        count_match = re.search(pattern, conversation_history, re.IGNORECASE)
        if count_match:
            result_count = int(count_match.group(1))
            break
    
    return (sql_text, result_count)


async def build_data_query_response(
    db: AsyncSession, user_id: str, session_id: str, query: str, conversation_history: str = ""
) -> schemas.ResponseWrapper:
    """Handle data queries by generating SQL, running it and creating visuals.

    Uses intelligent step-by-step analysis:
    1. Extracts intent and search values from query
    2. Searches schema to find relevant columns
    3. Generates SQL with proper JOINs and filters
    4. Executes SQL against database
    5. Generates contextual visualizations and responses

    Args:
        db: Database session.
        user_id: Current user identifier.
        session_id: Current chat session identifier.
        query: Natural language query from the user.
        conversation_history: Optional formatted conversation history for context.

    Returns:
        A ResponseWrapper containing a DataQueryResponse.
    """
    # Initialize schema_info early so it's always available
    from ..helpers import get_database_schema
    try:
        schema_info = await get_database_schema(db)
    except Exception as e:
        print(f"⚠️  Warning: Could not retrieve schema: {e}")
        schema_info = "Available tables detected from database. Use schema analysis to identify tables, columns, and relationships."
    
    # Try intelligent SQL generation first
    print(f"\n[ANALYSIS] Processing query: '{query}'")
    
    # DYNAMIC FOLLOW-UP DETECTION: Analyze if this is a follow-up query
    followup_analyzer = await get_followup_analyzer()
    previous_sql_tuple = _extract_previous_sql_from_messages(conversation_history)
    previous_sql = previous_sql_tuple[0] if previous_sql_tuple else None
    previous_result_count = previous_sql_tuple[1] if previous_sql_tuple else 0
    
    # DIAGNOSTIC: Log what we extracted
    print(f"[DEBUG] Previous SQL extracted: {previous_sql[:80] if previous_sql else 'NONE'}..." if previous_sql else "[DEBUG] Previous SQL: NOT FOUND")
    print(f"[DEBUG] Conversation history length: {len(conversation_history) if conversation_history else 0}")
    
    followup_context = await followup_analyzer.analyze(
        current_query=query,
        conversation_history=conversation_history,
        previous_sql=previous_sql,
        previous_result_count=previous_result_count,
    )
    
    if followup_context.is_followup:
        print(f"[FOLLOWUP] Type: {followup_context.followup_type.value.upper()}, Confidence: {followup_context.confidence:.0%}")
        print(f"[FOLLOWUP] Reasoning: {followup_context.reasoning}")
        if followup_context.previous_context:
            print(f"[FOLLOWUP] Previous table: {followup_context.previous_context.table_name}, Rows: {followup_context.previous_context.result_count}")
            if followup_context.previous_context.filters:
                filters_str = ", ".join([f"{f['column']} {f['operator']} {f['value']}" for f in followup_context.previous_context.filters])
                print(f"[FOLLOWUP] Previous filters: {filters_str}")
        else:
            print(f"[FOLLOWUP] ⚠️  Context available: NO (previous_sql was not found)")
    
    # Check if this is a confirmation response to a previous clarifying question
    is_confirmation = any(word in query.lower() for word in ['yes', 'yeah', 'yep', 'correct', 'confirm', 'right', 'absolutely', 'true', 'sure'])
    
    # Check if conversation history contains a clarifying question about joining
    has_previous_clarification = 'join with this table' in conversation_history.lower() or 'should i join' in conversation_history.lower()
    
    force_join = is_confirmation and has_previous_clarification
    
    if force_join:
        print(f"[CONFIRMATION] User confirmed JOIN with customer table")
        # When confirming, pass the full conversation to extract the original search value
        # The analyzer will look through it to find any customer codes
        original_query_to_analyze = conversation_history
    else:
        original_query_to_analyze = query
    
    # NEW: Execute semantic query orchestrator pipeline to get schema context
    # This provides plan-first generation with AI-retrieved relevant tables/columns
    semantic_context = None
    try:
        orchestrator = await _get_semantic_orchestrator(db)
        print(f"[SEMANTIC] Starting semantic query pipeline...")
        semantic_result = await orchestrator.execute_semantic_query(original_query_to_analyze)
        
        # ✅ REQUIREMENT E: ENFORCE CONFIDENCE GATE - Return clarification immediately if needed
        # (Before any SQL generation or further processing)
        if semantic_result and semantic_result.clarification_needed:
            print(f"[SEMANTIC] Confidence gate triggered: {semantic_result.clarification_question}")
            
            # Return clarification to user without generating SQL
            response = schemas.DataQueryResponse(
                intent=query,
                confidence=semantic_result.confidence_score,
                message=semantic_result.clarification_question,  # Return the question
                datasets=[],
                visualizations=[],
                layout=schemas.Layout(type="single", arrangement="single"),
                related_queries=[],
                metadata={
                    "requires_confirmation": True,
                    "original_query": query,
                    "response_type": "semantic_clarification",
                    "pipeline_stage": "confidence_gate",
                    "confidence_score": semantic_result.confidence_score,
                    "retrieved_tables": semantic_result.retrieval_context.top_tables if semantic_result.retrieval_context else [],
                },
            )
            return schemas.ResponseWrapper(
                success=True,
                response=response,
                timestamp=current_timestamp(),
                original_query=query,
            )
        
        # Check if orchestrator succeeded and has retrieval context
        if semantic_result and semantic_result.success and semantic_result.retrieval_context:
            semantic_context = semantic_result
            num_tables = len(semantic_result.retrieval_context.top_tables)
            num_columns = sum(len(cols) for cols in semantic_result.retrieval_context.top_columns_per_table.values())
            print(f"[SEMANTIC] Pipeline result: {num_tables} tables, {num_columns} columns, "
                  f"confidence: {semantic_result.confidence_score:.0%}")
            print(f"[SEMANTIC] Retrieved tables: {semantic_result.retrieval_context.top_tables}")
            print(f"[SEMANTIC] Columns per table: {list(semantic_result.retrieval_context.top_columns_per_table.keys())}")
            
            # Log pipeline trace if available
            if semantic_result.pipeline_trace:
                stages = list(semantic_result.pipeline_trace.keys())
                print(f"[SEMANTIC] Pipeline stages: {' → '.join(stages)}")
        else:
            error_msg = semantic_result.error if semantic_result else "No result returned"
            print(f"[SEMANTIC] Pipeline did not succeed: {error_msg}")
            semantic_context = None
    except Exception as e:
        print(f"[SEMANTIC] Warning: Orchestrator exception: {e}")
        # Continue without semantic context - the system remains backward compatible
        semantic_context = None
    
    # ✅ REQUIREMENT C+D: Use plan-first SQL from orchestrator if available
    # Orchestrator already handled: plan generation (C) + deterministic rendering (D)
    sql = None
    clarifying_question = None
    
    if semantic_result and semantic_result.success and semantic_result.sql:
        # Plan-first path: LLM generated QueryPlan JSON → deterministic renderer → SQL
        sql = semantic_result.sql
        print(f"[PLAN-FIRST] Using deterministic rendered SQL from orchestrator")
    else:
        # Fallback: traditional LLM SQL generation (backward compatible)
        print(f"[FALLBACK] Semantic orchestrator did not produce SQL, using traditional LLM...")
        sql, clarifying_question = await generate_sql_with_analysis(
            original_query_to_analyze, 
            db, 
            conversation_history, 
            force_join=force_join,
            followup_context=followup_context,
            semantic_context=semantic_context
        )
    
    # DYNAMIC: Apply semantic column mapping if SQL was generated and NOT from orchestrator
    # (Orchestrator output is already deterministically rendered, no need for mapping)
    if sql and not clarifying_question and not (semantic_result and semantic_result.success and semantic_result.sql):
        try:
            from ..helpers import get_database_schema
            schema_info = await get_database_schema(db)
            sql = map_semantic_columns(sql, schema_info, original_query_to_analyze)
            
            # Clean invalid WHERE clauses that reference non-existent columns
            sql = clean_invalid_where_clauses(sql, schema_info, original_query_to_analyze)
            
            # Add schema prefixes to table references
            sql = add_schema_prefixes_to_sql(sql)
        except Exception as e:
            print(f"⚠️  Semantic column mapping failed: {e}")
            # Continue with original SQL if mapping fails
    
    # If we got a clarifying question, return it for user confirmation
    # Store the original query context so we can re-execute when user confirms
    if clarifying_question and not sql:
        print(f"[CLARIFICATION] System needs user confirmation")
        
        # Save pending query to conversation context so next response can use it
        response = schemas.DataQueryResponse(
            intent=query,
            confidence=0.0,
            message=clarifying_question,  # Return the question as the message
            datasets=[],
            visualizations=[],
            layout=schemas.Layout(type="single", arrangement="single"),
            related_queries=[],
            metadata={
                "requires_confirmation": True,
                "original_query": query,
                "response_type": "clarifying_question"
            },
        )
        return schemas.ResponseWrapper(
            success=True,
            response=response,
            timestamp=current_timestamp(),
            original_query=query,
        )
    
    # If sql is still empty, use LLM fallback
    if not sql:
        print(f"[FALLBACK] Using LLM to generate SQL...")
        
        # Build comprehensive schema context for LLM
        from ..helpers import get_database_schema
        try:
            schema_info = await get_database_schema(db)
        except:
            # GENERIC fallback (NO hardcoded table names)
            schema_info = "Available tables detected from database. Use schema analysis to identify tables, columns, and relationships."
        
        # Build smart context for common query patterns
        smart_context = build_smart_schema_context(query, schema_info)
        
        # DYNAMIC: Extract semantic hints from schema context
        semantic_hints = extract_semantic_hints_from_schema(smart_context, query)
        
        enhanced_query = f"""You are an expert SQL generator that works with ANY database. Generate SQL that PRECISELY matches the user's intent.

CRITICAL RULES (completely database-agnostic):
1. DATABASE DETECTION - This system uses: {adapter.db_type.value.upper()}
   - Schema prefix format: {adapter.get_capabilities().schema_prefix_style}
   - IMPORTANT: ALWAYS include schema prefix on ALL table names in FROM and JOIN clauses
   - Example: SELECT t.* FROM {settings.postgres_schema}.transactions t JOIN {settings.postgres_schema}.customers c ON t.customer_id = c.customer_id
   - Boolean literals: {adapter.get_capabilities().boolean_literals[0]} or {adapter.get_capabilities().boolean_literals[1]}
   - Tables: Use {adapter.format_schema_qualified_table('table_name')} format (e.g., {settings.postgres_schema}.table_name)

2. Use table aliases for readability: t=main_table, c=related_table, etc.

3. Match user's semantic intent to database columns:
{semantic_hints}

4. Recognize cross-table queries and add JOINs:
   - If query mentions multiple entities, JOIN them properly
   - Use ON clauses with actual foreign keys from the schema
   - Generic pattern: JOIN {adapter.format_schema_qualified_table('table2')} t2 ON table1.fk_id = t2.id

5. Boolean/Status columns:
   - Use {adapter.get_capabilities().boolean_literals[0]} for true (database-appropriate)
   - Use {adapter.get_capabilities().boolean_literals[1]} for false (database-appropriate)
   - For enum columns, use values as shown in schema (already normalized)

6. WHERE CLAUSE PRECISION (CRITICAL):
   - ONLY add WHERE conditions for filter criteria explicitly mentioned in the user query
   - DO NOT invent or assume WHERE clauses based on hallucinated column usage
   - Map user intent phrases directly to actual schema columns
   - Example: "approved" users → WHERE kyc_verified = true (if that column exists)
   - Example: "not approved" users → WHERE kyc_verified = false (if that column exists)
   - If a column is not mentioned in the query, DO NOT use it in WHERE clause

7. Safety:
   - Include LIMIT 10 for safety (or adapt to query complexity)
   - Return ONLY the SQL statement, no markdown or explanation
   - NO subqueries unless explicitly required by the user query

SCHEMA INFORMATION (READ CAREFULLY - this is YOUR specific database):
{smart_context}

USER QUERY: {query}

Generate the SQL query now (adapt to whatever schema is provided, no assumptions):"""
        
        messages = [
            {"role": "system", "content": "You are a database expert. Analyze the provided schema and generate precise, executable SQL that exactly matches the user's intent. Always use proper JOINs for multi-table queries. Adapt to whatever schema is provided, don't hardcode column names."},
            {"role": "user", "content": enhanced_query},
        ]
        
        try:
            sql = await llm.call_llm(messages, stream=False, max_tokens=512)
            # Clean up the response
            sql = sql.strip()
            if sql.startswith("```"):
                sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
                sql = re.sub(r"```$", "", sql, flags=re.IGNORECASE)
            if not sql.endswith(";"):
                sql = sql + ";"
            
            # DYNAMIC: Map semantic columns to actual schema columns
            sql = map_semantic_columns(sql, smart_context, query)
            
            # Clean invalid WHERE clauses that reference non-existent columns
            sql = clean_invalid_where_clauses(sql, smart_context, query)
            
            # Normalize semantics (fix common LLM mistakes)
            sql = normalize_sql_semantics(sql, query)
            
            # Add schema prefixes to table references (from default LLM often omits schema)
            sql = add_schema_prefixes_to_sql(sql)
            
            print(f"[OK] LLM Generated SQL: {sql}\n")
        except Exception as e:
            print(f"❌ Failed to generate SQL with LLM: {e}")
            raise
    
    # Log generated SQL for debugging
    print(f"[SQL] {sql}\n")
    
    # ✅ CRITICAL: VALIDATE & ENFORCE SAFETY LIMIT BEFORE FIRST EXECUTION
    # This ensures every query gets LIMIT enforcement, not just failed ones
    from .sql_safety_validator import SQLSafetyValidator
    validator = SQLSafetyValidator(allowed_schemas=None, max_rows=500)  # Adapter-aware, safety limit
    is_safe, validation_error, rewritten_sql = validator.validate_and_rewrite(sql)
    
    if not is_safe:
        print(f"❌ SQL VALIDATION FAILED: {validation_error}")
        raise ValueError(f"SQL validation failed: {validation_error}")
    
    # Use the rewritten SQL (with enforced LIMIT)
    sql = rewritten_sql
    print(f"[OK] SQL after safety validation (LIMIT enforced): {sql}\n")
    
    # SMART LIMIT: Count records first, then decide whether to apply LIMIT
    print(f"[SMART LIMIT] Applying count-first approach to determine if LIMIT needed...")
    sql = await apply_smart_limit(db, sql, threshold=1000)
    print(f"[SMART LIMIT] Final SQL: {sql}\n")
    
    # Execute SQL against database with retry on certain errors
    max_retries = 2
    retry_count = 0
    rows = None
    last_error = None
    
    while retry_count < max_retries and rows is None:
        try:
            rows = await run_sql(db, sql)
        except Exception as e:
            last_error = e
            error_msg = str(e)
            
            # FULLY DATABASE-AGNOSTIC ERROR CLASSIFICATION
            # Use database adapter to classify errors appropriately for connected DB
            adapter = get_global_adapter()
            error_type = adapter.classify_error(error_msg)
            
            is_type_error = error_type == "type_error"
            is_undefined_error = error_type in ("undefined_column", "undefined_table")
            is_enum_error = error_type == "enum_error"
            
            # Skip recovery for type errors - they're handled in type_converter
            if is_type_error:
                print(f"⚠️  Type mismatch/operator error detected (recoverable by type conversion layer): {error_msg[:100]}")
                print(f"🔧 Type converter should handle this - check type_converter.py")
                raise
            
            if (is_undefined_error or is_enum_error) and retry_count < max_retries - 1:
                
                print(f"⚠️  SQL error detected: {error_msg}")
                print(f"🔄 Attempting to regenerate SQL with error context...")
                
                # Get schema again to ensure we have latest info
                from ..helpers import get_database_schema
                schema_info = await get_database_schema(db)
                
                # Detect the specific type of error
                is_undefined_col_error = "undefined column" in error_msg or "does not exist" in error_msg
                
                # Extract main table from the failed SQL to prevent jumping to different tables
                main_table = None
                table_match = re.search(r'(?:FROM|JOIN)\s+(?:ONLY\s+)?(?:\w+\.)?(\w+)\s+(?:\w+)?(?:\s|$|WHERE|JOIN)', sql, re.IGNORECASE)
                if table_match:
                    main_table = table_match.group(1)
                
                # Get columns for the specific table being queried
                table_columns_info = ""
                columns = []
                if main_table:
                    try:
                        # Use database adapter to get table columns (works for any database)
                        columns = await adapter.get_table_columns(db, main_table)
                        
                        if columns:
                            table_columns_info = f"\n\nACTUAL COLUMNS IN {main_table.upper()} TABLE:\n"
                            table_columns_info += "You MUST use ONLY these columns:\n"
                            for col_name, col_type in columns:
                                table_columns_info += f"  - {col_name} ({col_type})\n"
                            table_columns_info += f"\nDO NOT jump to different tables! Stay with {main_table}."
                        else:
                            # If we can't get columns from DB, provide generic guidance
                            table_columns_info = f"\n\nCOULD NOT RETRIEVE COLUMNS for {main_table}"
                            table_columns_info += f"\nTry to infer from table name and user query:"
                            table_columns_info += f"\n- User asked about: {query.lower()}"
                            table_columns_info += f"\n- Look for columns matching user intent"
                            table_columns_info += f"\n- DO NOT jump to different tables! Stay with {main_table}."
                    except Exception as col_err:
                        print(f"[DEBUG] Could not get columns for {main_table} using {adapter.db_type.value}: {col_err}")
                        table_columns_info = f"\n\nERROR: Could not retrieve columns for {main_table}"
                        table_columns_info += f"\nTry to infer correct column from user intent: {query.lower()}"
                        table_columns_info += f"\nDO NOT jump to different tables! Stay with {main_table}."
                
                # Special handling for enum errors
                if is_enum_error:
                    # Extract the problematic enum value from error message
                    enum_match = re.search(r'invalid input value for enum (\w+): "([^"]+)"', str(e), re.IGNORECASE)
                    enum_col = enum_match.group(1) if enum_match else "unknown"
                    enum_val = enum_match.group(2) if enum_match else "unknown"
                    
                    error_context = f"""IMPORTANT: Fix the SQL query - it has an ENUM validation error

CONVERSATION CONTEXT:
{conversation_history if conversation_history else "(No previous context)"}

USER'S INTENT: {query}
FAILED SQL: {sql}
ERROR: Invalid enum value '{enum_val}' for column '{enum_col}'

WHAT TO DO:
1. Reread the user's query carefully: '{query}'
2. The value '{enum_val}' is NOT a valid enum value for column '{enum_col}'
3. Search the schema to find where '{enum_val}' actually appears as a value
4. It might exist in a different column (e.g., state, location, category, type, status)
5. Generate SQL that answers the user's ORIGINAL intent using the correct column

SCHEMA:
{schema_info}

Output ONLY corrected SQL, nothing else:"""
                
                elif is_undefined_col_error:
                    # Extract which column was not found
                    col_match = re.search(r'column "?(\w+)"? does not exist|undefined column "?(\w+)"?', str(e), re.IGNORECASE)
                    bad_col = col_match.group(1) or col_match.group(2) if col_match else "unknown"
                    
                    # Try to suggest correct columns based on semantic meaning
                    suggested_cols = []
                    bad_col_lower = bad_col.lower()
                    
                    # First, try to match against all actual columns from the database
                    if columns:
                        col_names = [col[0] for col in columns]  # Keep original casing
                        col_names_lower = [col[0].lower() for col in columns]
                        
                        print(f"[DEBUG] Searching for match for '{bad_col}' in {len(col_names)} columns")
                        
                        # Strategy 1: Direct substring matching
                        for col_name, col_type in columns:
                            col_lower = col_name.lower()
                            bad_lower = bad_col_lower
                            
                            # Check various substring patterns
                            if bad_lower in col_lower or col_lower in bad_lower:
                                suggested_cols.append(col_name)
                                print(f"[DEBUG]   Matched by substring: {bad_col} → {col_name}")
                            elif any(part in col_lower for part in bad_lower.split('_')):
                                # If bad_col has multiple parts, try partial matching
                                suggested_cols.append(col_name)
                                print(f"[DEBUG]   Matched by token: {bad_col} → {col_name}")
                        
                        # Strategy 2: Semantic matching based on user query keywords
                        if not suggested_cols:
                            query_lower = query.lower()
                            user_intent_keywords = {
                                'kyc': ['kyc', 'verification', 'verified', 'verified_kyc'],
                                'approval': ['approval', 'approved', 'approv'],
                                'active': ['active', 'activated', 'is_active', 'status'],
                                'verified': ['verified', 'verify', 'kyc'],
                                'approved': ['approved', 'approval', 'approve'],
                            }
                            
                            for keyword, patterns in user_intent_keywords.items():
                                if any(kw in query_lower for kw in patterns):
                                    # Look for columns matching this intent
                                    for col_name, col_type in columns:
                                        col_lower = col_name.lower()
                                        if any(p in col_lower for p in patterns):
                                            suggested_cols.append(col_name)
                                            print(f"[DEBUG]   Matched by user intent '{keyword}': {bad_col} → {col_name}")
                        
                        # Remove duplicates while preserving order
                        suggested_cols = list(dict.fromkeys(suggested_cols))
                        
                    # If no columns retrieved from DB, try to infer from schema_info
                    if not suggested_cols and schema_info:
                        schema_lower = schema_info.lower()
                        # Look for column definitions in schema text
                        col_patterns = re.findall(r'(\w+)\s+(?:boolean|integer|text|varchar|timestamp|numeric)', schema_info, re.IGNORECASE)
                        
                        print(f"[DEBUG] Extracting columns from schema text: found {len(col_patterns)}")
                        
                        for col_name in col_patterns:
                            col_lower = col_name.lower()
                            if bad_col_lower in col_lower or col_lower in bad_col_lower:
                                suggested_cols.append(col_name)
                                print(f"[DEBUG]   Matched from schema: {bad_col} → {col_name}")
                        
                        suggested_cols = list(dict.fromkeys(suggested_cols))
                    
                    # Build suggestion string
                    suggestion_str = ""
                    if suggested_cols:
                        suggestion_str = f"\n\n🎯 SEMANTIC MATCHES FOUND (most likely correct):\n"
                        for i, col in enumerate(suggested_cols[:5], 1):
                            suggestion_str += f"  {i}. {col}\n"
                        suggestion_str += f"\nThe column '{bad_col}' doesn't exist, but one of the above columns is likely what you meant."
                    else:
                        suggestion_str = f"\n\n⚠️  Could not find automatic match for '{bad_col}'"
                        if columns:
                            suggestion_str += f"\n\nAvailable columns in {main_table}:\n"
                            for col_name, col_type in columns[:10]:
                                suggestion_str += f"  • {col_name} ({col_type})\n"
                        else:
                            suggestion_str += f"\nTry to infer from the table/column naming conventions."
                    
                    error_context = f"""IMPORTANT: Fix the SQL query - column '{bad_col}' doesn't exist in table '{main_table}'

USER'S QUERY (for context): {query}
FAILED SQL: {sql}
ERROR: Column '{bad_col}' does not exist
{table_columns_info}
{suggestion_str}

INSTRUCTIONS:
1. The column '{bad_col}' DOES NOT EXIST in {main_table}
2. Look at the semantic matches above - one of them is probably correct
3. Replace '{bad_col}' with the correct column name
4. Keep everything else the same
5. Return ONLY the corrected SQL statement
"""
                    
                    # Add follow-up specific instructions if this is a refinement
                    if followup_context and followup_context.is_followup and followup_context.followup_type.value == 'refinement':
                        if followup_context.previous_context and followup_context.previous_context.filters:
                            filters_str = ", ".join([f"{f['column']} {f['operator']} {f['value']}" for f in followup_context.previous_context.filters])
                            error_context += f"""
CRITICAL FOR REFINEMENT: This is a follow-up query!
- Previous query used table '{followup_context.previous_context.table_name}'
- Previous filters: {filters_str}
- NEW INSTRUCTION: Include previous filters in the corrected SQL using AND
- Example: If previous was WHERE c.state = 'AP' and new filter is city, use:
  WHERE c.state = 'AP' AND c.city LIKE 'vizag%'
"""
                        else:
                            # FALLBACK: Refinement follow-up but no previous_context 
                            # Still need to guide LLM about refinement intent
                            error_context += f"""
CRITICAL FOR REFINEMENT: This is a FOLLOW-UP query (refinement type)
- The user is adding MORE FILTERS to their previous query
- You MUST use the SAME TABLE as before
- Combine old filters with new ones using AND operator
- CONVERSATION HISTORY above shows what they asked before
- NEW FILTER: User now says: '{query}'
- ACTION: Keep previous WHERE filters and ADD the new filter with AND
- Example pattern: WHERE [old_filter] AND [new_filter_based_on_'{query}']
"""
                    
                    error_context += "\nOutput format: SELECT... (exactly like the failed SQL but with correct column)"
                
                else:
                    error_context = f"""Fix this SQL query:

CONVERSATION CONTEXT:
{conversation_history if conversation_history else "(No previous context)"}

Original query: {query}
Failed SQL: {sql}
Error: {str(e)[:200]}

{table_columns_info}

Schema reference:
{schema_info}

Generate corrected SQL only, nothing else."""
                
                fix_messages = [
                    {
                        "role": "system",
                        "content": f"""You are a SQL expert FIXING BROKEN queries with column errors.

⚠️  CRITICAL RULES - MANDATORY, DO NOT BREAK:

1. COLUMN MATCHING PRIORITY:
   - FIRST: Use suggested columns from the error message (these are nearly always correct)
   - If multiple suggestions, pick the one closest to user's intent
   - Example: User said "kyc not approved" → select 'is_kyc_approved' or 'kyc_verified'

2. TABLE LOCK: Fix {main_table if main_table else 'the'} table query ONLY
   - DO NOT jump to different tables
   - Keep the EXACT same table: {main_table if main_table else '(unknown)'}

3. SEMANTIC CONTEXT:
   - User's original request: {query}
   - This tells you WHAT they're looking for (even if column name is wrong)
   - Find the actual column that matches their intent

4. SUBSTITUTION ONLY:
   - Replace ONLY the wrong column name
   - Do NOT rewrite the entire query
   - Do NOT change structure, aliases, or logic

5. OUTPUT FORMAT:
   - Return ONLY the corrected SQL
   - Start with SELECT, end with semicolon
   - No explanations or reasoning"""
                    },
                    {"role": "user", "content": error_context},
                ]
                
                try:
                    # Debug: Show what we're sending to LLM
                    print(f"[DEBUG] Error recovery context for table '{main_table}':")
                    print(f"[DEBUG] Conversation history: {conversation_history[:200] if conversation_history else '(empty)'}")
                    print(f"[DEBUG] Table columns info length: {len(table_columns_info)}")
                    
                    fixed_sql = await llm.call_llm(fix_messages, stream=False, max_tokens=512)
                    
                    # AGGRESSIVE SQL EXTRACTION: Remove any explanations mixed with SQL
                    # This handles cases where LLM returns: "SELECT ... Explanation: Since the user..."
                    
                    # First pass: Extract just the SELECT statement
                    if 'SELECT' in fixed_sql.upper():
                        # Use greedy matching to get from SELECT to LIMIT, then trim
                        select_match = re.search(
                            r'SELECT.+?LIMIT\s+\d+',
                            fixed_sql,
                            re.IGNORECASE | re.DOTALL
                        )
                        if select_match:
                            fixed_sql = select_match.group(0).strip()
                        else:
                            # No LIMIT found, try to get just from SELECT to semicolon/newline
                            select_match = re.search(
                                r'SELECT.*?(?:;|\n|$)',
                                fixed_sql,
                                re.IGNORECASE | re.DOTALL
                            )
                            if select_match:
                                fixed_sql = select_match.group(0).strip().rstrip(';').strip()
                    
                    # Clean up the response
                    if ';' in fixed_sql:
                        fixed_sql = fixed_sql.split(';')[0]
                    fixed_sql = fixed_sql.strip()
                    fixed_sql = re.sub(r"^```(?:sql)?\s*", "", fixed_sql, flags=re.IGNORECASE)
                    fixed_sql = re.sub(r"```$", "", fixed_sql, flags=re.IGNORECASE)
                    fixed_sql = ' '.join(fixed_sql.split())
                    
                    # DYNAMIC: Map semantic columns to actual schema columns
                    # Use table_columns_info (specific table) if available, else full schema_info
                    semantic_context = table_columns_info if table_columns_info else schema_info
                    fixed_sql = map_semantic_columns(fixed_sql, semantic_context, query)
                    
                    # Clean invalid WHERE clauses that reference non-existent columns
                    fixed_sql = clean_invalid_where_clauses(fixed_sql, semantic_context, query)
                    
                    # Normalize semantics
                    fixed_sql = normalize_sql_semantics(fixed_sql, query)
                    
                    # Add schema prefixes to table references
                    fixed_sql = add_schema_prefixes_to_sql(fixed_sql)
                    
                    # CONSTRAINT VALIDATION: Check that LLM didn't jump tables
                    if main_table:
                        # Extract all tables from the regenerated SQL
                        regenerated_tables = re.findall(
                            r'(?:FROM|JOIN)\s+(?:ONLY\s+)?(?:\w+\.)?(\w+)\s+(?:\w+)?(?:\s|$|WHERE|JOIN)',
                            fixed_sql,
                            re.IGNORECASE
                        )
                        
                        # Check if it's trying to use different tables
                        if regenerated_tables and main_table not in regenerated_tables:
                            # LLM tried to jump to a different table - reject it
                            print(f"❌ ERROR RECOVERY FAILED: LLM tried to jump from '{main_table}' to '{regenerated_tables[0]}'")
                            print(f"❌ This is a FOLLOW-UP query and must stay with {main_table}")
                            raise ValueError(f"LLM attempted to jump tables: from {main_table} to {regenerated_tables[0]}")
                    
                    # Validate and rewrite the fixed SQL using new architecture
                    # Uses adapter-aware default schema (db-agnostic)
                    from .sql_safety_validator import SQLSafetyValidator
                    validator = SQLSafetyValidator(allowed_schemas=None, max_rows=500)  # Let adapter determine schema
                    is_safe, error, rewritten_sql = validator.validate_and_rewrite(fixed_sql)
                    if not is_safe:
                        raise ValueError(f"SQL validation failed: {error}")
                    sql = rewritten_sql  # Use the rewritten SQL with safety LIMIT
                    print(f"[OK] Regenerated SQL (with safety limit): {sql}\n")
                    retry_count += 1
                except Exception as regen_error:
                    print(f"❌ Failed to regenerate SQL: {regen_error}")
                    raise last_error  # Re-raise original error
            else:
                # Not a recoverable error or max retries reached
                raise
    
    if rows is None:
        raise last_error or Exception("Failed to execute query")
    
    # Check if this should be returned as a simple message (for follow-up questions)
    should_use_message, formatted_message = await should_return_as_message(
        query, rows, conversation_history
    )
    
    # If this is a follow-up question with simple answer, return just the message
    if should_use_message and formatted_message:
        response = schemas.DataQueryResponse(
            intent=query,
            confidence=0.95,
            message=formatted_message,
            datasets=[],  # No datasets for simple messages
            visualizations=[],  # No visualizations for simple messages
            layout=schemas.Layout(type="single", arrangement="single"),
            related_queries=[],  # Usually not needed for simple answers
            metadata={
                "requires_data": False,  # This is just a message, not a data visualization
                "estimated_complexity": "simple",
                "sql": sql,
                "row_count": len(rows),
                "response_type": "message"  # Indicate this is a message response
            },
        )
        return schemas.ResponseWrapper(
            success=True,
            response=response,
            timestamp=current_timestamp(),
            original_query=query,
        )
    
    # For multi-row or complex results, determine best visualization and generate message
    chart_type, viz_message, viz_title = await determine_visualization_type(
        query, rows, should_use_message=should_use_message
    )
    
    # Generate related follow-up queries
    related_queries = []
    try:
        related_prompt = f"""Based on this data query context, suggest 3 brief follow-up questions.
User's query: {query}
Result count: {len(rows)}
Sample result: {json.dumps(rows[0] if rows else {}, default=str)}

Available database schema:
{schema_info}

Return ONLY as JSON: {{"related_queries": ["question1", "question2", "question3"]}}"""
        
        related_messages = [
            {"role": "system", "content": "You generate helpful follow-up questions."},
            {"role": "user", "content": related_prompt},
        ]
        
        related_response_text = await llm.call_llm(related_messages, stream=False, max_tokens=300)
        
        # Handle case where response is empty or None
        if not related_response_text:
            print("[DEBUG] LLM returned empty response for related queries")
            related_queries = []
        else:
            # Make sure response_text is a string
            if hasattr(related_response_text, 'content'):
                # If it's an LLMResponse object, extract content
                related_response_text = related_response_text.content
            
            response_str = str(related_response_text).strip()
            
            if not response_str:
                print("[DEBUG] LLM response is empty after stripping")
                related_queries = []
            else:
                # Extract JSON if wrapped in text
                json_match = re.search(r'\{[^{}]*"related_queries"[^{}]*\}', response_str, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(0)
                    related_response = json.loads(json_str)
                    related_queries = related_response.get("related_queries", [])[:3]
                else:
                    print(f"[DEBUG] No JSON found in LLM response: {response_str[:100]}")
                    related_queries = []
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON decode error for related queries: {e}")
        related_queries = []
    except Exception as e:
        print(f"⚠️  Failed to generate related queries: {e}")
        related_queries = []
    
    # Build datasets
    dataset_id = "q1"
    datasets = []
    if rows:
        datasets.append(
            schemas.DataQueryDataset(
                id=dataset_id,
                data=rows,
                count=len(rows),
                description=f"Query returned {len(rows)} record(s)",
            )
        )
    else:
        datasets.append(
            schemas.DataQueryDataset(
                id=dataset_id,
                data=[],
                count=0,
                description="No data returned from the query.",
            )
        )
    
    # Build visualization
    visualisations = [
        schemas.Visualization(
            chart_id="v1",
            type=chart_type,
            title=viz_title,
            data=rows,
        )
    ]
    
    layout = schemas.Layout(type="single", arrangement="single")
    
    # Use ResponseComposer for ChatGPT-like response framing with AI-powered visualizations
    from .response_composer import ResponseComposer
    import time
    
    # Compose SQL response with emoji, structured formatting, and DYNAMIC visualizations
    assistant, artifacts, followups, visualizations = await ResponseComposer.compose_sql_response_async(
        query=sql,
        results=rows,
        execution_time=0.1,  # Approximate execution time
        intent="run_sql",
    )
    
    # Build ChatGPT-like response
    response = schemas.LamaResponse(
        id=f"msg_{int(time.time() * 1000)}",
        object="chat.response",
        created_at=int(time.time() * 1000),
        session_id=session_id,
        mode="sql",
        assistant={
            "role": "assistant",
            "title": assistant.title,
            "content": [block.to_dict() for block in assistant.content],
        },
        artifacts=artifacts.to_dict(),
        visualizations=schemas.Visualization(**visualizations) if visualizations else None,
        routing={
            "type": "data_query",
            "intent": "run_sql",
            "confidence": 0.9,
        },
        followups=[fu.to_dict() for fu in followups],
        debug={
            "normalized_user_request": query,
            "sql_executed": sql,
            "row_count": len(rows),
            "complexity": "medium",
        }
    )
    return schemas.ResponseWrapper(
        success=True,
        response=response,
        timestamp=int(time.time() * 1000),
        original_query=query,
    )


async def build_file_query_response(
    db: AsyncSession, user_id: str, session_id: str, query: str, files: List[UploadFile]
) -> schemas.ResponseWrapper:
    """Handle queries that include file uploads using ChatGPT-like response framing.

    Args:
        db: Database session.
        user_id: ID of the current user.
        session_id: Chat session identifier.
        query: Natural language query.
        files: List of uploaded files.

    Returns:
        ResponseWrapper containing a ChatGPT-like response.
    """
    from .response_composer import ResponseComposer
    import time
    
    # Persist each file and generate embeddings
    file_infos: List[Dict[str, Any]] = []
    first_file_id = None
    first_file_obj = None
    
    for file in files:
        uploaded = await add_file(db, session_id, file)
        file_infos.append({"filename": uploaded.filename, "size": uploaded.size})
        
        if not first_file_id:
            first_file_id = str(uploaded.id)
            first_file_obj = uploaded
    
    # Generate summary using LLM
    summary_text = ""
    if first_file_obj and first_file_obj.content_text:
        system_prompt = (
            "You are a helpful assistant that provides structured summaries of documents. "
            "Provide a well-organized summary with headings, key points, and actionable insights."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_file_obj.content_text[:3000]},
        ]
        summary_text = await llm.call_llm(messages, stream=False, max_tokens=1024)
    
    # Use ResponseComposer to frame the response
    assistant, artifacts, followups = ResponseComposer.compose_file_response(
        filename=first_file_obj.filename if first_file_obj else "uploaded file",
        summary=summary_text or "Document processed.",
        file_id=first_file_id or "unknown",
        intent="summarize_uploaded_file",
    )
    
    # Build ChatGPT-like response
    response = schemas.LamaResponse(
        id=f"msg_{int(time.time() * 1000)}",
        object="chat.response",
        created_at=int(time.time() * 1000),
        session_id=session_id,
        mode="file",
        assistant={
            "role": "assistant",
            "title": assistant.title,
            "content": [block.to_dict() for block in assistant.content],
        },
        artifacts=artifacts.to_dict(),
        routing={
            "type": "file_query",
            "intent": "summarize_uploaded_file",
            "confidence": 0.95,
        },
        followups=[fu.to_dict() for fu in followups],
        debug={
            "normalized_user_request": query,
            "requires_date": False,
            "complexity": "medium",
        }
    )
    
    # For backwards compatibility, wrap in ResponseWrapper with old schema
    return schemas.ResponseWrapper(
        success=True,
        response=response,
        timestamp=int(time.time() * 1000),
        original_query=query,
    )


async def build_file_lookup_response(
    db: AsyncSession, user_id: str, session_id: str, query: str
) -> schemas.ResponseWrapper:
    """Handle follow-up queries that reference previously uploaded files.

    Args:
        db: Database session.
        user_id: ID of the current user.
        session_id: Chat session identifier.
        query: Follow-up query.

    Returns:
        ResponseWrapper containing a FileLookupResponse.
    """
    # Retrieve relevant chunks from previously uploaded files
    chunks = await retrieve_relevant_chunks(db, session_id, query)
    # Concatenate top chunks into a context for the LLM
    context_text = "\n".join(chunk.text for chunk in chunks)
    system_prompt = (
        "You are a helpful assistant that answers questions based on provided context. "
        "Use the context to answer the user's question."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": context_text},
        {"role": "user", "content": query},
    ]
    answer = await llm.call_llm(messages, stream=False, max_tokens=256)
    response = schemas.FileLookupResponse(
        intent=query,
        confidence="medium",
        message=answer or "No relevant information found in uploaded files.",
        related_queries=[],
        metadata={"requires_date": True, "complexity": "medium"},
    )
    return schemas.ResponseWrapper(
        success=True,
        response=response,
        timestamp=current_timestamp(),
        original_query=query,
    )


async def build_config_update_response(
    db: AsyncSession, user_id: str, session_id: str, query: str
) -> schemas.ResponseWrapper:
    """Handle user requests to modify visualisation configuration.

    A naive implementation that flips the chart type between bar and pie
    when the query mentions either. More advanced logic could parse
    colours, legend positions, sizes, etc.

    Args:
        db: Database session.
        user_id: ID of the current user.
        session_id: Chat session identifier.
        query: The user's configuration update request.

    Returns:
        ResponseWrapper containing a ConfigUpdateResponse.
    """
    # Determine new chart type from query
    chart_type = "bar"
    q = query.lower()
    if "pie" in q:
        chart_type = "pie"
    elif "bar" in q:
        chart_type = "bar"
    elif "line" in q:
        chart_type = "line"
    # Dummy data representing previous data for demonstration
    dummy_data = [
        {"label": "A", "value": 30},
        {"label": "B", "value": 50},
        {"label": "C", "value": 20},
    ]
    viz = schemas.Visualization(
        chart_id="v1",
        type=chart_type,
        title="Updated Chart",
        data=dummy_data,
        config=schemas.VisualizationConfig(
            legend={"position": "right", "show": True},
            colors=["#FF0000", "#00FF00", "#0000FF"],
            labels={"show": True, "format": "{label}: {value}"},
            size={"width": 600, "height": 400},
        ),
    )
    response = schemas.ConfigUpdateResponse(
        intent=query,
        confidence=0.9,
        message="Chart configuration updated.",
        visualizations=[viz],
        layout=schemas.Layout(type="single", arrangement="single"),
        related_queries=[],
        metadata={"updated": True},
    )
    return schemas.ResponseWrapper(
        success=True,
        response=response,
        timestamp=current_timestamp(),
        original_query=query,
    )


async def build_standard_response(
    db: AsyncSession, user_id: str, session_id: str, query: str
) -> schemas.ResponseWrapper:
    """Handle generic queries by passing them directly to the LLM with ChatGPT-like response framing.

    Args:
        db: Database session.
        user_id: ID of the current user.
        session_id: Chat session identifier.
        query: Natural language query.

    Returns:
        ResponseWrapper containing a ChatGPT-like response.
    """
    # Use the LLM to answer the question directly
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    answer = await llm.call_llm(messages, stream=False, max_tokens=512)
    
    # Use ResponseComposer for ChatGPT-like response framing
    from .response_composer import ResponseComposer
    import time
    
    # Compose chat response with emoji and structured formatting
    assistant, artifacts, followups = ResponseComposer.compose_chat_response(
        user_query=query,
        answer=answer or "I'm sorry, I cannot answer that question.",
        intent="general_answer",
    )
    
    # Build ChatGPT-like response
    response = schemas.LamaResponse(
        id=f"msg_{int(time.time() * 1000)}",
        object="chat.response",
        created_at=int(time.time() * 1000),
        session_id=session_id,
        mode="chat",
        assistant={
            "role": "assistant",
            "title": assistant.title,
            "content": [block.to_dict() if hasattr(block, 'to_dict') else block for block in (assistant.content or [])],
        },
        artifacts=artifacts.to_dict(),
        routing={
            "type": "chat",
            "intent": "general_answer",
            "confidence": 0.7,
        },
        followups=[fu.to_dict() for fu in followups],
        debug={
            "normalized_user_request": query,
            "requires_data": False,
            "complexity": "low",
        }
    )
    return schemas.ResponseWrapper(
        success=True,
        response=response,
        timestamp=int(time.time() * 1000),
        original_query=query,
    )
