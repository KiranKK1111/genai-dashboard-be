"""Database schema discovery and caching utilities."""

from __future__ import annotations

from typing import Dict, List
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings


async def get_business_table_names(session: AsyncSession) -> List[str]:
    """Return the list of business-visible table names (excludes internal tables).

    Lightweight — only fetches table names, no column or sample-value queries.
    Used to pass the table list to the LLM concept extractor so it maps
    user synonyms (e.g. 'clients') to real table names (e.g. 'customers').
    """
    target_schema = settings.postgres_schema
    result = await session.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = :schema ORDER BY table_name"
        ),
        {"schema": target_schema},
    )
    all_tables = [row[0] for row in result.fetchall()]

    if settings.discover_all_tables:
        return all_tables
    if settings.allowed_tables:
        allowed = set(settings.allowed_tables)
        return [t for t in all_tables if t in allowed]
    # Default: exclude internal tables
    internal = set(settings.internal_tables or [])
    return [t for t in all_tables if t not in internal]


async def get_business_table_schemas(session: AsyncSession) -> Dict[str, List[str]]:
    """Return {table_name: [column_names]} for business tables.

    Lightweight — only fetches column names (no types, no sample values).
    Used to give the concept extractor actual column names so it maps
    user concepts (e.g. 'city') to real columns instead of guessing.
    """
    tables = await get_business_table_names(session)
    if not tables:
        return {}
    target_schema = settings.postgres_schema
    result = await session.execute(
        text(
            "SELECT table_name, column_name "
            "FROM information_schema.columns "
            "WHERE table_schema = :schema AND table_name = ANY(:tables) "
            "ORDER BY table_name, ordinal_position"
        ),
        {"schema": target_schema, "tables": tables},
    )
    schemas: Dict[str, List[str]] = {}
    for row in result.fetchall():
        schemas.setdefault(row[0], []).append(row[1])
    return schemas


async def get_database_schema(session: AsyncSession) -> str:
    """Fetch database schema dynamically from PostgreSQL.
    
    Retrieves all tables and their columns from the configured schema
    including sample values, primary keys, and foreign keys to help LLM understand data relationships.
    
    Args:
        session: Async database session.
        
    Returns:
        Formatted schema description for LLM context.
    """
    try:
        # Get the target schema from settings
        target_schema = settings.postgres_schema
        
        # Get all tables in the configured schema
        tables_query = text(f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = '{target_schema}'
            ORDER BY table_name;
        """)
        
        tables_result = await session.execute(tables_query)
        all_tables = [row[0] for row in tables_result.fetchall()]
        
        # Apply table filtering based on configuration
        if settings.discover_all_tables:
            # Use ALL tables in the schema
            tables = all_tables
            print(f"📊 Discovering ALL tables in schema '{target_schema}': {tables}")
        elif settings.internal_tables:
            # Filter out internal application tables
            internal_set = set(settings.internal_tables)
            tables = [t for t in all_tables if t not in internal_set]
            print(f"📊 Found {len(all_tables)} tables, filtered to {len(tables)} business tables")
        elif settings.allowed_tables:
            # Use only explicitly allowed tables
            allowed_set = set(settings.allowed_tables)
            tables = [t for t in all_tables if t in allowed_set]
            print(f"📊 Using {len(tables)} allowed tables from configuration")
        else:
            # Default: filter out internal tables
            internal_set = set(settings.internal_tables)
            tables = [t for t in all_tables if t not in internal_set]
            print(f"📊 Found {len(all_tables)} tables, filtered to {len(tables)} business tables")
        
        if not tables:
            return f"No tables found in schema '{target_schema}'. Configure DISCOVER_ALL_TABLES=true to include all tables, or set ALLOWED_TABLES to specify which tables to use."
        
        schema_description = f"Database Schema ('{target_schema}' schema):\n\n"
        
        # For each table, get its columns and their types with sample values and key info
        for table_name in tables:
            # Get column information with constraint details
            columns_query = text(f"""
                SELECT 
                    c.column_name, 
                    c.data_type, 
                    c.is_nullable,
                    CASE WHEN pk.column_name IS NOT NULL THEN 'PRIMARY KEY' ELSE '' END as constraint_type
                FROM information_schema.columns c
                LEFT JOIN information_schema.table_constraints tc ON c.table_name = tc.table_name AND tc.constraint_type = 'PRIMARY KEY'
                LEFT JOIN information_schema.key_column_usage kcu ON c.column_name = kcu.column_name AND c.table_name = kcu.table_name AND kcu.constraint_name = tc.constraint_name
                LEFT JOIN information_schema.columns pk ON kcu.column_name = pk.column_name
                WHERE c.table_schema = '{target_schema}' AND c.table_name = '{table_name}'
                ORDER BY c.ordinal_position;
            """)
            
            columns_result = await session.execute(columns_query)
            columns = columns_result.fetchall()
            
            schema_description += f"Table: {table_name}\n"
            schema_description += "Columns (ACTUAL DATABASE COLUMN NAMES):\n"
            for col_name, col_type, is_nullable, constraint in columns:
                nullable = "NOT NULL" if is_nullable == "NO" else "NULL"
                constraint_str = f"[{constraint}]" if constraint and constraint.strip() else ""
                # Highlight actual column name in bold-like format for LLM readability
                schema_description += f"  *** {col_name} *** ({col_type}) {nullable} {constraint_str}\n"
                
                # Get sample values for key columns
                try:
                    if col_type in ['character varying', 'text', 'boolean', 'integer', 'bigint', 'numeric']:
                        sample_query = text(f"""
                            SELECT DISTINCT {col_name} 
                            FROM {target_schema}.{table_name}
                            WHERE {col_name} IS NOT NULL
                            LIMIT 5;
                        """)
                        sample_result = await session.execute(sample_query)
                        samples = [str(row[0]) for row in sample_result.fetchall()]
                        if samples:
                            # Infer column purpose dynamically from name patterns
                            # NO HARDCODED DOMAIN KEYWORDS - use generic patterns
                            col_lower = col_name.lower()
                            purpose = ""
                            
                            # Generic pattern-based purpose inference
                            if col_lower.endswith('_id') or col_lower == 'id':
                                purpose = "Identifier/foreign key"
                            elif any(p in col_lower for p in ['amount', 'value', 'price', 'cost', 'total', 'sum']):
                                purpose = "Amount/monetary value"
                            elif any(p in col_lower for p in ['time', 'date', 'created', 'updated', 'timestamp', '_at']):
                                purpose = "Date/time information"
                            elif any(p in col_lower for p in ['status', 'state', 'type', 'category', 'kind']):
                                purpose = "Status/classification"
                            elif any(p in col_lower for p in ['name', 'title', 'description', 'label']):
                                purpose = "Name/description"
                            elif any(p in col_lower for p in ['is_', 'has_', 'can_', 'flag', 'active', 'enabled']):
                                purpose = "Boolean flag"
                            elif any(p in col_lower for p in ['email', 'phone', 'address', 'url', 'link']):
                                purpose = "Contact/reference information"
                            
                            sample_str = ', '.join(samples[:3])
                            if purpose:
                                schema_description += f"    Purpose: {purpose}\n"
                            schema_description += f"    Sample values: {sample_str}\n"
                except Exception:
                    pass  # Skip if we can't get samples
            
            # Get a complete sample row to show structure
            try:
                sample_row_query = text(f"""
                    SELECT * FROM {target_schema}.{table_name} LIMIT 1;
                """)
                sample_row_result = await session.execute(sample_row_query)
                sample_row = sample_row_result.first()
                if sample_row:
                    schema_description += f"SAMPLE DATA ROW:\n"
                    col_names = [col[0] for col in columns]
                    for col_name, value in zip(col_names, sample_row):
                        if value is not None:
                            # Show actual column name with actual value from database
                            schema_description += f"  {col_name}: {value}\n"
            except Exception:
                pass
            
            # Get foreign key relationships
            try:
                fk_query = text(f"""
                    SELECT 
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc 
                    JOIN information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = '{table_name}';
                """)
                fk_result = await session.execute(fk_query)
                fk_rows = fk_result.fetchall()
                if fk_rows:
                    schema_description += "Foreign Keys:\n"
                    for fk_col, fk_table, fk_ref_col in fk_rows:
                        schema_description += f"  {fk_col} -> {fk_table}.{fk_ref_col}\n"
            except Exception:
                pass
            
            schema_description += "\n"
        
        # Add generic guidance for LLM (no hardcoded table/column names)
        schema_description += "\n" + "="*70 + "\n"
        schema_description += "SQL GENERATION GUIDELINES:\n"
        schema_description += "="*70 + "\n"
        schema_description += """
CRITICAL RULES:

1. USE ACTUAL COLUMN NAMES from the schema above (marked with *** ***)
   - If schema shows *** txn_id ***, use 'txn_id' NOT 'transaction_id'
   - If schema shows *** first_name ***, use 'first_name' NOT 'name'
   - Always match exact column names from the discovered schema

2. EXTRACT VALUES FROM USER QUERY:
   - When user provides a specific value, use it EXACTLY in the WHERE clause
   - Example: If user says "show records for ABC123", use WHERE col = 'ABC123'
   - NEVER use placeholder text like '[VALUE]' or '[EXTRACT_FROM_USER_QUERY]'

3. USE FOREIGN KEYS FOR JOINS:
   - Check the Foreign Keys section shown above for each table
   - Join tables using the exact FK relationships discovered
     - Example: If FK shows "entity_id -> entities.id", use:
         JOIN entities e ON main_table.entity_id = e.id

4. FILTERING PATTERNS:
   - For identifier columns (ending in _id, _code, _no): use exact matching (=)
   - For text columns: use LIKE with wildcards if doing partial match
   - For date columns: use appropriate date comparisons

The schema shown above with *** *** around column names shows EXACTLY what columns exist.
Use those exact names without modification.
"""
        schema_description += "="*70 + "\n"
        
        return schema_description
    except Exception as e:
        return f"Error fetching schema: {str(e)}"
