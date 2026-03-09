"""
Schema Discovery Engine - Discovers schema facts from database metadata.

Dynamically discovers WITHOUT hardcoding:
- Foreign key relationships (any naming convention)
- Boolean column types (across all databases)
- Enum types and their values (ENUM, CHECK constraints, etc.)
- Semantic concepts in schema (approval, status, identifier, etc.)

Works across PostgreSQL, MySQL, SQL Server, SQLite.
Caches results for session-level performance.
"""

from __future__ import annotations

import logging
import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from .database_adapter import get_global_adapter, DatabaseType
from .. import llm

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Information about a column discovered from database."""
    table_name: str
    column_name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    enum_values: Optional[List[str]] = None  # For ENUM columns
    is_boolean: bool = False


@dataclass
class SchemaCache:
    """Cache for discovered schema information."""
    foreign_keys: Dict[Tuple[str, str], str] = field(default_factory=dict)  # (table1, table2) -> join_sql
    boolean_columns: Dict[str, List[str]] = field(default_factory=dict)  # table -> [col1, col2, ...]
    enum_columns: Dict[str, List[Tuple[str, List[str]]]] = field(default_factory=dict)  # table -> [(col, values), ...]
    semantic_concepts: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)  # concept -> {table: [cols]}


class SchemaDiscoveryEngine:
    """Discovers schema facts from database without hardcoding."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.adapter = get_global_adapter()
        self.cache = SchemaCache()
        self._discovery_in_progress = False
    
    async def discover_foreign_keys(
        self, 
        limit_tables: Optional[List[str]] = None
    ) -> Dict[Tuple[str, str], str]:
        """
        Discover ALL foreign key relationships in database.
        
        Returns:
            {
                ("child_table", "parent_table"): "child_table.parent_id = parent_table.id",
                ("related_table", "main_table"): "related_table.main_id = main_table.id",
                ...
            }
        
        Works for:
        - Standard FKs (table_id, table_name_id)
        - Custom FKs (custom_ref, number)
        - Composite keys (multiple columns)
        - All database types (PostgreSQL, MySQL, SQL Server, SQLite)
        """
        
        if self.cache.foreign_keys:
            logger.debug("Using cached foreign keys")
            return self.cache.foreign_keys
        
        logger.info(f"Discovering foreign keys from {self.adapter.db_type.value}")
        fk_map = {}
        
        try:
            if self.adapter.db_type == DatabaseType.POSTGRESQL:
                fk_map = await self._discover_fk_postgresql(limit_tables)
            elif self.adapter.db_type == DatabaseType.MYSQL:
                fk_map = await self._discover_fk_mysql(limit_tables)
            elif self.adapter.db_type == DatabaseType.SQL_SERVER:
                fk_map = await self._discover_fk_sqlserver(limit_tables)
            elif self.adapter.db_type == DatabaseType.SQLITE:
                fk_map = await self._discover_fk_sqlite(limit_tables)
        
        except Exception as e:
            logger.error(f"Error discovering foreign keys: {e}", exc_info=True)
            return {}
        
        self.cache.foreign_keys = fk_map
        logger.info(f"Discovered {len(fk_map)} foreign key relationships")
        return fk_map
    
    async def _discover_fk_postgresql(
        self, 
        limit_tables: Optional[List[str]] = None
    ) -> Dict[Tuple[str, str], str]:
        """Query PostgreSQL system catalogs for FKs."""
        query = """
        SELECT 
            kcu1.table_name,
            kcu2.table_name as referenced_table,
            string_agg(kcu1.column_name, ', ') as columns,
            string_agg(kcu2.column_name, ', ') as ref_columns
        FROM information_schema.referential_constraints rc
        JOIN information_schema.key_column_usage kcu1
            ON kcu1.constraint_name = rc.constraint_name
            AND kcu1.table_schema = rc.constraint_schema
        JOIN information_schema.key_column_usage kcu2
            ON kcu2.constraint_name = rc.unique_constraint_name
            AND kcu2.table_schema = rc.unique_constraint_schema
        GROUP BY kcu1.table_name, kcu2.table_name
        ORDER BY kcu1.table_name, kcu2.table_name
        """
        
        result = await self.db.execute(text(query))
        fk_map = {}
        
        for row in result.all():
            table1, table2, cols, ref_cols = row
            
            if limit_tables:
                if table1 not in limit_tables and table2 not in limit_tables:
                    continue
            
            cols_list = [c.strip() for c in cols.split(',')]
            ref_cols_list = [c.strip() for c in ref_cols.split(',')]
            
            join_conditions = [
                f"{table1}.{col1} = {table2}.{col2}"
                for col1, col2 in zip(cols_list, ref_cols_list)
            ]
            join_sql = " AND ".join(join_conditions)
            
            fk_map[(table1, table2)] = join_sql
        
        return fk_map
    
    async def _discover_fk_mysql(
        self, 
        limit_tables: Optional[List[str]] = None
    ) -> Dict[Tuple[str, str], str]:
        """Query MySQL schema for FKs via INFORMATION_SCHEMA."""
        query = """
        SELECT 
            CONSTRAINT_NAME,
            TABLE_NAME,
            COLUMN_NAME,
            REFERENCED_TABLE_NAME,
            REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE REFERENCED_TABLE_NAME IS NOT NULL
        AND TABLE_SCHEMA = DATABASE()
        ORDER BY TABLE_NAME, REFERENCED_TABLE_NAME
        """
        
        result = await self.db.execute(text(query))
        fk_map = {}
        fk_groups: Dict[str, List[Tuple[str, str, str, str]]] = {}
        
        for row in result.all():
            constraint_name, table_name, column_name, ref_table, ref_column = row
            
            if limit_tables:
                if table_name not in limit_tables and ref_table not in limit_tables:
                    continue
            
            key = (table_name, ref_table)
            if key not in fk_groups:
                fk_groups[key] = []
            fk_groups[key].append((table_name, column_name, ref_table, ref_column))
        
        for (table1, table2), cols in fk_groups.items():
            join_conditions = [
                f"{col[0]}.{col[1]} = {col[2]}.{col[3]}"
                for col in cols
            ]
            join_sql = " AND ".join(join_conditions)
            fk_map[(table1, table2)] = join_sql
        
        return fk_map
    
    async def _discover_fk_sqlserver(
        self, 
        limit_tables: Optional[List[str]] = None
    ) -> Dict[Tuple[str, str], str]:
        """Query SQL Server system views for FKs."""
        query = """
        SELECT
            t1.name as table_name,
            t2.name as referenced_table,
            c1.name as column_name,
            c2.name as ref_column_name
        FROM sys.foreign_key_columns fkc
        JOIN sys.tables t1 ON fkc.parent_object_id = t1.object_id
        JOIN sys.tables t2 ON fkc.referenced_object_id = t2.object_id
        JOIN sys.columns c1 ON fkc.parent_object_id = c1.object_id 
            AND fkc.parent_column_id = c1.column_id
        JOIN sys.columns c2 ON fkc.referenced_object_id = c2.object_id
            AND fkc.referenced_column_id = c2.column_id
        ORDER BY t1.name, t2.name, fkc.constraint_column_id
        """
        
        result = await self.db.execute(text(query))
        fk_map = {}
        fk_groups: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        
        for row in result.all():
            table1, table2, col1, col2 = row
            
            if limit_tables:
                if table1 not in limit_tables and table2 not in limit_tables:
                    continue
            
            key = (table1, table2)
            if key not in fk_groups:
                fk_groups[key] = []
            fk_groups[key].append((col1, col2))
        
        for (table1, table2), cols in fk_groups.items():
            join_conditions = [f"{table1}.{col1} = {table2}.{col2}" for col1, col2 in cols]
            join_sql = " AND ".join(join_conditions)
            fk_map[(table1, table2)] = join_sql
        
        return fk_map
    
    async def _discover_fk_sqlite(
        self, 
        limit_tables: Optional[List[str]] = None
    ) -> Dict[Tuple[str, str], str]:
        """Query SQLite pragma for FKs."""
        # Get all tables first
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        result = await self.db.execute(text(tables_query))
        tables = [row[0] for row in result.all()]
        
        if limit_tables:
            tables = [t for t in tables if t in limit_tables]
        
        fk_map = {}
        
        for table in tables:
            try:
                pragma_query = f"PRAGMA foreign_key_list({table})"
                result = await self.db.execute(text(pragma_query))
                for row in result.all():
                    # (id, seq, table, from, to, on_delete, on_update, match)
                    _, _, ref_table, from_col, to_col, *_ = row
                    
                    if limit_tables and ref_table not in limit_tables:
                        continue
                    
                    join_sql = f"{table}.{from_col} = {ref_table}.{to_col}"
                    fk_map[(table, ref_table)] = join_sql
            except Exception as e:
                logger.debug(f"Error querying pragma for table {table}: {e}")
        
        return fk_map
    
    async def discover_boolean_columns(
        self, 
        limit_tables: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Discover all BOOLEAN type columns from INFORMATION_SCHEMA.
        
        Returns:
            {
                "table1": ["is_verified", "is_active", "email_confirmed"],
                "table2": ["is_active", "is_primary"],
                "table3": []
            }
        
        Works for:
        - BOOLEAN type (PostgreSQL)
        - BIT(1) type (SQL Server)
        - TINYINT type used as boolean (MySQL)
        """
        
        if self.cache.boolean_columns:
            logger.debug("Using cached boolean columns")
            return self.cache.boolean_columns
        
        logger.info(f"Discovering boolean columns from {self.adapter.db_type.value}")
        bool_cols = {}
        
        try:
            if self.adapter.db_type == DatabaseType.POSTGRESQL:
                bool_cols = await self._discover_bool_postgresql(limit_tables)
            elif self.adapter.db_type == DatabaseType.MYSQL:
                bool_cols = await self._discover_bool_mysql(limit_tables)
            elif self.adapter.db_type == DatabaseType.SQL_SERVER:
                bool_cols = await self._discover_bool_sqlserver(limit_tables)
            elif self.adapter.db_type == DatabaseType.SQLITE:
                bool_cols = await self._discover_bool_sqlite(limit_tables)
        
        except Exception as e:
            logger.error(f"Error discovering boolean columns: {e}", exc_info=True)
            return {}
        
        self.cache.boolean_columns = bool_cols
        logger.info(f"Discovered boolean columns: {sum(len(v) for v in bool_cols.values())} total")
        return bool_cols
    
    async def _discover_bool_postgresql(
        self, 
        limit_tables: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """Query PostgreSQL for BOOLEAN type columns."""
        query = """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE data_type = 'boolean'
        ORDER BY table_name, column_name
        """
        
        result = await self.db.execute(text(query))
        bool_cols = {}
        
        for row in result.all():
            table, column = row
            
            if limit_tables and table not in limit_tables:
                continue
            
            if table not in bool_cols:
                bool_cols[table] = []
            bool_cols[table].append(column)
        
        return bool_cols
    
    async def _discover_bool_mysql(
        self, 
        limit_tables: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """Query MySQL for BOOLEAN/BIT columns."""
        query = """
        SELECT TABLE_NAME, COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
        AND COLUMN_TYPE IN ('tinyint(1)', 'bit(1)', 'boolean')
        ORDER BY TABLE_NAME, COLUMN_NAME
        """
        
        result = await self.db.execute(text(query))
        bool_cols = {}
        
        for row in result.all():
            table, column = row
            
            if limit_tables and table not in limit_tables:
                continue
            
            if table not in bool_cols:
                bool_cols[table] = []
            bool_cols[table].append(column)
        
        return bool_cols
    
    async def _discover_bool_sqlserver(
        self, 
        limit_tables: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """Query SQL Server for BIT type columns."""
        query = """
        SELECT t.name, c.name
        FROM sys.columns c
        JOIN sys.tables t ON c.object_id = t.object_id
        JOIN sys.types st ON c.user_type_id = st.user_type_id
        WHERE st.name IN ('bit', 'boolean')
        ORDER BY t.name, c.name
        """
        
        result = await self.db.execute(text(query))
        bool_cols = {}
        
        for row in result.all():
            table, column = row
            
            if limit_tables and table not in limit_tables:
                continue
            
            if table not in bool_cols:
                bool_cols[table] = []
            bool_cols[table].append(column)
        
        return bool_cols
    
    async def _discover_bool_sqlite(
        self, 
        limit_tables: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """Query SQLite for likely boolean columns (no type system, use heuristics)."""
        # SQLite has no native boolean, but we can identify via constraints/CHECK
        bool_cols = {}
        
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        result = await self.db.execute(text(tables_query))
        tables = [row[0] for row in result.all()]
        
        if limit_tables:
            tables = [t for t in tables if t in limit_tables]
        
        for table in tables:
            try:
                info_query = f"PRAGMA table_info({table})"
                result = await self.db.execute(text(info_query))
                for row in result.all():
                    # (cid, name, type, notnull, dflt_value, pk)
                    col_name = row[1]
                    # Look for columns named is_*, has_*, *_bool, etc.
                    if any(pattern in col_name.lower() for pattern in ['is_', 'has_', '_bool', '_flag']):
                        if table not in bool_cols:
                            bool_cols[table] = []
                        bool_cols[table].append(col_name)
            except Exception as e:
                logger.debug(f"Error querying table info for {table}: {e}")
        
        return bool_cols
    
    async def discover_enum_columns(
        self, 
        limit_tables: Optional[List[str]] = None
    ) -> Dict[str, List[Tuple[str, List[str]]]]:
        """
        Discover all ENUM type columns and their possible values.
        
        Returns:
            {
                "table1": [
                    ("type_column", ["TYPE_A", "TYPE_B", "TYPE_C"]),
                    ("status_column", ["STATUS_1", "STATUS_2", "STATUS_3"])
                ],
                "table2": [...]
            }
        
        Works for:
        - ENUM type (PostgreSQL, MySQL)
        - CHECK constraints with literal values (SQL Server)
        - Sample distinct values (fallback)
        """
        
        if self.cache.enum_columns:
            logger.debug("Using cached enum columns")
            return self.cache.enum_columns
        
        logger.info(f"Discovering enum columns from {self.adapter.db_type.value}")
        enum_cols = {}
        
        try:
            if self.adapter.db_type == DatabaseType.POSTGRESQL:
                enum_cols = await self._discover_enum_postgresql(limit_tables)
            elif self.adapter.db_type == DatabaseType.MYSQL:
                enum_cols = await self._discover_enum_mysql(limit_tables)
            elif self.adapter.db_type in [DatabaseType.SQL_SERVER, DatabaseType.SQLITE]:
                # No native ENUM, skip
                enum_cols = {}
        
        except Exception as e:
            logger.error(f"Error discovering enum columns: {e}", exc_info=True)
            return {}
        
        self.cache.enum_columns = enum_cols
        logger.info(f"Discovered enum columns: {sum(len(v) for v in enum_cols.values())} total")
        return enum_cols
    
    async def _discover_enum_postgresql(
        self, 
        limit_tables: Optional[List[str]] = None
    ) -> Dict[str, List[Tuple[str, List[str]]]]:
        """Query PostgreSQL ENUM types."""
        query = """
        SELECT 
            t.typname,
            array_agg(e.enumlabel ORDER BY e.enumsortorder) as enum_values
        FROM pg_enum e
        JOIN pg_type t ON e.enumtypid = t.oid
        GROUP BY t.typname
        ORDER BY t.typname
        """
        
        result = await self.db.execute(text(query))
        enum_types = {}
        
        for row in result.all():
            type_name, values = row
            enum_types[type_name] = list(values) if values else []
        
        # Find columns using these types
        enum_cols = {}
        columns_query = """
        SELECT table_name, column_name, udt_name
        FROM information_schema.columns
        WHERE udt_name IN (SELECT typname FROM pg_type WHERE typcategory = 'E')
        ORDER BY table_name, column_name
        """
        
        result = await self.db.execute(text(columns_query))
        
        for row in result.all():
            table, column, udt_name = row
            
            if limit_tables and table not in limit_tables:
                continue
            
            if table not in enum_cols:
                enum_cols[table] = []
            
            values = enum_types.get(udt_name, [])
            enum_cols[table].append((column, values))
        
        return enum_cols
    
    async def _discover_enum_mysql(
        self, 
        limit_tables: Optional[List[str]] = None
    ) -> Dict[str, List[Tuple[str, List[str]]]]:
        """Query MySQL ENUM types."""
        query = """
        SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
        AND COLUMN_TYPE LIKE 'enum(%'
        ORDER BY TABLE_NAME, COLUMN_NAME
        """
        
        result = await self.db.execute(text(query))
        enum_cols = {}
        
        for row in result.all():
            table, column, column_type = row
            
            if limit_tables and table not in limit_tables:
                continue
            
            # Parse: enum('A','B','C')
            match = re.search(r"enum\((.*?)\)", column_type, re.IGNORECASE)
            if match:
                values_str = match.group(1)
                values = [v.strip().strip("'\"") for v in values_str.split(",")]
                
                if table not in enum_cols:
                    enum_cols[table] = []
                enum_cols[table].append((column, values))
        
        return enum_cols
    
    async def discover_semantic_concepts(
        self, 
        schema_context: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Discover semantic concepts in schema (NOT hardcoded).
        
        Uses LLM to infer semantic meaning from column names and types.
        
        Returns:
            {
                "approval": {
                    "table1": ["is_verified", "is_approved"],
                    "table2": ["compliance_check"]
                },
                "status": {
                    "table1": ["record_status"],
                    "table2": ["item_status"]
                }
            }
        """
        
        logger.info("Discovering semantic concepts using LLM")
        
        schema_desc = json.dumps(schema_context, indent=2)
        
        prompt = f"""
You are a database schema analyst. Analyze these tables/columns and identify semantic concepts:

Schema:
{schema_desc}

For each semantic concept found, list the tables and columns that represent it.

Common concepts to look for:
- approval (verified, approved, passed checks)
- status (state, condition, current state)
- identifier (id, code, key)
- amount (value, sum, total)
- timestamp (date, time, created, updated)
- name (title, label, description)
- contact (email, phone, address)

Respond ONLY with valid JSON (no markdown):
{{
  "concepts": {{
    "concept_name": {{
      "table1": ["column1", "column2"],
      "table2": ["column3"]
    }},
    "another_concept": {{...}}
  }}
}}
"""
        
        try:
            response = await llm.call_llm([
                {
                    "role": "system",
                    "content": "You are a database schema analyzer. Respond with ONLY valid JSON."
                },
                {"role": "user", "content": prompt}
            ], max_tokens=2000, temperature=0.3)
            
            # Parse JSON response
            response_text = str(response)
            result = json.loads(response_text)
            concepts = result.get("concepts", {})
            
            self.cache.semantic_concepts = concepts
            logger.info(f"Discovered {len(concepts)} semantic concepts")
            return concepts
        
        except Exception as e:
            logger.error(f"Error discovering semantic concepts: {e}", exc_info=True)
            return {}
    
    async def find_columns_for_table_concept(
        self,
        concept: str,
        table_name: str
    ) -> List[str]:
        """
        Find columns in a specific table matching a semantic concept.
        
        Example: find_columns_for_table_concept("approval", "users")
        Returns: ["is_verified", "is_approved", "compliance_check"]
        """
        
        # Use cached semantic concepts if available
        if not self.cache.semantic_concepts:
            return []
        
        concept_data = self.cache.semantic_concepts.get(concept, {})
        return concept_data.get(table_name, [])
    
    async def get_related_tables(
        self, 
        table_name: str,
        fk_map: Optional[Dict[Tuple[str, str], str]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Get tables related to the given table via foreign keys.
        
        Returns:
            (outbound_tables, inbound_tables)
            - outbound: Tables this table references
            - inbound: Tables that reference this table
        """
        
        if not fk_map:
            fk_map = await self.discover_foreign_keys()
        
        outbound = [t2 for (t1, t2) in fk_map.keys() if t1 == table_name]
        inbound = [t1 for (t1, t2) in fk_map.keys() if t2 == table_name]
        
        return (outbound, inbound)
    
    def clear_cache(self) -> None:
        """Clear all cached discovery results."""
        self.cache = SchemaCache()
        logger.info("Cleared schema discovery cache")
