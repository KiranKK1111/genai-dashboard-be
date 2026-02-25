"""
Database Analyzer - Dynamically understands table structure and data.

Provides comprehensive metadata about every table:
- Column names, types, constraints
- Sample data (5 rows) to understand data format
- Foreign key relationships
- Table descriptions based on data analysis
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, inspect
import json

from ..config import settings


class DatabaseAnalyzer:
    """Analyzes database structure and provides intelligent context for SQL generation."""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.table_metadata_cache: Dict[str, Dict[str, Any]] = {}
        
    async def analyze_all_tables(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze all tables in the schema.
        
        Returns dictionary with table metadata:
        {
            'table_name': {
                'columns': [...],
                'sample_data': [...],
                'relationships': [...],
                'description': '...'
            }
        }
        """
        # Get all table names
        tables_query = f"""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = :schema
        ORDER BY table_name
        """
        
        result = await self.db_session.execute(
            text(tables_query),
            {"schema": settings.postgres_schema}
        )
        table_names = [row[0] for row in result.fetchall()]
        
        print(f"[DB_ANALYZER] Found {len(table_names)} tables: {', '.join(table_names)}")
        
        # Analyze each table
        for table_name in table_names:
            self.table_metadata_cache[table_name] = await self._analyze_table(table_name)
        
        return self.table_metadata_cache
    
    async def get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """Get metadata for a specific table (cached)."""
        if table_name not in self.table_metadata_cache:
            self.table_metadata_cache[table_name] = await self._analyze_table(table_name)
        return self.table_metadata_cache[table_name]
    
    async def _analyze_table(self, table_name: str) -> Dict[str, Any]:
        """Analyze a single table's structure and data."""
        
        # Step 1: Get column information
        columns_query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
        """
        
        result = await self.db_session.execute(
            text(columns_query),
            {"schema": settings.postgres_schema, "table": table_name}
        )
        
        columns = []
        for row in result.fetchall():
            columns.append({
                "name": row[0],
                "type": row[1],
                "nullable": row[2] == "YES",
                "default": row[3]
            })
        
        # Step 2: Get foreign key relationships
        fk_query = f"""
        SELECT 
            constraint_name,
            column_name,
            referenced_table_name, 
            referenced_column_name
        FROM information_schema.referential_constraints rc
        JOIN information_schema.key_column_usage kcu 
            ON rc.constraint_name = kcu.constraint_name
            AND rc.table_schema = kcu.table_schema
        LEFT JOIN information_schema.constraint_column_usage ccu
            ON rc.unique_constraint_name = ccu.constraint_name
            AND rc.constraint_schema = ccu.table_schema
        WHERE rc.constraint_schema = :schema 
            AND rc.table_name = :table
        """
        
        try:
            result = await self.db_session.execute(
                text(fk_query),
                {"schema": settings.postgres_schema, "table": table_name}
            )
            relationships = [
                {
                    "column": row[1],
                    "references_table": row[2],
                    "references_column": row[3]
                }
                for row in result.fetchall()
            ]
        except Exception as e:
            # If FK query fails, rollback to prevent transaction corruption
            try:
                await self.db_session.rollback()
            except:
                pass
            print(f"[DB_ANALYZER] Could not fetch relationships for {table_name}: {str(e)[:80]}")
            relationships = []
        
        # Step 3: Get sample data (first 5 rows)
        # Note: If this fails, we continue without sample data
        sample_data = []
        try:
            sample_data_query = f"""
            SELECT * FROM {settings.postgres_schema}.{table_name} LIMIT 5
            """
            result = await self.db_session.execute(text(sample_data_query))
            rows = result.fetchall()
            
            # Get column names from result keys
            column_names = result.keys()
            
            for row in rows:
                # Convert row to dict using column names
                row_dict = {}
                for col_name, value in zip(column_names, row):
                    row_dict[col_name] = value
                sample_data.append(row_dict)
        except Exception as e:
            # If sample data fetch fails, rollback and continue
            # This prevents transaction state corruption
            try:
                await self.db_session.rollback()
            except:
                pass
            print(f"[DB_ANALYZER] Could not fetch sample data for {table_name}: {str(e)[:100]}")
            sample_data = []
        
        # Step 4: Generate intelligent description
        description = await self._generate_table_description(table_name, columns, sample_data, relationships)
        
        metadata = {
            "table_name": table_name,
            "columns": columns,
            "sample_data": sample_data,
            "relationships": relationships,
            "description": description,
            "column_names": [col["name"] for col in columns],
            "column_types": {col["name"]: col["type"] for col in columns}
        }
        
        print(f"[DB_ANALYZER] Analyzed {table_name}: {len(columns)} columns, {len(sample_data)} sample rows")
        
        return metadata
    
    async def _generate_table_description(
        self, 
        table_name: str, 
        columns: List[Dict], 
        sample_data: List[Dict],
        relationships: List[Dict]
    ) -> str:
        """
        Generate intelligent description of table based on its structure and data.
        """
        description_parts = []
        
        # Table purpose based on name
        description_parts.append(f"Table: {table_name}")
        
        # Key columns
        key_cols = [col for col in columns if col["name"].endswith("_id") or col["name"].endswith("_code")]
        if key_cols:
            description_parts.append(f"Keys: {', '.join(c['name'] for c in key_cols)}")
        
        # Relationships
        if relationships:
            for rel in relationships:
                description_parts.append(f"Links to: {rel['references_table']}.{rel['references_column']} via {rel['column']}")
        
        # Data examples
        if sample_data:
            description_parts.append("Sample records:")
            for i, record in enumerate(sample_data[:2], 1):  # Show first 2 records
                sample_str = ", ".join(f"{k}={v}" for k, v in list(record.items())[:3])
                description_parts.append(f"  {i}. {sample_str}")
        
        return "\n".join(description_parts)
    
    async def get_table_context_for_llm(self, table_names: List[str]) -> str:
        """
        Create formatted context about tables for LLM SQL generation.
        
        This describes table structure, relationships, and sample data
        so LLM understands what data is actually in the database.
        
        Tables are analyzed on-demand if not already cached.
        """
        if not table_names:
            return ""
        
        # Ensure requested tables are analyzed (on-demand analysis)
        for table_name in table_names:
            if table_name not in self.table_metadata_cache:
                try:
                    await self._analyze_table(table_name)
                except Exception as e:
                    print(f"[DB_ANALYZER] Warning: Failed to analyze table {table_name}: {str(e)}")
                    # Continue - we'll use whatever metadata we have
        
        context_lines = ["DATABASE SCHEMA & DATA CONTEXT:"]
        context_lines.append("=" * 70)
        
        for table_name in table_names:
            if table_name not in self.table_metadata_cache:
                continue
            
            metadata = self.table_metadata_cache[table_name]
            
            # Table header
            context_lines.append(f"\n📊 TABLE: {table_name.upper()}")
            context_lines.append("-" * 70)
            
            # Columns
            context_lines.append("Columns:")
            for col in metadata["columns"]:
                nullable = "(nullable)" if col["nullable"] else "(required)"
                context_lines.append(f"  • {col['name']}: {col['type']} {nullable}")
            
            # Relationships
            if metadata["relationships"]:
                context_lines.append("\nRelationships:")
                for rel in metadata["relationships"]:
                    context_lines.append(
                        f"  • {rel['column']} → {rel['references_table']}.{rel['references_column']}"
                    )
            
            # Sample data
            if metadata["sample_data"]:
                context_lines.append("\nSample Data:")
                for i, record in enumerate(metadata["sample_data"][:3], 1):
                    record_str = " | ".join(f"{k}={v}" for k, v in list(record.items())[:4])
                    context_lines.append(f"  {i}. {record_str}")
            
            # Data description
            context_lines.append(f"\nData Description:\n{metadata['description']}")
        
        context_lines.append("\n" + "=" * 70)
        return "\n".join(context_lines)
    
    async def get_column_relationships(self, table_name: str, column_name: str) -> Optional[Dict]:
        """
        Check if a column is a foreign key and return relationship info.
        """
        if table_name not in self.table_metadata_cache:
            await self._analyze_table(table_name)
        
        metadata = self.table_metadata_cache[table_name]
        
        for rel in metadata["relationships"]:
            if rel["column"] == column_name:
                return rel
        
        return None


async def create_database_analyzer(db: AsyncSession) -> DatabaseAnalyzer:
    """Factory function to create and initialize database analyzer."""
    analyzer = DatabaseAnalyzer(db)
    await analyzer.analyze_all_tables()
    return analyzer
