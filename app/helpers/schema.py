"""Database schema discovery and caching utilities."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings


async def get_database_schema(session: AsyncSession) -> str:
    """Fetch database schema dynamically from PostgreSQL.
    
    Retrieves all tables and their columns from the 'genai' schema
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
                WHERE c.table_schema = 'genai' AND c.table_name = '{table_name}'
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
                            # Infer column purpose based on name and sample values
                            col_lower = col_name.lower()
                            purpose = ""
                            if any(keyword in col_lower for keyword in ['cust', 'customer', 'client']):
                                purpose = "Customer identifier"
                            elif any(keyword in col_lower for keyword in ['merchant', 'vendor', 'seller']):
                                purpose = "Merchant/Vendor identifier"
                            elif any(keyword in col_lower for keyword in ['txn', 'transaction', 'trans']):
                                purpose = "Transaction identifier"
                            elif any(keyword in col_lower for keyword in ['amount', 'value', 'price', 'cost']):
                                purpose = "Amount/monetary value"
                            elif any(keyword in col_lower for keyword in ['time', 'date', 'created', 'timestamp']):
                                purpose = "Date/time information"
                            elif any(keyword in col_lower for keyword in ['status', 'state', 'type']):
                                purpose = "Status/classification"
                            elif any(keyword in col_lower for keyword in ['name', 'title', 'description']):
                                purpose = "Name/description"
                            
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
        
        # Add entity mapping guide to help LLM match search criteria to actual columns
        schema_description += "\n" + "="*70 + "\n"
        schema_description += "ENTITY MAPPING GUIDE - INTELLIGENT ENTITY MATCHING:\n"
        schema_description += "="*70 + "\n"
        schema_description += """
CRITICAL: Use ACTUAL column names from the schema above (marked with ***)

REAL COLUMN NAME MAPPING FOR COMMON SEARCHES:

1. TRANSACTION SEARCHES - Use ACTUAL transaction table columns:
   - Transaction ID: use 'txn_id' (NOT transaction_id)
   - Transaction Time: use 'txn_time' (NOT transaction_time)
   - Transaction Type: use 'txn_type'
   - Customer in transaction: use 'customer_id'
   - Amount: use 'amount'
   - Merchant: use 'merchant_id'
   Example SQL: SELECT t.txn_id, t.amount, t.txn_time FROM transactions t 
                WHERE t.customer_id = 1868 ORDER BY t.txn_time DESC

2. CUSTOMER SEARCHES - Use ACTUAL customer table columns:
   - Customer Code (the CUST#### format): use 'customer_code' (NOT customer_id)
   - Customer ID (numeric): use 'customer_id'
   - Name: use 'first_name' or 'last_name'
   - Email: use 'email'
   - Phone: use 'phone'
   CRITICAL: Extract the exact value from the user query and use it directly
   Example: If user says "customer CUST0000001", extract "CUST0000001" and use it
   Correct: SELECT c.customer_code, c.first_name, c.email FROM customers c 
            WHERE c.customer_code = 'CUST0000001'
   Wrong: SELECT c.customer_code, c.first_name, c.email FROM customers c 
          WHERE c.customer_code = '[PLACEHOLDER]'

3. JOINING TABLES - Match actual foreign keys:
   - To find transactions for a customer with a specific code:
     a) Find customer_id by looking up in customers table using their code
     b) Join transactions on customer_id
     c) Use the EXACT customer code from the user query in the WHERE clause
   Example: If user says "customer CUST0000001", do this:
            SELECT t.txn_id, t.amount, t.txn_time 
            FROM transactions t 
            JOIN customers c ON t.customer_id = c.customer_id 
            WHERE c.customer_code = 'CUST0000001'  ← Use the actual value from user input

KEY RULE: Always use the actual column names shown above (txn_id, txn_time, customer_code, etc.)
Do NOT use generic names like transaction_id or transaction_time - they don't exist!
Do NOT use placeholder text like [EXTRACT_FROM_USER_QUERY] - extract and use the REAL value!

The schema shown above with *** *** around column names shows EXACTLY what columns exist.
Use those exact names without modification.
When user provides a specific value (like CUST0000001), always use that exact value in the WHERE clause.
"""
        schema_description += "="*70 + "\n"
        
        return schema_description
    except Exception as e:
        return f"Error fetching schema: {str(e)}"
