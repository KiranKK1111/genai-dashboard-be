"""SQL generation service - converts natural language to SQL."""

from __future__ import annotations

import re
from typing import List, Dict, Tuple, Optional

from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import AsyncSession

from .. import llm
from ..config import settings
from ..helpers import get_database_schema
from .sql_safety_validator import SQLSafetyValidator
from .schema_discovery import SchemaCatalog
from .adaptive_schema_analyzer import AdaptiveSchemaAnalyzer
from .dialect_sql_engine import DialectSqlEngine
from .database_adapter import DatabaseAdapter
from .schema_grounding import SchemaGroundingContext
from .filter_value_grounding import FilterValueGrounding, FilterValidator
from .relationship_aware_table_selector import JoinGraphBuilder


def _extract_query_identifiers(query: str) -> List[Dict[str, str]]:
    """
    Extract identifiers (IDs, codes) from query using generic pattern matching.
    
    Replaces legacy domain-specific identifier logic with generic identifier detection.
    Detects patterns like: CUST123, INV456, ACC789, etc.
    
    Returns:
        List of dicts with 'pattern' and 'value' keys
    """
    identifiers = []
    
    # Generic pattern: 3-6 uppercase letters followed by digits
    # This matches CUST123, INV456, ACC789, etc. without hardcoding
    id_pattern = r'\b([A-Z]{3,6}\d+)\b'
    matches = re.finditer(id_pattern, query, re.IGNORECASE)
    
    for match in matches:
        id_code = match.group(1).upper()
        # Infer the type from prefix (first alpha characters)
        id_type = ''.join(c for c in id_code if c.isalpha())
        identifiers.append({
            'type': id_type,
            'value': id_code,
            'full': id_code
        })
        print(f"[ID DETECTOR] Found {id_type} code: {id_code}")
    
    return identifiers


async def generate_sql_with_analysis(
    query: str, 
    session: AsyncSession, 
    conversation_history: str = "",
    force_join: bool = False,  # Legacy: force JOIN when clarification implies related entities
    followup_context = None,  # Optional FollowUpContext from followup_manager
    semantic_context = None,  # Optional SemanticQueryResult from semantic orchestrator (NEW - FOR SEMANTIC PIPELINE)
) -> Tuple[str, Optional[str]]:
    """Generate SQL using LLM-based analysis with new architecture.
    
    This function now delegates to the main generate_sql() function
    which uses the new modular architecture services.
    
    Args:
        query: The user's natural language query
        session: Async database session
        conversation_history: Optional previous context
        force_join: If True, force JOIN even if asking clarifying question (legacy parameter)
        followup_context: Optional FollowUpContext for dynamic follow-up management
        semantic_context: Optional SemanticQueryResult with retrieved schema candidates (NEW - FOR SEMANTIC PIPELINE)
    
    Returns:
        Tuple of (sql_query, clarifying_question or None)
    """
    print(f"\n[STEP 1] Analyzing user query: {query}")
    
    # Check if this is a confirmation response
    is_confirmation = any(word in query.lower() for word in ['yes', 'yeah', 'yep', 'correct', 'confirm', 'right', 'absolutely', 'sure'])
    if is_confirmation:
        print(f"[CONFIRMATION] User confirmed with: '{query}'")
    
    # Extract identifiers from the query (GENERIC approach, not hardcoded domain-specific codes)
    # This detects patterns like CUST0001, INV123, ACC456, etc. without hardcoding specific prefixes
    extracted_identifiers = _extract_query_identifiers(query)
    if extracted_identifiers:
        print(f"[IDENTIFIERS] Found in query: {extracted_identifiers}")
    
    # Use the main LLM-based generation function
    # The new architecture (entity_parser, schema_discovery, hybrid_matcher) 
    # handles all the analysis automatically
    sql = await generate_sql(query, session, conversation_history, followup_context=followup_context, semantic_context=semantic_context)
    
    print(f"[STEP 5] Generated SQL: {sql}")
    return sql, None


async def generate_sql(query: str, session: AsyncSession, conversation_history: str = "", followup_context = None, semantic_context = None) -> str:
    """Generate SQL statement from natural language query using LLM.
    
    Supports semantic query pipeline for plan-first generation with schema context.
    Uses dynamic schema discovery to understand actual database structure
    and generates accurate SQL queries. The LLM handles all context awareness
    and query understanding without hardcoded keyword checks.
    
    Supports intelligent follow-up query handling via followup_context which
    provides information about previous queries, tables, and filters.
    
    Args:
        query: The user's natural language query
        session: Async database session for schema discovery
        conversation_history: Optional formatted conversation history for context
        followup_context: Optional FollowUpContext for dynamic follow-up management
        semantic_context: Optional SemanticQueryResult with retrieved schema candidates (NEW - FROM SEMANTIC ORCHESTRATOR)
    
    Returns:
        A SQL SELECT statement as a string
    """
    # CRITICAL: Extract identifiers (GENERIC approach, not hardcoded like entity_code)
    # This detects CUST123, INV456, etc. without hardcoding specific prefixes
    extracted_ids = _extract_query_identifiers(query)
    identifiers_context = ""
    if extracted_ids:
        id_list = ", ".join([f"{x['type']}({x['value']})" for x in extracted_ids])
        identifiers_context = f"Identifiers in query: {id_list}"
        print(f"[IDENTIFIERS] Contextualizing for LLM: {identifiers_context}")
    
    # Fetch actual database schema
    schema_info = await get_database_schema(session)
    
    # ADAPTIVE SCHEMA ANALYSIS: Determine target table dynamically using LLM (NO HARDCODING)
    target_table = None
    intelligent_schema_context = None
    schema_grounding = None
    filter_value_grounding = None
    join_graph_builder = None
    
    try:
        # Initialize schema catalog for dynamic table discovery (replaces hardcoded keywords)
        # First, detect the database dialect from the connection
        bind = session.get_bind()
        dialect_name = bind.dialect.name if hasattr(bind, 'dialect') else 'postgresql'
        dialect_engine = DialectSqlEngine(dialect_name)
        print(f"[DIALECT DETECTION] Database dialect: {dialect_name}")
        
        # Initialize schema catalog with explicit schema name from settings
        schema_name = getattr(settings, 'postgres_schema', 'genai')
        schema_catalog = SchemaCatalog(schema_name=schema_name)
        await schema_catalog.initialize(session)
        available_tables = schema_catalog.get_all_tables()
        table_names = sorted(list(available_tables.keys()))  # Sorted for deterministic matching
        
        print(f"[DYNAMIC TABLE DETECTION] Available tables in database: {table_names}")
        
        # ============================================================================
        # PRINCIPLES 2 & 3 INTEGRATION (DEFERRED AFTER SCHEMA CATALOG INIT)
        # Initialize schema grounding (Principle 1) to support Principles 2 & 3
        # ============================================================================
        try:
            print(f"[GROUNDING] Initializing schema context for constraint generation...")
            
            # Principle 1: Build schema grounding context from schema_catalog data
            # (Avoids greenlet issue by not using inspect() directly)
            schema_grounding = SchemaGroundingContext(schema_name=schema_name)
            
            # Manually populate from schema_catalog instead of using inspect()
            # This is async-safe and reuses the already-discovered schema
            for table_name in table_names:
                if table_name in available_tables:
                    table_data = available_tables[table_name]
                    table_info = {
                        "name": table_name,
                        "full_name": f"{schema_name}.{table_name}",
                        "columns": {},
                        "primary_key": None,
                        "foreign_keys": [],
                        "sample_values": {}
                    }
                    
                    # Populate columns from schema_catalog (if available)
                    if hasattr(table_data, 'get'):
                        for col_name, col_data in table_data.items():
                            table_info["columns"][col_name] = {
                                "name": col_name,
                                "type": str(col_data.get('type', 'unknown')),
                                "nullable": col_data.get('nullable', True),
                                "is_enum": 'enum' in str(col_data.get('type', '')).lower(),
                                "sample_values": []
                            }
                    
                    schema_grounding.tables[table_name] = table_info
            
            # Populate enum values and sample values from database
            await schema_grounding.populate_enum_and_samples(session, sample_limit=10)
            
            # Principle 2: Initialize filter value grounding
            filter_value_grounding = FilterValueGrounding(schema_name=schema_name)
            await filter_value_grounding.populate_from_schema(schema_grounding, session)
            filter_value_grounding.extract_user_query_terms(query)
            print(f"[GROUNDING] Filter value constraints loaded ({len(filter_value_grounding.filter_metadata)} columns)")
            
            # Principle 3: Initialize join graph for relationship awareness
            join_graph_builder = JoinGraphBuilder(schema_grounding)
            print(f"[GROUNDING] Join graph built with {len(join_graph_builder.adjacency)} tables")
            
        except Exception as e:
            print(f"[GROUNDING] Warning: Could not initialize grounding principles: {e}")
            # Continue without grounding - system remains backward compatible
            import traceback
            traceback.print_exc()
        
        # Use LLM to determine which table user is asking about (SEMANTIC MATCHING)
        # This replaces hardcoded keyword mappings
        table_names_str = ", ".join(table_names)
        table_detection_prompt = f"""Analyze the user's query and identify the PRIMARY ENTITY they want to retrieve.

Available tables: {table_names_str}

User query: "{query}"

CRITICAL DISTINCTION:
- Identify WHAT ENTITY the user is asking about, not just what filters they mention.
- If the user asks about "X with Y", usually they want X data, filtered by Y.
- If the user asks to "find X" or "show X" or "list X", return the table that matches X.

KEY RULE: What does the user want to SEE/GET/FIND IN THE RESULT?
Look at the subject of the query and match it to the most semantically similar table name.

Your response MUST be EXACTLY ONE table name from the list above. Nothing else.
Return ONLY the table name. No explanation."""
        
        messages = [
            {"role": "system", "content": "You are a semantic entity analyzer. Identify what the user wants to see/retrieve. Return ONLY a table name from the provided list."},
            {"role": "user", "content": table_detection_prompt}
        ]
        
        llm_response = await llm.call_llm(messages, stream=False, max_tokens=50, temperature=0.0)
        detected_table = llm_response.strip().lower()
        
        print(f"[DYNAMIC] LLM response: '{detected_table}'")
        
        # Clean response (remove quotes, backticks, punctuation)
        detected_table = detected_table.strip('"`\'.,;:!? ')
        
        # Handle common LLM response patterns
        # Try to extract table name from conversational responses
        # Priority: exact match → pluralization → substring match
        
        matched_table = None
        table_names_lower = {t.lower(): t for t in table_names}
        
        # 1. Exact match (case-insensitive)
        if detected_table in table_names_lower:
            matched_table = table_names_lower[detected_table]
            print(f"[DYNAMIC] ✅ Exact match found: {matched_table}")
        
        # 2. Try adding 's' for pluralization (entity → entities)
        elif detected_table + 's' in table_names_lower:
            matched_table = table_names_lower[detected_table + 's']
            print(f"[DYNAMIC] ✅ Pluralization match found: {detected_table} → {matched_table}")
        
        # 3. Try removing 's' for singularization (entities → entity)
        elif detected_table.endswith('s') and detected_table[:-1] in table_names_lower:
            matched_table = table_names_lower[detected_table[:-1]]
            print(f"[DYNAMIC] ✅ Singularization match found: {detected_table} → {matched_table}")
        
        # 4. Substring match - find table containing the detected word
        else:
            for table in table_names:
                if detected_table in table.lower() or table.lower() in detected_table:
                    matched_table = table
                    print(f"[DYNAMIC] ✅ Substring match found: {detected_table} → {matched_table}")
                    break
        
        if matched_table:
            target_table = matched_table
            print(f"[DYNAMIC] Selected target table: {target_table}")
        else:
            print(f"[DYNAMIC] ❌ Could not match '{detected_table}' to any table")
            print(f"[DYNAMIC]    LLM response: '{llm_response}'")
            print(f"[DYNAMIC]    Available tables: {table_names}")
        
        # If we identified a target table, build intelligent schema context
        if target_table:
            print(f"[ADAPTIVE] Building intelligent schema context for: {target_table}")
            
            # Create analyzer and build intelligent context
            analyzer = AdaptiveSchemaAnalyzer(schema_catalog=schema_catalog)
            intelligent_context = await analyzer.build_intelligent_llm_context(target_table, query)
            
            if intelligent_context:
                intelligent_schema_context = intelligent_context
                print(f"[ADAPTIVE] Built intelligent schema context for {target_table}")
                print(f"[ADAPTIVE] Context includes column purposes and sample values for better LLM understanding")
        else:
            print(f"[ADAPTIVE] Could not determine target table - using generic schema context")
    
    except Exception as e:
        print(f"[WARN] Dynamic schema analysis failed: {e}")
        print(f"[WARN] Falling back to generic schema context")
        import traceback
        traceback.print_exc()
        intelligent_schema_context = None
    
    # Use intelligent context if available, otherwise fallback to generic schema
    schema_for_prompt = intelligent_schema_context if intelligent_schema_context else schema_info
    
    # ============================================================================
    # Build grounding constraints for LLM prompt (Principles 2 & 3)
    # ============================================================================
    grounding_constraints = ""
    
    # Principle 2: Filter Value Grounding - Inject valid enum values
    if filter_value_grounding:
        try:
            filter_constraint = filter_value_grounding.generate_filter_constraint_for_llm()
            if filter_constraint:
                grounding_constraints += filter_constraint + "\n\n"
                print(f"[GROUNDING] Injected filter value constraints into prompt")
        except Exception as e:
            print(f"[GROUNDING] Warning: Could not generate filter constraints: {e}")
    
    # Principle 3: Relationship-Aware Table Selection - Inject join guidance (DYNAMIC - ZERO HARDCODING)
    if join_graph_builder and schema_grounding:
        try:
            join_guidance = "\nJOIN PATH GUIDANCE (Use for relationships):\n"
            join_guidance += "=" * 60 + "\n"
            
            # Discover join paths DYNAMICALLY from the join graph (NO hardcoded pairs)
            available_tables = list(join_graph_builder.adjacency.keys()) if hasattr(join_graph_builder, 'adjacency') else []
            
            # Generate common_pairs dynamically from available tables
            common_pairs = []
            for i, source in enumerate(available_tables):
                for target in available_tables[i+1:]:
                    common_pairs.append((source, target))
            
            for source, target in common_pairs:
                # Only add if both tables exist
                if source in join_graph_builder.adjacency and target in join_graph_builder.adjacency:
                    try:
                        join_path = join_graph_builder.find_join_path(source, target)
                        if join_path:
                            join_clause = " AND ".join([step['join_clause'] for step in join_path.steps])
                            if join_clause:
                                join_guidance += f"\n✓ {source} + {target}:\n  JOIN clause: {join_clause}\n"
                    except:
                        pass
            
            if len(join_guidance) > 150:  # Only add if we have meaningful content
                grounding_constraints += join_guidance + "\n"
                print(f"[GROUNDING] Injected join path guidance into prompt")
        except Exception as e:
            print(f"[GROUNDING] Warning: Could not generate join guidance: {e}")
    
    # Build system prompt - SIMPLIFIED and STRICT for reliability
    using_adaptive_context = intelligent_schema_context is not None
    
    system_prompt = (
        "OUTPUT ONLY: Pure SQL query. One line. No explanations.\n"
        "START with: SELECT\n\n"
        
        "╔════════════════════════════════════════════════════════════════════╗\n"
        "║              SQL MODIFIER RULES - KEYWORD DRIVEN                   ║\n"
        "╚════════════════════════════════════════════════════════════════════╝\n\n"
        
        "1. WHERE CLAUSE - CRITICAL LOGIC:\n"
        "   USE WHERE if: User mentions ANY specific value, filter, or characteristic\n"
        "   ► IMPORTANT: 'show all X' where X is a FILTER = needs WHERE, not DISTINCT\n"
        "   ► ONLY use DISTINCT (no WHERE) if asking for UNIQUE VALUES of a single column\n"
        "   \n"
        "   Filter Pattern (USE WHERE):\n"
        "   - 'show all [type] [entity]' → WHERE type_column = 'TYPE_VALUE' (type is the filter)\n"
        "   - 'get [status] [entity]' → WHERE status = 'STATUS_VALUE' (status is the filter)\n"
        "   - 'show [entity] from [location]' → WHERE location_column = 'LOCATION'\n"
        "   \n"
        "   Non-Filter Pattern (NO WHERE, use DISTINCT):\n"
        "   - 'what are all [column_name]' → SELECT DISTINCT column_name (asking for list of values)\n"
        "   - 'get all different [column_name]' → SELECT DISTINCT column_name\n"
        "   - 'show me all available [column_name]' → SELECT DISTINCT column_name\n\n"
        
        "2. DISTINCT:\n"
        "   USE DISTINCT ONLY if:\n"
        "   a) User asks for UNIQUE VALUES of a SINGLE COLUMN\n"
        "   b) NOT asking for filtered records\n"
        "   c) NOT selecting entire rows (*)\n"
        "   \n"
        "   Valid DISTINCT queries (pattern-based):\n"
        "   - 'what are all the [column]' → SELECT DISTINCT column FROM table\n"
        "   - 'show the different [column]' → SELECT DISTINCT column FROM table\n"
        "   - 'list all [column] values' → SELECT DISTINCT column FROM table\n"
        "   \n"
        "   DO NOT use DISTINCT if:\n"
        "   - User mentions a filter/criterion: any type, status, location, or value\n"
        "   - User asks for full records: 'show me', 'get details', 'list records'\n"
        "   - Any WHERE condition could apply\n\n"
        
        "3. LIMIT:\n"
        "   DEFAULT SAFETY LIMIT: If not specified by LLM, a safety validator will add LIMIT 500\n"
        "   USE EXPLICIT LIMIT ONLY if: User explicitly asks for 'top N', 'first N', 'limit N', or specific count\n"
        "   Keywords that trigger LIMIT: 'top N', 'limit N', 'first N', 'show me N', 'get top', 'last N'\n"
        "   DO NOT add LIMIT: for 'all', 'list', 'get', 'show' without a specific number\n"
        "   Pattern examples:\n"
        "   - 'top N [entity]' → ... LIMIT N\n"
        "   - 'get all [entity]' → NO LIMIT (validator will add default LIMIT 500)\n"
        "   - 'first N [entity]' → ... LIMIT N\n"
        "   - 'show all [type] [entity]' → NO LIMIT (validator will add default LIMIT 500)\n\n"
        
        "4. COUNT(*) vs SELECT *:\n"
        "   USE COUNT(*) if: User asks 'how many', 'count', 'total', 'number of'\n"
        "   USE SELECT * if: User asks for 'details', 'show', 'get me', 'display', 'list', or full records\n"
        "   Pattern examples:\n"
        "   - 'how many [entity]' → SELECT COUNT(*) FROM table\n"
        "   - 'show me [entity] details' → SELECT * FROM table\n\n"
        
        "5. GROUP BY & ORDER BY:\n"
        "   GROUP BY: Only if user asks for 'group', 'grouped by', 'per', 'each', or 'summary'\n"
        "   ORDER BY: Only if user asks for 'sort', 'sorted by', 'in order', 'ascending', 'descending'\n"
        "   DO NOT add these unless explicitly requested\n\n"
        
        "6. COLUMN SELECTION (CRITICAL - DYNAMIC BASED ON QUERY):\n"
        "   ⚠️ Match column count to query specificity - NOT table structure\n\n"
        
        "   USE SELECT * when:\n"
        "   - User asks for 'details', 'all details', 'information', 'full record', 'complete data'\n"
        "   - User asks for 'all [entity]' like 'all records', 'all items', 'all entries'\n"
        "   - User wants comprehensive data regardless of filters applied\n"
        "   - Pattern: 'get all [entity] details' → SELECT * FROM table\n"
        "             'show me all [type] [entity]' → SELECT * FROM table WHERE ...\n"
        "             'get complete information about [entity]' → SELECT * FROM table\n\n"
        
        "   USE SPECIFIC COLUMNS when:\n"
        "   - User asks for one/few specific fields by name\n"
        "   - User asks for UNIQUE VALUES: 'what are all [column]' → SELECT DISTINCT column\n"
        "   - Pattern: 'get [entity] [column]' → SELECT column FROM table (NOT all columns)\n\n"
        
        "   RULE: 'all X' referring to entity records with optional filter = ALL COLUMNS\n"
        "   RULE: 'all X' referring to unique values of a property = DISTINCT + specific column\n"
        "   Pattern examples:\n"
        "   - WRONG: 'list all [column]' with SELECT col1, col2, col3 (over-selection)\n"
        "   - RIGHT: 'list all [column]' with SELECT column FROM table\n"
        "   - WRONG: 'show all [type] details' with SELECT col1, col2 (incomplete)\n"
        "   - RIGHT: 'show all [type] details' with SELECT * FROM table WHERE type='VALUE'\n\n"
        
        f"{grounding_constraints}"
        f"DATABASE SCHEMA:\n{schema_for_prompt}\n\n"
        
        f"DATABASE DIALECT: {dialect_engine.get_dialect_prompt_hint()}\n\n"
        
        "CRITICAL RULES:\n"
        "• One line SQL only\n"
        "• Use exact column names and values from schema\n"
        "• READ THE QUERY CAREFULLY: identify if asking for VALUES or RECORDS\n"
        "  - VALUES = 'what are', 'list', 'show me the', 'all different' → DISTINCT\n"
        "  - RECORDS = 'show me', 'get me', with filter word → WHERE clause\n"
        "• No explanations, no markdown, no extra text\n"
        "• First character MUST be: S (from SELECT)\n"
        "• COLUMN MINIMIZATION: Generate SELECT with minimum necessary columns\n"
        "  - Do NOT select all table columns; choose only what's relevant\n"
        "  - 'show [column]' = SELECT column (not SELECT all columns)\n"
        "  - 'get details' = SELECT * (all fields are relevant)\n"
    )
    
    # Add minimal adaptive schema guidance if available
    if using_adaptive_context:
        system_prompt += (
            "TARGET TABLE: Use the columns shown above. "
            "MATCHING STRATEGY: (1) Check if user is asking for 'all' or specific filters. "
            "(2) If 'all': NO WHERE clause, use DISTINCT if appropriate. "
            "(3) If specific filter: Add WHERE with exact values from schema samples. "
            "(4) DO NOT invent column names or values. Use only what's shown in the schema.\n"
        )
    
    # NEW: Add semantic context guidance if available (from semantic orchestrator)
    if semantic_context and semantic_context.retrieval_context:
        retrieval = semantic_context.retrieval_context
        num_tables = len(retrieval.top_tables)
        num_columns = sum(len(cols) for cols in retrieval.top_columns_per_table.values())
        print(f"[SEMANTIC] Using semantic context with {num_tables} tables, {num_columns} columns")
        
        semantic_guidance = "\n╔════════════════════════════════════════════════════════════════════╗\n"
        semantic_guidance += "║        SEMANTIC QUERY PLANNING - AI-RETRIEVED SCHEMA CONTEXT       ║\n"
        semantic_guidance += "╚════════════════════════════════════════════════════════════════════╝\n\n"
        
        if retrieval.top_tables:
            semantic_guidance += f"RELEVANT TABLES (by relevance):\n"
            for i, table_name in enumerate(retrieval.top_tables[:5], 1):
                semantic_guidance += f"  {i}. {table_name}\n"
            semantic_guidance += "\n"
        
        if retrieval.top_columns_per_table:
            semantic_guidance += f"RELEVANT COLUMNS (by relevance - USE ONLY WHAT'S NEEDED):\n"
            col_count = 0
            for table_name, column_names in retrieval.top_columns_per_table.items():
                for col_name in column_names[:3]:  # Limit to 3 per table
                    col_count += 1
                    semantic_guidance += f"  {col_count}. {table_name}.{col_name}\n"
                    if col_count >= 6:  # Max 6 columns total
                        break
            semantic_guidance += "\nCOLUMN SELECTION STRATEGY: Choose only columns relevant to the query.\n"
            semantic_guidance += "Do NOT select all columns - pick the minimum needed columns for the query.\n\n"
        
        if semantic_context.plan:
            semantic_guidance += f"SEMANTIC PLAN:\n"
            plan = semantic_context.plan
            semantic_guidance += f"  Primary table: {plan.primary_table}\n"
            if plan.joins:
                semantic_guidance += f"  Joins: {', '.join(plan.joins)}\n"
            if plan.filters:
                semantic_guidance += f"  Suggested filters: {', '.join(plan.filters)}\n"
            semantic_guidance += "\n"
        
        semantic_guidance += "STRATEGY: Prioritize tables and columns from above when building SQL.\n\n"
        system_prompt += semantic_guidance
    
    # NEW: Add intelligently discovered value-to-column mappings (from intelligent_followup_value_mapper)
    if followup_context and hasattr(followup_context, 'value_mappings') and followup_context.value_mappings:
        print(f"[SQL_GEN] 🧠 Applying {len(followup_context.value_mappings)} discovered value-to-column mappings")
        value_mappings_guidance = "\n╔════════════════════════════════════════════════════════════════════╗\n"
        value_mappings_guidance += "║    INTELLIGENTLY DISCOVERED VALUE-TO-COLUMN MAPPINGS               ║\n"
        value_mappings_guidance += "║    (Automatically found from user query and database analysis)       ║\n"
        value_mappings_guidance += "╚════════════════════════════════════════════════════════════════════╝\n\n"
        
        for i, mapping in enumerate(followup_context.value_mappings[:5], 1):  # Show top 5 mappings
            user_val = mapping.get('user_value', '?')
            col_name = mapping.get('column', '?')
            confidence = mapping.get('confidence', 0)
            strategy = mapping.get('strategy', 'unknown')
            reasoning = mapping.get('reasoning', 'Intelligent discovery')
            
            value_mappings_guidance += f"{i}. User value '{user_val}' → Column '{col_name}'\n"
            value_mappings_guidance += f"   Confidence: {confidence:.0%} | Strategy: {strategy}\n"
            value_mappings_guidance += f"   Reasoning: {reasoning}\n\n"
        
        value_mappings_guidance += "INSTRUCTION: Use these mappings to build accurate WHERE clauses.\n"
        value_mappings_guidance += "These values have been intelligently discovered and matched to columns.\n\n"
        
        system_prompt += value_mappings_guidance
    
    # Extract ANY mentioned values from the query (not just a domain-specific code)
    # This helps LLM use the actual user values, not example values
    mentioned_values = []
    
    # Extract codes/IDs (CUST0000001, INV123, etc.)
    codes = re.findall(r'\b[A-Z]{3,6}\d+\b', query)
    mentioned_values.extend(codes)
    
    # ✅ NEW ARCHITECTURE: Column selection is NOW LLM-DRIVEN via semantic orchestrator
    # Instead of hardcoded keywords, we use the semantic plan's column_selection analysis
    # This makes it dynamic, database-agnostic, and driven by LLM semantic understanding
    
    column_selection_intent = None
    wants_all_columns = False  # Will be set based on semantic plan
    is_get_all_query = False    # Will be set based on semantic plan
    is_count_query = False      # Will be set based on semantic plan
    
    # Extract semantic guidance from plan if available
    print(f"\n[DEBUG] Checking semantic_context for column selection...")
    print(f"  semantic_context type: {type(semantic_context)}")
    print(f"  semantic_context is None: {semantic_context is None}")
    if semantic_context:
        print(f"  semantic_context has 'plan' attr: {hasattr(semantic_context, 'plan')}")
        if hasattr(semantic_context, 'plan'):
            print(f"  semantic_context.plan type: {type(semantic_context.plan)}")
            print(f"  semantic_context.plan is None: {semantic_context.plan is None}")
            if semantic_context.plan:
                print(f"  plan has 'column_selection' attr: {hasattr(semantic_context.plan, 'column_selection')}")
                if hasattr(semantic_context.plan, 'column_selection'):
                    print(f"  plan.column_selection value: {semantic_context.plan.column_selection}")
    
    if semantic_context and semantic_context.plan and hasattr(semantic_context.plan, 'column_selection'):
        if semantic_context.plan.column_selection:
            column_selection_intent = semantic_context.plan.column_selection
            from .query_plan_generator import ColumnSelectionIntent
            
            print(f"\n[SEMANTIC] ✅ Using LLM-determined column selection from plan:")
            intent_val = column_selection_intent.intent.value if hasattr(column_selection_intent.intent, 'value') else str(column_selection_intent.intent)
            print(f"  Intent: {intent_val}")
            print(f"  Reasoning: {column_selection_intent.reasoning}")
            print(f"  Confidence: {column_selection_intent.confidence:.2f}\n")
            
            # Map semantic intent to our query processing flags
            if column_selection_intent.intent == ColumnSelectionIntent.ALL_COLUMNS:
                wants_all_columns = True
            elif column_selection_intent.intent == ColumnSelectionIntent.DISTINCT_VALUES:
                is_get_all_query = True
            elif column_selection_intent.intent == ColumnSelectionIntent.COUNT_ONLY:
                is_count_query = True
    
    # Fallback: If no semantic analysis available, use basic heuristic
    if not column_selection_intent:
        print("[WARN] ❌ No semantic column analysis available, using fallback heuristic")
        # Very basic fallback - prefer all columns unless specific columns mentioned
        if any(word in query.lower() for word in ["count", "how many", "total"]):
            is_count_query = True
        elif any(word in query.lower() for word in ["distinct", "unique", "what are all", "list all different"]):
            is_get_all_query = True
        else:
            wants_all_columns = True  # Default to all columns for safety
    
  
    # SEMANTIC INTENT MAPPER: Extract keywords dynamically from user query
    # No hardcoded domain terms - let LLM infer from schema context
    semantic_keywords = []
    
    # Dynamic keyword extraction - no domain-specific hardcoding
    # The LLM will understand user intent from context and schema
    # Keywords and values are extracted generically and matched against actual database values
    
    # Extract quoted strings (values in quotes)
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", query)
    mentioned_values.extend(quoted)
    
    # Extract all-caps words (likely domain values)
    uppercased = re.findall(r'\b([A-Z]{4,})\b', query)
    mentioned_values.extend(uppercased)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_values = []
    for val in mentioned_values:
        if val not in seen and len(val) > 0:
            seen.add(val)
            unique_values.append(val)
    
    # Build user message with extracted context
    context_lines = []
    
    # KEEP IT SIMPLE: Only add essential context
    if intelligent_schema_context:
        context_lines.append(intelligent_schema_context)
        context_lines.append("")
    
    # ✅ NEW: Use semantic plan guidance instead of hardcoded checks
    if column_selection_intent:
        from .query_plan_generator import ColumnSelectionIntent
        
        if column_selection_intent.intent == ColumnSelectionIntent.ALL_COLUMNS:
            context_lines.append("[SEMANTIC] User wants ALL COLUMNS")
            context_lines.append("→ USE: SELECT * FROM table (all columns)")
            context_lines.append("→ If filter keywords present, add WHERE clause")
            context_lines.append("→ Confidence: {:.1%}".format(column_selection_intent.confidence))
            context_lines.append("")
        elif column_selection_intent.intent == ColumnSelectionIntent.DISTINCT_VALUES:
            context_lines.append("[SEMANTIC] User wants DISTINCT/UNIQUE values")
            context_lines.append("→ USE: SELECT DISTINCT column_name")
            context_lines.append("→ Only return unique values, not all records")
            context_lines.append("")
        elif column_selection_intent.intent == ColumnSelectionIntent.COUNT_ONLY:
            context_lines.append("[SEMANTIC] User wants COUNT of records")
            context_lines.append("→ USE: SELECT COUNT(*)")
            context_lines.append("")
        elif column_selection_intent.intent == ColumnSelectionIntent.SPECIFIC_COLUMNS:
            cols_str = ", ".join(column_selection_intent.requested_columns[:5])
            context_lines.append(f"[SEMANTIC] User wants SPECIFIC COLUMNS: {cols_str}")
            context_lines.append(f"→ USE: SELECT {cols_str}")
            context_lines.append("")
    
    # ADD QUERY INTENT DETECTION: "Get All Values" queries (legacy, for safety)
    if is_get_all_query and not column_selection_intent:
        context_lines.append("Query Intent: User wants UNIQUE VALUES of a column (not all records)")
        context_lines.append("→ Use: SELECT DISTINCT column_name")
        context_lines.append("→ DO NOT use WHERE clause (unless asked for filtered unique values)")
        context_lines.append("")
    
    # ADD SEMANTIC INTENT MAPPER OUTPUT - MINIMAL, KEYWORD FOCUSED
    if semantic_keywords:
        context_lines.append("User Intent: Asking for items with specific characteristics")
        for intent in semantic_keywords:
            context_lines.append(f"- '{intent['keyword'].upper()}' appears in query")
        context_lines.append("→ Match these keywords to exact values in 'Sample values in DB' from schema above")
        context_lines.append("")
    
    # DYNAMIC FOLLOW-UP MANAGEMENT: Put followup context AFTER adaptive schema
    if followup_context and followup_context.is_followup:
        followup_prompt = followup_context.to_prompt_section()
        if followup_prompt:
            followup_type_val = followup_context.followup_type.value if hasattr(followup_context.followup_type, 'value') else str(followup_context.followup_type)
            print(f"[DEBUG] Including follow-up context ({followup_type_val})")
            context_lines.append(followup_prompt)
            context_lines.append("")
    
    # Add user query at the end
    if unique_values:
        context_lines.append(f"Values in query: {', '.join(unique_values)}")
    
    if identifiers_context:
        context_lines.append(identifiers_context)
    
    context_lines.append(f"Query: {query}")
    
    user_message = "\n".join(context_lines)
    if conversation_history:
        user_message = f"{conversation_history}\n\n" + user_message
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    # DEBUG: Log the exact messages being sent to LLM for troubleshooting
    print(f"\n[DEBUG] ===== LLM MESSAGE CONSTRUCTION =====")
    print(f"[DEBUG] System Prompt Length: {len(system_prompt)} chars")
    print(f"[DEBUG] System Prompt (first 500 chars):\n{system_prompt[:500]}...\n")
    print(f"[DEBUG] User Message Length: {len(user_message)} chars")
    print(f"[DEBUG] User Message (first 500 chars):\n{user_message[:500]}...\n")
    print(f"[DEBUG] Full User Message:\n{user_message}\n")
    print(f"[DEBUG] Calling LLM with {len(messages)} messages (temperature=0.0 for deterministic SQL)...")
    
    # Use temperature 0.0 for SQL generation - we want the most deterministic output possible
    sql = await llm.call_llm(messages, stream=False, max_tokens=512, temperature=0.0)
    
    # DEBUG: Log the raw response from LLM
    print(f"\n[DEBUG] ===== LLM RESPONSE RECEIVED =====")
    print(f"[DEBUG] Raw response length: {len(sql)} chars")
    print(f"[DEBUG] Raw response:\n{sql}\n")
    
    # VALIDATION & RETRY: If LLM returns non-SQL (chat response), retry with fallback prompt
    if 'SELECT' not in sql.upper():
        print(f"[WARN] LLM returned non-SQL response, attempting retry with fallback prompt...")
        print(f"[DEBUG] Original response: {sql[:100]}")
        
        # Fallback: Ultra-minimal prompt that forces SQL
        fallback_prompt = (
            "OUTPUT ONLY SQL. No text. No explanation. Just SQL.\n"
            "Your response must be ONE LINE starting with SELECT and ending with a number.\n"
            f"Database schema:\n{schema_for_prompt}\n\n"
            f"User request: {query}\n\n"
            "Generate ONLY the SQL query. Nothing else."
        )
        
        fallback_messages = [
            {"role": "system", "content": "You output only SQL queries. One line. SELECT...LIMIT."},
            {"role": "user", "content": fallback_prompt},
        ]
        
        print(f"[DEBUG] Retrying LLM with fallback prompt...")
        sql = await llm.call_llm(fallback_messages, stream=False, max_tokens=512, temperature=0.0)
        print(f"[DEBUG] Retry response: {sql[:100]}\n")
        
        # If still no SELECT after retry, we'll use fallback
        if 'SELECT' not in sql.upper():
            print(f"[ERROR] LLM retry also failed (no SELECT found), proceeding to extraction + fallback logic")
    
    # CRITICAL: Clean the response to ensure it's ONLY SQL
    # Remove any explanations, markdown, or extra text
    sql_cleaned = sql.strip()
    
    print(f"[DEBUG] ===== SQL EXTRACTION STARTED =====")
    print(f"[DEBUG] Step 1: Strip whitespace\n  Length: {len(sql_cleaned)}\n  Content: {sql_cleaned[:100]}\n")
    
    # Step 1: Try to extract from markdown code blocks first
    code_block_match = re.search(r"```(?:sql)?\s*\n?(.*?)\n?```", sql_cleaned, re.IGNORECASE | re.DOTALL)
    if code_block_match:
        sql_cleaned = code_block_match.group(1).strip()
        print(f"[DEBUG] Step 2: Extracted from markdown code block\n  Length: {len(sql_cleaned)}\n  Content: {sql_cleaned[:100]}\n")
    else:
        print(f"[DEBUG] Step 2: No markdown code block found\n")
        # Remove markdown code blocks anyway (in case they format differently)
        sql_cleaned = re.sub(r"^```+\s*(?:sql)?\s*\n?", "", sql_cleaned, flags=re.IGNORECASE)
        sql_cleaned = re.sub(r"\n?```+\s*$", "", sql_cleaned, flags=re.IGNORECASE)
        sql_cleaned = sql_cleaned.strip()
    
    # Step 2: Check if we have SELECT keyword at all
    if 'SELECT' not in sql_cleaned.upper():
        print(f"[DEBUG] Step 3: NO SELECT keyword found - attempting aggressive extraction\n")
        print(f"[DEBUG] Raw response: {sql[:200]}\n")
        
        # Try to find SELECT somewhere in the original response (case-insensitive)
        select_index = sql.upper().find('SELECT')
        if select_index >= 0:
            # Found SELECT somewhere - extract from there
            sql_cleaned = sql[select_index:].strip()
            print(f"[DEBUG] Found SELECT at position {select_index}, extracting from there\n")
            print(f"[DEBUG] Extracted: {sql_cleaned[:100]}\n")
        else:
            # No SELECT keyword anywhere - LLM completely failed
            print(f"[WARN] CRITICAL: LLM returned no SQL at all!")
            print(f"[WARN] Response content: {sql[:200]}")
            print(f"[DEBUG] Step 3: Using intelligent fallback SQL\n")
            
            # Detect database dialect
            bind = session.get_bind()
            dialect_name = bind.dialect.name if hasattr(bind, 'dialect') else 'postgresql'
            dialect_engine = DialectSqlEngine(dialect_name)
            
            # Use the target_table determined by LLM earlier (zero-hardcoding!)
            # If we couldn't determine it, we cannot generate fallback
            if not target_table:
                print("[WARN] Cannot generate fallback SQL - no target table determined")
                return "SELECT 1 AS error_no_table_detected"  # Signal error condition
            
            fallback_table = target_table
            
            # Build intelligent fallback SQL using dialect-aware qualification
            # Get the configured schema (if applicable for this dialect)
            configured_schema = settings.postgres_schema if hasattr(settings, 'postgres_schema') else None
            
            # Use generic identifier matching instead of hardcoded entity_code logic
            # This allows filtering by any identifier found in the query
            if extracted_ids:
                # Try to use first extracted identifier for filtering
                first_id = extracted_ids[0] if extracted_ids else None
                if first_id:
                    # Build query with the generic identifier using dialect-aware qualification
                    qualified_table = dialect_engine.qualify_table(fallback_table, configured_schema)
                    fallback_sql = (
                        f"SELECT * FROM {qualified_table} "
                        f"WHERE reference_id LIKE '%{first_id['value']}%' "
                    )
                    # Apply dialect-specific row limiting
                    fallback_sql = dialect_engine.enforce_row_limit(fallback_sql, 500)
                    print(f"[DEBUG] Fallback with identifier {first_id['type']}: {fallback_sql[:80]}\n")
                else:
                    # Generic fallback using dialect-aware qualification
                    qualified_table = dialect_engine.qualify_table(fallback_table, configured_schema)
                    fallback_sql = f"SELECT * FROM {qualified_table}"
                    fallback_sql = dialect_engine.enforce_row_limit(fallback_sql, 500)
            else:
                # Generic fallback using dialect-aware qualification
                qualified_table = dialect_engine.qualify_table(fallback_table, configured_schema)
                fallback_sql = f"SELECT * FROM {qualified_table}"
                fallback_sql = dialect_engine.enforce_row_limit(fallback_sql, 500)
                print(f"[DEBUG] Generic fallback: {fallback_sql}\n")
            
            return fallback_sql
    
    print(f"[DEBUG] Step 3: SELECT keyword found\n")
    
    # Step 3: Remove common conversational prefixes
    conversational_prefixes = [
        r'^Sure,?\s+here\s+(?:are|is).*?:\s*',  # "Sure, here are the ... :"
        r"^Here(?:'s)?\s+(?:are|is).*?:\s*",  # "Here's the SQL:" or "Here are the results:"
        r'^Here[^:]*:\s*',  # More generic: Any "Here...:" pattern
        r'^The\s+(?:SQL|query|following).*?:\s*',  # "The SQL is:" or "The following query:"
        r'^I\s+(?:will\s+)?(?:generate|create|provide).*?:\s*',  # "I will generate:" or "I provide:"
        r'^This\s+query.*?:\s*',  # "This query will..." then colon
        r'^Certainly!.*?:\s*',  # "Certainly! Here's the query:" 
        r'^Based\s+on.*?:\s*',  # "Based on the schema..."
        r'^To\s+(?:fetch|get|retrieve).*?:\s*',  # "To fetch the records:"
        r'^\s*\|+\s*',  # Remove leading pipe characters
        r'\n\s*\|+\s*',  # Remove pipes after newlines
        r'^[\s\-]*',  # Remove leading dashes or whitespace
    ]
    
    for pattern in conversational_prefixes:
        sql_cleaned = re.sub(pattern, '', sql_cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    sql_cleaned = sql_cleaned.strip()
    print(f"[DEBUG] Step 4: Removed conversational prefixes\n  Length: {len(sql_cleaned)}\n  Content: {sql_cleaned[:100]}\n")
    
    # Step 4: Find the actual SELECT statement
    select_index = sql_cleaned.upper().find('SELECT')
    if select_index > 0:
        # Found SELECT somewhere in the middle - extract from there
        sql_cleaned = sql_cleaned[select_index:]
        print(f"[DEBUG] Step 5: Removed text before SELECT\n  Length: {len(sql_cleaned)}\n  Content: {sql_cleaned[:100]}\n")
    else:
        print(f"[DEBUG] Step 5: SELECT is already at start\n")
    
    # Step 5: Extract up to natural end boundaries (LIMIT, semicolon, or double newline)
    # PRIORITY: Look for complete SQL statements, not just first line
    
    # Strategy 1: Look for LIMIT clause (most reliable end marker for SELECT)
    limit_match = re.search(r'\bLIMIT\b\s+\d+', sql_cleaned, re.IGNORECASE)
    if limit_match:
        limit_end = limit_match.end()
        sql_cleaned = sql_cleaned[:limit_end].strip()
        print(f"[DEBUG] Step 6: Extracted up to LIMIT clause\n  Length: {len(sql_cleaned)}\n  Content: {sql_cleaned}\n")
    else:
        # Strategy 2: Look for semicolon (SQL statement terminator)
        semicolon_pos = sql_cleaned.find(';')
        if semicolon_pos > 10:  # Must have substantial content before semicolon
            sql_cleaned = sql_cleaned[:semicolon_pos].strip()
            print(f"[DEBUG] Step 6: Extracted up to semicolon\n  Length: {len(sql_cleaned)}\n  Content: {sql_cleaned}\n")
        else:
            # Strategy 3: Look for paragraph break (double newline)
            paragraph_pos = sql_cleaned.find('\n\n')
            if paragraph_pos > 10:
                sql_cleaned = sql_cleaned[:paragraph_pos].strip()
                print(f"[DEBUG] Step 6: Extracted up to double newline\n  Length: {len(sql_cleaned)}\n  Content: {sql_cleaned}\n")
            else:
                # Strategy 4: Look for explanation text patterns (Explanation:, Note:, etc.)
                explanation_pos = re.search(r'\b(?:Explanation|Note|This query|The above|In summary):', sql_cleaned, re.IGNORECASE)
                if explanation_pos:
                    sql_cleaned = sql_cleaned[:explanation_pos.start()].strip()
                    print(f"[DEBUG] Step 6: Extracted up to explanation text\n  Length: {len(sql_cleaned)}\n  Content: {sql_cleaned}\n")
                else:
                    # Strategy 5: Multi-line SQL detection - capture complete JOINs and WHERE clauses
                    # A complete SQL should have balanced parentheses and keywords on separate lines
                    lines = sql_cleaned.split('\n')
                    sql_lines = []
                    for line in lines:
                        line = line.strip()
                        if line:  # Skip empty lines
                            sql_lines.append(line)
                            # Check if this line ends the SQL statement logically
                            # Common end markers: ends with number (LIMIT 500), semicolon, or specific keywords
                            if (re.search(r'\d+\s*$', line) or  # Ends with number (LIMIT value)
                                line.endswith(';') or
                                line.upper().startswith('--')):  # Comment line
                                break
                    
                    if sql_lines:
                        sql_cleaned = '\n'.join(sql_lines).strip()
                        print(f"[DEBUG] Step 6: Extracted multi-line SQL ({len(sql_lines)} lines)\n  Length: {len(sql_cleaned)}\n  Content: {sql_cleaned[:100]}...\n")
                    else:
                        print(f"[DEBUG] Step 6: No clear boundary found, using as-is (up to 500 chars)\n")
                        sql_cleaned = sql_cleaned[:500].strip()
    
    # Step 6: Remove any trailing explanation text
    explanation_patterns = [
        r'\bNote:\s*.*$',
        r'\bExplanation:\s*.*$',
        r'\bThis query.*$',
        r'\bThis SQL.*$',
        r'\bThe above query.*$',
        r'\bWHERE clause.*$',
        r'\d+[\.\)]\s+(?:SELECT|FROM|WHERE|JOIN).*$',
    ]
    for pattern in explanation_patterns:
        if re.search(pattern, sql_cleaned, re.IGNORECASE | re.MULTILINE):
            sql_cleaned = re.split(pattern, sql_cleaned, flags=re.IGNORECASE | re.MULTILINE)[0].strip()
    
    print(f"[DEBUG] Step 7: Removed trailing explanation\n  Length: {len(sql_cleaned)}\n  Content: {sql_cleaned[:100]}\n")
    
    # Step 7: Clean up whitespace but preserve SQL structure
    # Replace multiple spaces with single space (but preserve newlines for now)
    sql_cleaned = ' '.join(sql_cleaned.split())
    
    print(f"[DEBUG] Step 8: Normalized whitespace\n  Length: {len(sql_cleaned)}\n  Content: {sql_cleaned[:100]}\n")
    
    # Final validation: ensure starts with SELECT
    if not sql_cleaned.upper().startswith('SELECT'):
        print(f"[ERROR] CRITICAL: After extraction, SQL still doesn't start with SELECT")
        print(f"[ERROR] Extracted content: {sql_cleaned[:100]}")
        raise ValueError(f"Failed to extract valid SQL from LLM response: {sql[:150]}")
    
    print(f"[DEBUG] ===== SQL EXTRACTION COMPLETE =====\n")
    
    sql = sql_cleaned
    
    # FIX: Convert type mismatches using DYNAMIC schema analysis
    # Instead of hardcoding column names, read the actual database schema
    print(f"[DEBUG] ===== DYNAMIC TYPE CONVERSION =====")
    
    try:
        from .type_converter import get_type_converter, initialize_type_converter
        
        # Initialize type converter on first use
        type_converter = await get_type_converter()
        if not type_converter.db_type:
            await initialize_type_converter(session)
        
        # Convert SQL based on actual schema
        converted_sql = await type_converter.convert_sql(sql, session)
        
        if converted_sql != sql:
            print(f"[OK] SQL type conversion applied\n  Before: {sql[:80]}\n  After: {converted_sql[:80]}\n")
            sql = converted_sql
        else:
            print(f"[DEBUG] No type conversions needed (schema analysis complete)\n")
    
    except Exception as e:
        print(f"[WARN] Type conversion failed (continuing anyway): {e}")
        # Continue without conversion rather than failing
    
    sql_cleaned = sql
    sql = sql_cleaned
    
    # VALIDATION: Check if user asked for ALL columns vs specific columns
    # If user didn't specify columns, fix SQL to use SELECT t.* or SELECT c.*
    # More precise: look for keywords that indicate column selection
    specific_column_patterns = [
        r'\bjust\s+(?:the\s+)?(?:amount|date|time|id|name|code|type)',  # "just the amount", "just date"
        r'\bonly\s+(?:the\s+)?(?:amount|date|time|id|name|code|type)',  # "only amount", "only the date"
        r'\b(?:amount|date|time)\s+and\s+(?:amount|date|time)',  # "amount and date"
        r'\bspecific\s+column',  # "specific columns"
        r'\bselect\s+which\s+column',  # "select which columns to show"
    ]
    user_asked_for_specific_cols = any(re.search(pattern, query, re.IGNORECASE) for pattern in specific_column_patterns)
    
    # Check current SELECT clause
    select_match = re.search(r'\bSELECT\s+(\w+\.\w+(?:\s*,\s*\w+\.\w+)*|\w+\.\*)', sql, re.IGNORECASE)
    if select_match:
        current_select = select_match.group(1)
        is_select_star = '.*' in current_select  # Matches t.* or c.*
        
        # If user didn't ask for specific columns but SQL has specific column list, fix it
        if not user_asked_for_specific_cols and not is_select_star:
            # Extract table alias (e.g., 't' from 't.txn_id, t.amount')
            table_alias_match = re.match(r'(\w+)\.', current_select)
            if table_alias_match:
                table_alias = table_alias_match.group(1)
                # Replace specific columns with SELECT t.* or SELECT c.*
                sql = re.sub(
                    r'\bSELECT\s+' + re.escape(table_alias) + r'\.\w+(?:\s*,\s*' + re.escape(table_alias) + r'\.\w+)*',
                    f'SELECT {table_alias}.*',
                    sql,
                    flags=re.IGNORECASE
                )
                print(f"[OK] Fixed column selection - using SELECT {table_alias}.* (all columns)")
    
    # CRITICAL FIX: Ensure table aliases used in SELECT are properly defined in FROM
    # Check if SELECT uses an alias (like t.*) but FROM doesn't define it
    select_aliases = re.findall(r'SELECT\s+(?:DISTINCT\s+)?(\w+)\.', sql, re.IGNORECASE)
    if select_aliases:
        # We found aliases used in SELECT (e.g., 't' from 'SELECT t.*')
        for alias in select_aliases:
            # Check if this alias is defined in a FROM or JOIN clause
            # Pattern: FROM table_name ALIAS or JOIN table_name ALIAS
            alias_defined = re.search(
                rf'(?:FROM|JOIN)\s+(?:\w+\.)?(\w+)(?:\s+(?:AS\s+)?{re.escape(alias)}\b|\s+{re.escape(alias)}\b)',
                sql,
                re.IGNORECASE
            )
            
            if not alias_defined:
                # The alias is used but not defined! Add it
                print(f"[WARN] Alias '{alias}' used in SELECT but not defined in FROM/JOIN")
                
                # Try to fix: Find the table name and add the alias
                # Match: FROM schema.table OR FROM table (without existing alias)
                from_match = re.search(
                    rf'FROM\s+(?:(\w+)\.)?(\w+)(?!\s+\w)\s*(?=WHERE|JOIN|ORDER|GROUP|LIMIT|;|$)',
                    sql,
                    re.IGNORECASE
                )
                if from_match:
                    # Replace FROM table with FROM table alias
                    original = from_match.group(0)
                    sql = sql.replace(original, f"{original} {alias}", 1)
                    print(f"[OK] Fixed: Added '{alias}' alias to FROM clause")
    
    # DEBUG: Log SQL before validation
    print(f"\n[DEBUG] SQL BEFORE validation ({len(sql)} chars): {sql}\n")
    
    # Validate SQL safety and enforce LIMIT using new architecture
    # Uses adapter-aware default schema (db-agnostic, not hardcoded postgres_schema)
    try:
        validator = SQLSafetyValidator(allowed_schemas=None, max_rows=500)  # Let adapter determine schema
        print(f"[INFO] Validating SQL (adapter-aware schemas): {sql[:150]}...")
        is_safe, error, safe_sql = validator.validate_and_rewrite(sql)
        if not is_safe:
            raise ValueError(f"SQL validation failed: {error}")
        sql = safe_sql  # Use the rewritten SQL with safety LIMIT enforced
        print(f"[OK] Validated and rewritten SQL: {sql[:150]}...")
    except ValueError as e:
        print(f"[ERROR] SQL Validation Error: {str(e)}")
        print(f"Raw SQL from LLM: {sql[:200]}")
        raise ValueError(f"Invalid SQL generated: {str(e)}")
    
    # DEBUG: Log SQL after validation
    print(f"\n[DEBUG] SQL AFTER validation with enforced LIMIT ({len(sql)} chars): {sql}\n")
    
    
    # CRITICAL: Replace [EXTRACT_FROM_USER_QUERY] placeholder with actual values from user input
    # The schema examples contain this placeholder, but we need to extract the REAL VALUES from the query
    if "[EXTRACT_FROM_USER_QUERY]" in sql:
        print("🔍 Detected [EXTRACT_FROM_USER_QUERY] placeholder - extracting actual value from user query...")
        
        # Pattern 1: Extract any alphanumeric code pattern (e.g., ABC123, ITEM001, CODE-456)
        # Generic pattern that matches common identifier formats without domain assumptions
        code_match = re.search(r'\b([A-Z]{2,}[-_]?\d+|\d+[-_]?[A-Z]{2,})\b', query, re.IGNORECASE)
        if code_match:
            code_value = code_match.group(1)
            # Replace the placeholder with the actual code
            sql = sql.replace("[EXTRACT_FROM_USER_QUERY]", code_value)
            print(f"[OK] Replaced [EXTRACT_FROM_USER_QUERY] with identifier code: {code_value}")
        else:
            # Pattern 2: Try to extract numeric IDs
            id_match = re.search(r'(?:id|code|with\s+id|number)?\s*[:#]?\s*(\d+)', query, re.IGNORECASE)
            if id_match:
                value = id_match.group(1)
                sql = sql.replace("[EXTRACT_FROM_USER_QUERY]", value)
                print(f"[OK] Replaced [EXTRACT_FROM_USER_QUERY] with extracted value: {value}")
            else:
                # Pattern 3: Try to extract any quoted string or significant word
                quote_match = re.search(r"['\"]([^'\"]+)['\"]", query)
                if quote_match:
                    value = quote_match.group(1)
                    sql = sql.replace("[EXTRACT_FROM_USER_QUERY]", value)
                    print(f"[OK] Replaced [EXTRACT_FROM_USER_QUERY] with quoted value: {value}")
                else:
                    print(f"[WARN] Could not extract value for [EXTRACT_FROM_USER_QUERY] placeholder")
                    print(f"   Query: {query}")
    
    # NOTE: Domain-specific JOIN logic removed - LLM handles semantic understanding
    # If user mentions an identifier code in their query, the LLM should include
    # proper JOIN and WHERE clauses in the generated SQL.
    # Any missing filters should be addressed through prompt engineering, not hardcoded post-processing.

    
    return sql
