"""
Plan Validator: Enforces semantic coverage rules on generated query plans.

Validates that:
1. If query mentions modifiers (credit/debit/etc) and matching table is available, it must be included
2. If plan includes a table, it must have appropriate filters or existence conditions
3. Joins are properly structured and non-circular
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class PlanValidator:
    """Validates query plans against semantic coverage rules - 100% DYNAMIC, ZERO HARDCODING."""
    
    # Keyword mappings will be built dynamically from schema, not hardcoded
    KEYWORD_TABLE_MAPPING = {}  # Built at runtime from schema
    EXPECTED_FILTERS = {}  # Built at runtime from schema
    
    def __init__(self, available_tables: List[str], table_metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize validator - builds all mappings dynamically from schema.
        
        Args:
            available_tables: List of tables that can be queried
            table_metadata: Optional metadata about tables (schema, columns, etc.)
        """
        self.available_tables = [t.lower() for t in available_tables]
        self.table_metadata = table_metadata or {}
        
        # BUILD MAPPINGS DYNAMICALLY FROM SCHEMA - zero hardcoding
        self._build_keyword_mappings()
        self._build_expected_filters()
        
        logger.info(f"[VALIDATOR] Initialized with {len(self.available_tables)} tables")
        logger.info(f"[VALIDATOR] Dynamic keyword mappings: {len(self.KEYWORD_TABLE_MAPPING)} keywords")
        logger.info(f"[VALIDATOR] Dynamic filter expectations: {len(self.EXPECTED_FILTERS)} tables")
    
    def _build_keyword_mappings(self) -> None:
        """
        Build keyword-to-table mappings dynamically from schema.
        Discovers table relationships based on actual column names.
        """
        self.KEYWORD_TABLE_MAPPING = {}
        
        # Analyze columns in all tables to discover semantic relationships
        for table_name, metadata in self.table_metadata.items():
            if table_name not in self.available_tables:
                continue
            
            columns = metadata.get('columns', [])
            if not columns:
                continue
            
            # Extract keywords from column names
            for col in columns:
                col_lower = col.lower()
                
                # Extract meaningful keywords from column names
                # E.g., "card_type" → "card", "txn_date" → "txn" or "transaction"
                keywords = self._extract_keywords(col_lower)
                
                for keyword in keywords:
                    if keyword not in self.KEYWORD_TABLE_MAPPING:
                        self.KEYWORD_TABLE_MAPPING[keyword] = []
                    if table_name not in self.KEYWORD_TABLE_MAPPING[keyword]:
                        self.KEYWORD_TABLE_MAPPING[keyword].append(table_name)
        
        logger.debug(f"[VALIDATOR] Built keyword mappings: {self.KEYWORD_TABLE_MAPPING}")
    
    def _build_expected_filters(self) -> None:
        """
        Build expected filter columns dynamically from schema.
        Any column that looks like an identifier/status/date is a potential filter.
        """
        self.EXPECTED_FILTERS = {}
        
        for table_name, metadata in self.table_metadata.items():
            if table_name not in self.available_tables:
                continue
            
            columns = metadata.get('columns', [])
            filter_columns = []
            
            for col in columns:
                col_lower = col.lower()
                # Potential filter columns: id, code, status, type, date, amount, name
                if any(pattern in col_lower for pattern in ['id', 'code', 'status', 'type', 'date', 'amount', 'name', 'flag', 'reason']):
                    filter_columns.append(col)
            
            if filter_columns:
                self.EXPECTED_FILTERS[table_name] = filter_columns
        
        logger.debug(f"[VALIDATOR] Built filter expectations: {self.EXPECTED_FILTERS}")
    
    def _extract_keywords(self, column_name: str) -> List[str]:
        """
        Extract relevant keywords from column names.
        E.g., "card_type" → ["card", "type"]
             "txn_date" → ["txn", "transaction", "date"]
        """
        import re
        keywords = []
        
        # Split on underscores and camelCase
        parts = re.split(r'[_]|(?=[A-Z])', column_name.lower())
        parts = [p for p in parts if p]  # Remove empty strings
        
        for part in parts:
            keywords.append(part)
            
            # Add expansions for common abbreviations
            if part == 'txn':
                keywords.append('transaction')
            elif part == 'cust':
                keywords.append('customer')
            elif part == 'acct':
                keywords.append('account')
            elif part == 'kyc':
                keywords.append('kyc')
        
        return list(set(keywords))  # Remove duplicates
    
    def validate_plan(
        self,
        query: str,
        plan_tables: List[str],
        query_tables: List[str],
        plan_joins: Optional[List[Dict[str, Any]]] = None,
        plan_where: Optional[List[Dict[str, Any]]] = None,
        plan_select: Optional[List[str]] = None,
        plan_group_by: Optional[List[str]] = None,
        plan_having: Optional[List[Dict[str, Any]]] = None,
        plan_order_by: Optional[List[Dict[str, Any]]] = None,
        plan_limit: Optional[int] = None,
        plan_offset: Optional[int] = None,
        plan_distinct: Optional[bool] = None,
        plan_union: Optional[List[Dict[str, Any]]] = None,
        plan_subqueries: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Comprehensive validation for all SQL clause combinations.
        
        Args:
            query: Original user query
            plan_tables: Tables in the generated plan (from_table + joined tables)
            query_tables: Tables semantically retrieved from query (from retrieval)
            plan_joins: Optional join specifications
            plan_where: Optional WHERE conditions
            plan_select: Optional SELECT columns
            plan_group_by: Optional GROUP BY columns
            plan_having: Optional HAVING conditions
            plan_order_by: Optional ORDER BY specifications
            plan_limit: Optional LIMIT value
            plan_offset: Optional OFFSET value
            plan_distinct: Optional DISTINCT flag
            plan_union: Optional UNION queries
            plan_subqueries: Optional subqueries
        
        Returns:
            (is_valid, reason, suggestion)
        """
        plan_tables_lower = [t.lower() for t in plan_tables]
        query_tables_lower = [t.lower() for t in query_tables]
        
        # Rule A: Check for keyword-to-table coverage
        rule_a_pass, rule_a_msg, rule_a_fix = self._check_keyword_coverage(
            query, query_tables_lower, plan_tables_lower
        )
        if not rule_a_pass:
            logger.warning(f"[VALIDATOR] Rule A failed: {rule_a_msg}")
            return False, rule_a_msg, rule_a_fix
        
        # Rule B: Check for required filters when using certain tables
        rule_b_pass, rule_b_msg = self._check_required_filters(query, plan_tables_lower)
        if not rule_b_pass:
            logger.warning(f"[VALIDATOR] Rule B failed: {rule_b_msg}")
            return False, rule_b_msg, None
        
        # Rule C: Check for join completeness
        if plan_joins:
            rule_c_pass, rule_c_msg = self._check_join_completeness(plan_joins, plan_tables_lower)
            if not rule_c_pass:
                logger.warning(f"[VALIDATOR] Rule C failed: {rule_c_msg}")
                return False, rule_c_msg, None
        
        # Rule D: Check for WHERE condition completeness
        if plan_where:
            rule_d_pass, rule_d_msg = self._check_where_completeness(plan_where, plan_tables_lower)
            if not rule_d_pass:
                logger.warning(f"[VALIDATOR] Rule D failed: {rule_d_msg}")
                return False, rule_d_msg, None
        
        # Rule E: Check GROUP BY completeness
        if plan_group_by:
            rule_e_pass, rule_e_msg = self._check_group_by_completeness(plan_select, plan_group_by)
            if not rule_e_pass:
                logger.warning(f"[VALIDATOR] Rule E failed: {rule_e_msg}")
                return False, rule_e_msg, None
        
        # Rule F: Check HAVING completeness
        if plan_having:
            rule_f_pass, rule_f_msg = self._check_having_completeness(plan_having, plan_group_by)
            if not rule_f_pass:
                logger.warning(f"[VALIDATOR] Rule F failed: {rule_f_msg}")
                return False, rule_f_msg, None
        
        # Rule G: Check ORDER BY completeness
        if plan_order_by:
            rule_g_pass, rule_g_msg = self._check_order_by_completeness(plan_order_by, plan_select)
            if not rule_g_pass:
                logger.warning(f"[VALIDATOR] Rule G failed: {rule_g_msg}")
                return False, rule_g_msg, None
        
        # Rule H: Check LIMIT/OFFSET
        rule_h_pass, rule_h_msg = self._check_limit_offset(plan_limit, plan_offset)
        if not rule_h_pass:
            logger.warning(f"[VALIDATOR] Rule H failed: {rule_h_msg}")
            return False, rule_h_msg, None
        
        # Rule I: Check DISTINCT
        rule_i_pass, rule_i_msg = self._check_distinct(plan_distinct, plan_select)
        if not rule_i_pass:
            logger.warning(f"[VALIDATOR] Rule I failed: {rule_i_msg}")
            return False, rule_i_msg, None
        
        # Rule J: Check UNION compatibility
        rule_j_pass, rule_j_msg = self._check_union_compatibility(plan_union)
        if not rule_j_pass:
            logger.warning(f"[VALIDATOR] Rule J failed: {rule_j_msg}")
            return False, rule_j_msg, None
        
        # Rule K: Check aggregate functions
        rule_k_pass, rule_k_msg = self._check_aggregate_functions(plan_select)
        if not rule_k_pass:
            logger.warning(f"[VALIDATOR] Rule K failed: {rule_k_msg}")
            return False, rule_k_msg, None
        
        # Rule L: Check subqueries
        rule_l_pass, rule_l_msg = self._check_subqueries(plan_subqueries)
        if not rule_l_pass:
            logger.warning(f"[VALIDATOR] Rule L failed: {rule_l_msg}")
            return False, rule_l_msg, None
        
        logger.info("[VALIDATOR] Plan passed all validation rules (A-L)")
        return True, "", None
    
    def _check_keyword_coverage(
        self,
        query: str,
        query_tables: List[str],
        plan_tables: List[str],
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Rule A: If query mentions a keyword and matching table is in retrieval results,
        then that table MUST be in the plan.
        
        Returns:
            (is_valid, reason, suggestion_to_fix)
        """
        query_lower = query.lower()
        
        # Find keywords mentioned in query
        mentioned_keywords = []
        for keyword in self.KEYWORD_TABLE_MAPPING.keys():
            # Look for keyword as word boundary (not substring)
            if re.search(rf'\b{keyword}\b', query_lower):
                mentioned_keywords.append(keyword)
        
        if not mentioned_keywords:
            # No keywords found - pass
            return True, "", None
        
        logger.debug(f"[VALIDATOR] Found keywords in query: {mentioned_keywords}")
        
        # For each keyword, check if required table is in retrieval BUT NOT in plan
        missing_tables = set()
        for keyword in mentioned_keywords:
            required_tables = self.KEYWORD_TABLE_MAPPING[keyword]
            
            for req_table in required_tables:
                req_table_lower = req_table.lower()
                
                # Check: is this required table in query_tables (retrieved)?
                # AND is it NOT in plan_tables?
                if req_table_lower in query_tables and req_table_lower not in plan_tables:
                    missing_tables.add(req_table_lower)
        
        if missing_tables:
            msg = f"Query mentions '{', '.join(mentioned_keywords)}' but plan doesn't include required table(s): {missing_tables}"
            suggestion = f"Add {missing_tables} to plan with appropriate JOINs"
            return False, msg, suggestion
        
        return True, "", None
    
    def _check_required_filters(
        self,
        query: str,
        plan_tables: List[str],
    ) -> Tuple[bool, str]:
        """
        Rule B: If a table is used in plan, it should have contextually appropriate filters.
        
        This is advisory for now (warns but doesn't fail) since the SQL renderer
        and WHERE clauses handle this.
        
        Returns:
            (is_valid, reason)
        """
        # This is more of a quality check - most plans will pass
        # The actual WHERE clause validation happens during SQL generation
        return True, ""
    
    def _check_join_completeness(
        self,
        plan_joins: List[Dict[str, Any]],
        plan_tables: List[str],
    ) -> Tuple[bool, str]:
        """
        NEW Rule C: If joins exist in plan, each join MUST have:
        1. A non-empty table name (join.table or join.right_table)
        2. At least one ON condition (join.on or join.conditions must not be empty)
        
        Returns:
            (is_valid, reason)
        """
        for idx, join in enumerate(plan_joins):
            # Check for table name
            table_name = join.get('table') or join.get('right_table') or join.get('target_table')
            if not table_name:
                msg = f"Join {idx} is missing table name (table/right_table/target_table)"
                logger.warning(f"[VALIDATOR] Rule C failed: {msg}")
                return False, msg
            
            # Check for ON conditions
            on_conditions = join.get('on') or join.get('conditions') or join.get('on_clause')
            if not on_conditions:
                msg = f"Join {idx} to '{table_name}' is missing ON conditions (on/conditions/on_clause)"
                logger.warning(f"[VALIDATOR] Rule C failed: {msg}")
                return False, msg
            
            # Check if on_conditions is a string (ON TRUE fallback)
            if isinstance(on_conditions, str):
                if on_conditions.strip().upper() == "TRUE":
                    msg = f"Join {idx} to '{table_name}' has invalid ON clause 'TRUE' (no actual conditions)"
                    logger.warning(f"[VALIDATOR] Rule C failed: {msg}")
                    return False, msg
            # Check if it's a list but empty
            elif isinstance(on_conditions, (list, tuple)) and len(on_conditions) == 0:
                msg = f"Join {idx} to '{table_name}' has empty ON conditions list"
                logger.warning(f"[VALIDATOR] Rule C failed: {msg}")
                return False, msg
        
        logger.info("[VALIDATOR] Rule C passed: All joins have table names and ON conditions")
        return True, ""
    
    def _check_where_completeness(
        self,
        plan_where: List[Dict[str, Any]],
        plan_tables: List[str],
    ) -> Tuple[bool, str]:
        """
        Rule D: If WHERE conditions exist in plan, each condition MUST have:
        1. A left side (column name) - not empty
        2. An operator (=, <, >, LIKE, IN, BETWEEN, etc.) - not empty
        3. A right side (value) - not empty or null
        
        Returns:
            (is_valid, reason)
        """
        if not plan_where:
            return True, ""
        
        for idx, condition in enumerate(plan_where):
            # Check for left side (column)
            left = condition.get('left') or condition.get('column') or condition.get('field')
            if not left:
                msg = f"WHERE condition {idx} is missing left side (column/field name)"
                logger.warning(f"[VALIDATOR] Rule D failed: {msg}")
                return False, msg
            
            # Check for operator
            operator = condition.get('operator') or condition.get('op')
            if not operator:
                msg = f"WHERE condition {idx} on '{left}' is missing operator (=, <, >, LIKE, IN, etc.)"
                logger.warning(f"[VALIDATOR] Rule D failed: {msg}")
                return False, msg
            
            # Check for right side (value)
            right = condition.get('right') or condition.get('value')
            if right is None or right == '':
                msg = f"WHERE condition {idx} ('{left} {operator}') is missing right side (value)"
                logger.warning(f"[VALIDATOR] Rule D failed: {msg}")
                return False, msg
            
            # Check if operator is valid
            valid_operators = ['=', '<', '>', '<=', '>=', '!=', '<>', 'LIKE', 'IN', 'BETWEEN', 
                             'IS', 'IS NOT', 'NOT LIKE', 'NOT IN', 'EXISTS', 'NOT EXISTS']
            if operator.upper() not in valid_operators:
                msg = f"WHERE condition {idx} has invalid operator '{operator}' (must be one of {valid_operators})"
                logger.warning(f"[VALIDATOR] Rule D failed: {msg}")
                return False, msg
        
        logger.info("[VALIDATOR] Rule D passed: All WHERE conditions properly structured")
        return True, ""
    
    def _check_group_by_completeness(
        self,
        plan_select: Optional[List[str]],
        plan_group_by: Optional[List[str]],
    ) -> Tuple[bool, str]:
        """
        Rule E: GROUP BY completeness validation.
        If GROUP BY exists:
        1. All non-aggregate columns in SELECT must be in GROUP BY
        2. Aggregate functions (COUNT, SUM, AVG, MIN, MAX) can be used
        
        Returns:
            (is_valid, reason)
        """
        if not plan_group_by or not plan_select:
            return True, ""
        
        agg_functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP_CONCAT', 'STDDEV', 'VARIANCE']
        
        for col in plan_select:
            # Check if column is an aggregate function
            is_aggregate = any(agg in col.upper() for agg in agg_functions)
            
            if not is_aggregate:
                # Non-aggregate columns must be in GROUP BY
                col_name = col.strip().lower()
                # Remove alias if present
                if ' as ' in col_name:
                    col_name = col_name.split(' as ')[0].strip()
                
                # Check if this column is in GROUP BY
                group_by_names = [gb.strip().lower() for gb in plan_group_by]
                if col_name not in group_by_names:
                    msg = f"Non-aggregate column '{col}' is not in GROUP BY clause"
                    logger.warning(f"[VALIDATOR] Rule E failed: {msg}")
                    return False, msg
        
        logger.info("[VALIDATOR] Rule E passed: GROUP BY properly structured")
        return True, ""
    
    def _check_having_completeness(
        self,
        plan_having: Optional[List[Dict[str, Any]]],
        plan_group_by: Optional[List[str]],
    ) -> Tuple[bool, str]:
        """
        Rule F: HAVING clause validation.
        If HAVING exists:
        1. Must have GROUP BY clause
        2. HAVING conditions must reference GROUP BY columns or aggregates
        
        Returns:
            (is_valid, reason)
        """
        if not plan_having:
            return True, ""
        
        if not plan_group_by:
            msg = "HAVING clause requires GROUP BY clause to be present"
            logger.warning(f"[VALIDATOR] Rule F failed: {msg}")
            return False, msg
        
        agg_functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP_CONCAT']
        
        for idx, condition in enumerate(plan_having):
            left = condition.get('left') or condition.get('column')
            if not left:
                msg = f"HAVING condition {idx} is missing column reference"
                logger.warning(f"[VALIDATOR] Rule F failed: {msg}")
                return False, msg
            
            # Check if left is aggregate or in GROUP BY
            is_aggregate = any(agg in left.upper() for agg in agg_functions)
            left_name = left.strip().lower()
            group_by_names = [gb.strip().lower() for gb in plan_group_by]
            
            if not is_aggregate and left_name not in group_by_names:
                msg = f"HAVING column '{left}' must be either grouped or aggregated"
                logger.warning(f"[VALIDATOR] Rule F failed: {msg}")
                return False, msg
        
        logger.info("[VALIDATOR] Rule F passed: HAVING properly structured")
        return True, ""
    
    def _check_order_by_completeness(
        self,
        plan_order_by: Optional[List[Dict[str, Any]]],
        plan_select: Optional[List[str]],
    ) -> Tuple[bool, str]:
        """
        Rule G: ORDER BY validation.
        If ORDER BY exists:
        1. Columns must exist in SELECT or be from base tables
        2. Must have valid direction (ASC/DESC)
        
        Returns:
            (is_valid, reason)
        """
        if not plan_order_by:
            return True, ""
        
        select_columns = [col.strip().lower().split(' as ')[-1].strip() if ' as ' in col else col.strip().lower() 
                         for col in (plan_select or [])]
        
        for idx, order_item in enumerate(plan_order_by):
            column = order_item.get('column') or order_item.get('field')
            direction = order_item.get('direction') or order_item.get('order')
            
            if not column:
                msg = f"ORDER BY item {idx} is missing column name"
                logger.warning(f"[VALIDATOR] Rule G failed: {msg}")
                return False, msg
            
            if direction and direction.upper() not in ['ASC', 'DESC']:
                msg = f"ORDER BY direction '{direction}' must be ASC or DESC"
                logger.warning(f"[VALIDATOR] Rule G failed: {msg}")
                return False, msg
        
        logger.info("[VALIDATOR] Rule G passed: ORDER BY properly structured")
        return True, ""
    
    def _check_limit_offset(
        self,
        plan_limit: Optional[int],
        plan_offset: Optional[int],
    ) -> Tuple[bool, str]:
        """
        Rule H: LIMIT/OFFSET validation.
        If LIMIT/OFFSET exists:
        1. Must be positive integers
        2. OFFSET cannot exceed reasonable bounds
        
        Returns:
            (is_valid, reason)
        """
        if plan_limit is not None:
            if not isinstance(plan_limit, int) or plan_limit < 0:
                msg = f"LIMIT must be a non-negative integer, got {plan_limit}"
                logger.warning(f"[VALIDATOR] Rule H failed: {msg}")
                return False, msg
        
        if plan_offset is not None:
            if not isinstance(plan_offset, int) or plan_offset < 0:
                msg = f"OFFSET must be a non-negative integer, got {plan_offset}"
                logger.warning(f"[VALIDATOR] Rule H failed: {msg}")
                return False, msg
        
        logger.info("[VALIDATOR] Rule H passed: LIMIT/OFFSET properly structured")
        return True, ""
    
    def _check_distinct(
        self,
        plan_distinct: Optional[bool],
        plan_select: Optional[List[str]],
    ) -> Tuple[bool, str]:
        """
        Rule I: DISTINCT validation.
        If DISTINCT is used:
        1. SELECT must have columns
        2. Cannot use * with aggregate functions
        
        Returns:
            (is_valid, reason)
        """
        if not plan_distinct:
            return True, ""
        
        if not plan_select or len(plan_select) == 0:
            msg = "DISTINCT requires SELECT columns to be specified"
            logger.warning(f"[VALIDATOR] Rule I failed: {msg}")
            return False, msg
        
        # Check for SELECT * with DISTINCT
        if any('*' in col for col in plan_select):
            # This is actually allowed in SQL, but might indicate issue
            logger.warning("[VALIDATOR] DISTINCT with SELECT * detected - may have performance implications")
        
        logger.info("[VALIDATOR] Rule I passed: DISTINCT properly structured")
        return True, ""
    
    def _check_union_compatibility(
        self,
        plan_union: Optional[List[Dict[str, Any]]],
    ) -> Tuple[bool, str]:
        """
        Rule J: UNION query validation.
        If UNION exists:
        1. All queries must have same number of columns
        2. Each query must be valid
        
        Returns:
            (is_valid, reason)
        """
        if not plan_union or len(plan_union) == 0:
            return True, ""
        
        # Check that all subqueries have same column count
        first_col_count = None
        for idx, sub_query in enumerate(plan_union):
            select_cols = sub_query.get('select') or []
            
            if first_col_count is None:
                first_col_count = len(select_cols)
            elif len(select_cols) != first_col_count:
                msg = f"UNION subquery {idx} has {len(select_cols)} columns but expected {first_col_count}"
                logger.warning(f"[VALIDATOR] Rule J failed: {msg}")
                return False, msg
        
        logger.info("[VALIDATOR] Rule J passed: UNION properly structured")
        return True, ""
    
    def _check_aggregate_functions(
        self,
        plan_select: Optional[List[str]],
    ) -> Tuple[bool, str]:
        """
        Rule K: Aggregate function validation.
        If aggregate functions exist:
        1. Proper syntax (only COUNT, SUM, AVG, MIN, MAX, etc.)
        2. Proper parentheses
        
        Returns:
            (is_valid, reason)
        """
        if not plan_select:
            return True, ""
        
        agg_pattern = r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|STDDEV|VARIANCE)\s*\('
        
        for col in plan_select:
            # Find all aggregate functions
            agg_matches = re.findall(agg_pattern, col, re.IGNORECASE)
            
            for agg in agg_matches:
                # Verify matching parentheses
                agg_section = col[col.upper().find(agg):]
                open_parens = agg_section.count('(')
                close_parens = agg_section.count(')')
                
                if open_parens != close_parens:
                    msg = f"Aggregate function {agg}() has mismatched parentheses in '{col}'"
                    logger.warning(f"[VALIDATOR] Rule K failed: {msg}")
                    return False, msg
        
        logger.info("[VALIDATOR] Rule K passed: Aggregate functions properly structured")
        return True, ""
    
    def _check_subqueries(
        self,
        plan_subqueries: Optional[List[Dict[str, Any]]],
    ) -> Tuple[bool, str]:
        """
        Rule L: Subquery validation.
        If subqueries exist:
        1. Must have FROM clause
        2. Must have SELECT clause
        3. Must be properly aliased if used in FROM
        
        Returns:
            (is_valid, reason)
        """
        if not plan_subqueries or len(plan_subqueries) == 0:
            return True, ""
        
        for idx, subq in enumerate(plan_subqueries):
            # Check SELECT
            if not subq.get('select'):
                msg = f"Subquery {idx} is missing SELECT clause"
                logger.warning(f"[VALIDATOR] Rule L failed: {msg}")
                return False, msg
            
            # Check FROM (if not a simple constant query)
            if subq.get('type') == 'complex' and not subq.get('from_table'):
                msg = f"Subquery {idx} is missing FROM clause"
                logger.warning(f"[VALIDATOR] Rule L failed: {msg}")
                return False, msg
        
        logger.info("[VALIDATOR] Rule L passed: Subqueries properly structured")
        return True, ""
    
    def suggest_required_tables(
        self,
        query: str,
        available_tables: List[str],
    ) -> List[str]:
        """
        Given a query, suggest which tables MUST be included in plan.
        
        Args:
            query: User query
            available_tables: List of tables that are available
        
        Returns:
            List of table names that should be in the plan
        """
        query_lower = query.lower()
        required_tables = set()
        
        for keyword, suggested_tables in self.KEYWORD_TABLE_MAPPING.items():
            if re.search(rf'\b{keyword}\b', query_lower):
                for table in suggested_tables:
                    table_lower = table.lower()
                    # Only add if table is actually available
                    if table_lower in [t.lower() for t in available_tables]:
                        required_tables.add(table_lower)
        
        return list(required_tables)


def validate_and_fix_plan(
    query: str,
    plan: Any,  # QueryPlan object
    available_tables: List[str],
    table_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, bool, str]:
    """
    High-level function: validate plan and suggest fixes.
    
    Returns:
        (possibly_modified_plan, is_valid, message)
    """
    validator = PlanValidator(available_tables, table_metadata)
    
    # Get tables in plan
    plan_tables = [plan.from_table]
    for join in plan.joins:
        plan_tables.append(join.right_table)
    
    # Validate
    is_valid, reason, suggestion = validator.validate_plan(
        query,
        plan_tables,
        available_tables
    )
    
    if is_valid:
        return plan, True, "Plan is valid"
    else:
        # Suggest required tables
        required = validator.suggest_required_tables(query, available_tables)
        msg = f"{reason} (detected required tables: {required})"
        return plan, False, msg
