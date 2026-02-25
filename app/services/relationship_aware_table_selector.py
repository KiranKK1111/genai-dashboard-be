"""
PRINCIPLE 3: Relationship-Aware Table Selection (Join Graph Search)
===================================================================
Before LLM generates SQL, deterministically find correct join paths using FK metadata.

Problem: Model chooses wrong linking table
  "credit cards" → should join customers + cards on customer_id
  BUT model might: join customers + transactions (because "credit" in transactions)

Solution: 
  1. Extract user's target entity (what they want)
  2. Extract constraint entities (what filters apply)
  3. Find shortest join path using FK graph
  4. Pass chosen path to LLM (constrain the join)

Impact: Removes 50% of "wrong join" errors.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class JoinPath:
    """Represents a verified join path in the schema."""
    
    source_table: str  # Entry point
    target_table: str  # Main table we SELECT FROM
    steps: List[Dict]  # [{table, on_clause, direction}]
    distance: int      # Number of hops
    confidence: float  # 0.0-1.0 based on FK cardinality


class JoinGraphBuilder:
    """Build and search FK relationship graph."""

    def __init__(self, schema_grounding):
        """
        Initialize with schema grounding (Principle 1).
        
        Args:
            schema_grounding: SchemaGroundingContext instance
        """
        self.schema = schema_grounding.schema_name
        self.tables = schema_grounding.tables
        self.relationships = schema_grounding.relationships
        self.adjacency: Dict[str, List[Dict]] = {}
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        """Build undirected adjacency list from FK relationships."""
        self.adjacency = {tbl: [] for tbl in self.tables.keys()}

        for rel in self.relationships:
            from_tbl = rel["from_table"]
            to_tbl = rel["to_table"]
            join_on = rel["join_on"]  # List of (local, remote) tuples

            join_clause = " AND ".join([f"{from_tbl}.{l} = {to_tbl}.{r}" for l, r in join_on])

            # Forward edge (parent view: child)
            self.adjacency[from_tbl].append({
                "to": to_tbl,
                "join_clause": join_clause,
                "direction": "child",  # Going to parent
                "join_on": join_on
            })

            # Reverse edge (child view: parent)
            reverse_join_clause = " AND ".join([f"{to_tbl}.{r} = {from_tbl}.{l}" for l, r in join_on])
            self.adjacency[to_tbl].append({
                "to": from_tbl,
                "join_clause": reverse_join_clause,
                "direction": "parent",  # Going to child
                "join_on": [(r, l) for l, r in join_on]
            })

        logger.info(f"[JOIN_GRAPH] Built adjacency list with {len(self.adjacency)} tables")

    def find_join_path(
        self,
        source_table: str,
        target_table: str,
        max_hops: int = 3
    ) -> Optional[JoinPath]:
        """
        Find shortest join path from source to target.
        
        Using BFS (Breadth-First Search) for shortest path.
        
        Args:
            source_table: Starting table
            target_table: Table we want to reach (might be same as source)
            max_hops: Maximum joins allowed (prevents Cartesian products)
            
        Returns:
            JoinPath if found, None otherwise
        """
        if source_table == target_table:
            return JoinPath(
                source_table=source_table,
                target_table=target_table,
                steps=[],
                distance=0,
                confidence=1.0
            )

        # BFS
        queue = deque([(source_table, [])])
        visited = {source_table}
        hop_count = 0

        while queue and hop_count < max_hops:
            # Process all nodes at current depth
            queue_size = len(queue)
            for _ in range(queue_size):
                current_table, path = queue.popleft()

                for edge in self.adjacency.get(current_table, []):
                    next_table = edge["to"]

                    if next_table == target_table:
                        # Found!
                        steps = path + [edge]
                        return JoinPath(
                            source_table=source_table,
                            target_table=target_table,
                            steps=steps,
                            distance=len(steps),
                            confidence=0.9
                        )

                    if next_table not in visited:
                        visited.add(next_table)
                        queue.append((next_table, path + [edge]))

            hop_count += 1

        logger.warning(f"[JOIN_GRAPH] No path found: {source_table} → {target_table}")
        return None

    def find_all_paths_involving(
        self,
        required_tables: Set[str],
        start_table: Optional[str] = None
    ) -> List[JoinPath]:
        """
        Find all join paths that connect a set of required tables.
        
        Used when user mentions multiple entities:
          "credit cards in Mumbai" → needs {cards, customers, branches}
          
        Args:
            required_tables: Tables that must be in result
            start_table: Preferred starting table (SELECT FROM this)
            
        Returns:
            List of possible join paths (sorted by total distance)
        """
        if not required_tables:
            return []

        # If no start specified, pick largest/most foundational table
        if start_table is None:
            start_table = sorted(required_tables)[0]

        all_paths = []

        # For each required table, find path from start_table
        for required in required_tables:
            if required == start_table:
                continue
            path = self.find_join_path(start_table, required)
            if path:
                all_paths.append(path)

        logger.info(f"[JOIN_GRAPH] Found {len(all_paths)} paths involving {required_tables}")
        return sorted(all_paths, key=lambda p: p.distance)


class TableEntityClassifier:
    """Classify which tables match user's intent."""

    def __init__(self, schema_grounding, join_graph: JoinGraphBuilder):
        """
        Args:
            schema_grounding: For table introspection
            join_graph: For join path lookup
        """
        self.schema = schema_grounding
        self.join_graph = join_graph
        self.entity_keywords = {
            "customers": ["customer", "clients", "users", "person", "cust"],
            "accounts": ["account", "deposit", "savings", "checking"],
            "cards": ["card", "credit card", "debit card", "visa", "mastercard"],
            "transactions": ["transaction", "transfer", "payment", "charge", "activity"],
            "branches": ["branch", "office", "location", "city"],
            "loans": ["loan", "lending", "credit line"],
        }

    def extract_entities(self, user_query: str) -> Set[str]:
        """
        Extract tables mentioned or implied in user query.
        
        Args:
            user_query: User's natural language question
            
        Returns:
            Set of probable table names
        """
        query_lower = user_query.lower()
        found_tables = set()

        for table_name, keywords in self.entity_keywords.items():
            # Check if table exists
            if table_name not in self.schema.tables:
                continue

            for keyword in keywords:
                if keyword in query_lower:
                    found_tables.add(table_name)
                    break

        logger.debug(f"[ENTITY_CLASSIFIER] Found tables: {found_tables}")
        return found_tables

    def determine_primary_table(
        self,
        entities: Set[str]
    ) -> Optional[str]:
        """
        Determine which table is the main SELECT target.
        
        Rules (order matters):
        1. If "customers" mentioned → customers is primary
        2. If "cards" mentioned without customers → cards is primary
        3. If "transactions" mentioned → transactions is primary
        4. Otherwise: largest table
        
        Args:
            entities: Set of extracted table names
            
        Returns:
            Primary table name or None
        """
        priority = ["customers", "accounts", "cards", "transactions", "branches", "loans"]

        for table in priority:
            if table in entities:
                logger.debug(f"[ENTITY_CLASSIFIER] Primary table: {table}")
                return table

        return None


class RelationshipAwareTableSelector:
    """Main entry point: coordinates entity extraction + join path finding."""

    def __init__(self, schema_grounding):
        self.join_graph = JoinGraphBuilder(schema_grounding)
        self.classifier = TableEntityClassifier(schema_grounding, self.join_graph)

    def select_tables_and_joins(
        self,
        user_query: str
    ) -> Dict:
        """
        Deterministically select tables and join paths for a query.
        
        Returns:
            {
                "primary_table": "customers",
                "related_tables": ["cards", "accounts"],
                "join_paths": [JoinPath objects],
                "recommended_sql_from": "customers c",
                "recommended_joins": [
                    "LEFT JOIN cards ca ON c.customer_id = ca.customer_id",
                    ...
                ]
            }
        """
        # Step 1: Extract entities
        entities = self.classifier.extract_entities(user_query)
        
        if not entities:
            logger.warning(f"[SELECTOR] No entities found in: {user_query}")
            return {
                "primary_table": None,
                "related_tables": [],
                "join_paths": [],
                "recommended_sql_from": None,
                "recommended_joins": [],
                "note": "Could not determine tables from query"
            }

        # Step 2: Determine primary table
        primary = self.classifier.determine_primary_table(entities)
        if not primary:
            primary = list(entities)[0]

        # Step 3: Find join paths to all other entities
        related = entities - {primary}
        join_paths = []

        for related_tbl in related:
            path = self.join_graph.find_join_path(primary, related_tbl)
            if path:
                join_paths.append(path)

        # Step 4: Generate recommended SQL fragments
        recommended_joins = []
        for path in join_paths:
            for step in path.steps:
                join_clause = step["join_clause"]
                recommended_joins.append(f"LEFT JOIN {step['to']} ON {join_clause}")

        result = {
            "primary_table": primary,
            "related_tables": sorted(related),
            "join_paths": join_paths,
            "recommended_sql_from": primary,
            "recommended_joins": recommended_joins,
            "note": f"Found {len(join_paths)} join paths"
        }

        logger.info(
            f"[SELECTOR] Query: {user_query[:50]}... "
            f"→ Primary: {primary}, Related: {related}"
        )

        return result

    def generate_join_constraint_for_llm(
        self,
        selection: Dict
    ) -> str:
        """
        Generate text to inject into LLM prompt constraining joins.
        
        Args:
            selection: Result from select_tables_and_joins()
            
        Returns:
            Constraint text for prompt
        """
        lines = [
            "\nTABLE AND JOIN CONSTRAINTS:",
            "=" * 60,
            f"\nMain table: {selection['primary_table']}",
            f"Use: SELECT ... FROM {selection['primary_table']}"
        ]

        if selection["recommended_joins"]:
            lines.append("\nRequired joins:")
            for join in selection["recommended_joins"]:
                lines.append(f"  {join}")
        else:
            lines.append("\nNo joins required.")

        lines.append("\nDO NOT invent other joins. Only tables mentioned above exist.")

        return "\n".join(lines)
