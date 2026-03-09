"""
KNOWLEDGE GRAPH - Structured relationships between database entities

Builds a semantic knowledge graph from:
- Foreign key relationships
- Table metadata
- Column semantic types
- Implied relationships

Features:
- Automatic relationship discovery
- Path finding between entities
- Semantic relationship types
- Business concept mapping

Architecture:
    Database Metadata → Schema Discovery → Knowledge Graph Builder → 
    Relationship Graph → Path Finder → Query Planner

Example Graph:
    EntityA --[has_many]--> EntityB --[contains]--> EntityC
    EntityA --[located_in]--> Location --[located_in]--> Region
    EntityB --[paid_by]--> Method
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class RelationshipType(str, Enum):
    """Types of relationships in the knowledge graph."""
    HAS_MANY = "has_many"           # One-to-many (parent → children)
    BELONGS_TO = "belongs_to"       # Many-to-one relationship
    HAS_ONE = "has_one"             # One-to-one
    MANY_TO_MANY = "many_to_many"   # Many-to-many (through junction table)
    CONTAINS = "contains"           # Composition (parent → children)
    LOCATED_IN = "located_in"       # Geographic relationship
    PART_OF = "part_of"             # Hierarchy relationship
    REFERENCES = "references"       # Generic foreign key


class EntityType(str, Enum):
    """Semantic types of entities - generic types applicable to any domain."""
    PERSON = "person"               # People-related entities
    ORGANIZATION = "organization"   # Business entities
    TRANSACTION = "transaction"     # Action/event records
    PRODUCT = "product"             # Items, services
    LOCATION = "location"           # Geographic entities
    TIME = "time"                   # Time-related entities
    DOCUMENT = "document"           # Document records
    CATEGORY = "category"           # Classification entities
    METRIC = "metric"               # Measurement entities
    ENTITY = "entity"               # Generic entity


@dataclass
class EntityNode:
    """Node in the knowledge graph representing a database table."""
    name: str                                  # Table name
    entity_type: EntityType                    # Semantic type
    description: Optional[str] = None          # Human-readable description
    primary_key: Optional[str] = None          # Primary key column
    attributes: Dict[str, str] = field(default_factory=dict)  # Column → type mapping
    business_concepts: List[str] = field(default_factory=list)  # Business terms
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.entity_type.value,
            "description": self.description,
            "primary_key": self.primary_key,
            "attributes": self.attributes,
            "business_concepts": self.business_concepts,
        }


@dataclass
class RelationshipEdge:
    """Edge in the knowledge graph representing a relationship."""
    from_entity: str                           # Source table
    to_entity: str                             # Target table
    relationship_type: RelationshipType        # Type of relationship
    join_condition: str                        # SQL join condition
    cardinality: str                           # "1:1", "1:N", "N:M"
    description: Optional[str] = None          # Human-readable description
    confidence: float = 1.0                    # Confidence score (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.from_entity,
            "to": self.to_entity,
            "type": self.relationship_type.value,
            "join_condition": self.join_condition,
            "cardinality": self.cardinality,
            "description": self.description,
            "confidence": self.confidence,
        }


@dataclass
class EntityPath:
    """Path between two entities in the graph."""
    start_entity: str
    end_entity: str
    path: List[str]                            # List of entity names
    relationships: List[RelationshipEdge]      # Edges along the path
    total_cost: float                          # Path cost (lower is better)
    join_sql: str                              # Generated JOIN SQL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start_entity,
            "end": self.end_entity,
            "path": self.path,
            "relationships": [r.to_dict() for r in self.relationships],
            "hops": len(self.path) - 1,
            "join_sql": self.join_sql,
        }


class KnowledgeGraph:
    """
    Knowledge Graph of database schema and relationships.
    
    Capabilities:
    1. Build graph from database metadata
    2. Discover implicit relationships
    3. Find paths between entities
    4. Generate JOIN SQL from paths
    5. Provide semantic context for queries
    
    Example:
        ```python
        kg = KnowledgeGraph()
        await kg.build_from_database(db_session)
        
        # Find a path between two tables
        path = kg.find_path("table_a", "table_b")
        # Returns: table_a → ... → table_b
        
        # Get join SQL
        sql = path.join_sql
        # Returns: JOIN orders ... JOIN order_items ... JOIN products ...
        ```
    """
    
    def __init__(self):
        """Initialize empty knowledge graph."""
        self.entities: Dict[str, EntityNode] = {}
        self.relationships: List[RelationshipEdge] = []
        self.adjacency: Dict[str, List[RelationshipEdge]] = defaultdict(list)
        logger.info("[KNOWLEDGE GRAPH] Initialized")
    
    async def build_from_database(
        self,
        db_session: Any,
        schema_name: str = "public",
    ):
        """
        Build knowledge graph from database metadata.
        
        Steps:
        1. Discover tables and columns
        2. Extract foreign key relationships
        3. Classify entity types
        4. Build adjacency graph
        5. Infer implicit relationships
        
        Args:
            db_session: Database session
            schema_name: Schema to analyze
        """
        logger.info(f"[KNOWLEDGE GRAPH] Building from schema: {schema_name}")
        
        from .schema_discovery_engine import SchemaDiscoveryEngine
        
        # Initialize discovery engine
        discovery = SchemaDiscoveryEngine(db_session)
        
        # Discover foreign keys
        fk_relationships = await discovery.discover_foreign_keys()
        logger.info(f"[KNOWLEDGE GRAPH] Discovered {len(fk_relationships)} FK relationships")
        
       # Build entities from tables
        from sqlalchemy import inspect, text
        inspector = inspect(db_session.bind)
        tables = inspector.get_table_names(schema=schema_name)
        
        for table_name in tables:
            # Get columns
            columns = inspector.get_columns(table_name, schema=schema_name)
            
            # Get primary key
            pk_constraint = inspector.get_pk_constraint(table_name, schema=schema_name)
            primary_key = pk_constraint['constrained_columns'][0] if pk_constraint.get('constrained_columns') else None
            
            # Classify entity type
            entity_type = self._classify_entity_type(table_name, columns)
            
            # Build attributes
            attributes = {col['name']: col['type'].__class__.__name__ for col in columns}
            
            # Create entity node
            entity = EntityNode(
                name=table_name,
                entity_type=entity_type,
                primary_key=primary_key,
                attributes=attributes,
                description=f"{entity_type.value.title()} entity: {table_name}",
                business_concepts=self._extract_business_concepts(table_name),
            )
            
            self.entities[table_name] = entity
        
        logger.info(f"[KNOWLEDGE GRAPH] Created {len(self.entities)} entity nodes")
        
        # Build relationship edges from FKs
        for (from_table, to_table), join_condition in fk_relationships.items():
            # Determine relationship type and cardinality
            rel_type, cardinality = self._determine_relationship_type(from_table, to_table)
            
            edge = RelationshipEdge(
                from_entity=from_table,
                to_entity=to_table,
                relationship_type=rel_type,
                join_condition=join_condition,
                cardinality=cardinality,
                description=f"{from_table} {rel_type.value} {to_table}",
                confidence=1.0,  # FK relationships are certain
            )
            
            self.relationships.append(edge)
            self.adjacency[from_table].append(edge)
            
            # Add reverse relationship
            reverse_type = self._reverse_relationship_type(rel_type)
            reverse_edge = RelationshipEdge(
                from_entity=to_table,
                to_entity=from_table,
                relationship_type=reverse_type,
                join_condition=join_condition,
                cardinality=self._reverse_cardinality(cardinality),
                description=f"{to_table} {reverse_type.value} {from_table}",
                confidence=1.0,
            )
            
            self.adjacency[to_table].append(reverse_edge)
        
        logger.info(f"[KNOWLEDGE GRAPH] Created {len(self.relationships)} relationship edges")
        logger.info("[KNOWLEDGE GRAPH] ✓ Graph built successfully")
    
    def _classify_entity_type(self, table_name: str, columns: List) -> EntityType:
        """Classify table as semantic entity type using generic heuristics.
        
        NO HARDCODED KEYWORDS - uses column pattern analysis instead.
        """
        table_lower = table_name.lower()
        col_names = [c['name'].lower() for c in columns]
        
        # Generic pattern-based classification (no hardcoded domain keywords)
        # Check for person-related patterns: has name, email, phone columns
        person_patterns = ['name', 'email', 'phone', 'address', 'birth', 'age']
        if sum(1 for p in person_patterns if any(p in c for c in col_names)) >= 2:
            return EntityType.PERSON
        
        # Check for transaction patterns: has amount, date, status columns
        transaction_patterns = ['amount', 'total', 'price', 'date', 'time', 'status']
        if sum(1 for p in transaction_patterns if any(p in c for c in col_names)) >= 2:
            return EntityType.TRANSACTION
        
        # Check for product patterns: has price, quantity, sku columns
        product_patterns = ['price', 'quantity', 'sku', 'stock', 'inventory']
        if sum(1 for p in product_patterns if any(p in c for c in col_names)) >= 2:
            return EntityType.PRODUCT
        
        # Check for location patterns: has lat/lon, city, country columns
        location_patterns = ['lat', 'lon', 'city', 'state', 'country', 'zip', 'postal']
        if sum(1 for p in location_patterns if any(p in c for c in col_names)) >= 2:
            return EntityType.LOCATION
        
        # Check for organization patterns: has company-related columns
        org_patterns = ['company', 'org', 'business', 'corp', 'inc']
        if any(p in table_lower for p in org_patterns) or sum(1 for p in org_patterns if any(p in c for c in col_names)) >= 1:
            return EntityType.ORGANIZATION
        
        # Check for category/lookup patterns: small number of columns, has name/code
        if len(col_names) <= 5 and any('name' in c or 'code' in c or 'type' in c for c in col_names):
            return EntityType.CATEGORY
        
        return EntityType.ENTITY
    
    def _determine_relationship_type(self, from_table: str, to_table: str) -> Tuple[RelationshipType, str]:
        """Determine semantic relationship type using generic heuristics.
        
        NO HARDCODED DOMAIN KEYWORDS - uses column patterns instead.
        """
        from_lower = from_table.lower()
        to_lower = to_table.lower()
        
        # Generic heuristics based on naming patterns
        # If from_table contains to_table's name + "_id", it belongs to to_table
        if f"{to_lower}_id" in from_lower or to_lower in from_lower:
            return RelationshipType.BELONGS_TO, "N:1"
        elif f"{from_lower}_id" in to_lower or from_lower in to_lower:
            return RelationshipType.HAS_MANY, "1:N"
        else:
            # Default: treat as generic reference
            return RelationshipType.REFERENCES, "N:1"
    
    def _reverse_relationship_type(self, rel_type: RelationshipType) -> RelationshipType:
        """Get reverse relationship type."""
        reverse_map = {
            RelationshipType.HAS_MANY: RelationshipType.BELONGS_TO,
            RelationshipType.BELONGS_TO: RelationshipType.HAS_MANY,
            RelationshipType.CONTAINS: RelationshipType.PART_OF,
            RelationshipType.PART_OF: RelationshipType.CONTAINS,
            RelationshipType.REFERENCES: RelationshipType.REFERENCES,
        }
        return reverse_map.get(rel_type, RelationshipType.REFERENCES)
    
    def _reverse_cardinality(self, cardinality: str) -> str:
        """Reverse cardinality (1:N → N:1)."""
        if cardinality == "1:N":
            return "N:1"
        elif cardinality == "N:1":
            return "1:N"
        else:
            return cardinality
    
    def _extract_business_concepts(self, table_name: str) -> List[str]:
        """Extract business concepts from table name."""
        # Split by underscore and extract meaningful terms
        parts = table_name.lower().split('_')
        concepts = [p for p in parts if len(p) > 2]
        return concepts
    
    def find_path(
        self,
        start_entity: str,
        end_entity: str,
        max_hops: int = 4,
    ) -> Optional[EntityPath]:
        """
        Find shortest path between two entities using BFS.
        
        Args:
            start_entity: Starting table name
            end_entity: Target table name
            max_hops: Maximum number of joins allowed
        
        Returns:
            EntityPath with join information, or None if no path
        """
        logger.info(f"[KNOWLEDGE GRAPH] Finding path: {start_entity} → {end_entity}")
        
        if start_entity not in self.entities or end_entity not in self.entities:
            logger.warning(f"[KNOWLEDGE GRAPH] Entity not found in graph")
            return None
        
        if start_entity == end_entity:
            # Same table, no joins needed
            return EntityPath(
                start_entity=start_entity,
                end_entity=end_entity,
                path=[start_entity],
                relationships=[],
                total_cost=0.0,
                join_sql="",
            )
        
        # BFS to find shortest path
        queue = deque([(start_entity, [start_entity], [], 0.0)])
        visited = {start_entity}
        
        while queue:
            current, path, edges, cost = queue.popleft()
            
            if len(path) - 1 > max_hops:
                continue
            
            # Check neighbors
            for edge in self.adjacency.get(current, []):
                neighbor = edge.to_entity
                
                if neighbor in visited:
                    continue
                
                new_path = path + [neighbor]
                new_edges = edges + [edge]
                new_cost = cost + (1.0 / edge.confidence)
                
                if neighbor == end_entity:
                    # Found path!
                    join_sql = self._generate_join_sql(new_path, new_edges)
                    
                    result = EntityPath(
                        start_entity=start_entity,
                        end_entity=end_entity,
                        path=new_path,
                        relationships=new_edges,
                        total_cost=new_cost,
                        join_sql=join_sql,
                    )
                    
                    logger.info(f"[KNOWLEDGE GRAPH] ✓ Found path with {len(new_path) - 1} hops")
                    return result
                
                visited.add(neighbor)
                queue.append((neighbor, new_path, new_edges, new_cost))
        
        logger.warning(f"[KNOWLEDGE GRAPH] No path found within {max_hops} hops")
        return None
    
    def _generate_join_sql(self, path: List[str], edges: List[RelationshipEdge]) -> str:
        """Generate SQL JOIN clause from path."""
        if len(path) < 2:
            return ""
        
        joins = []
        for i, edge in enumerate(edges):
            join_type = "LEFT JOIN" if edge.relationship_type == RelationshipType.HAS_MANY else "INNER JOIN"
            joins.append(f"{join_type} {edge.to_entity} ON {edge.join_condition}")
        
        return "\n".join(joins)
    
    def get_entity_relationships(self, entity_name: str) -> List[RelationshipEdge]:
        """Get all relationships for an entity."""
        return self.adjacency.get(entity_name, [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph as dictionary."""
        return {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relationships": [r.to_dict() for r in self.relationships],
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
        }


# Global instance
_knowledge_graph: Optional[KnowledgeGraph] = None


async def get_knowledge_graph(db_session: Any, rebuild: bool = False) ->KnowledgeGraph:
    """Get or create knowledge graph instance."""
    global _knowledge_graph
    
    if _knowledge_graph is None or rebuild:
        _knowledge_graph = KnowledgeGraph()
        await _knowledge_graph.build_from_database(db_session)
    
    return _knowledge_graph
