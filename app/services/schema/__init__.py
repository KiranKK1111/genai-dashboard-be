"""app.services.schema — Schema discovery, introspection and caching."""

from ..schema_intelligence_service import SchemaIntelligenceService, get_schema_intelligence
from ..schema_discovery_engine import SchemaDiscoveryEngine
from ..schema_change_detector import start_schema_change_detector
from ..schema_derived_signals import SchemaDerivedSignals
from ..schema_rag import SchemaRAG
from ..semantic_schema_catalog import SemanticSchemaCatalog, get_catalog
from ..adaptive_schema_analyzer import AdaptiveSchemaAnalyzer

# Correct names from actual modules
from ..schema_discovery import SchemaCatalog                  # the catalog class
from ..schema_discovery_sqlalchemy import ColumnInfo          # representative export
from ..schema_metadata import ColumnMetadata, ColumnType
from ..schema_grounding import SchemaGroundingContext
from ..schema_normalizer import SchemaNormalizer

__all__ = [
    "SchemaIntelligenceService", "get_schema_intelligence",
    "SchemaDiscoveryEngine", "start_schema_change_detector",
    "SchemaDerivedSignals", "SchemaRAG", "SemanticSchemaCatalog", "get_catalog",
    "AdaptiveSchemaAnalyzer", "SchemaCatalog", "ColumnInfo",
    "ColumnMetadata", "ColumnType", "SchemaGroundingContext", "SchemaNormalizer",
]
