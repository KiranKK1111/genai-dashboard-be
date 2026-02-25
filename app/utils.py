"""
Backward-compatible utils module.

This module provides backward compatibility for code that imports from app.utils.
All functions have been modularized into services/ and helpers/ packages.
New code should import directly from those packages.

Legacy imports from this module are now deprecated but still work:
  from app.utils import classify_query  -> from app.services import classify_query
  from app.utils import generate_sql    -> from app.services import generate_sql
  from app.utils import make_json_serializable  -> from app.helpers import make_json_serializable
"""

# Re-export all services for backward compatibility
from .helpers import (
    build_capabilities,
    current_timestamp as _current_timestamp,
    format_conversation_context,
    get_database_schema,
    make_json_serializable,
)
from .services import (
    add_file,
    build_config_update_response,
    build_data_query_response,
    build_file_lookup_response,
    build_file_query_response,
    build_standard_response,
    classify_query,
    generate_sql,
    process_file_upload,
    retrieve_relevant_chunks,
    run_sql,
    validate_and_fix_sql,
)

__all__ = [
    "add_file",
    "build_capabilities",
    "build_config_update_response",
    "build_data_query_response",
    "build_file_lookup_response",
    "build_file_query_response",
    "build_standard_response",
    "classify_query",
    "format_conversation_context",
    "generate_sql",
    "get_database_schema",
    "make_json_serializable",
    "process_file_upload",
    "retrieve_relevant_chunks",
    "run_sql",
    "validate_and_fix_sql",
    "_current_timestamp",
]
