"""
app.routes package.

Exports:
  router          — FastAPI APIRouter with all endpoints (used by main.py)
  stream_events   — SSE generator (used by StreamHandler)
  DISPATCH_TABLE  — mode → handler class registry
  Handler classes — VizUpdateHandler, StreamHandler, StandardHandler,
                    AgenticHandler, VariationsHandler
"""

from .mode_viz_update import VizUpdateHandler
from .mode_stream import StreamHandler
from .mode_standard import StandardHandler
from .mode_agentic import AgenticHandler
from .mode_variations import VariationsHandler

#: Maps mode name → handler class.  Add new modes here without touching routes.
DISPATCH_TABLE: dict[str, type] = {
    "standard": StandardHandler,
    "agentic": AgenticHandler,
    "stream": StreamHandler,
    "viz_update": VizUpdateHandler,
    "variations": VariationsHandler,
}

# Import router and stream_events AFTER DISPATCH_TABLE is defined so that
# main_routes.py can import DISPATCH_TABLE from this package without a cycle.
from .main_routes import router, stream_events  # noqa: E402

__all__ = [
    "router",
    "stream_events",
    "DISPATCH_TABLE",
    "VizUpdateHandler",
    "StreamHandler",
    "StandardHandler",
    "AgenticHandler",
    "VariationsHandler",
]
