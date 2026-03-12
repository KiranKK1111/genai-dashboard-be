"""
VizUpdate mode handler.

Resolves an existing message's transform template and re-aggregates its
cached data without re-running SQL.  Useful for interactive chart switching
(bar → line → pie) on the frontend.
"""

from __future__ import annotations

import json

from fastapi import HTTPException
from sqlalchemy import select

from .. import models, schemas
from ..helpers import current_timestamp
from ._base import QueryModeHandler
from ._context import QueryContext


class VizUpdateHandler(QueryModeHandler):
    async def execute(self, ctx: QueryContext) -> schemas.ResponseWrapper:
        from ..services.aggregation_resolver import AggregationResolver

        try:
            viz_request = (
                json.loads(ctx.query) if isinstance(ctx.query, str) else ctx.query
            )
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON in query parameter for viz_update mode",
            )

        req_message_id = viz_request.get("message_id")
        view = viz_request.get("view", "bar")
        selection = viz_request.get("selection", {})

        if not req_message_id:
            raise HTTPException(
                status_code=400,
                detail="message_id required for viz_update mode",
            )

        msg_result = await ctx.db.execute(
            select(models.Message).where(
                models.Message.id == req_message_id,
                models.Message.session_id == ctx.session_id,
            )
        )
        message = msg_result.scalars().first()
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        response_data = message.response
        if isinstance(response_data, str):
            response_data = json.loads(response_data)

        visualization = response_data.get("visualizations")
        table_data = response_data.get("table_preview", []) or response_data.get("data", [])

        if not visualization:
            raise HTTPException(
                status_code=400,
                detail="No visualization data found for this message",
            )

        view_aggregators = visualization.get(view, {})
        transform_spec = view_aggregators.get("transform_template", {})

        if not transform_spec:
            raise HTTPException(
                status_code=400,
                detail=f"No transform template for view: {view}",
            )
        if not table_data:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for aggregation",
            )

        result = await AggregationResolver.resolve(
            transform_spec=transform_spec,
            selection=selection,
            query_results=table_data,
            db=ctx.db,
        )

        viz_response = schemas.StandardResponse(
            type="viz_update",
            intent="Visualization Update",
            confidence=1.0,
            message=result.get("message", "Visualization updated"),
            metadata={
                "viz_update": True,
                "view": view,
                "dataset": result.get("dataset", []),
                "echarts_option": result.get("echarts_option"),
                "selection_applied": result.get("selection_applied"),
                "success": result.get("success", True),
            },
        )

        return schemas.ResponseWrapper(
            success=True,
            response=viz_response,
            timestamp=current_timestamp(),
            original_query=ctx.query,
            session_id=ctx.session_id,
            message_id=ctx.message_id,
        )
