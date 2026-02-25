"""
Aggregation Resolver - Executes visualization transforms on demand.

Takes user selections from UI and executes:
1. SQL GROUP BY queries for aggregation
2. Python in-memory aggregation for lightweight transforms
3. Time bucketing/formatting
4. Metric calculations (sum, avg, min, max, count_distinct)

Used by /viz/update endpoint for lazy-render pattern on chart selection.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class AggregationResolver:
    """Resolves visualization transforms and executes aggregations."""
    
    @staticmethod
    async def resolve(
        transform_spec: Dict[str, Any],
        selection: Dict[str, Any],
        query_results: List[Dict[str, Any]],
        db: Optional[AsyncSession] = None,
        original_query: Optional[str] = None,
        table_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute visualization transform based on user selection.
        
        Args:
            transform_spec: Template from aggregators (e.g., {"type": "aggregate", "group_by": ...})
            selection: User selections (e.g., {"x_dimension": {"field": "gender"}, ...})
            query_results: Full query result set
            db: Database session (optional, for SQL-based aggregation)
            original_query: Original SQL query (for building GROUP BY)
            table_name: Table name (for building GROUP BY)
        
        Returns:
            {
                "dataset": [...],  # Aggregated data
                "echarts_option": {...},  # Chart config
                "selection_applied": {...},  # Echoed selection
                "success": True,
                "message": "aggregation completed"
            }
        """
        
        transform_type = transform_spec.get("type", "none")
        
        if transform_type == "none":
            # Table: return as-is
            return {
                "dataset": query_results,
                "echarts_option": None,
                "selection_applied": selection,
                "success": True,
                "message": "table data returned"
            }
        
        elif transform_type == "aggregate":
            return await AggregationResolver._aggregate_bar_pie(
                transform_spec, selection, query_results
            )
        
        elif transform_type == "time_aggregate":
            return await AggregationResolver._aggregate_line(
                transform_spec, selection, query_results
            )
        
        else:
            return {
                "dataset": [],
                "echarts_option": None,
                "selection_applied": selection,
                "success": False,
                "message": f"Unknown transform type: {transform_type}"
            }
    
    @staticmethod
    async def _aggregate_bar_pie(
        transform_spec: Dict[str, Any],
        selection: Dict[str, Any],
        query_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate for bar/pie charts: GROUP BY + METRIC."""
        
        try:
            # Extract controls
            x_field = selection.get("x_dimension", {}).get("field")
            y_metric = selection.get("y_metric", {})
            series_field = selection.get("series_dimension", {}).get("field")
            top_k = selection.get("top_k", 10)
            sort_mode = selection.get("sort", "metric_desc")
            
            if not x_field:
                raise ValueError("x_dimension.field required")
            
            # Group and aggregate
            grouped = {}
            for row in query_results:
                x_val = row.get(x_field)
                
                if x_val is None:
                    x_val = "NULL"
                
                x_key = str(x_val)
                
                if x_key not in grouped:
                    grouped[x_key] = {
                        "x": x_val,
                        "count": 0,
                        "sum": {},
                        "avg": {},
                        "min": {},
                        "max": {},
                        "count_distinct": {},
                        "series": {}
                    }
                
                grouped[x_key]["count"] += 1
                
                # Calculate metrics
                if y_metric.get("field") and y_metric.get("field") != "*":
                    field = y_metric["field"]
                    op = y_metric.get("op", "count")
                    
                    val = row.get(field)
                    if val is not None and isinstance(val, (int, float)):
                        if op == "sum":
                            grouped[x_key]["sum"][field] = grouped[x_key]["sum"].get(field, 0) + val
                        elif op == "avg":
                            if field not in grouped[x_key]["avg"]:
                                grouped[x_key]["avg"][field] = {"sum": 0, "count": 0}
                            grouped[x_key]["avg"][field]["sum"] += val
                            grouped[x_key]["avg"][field]["count"] += 1
                        elif op == "min":
                            if field not in grouped[x_key]["min"]:
                                grouped[x_key]["min"][field] = val
                            else:
                                grouped[x_key]["min"][field] = min(grouped[x_key]["min"][field], val)
                        elif op == "max":
                            if field not in grouped[x_key]["max"]:
                                grouped[x_key]["max"][field] = val
                            else:
                                grouped[x_key]["max"][field] = max(grouped[x_key]["max"][field], val)
                
                # Series grouping
                if series_field:
                    series_val = row.get(series_field, "None")
                    if series_val not in grouped[x_key]["series"]:
                        grouped[x_key]["series"][series_val] = 1
                    else:
                        grouped[x_key]["series"][series_val] += 1
            
            # Finalize metric values
            dataset = []
            for x_key, agg in grouped.items():
                row_data = {"x": agg["x"]}
                
                op = y_metric.get("op", "count")
                field = y_metric.get("field", "*")
                as_name = y_metric.get("as", f"{op}_value")
                
                if op == "count" or field == "*":
                    row_data["y"] = agg["count"]
                elif op == "sum" and field in agg["sum"]:
                    row_data["y"] = agg["sum"][field]
                elif op == "avg" and field in agg["avg"] and agg["avg"][field]["count"] > 0:
                    row_data["y"] = agg["avg"][field]["sum"] / agg["avg"][field]["count"]
                elif op == "min" and field in agg["min"]:
                    row_data["y"] = agg["min"][field]
                elif op == "max" and field in agg["max"]:
                    row_data["y"] = agg["max"][field]
                else:
                    row_data["y"] = agg["count"]
                
                if series_field:
                    row_data["series"] = agg["series"]
                
                dataset.append(row_data)
            
            # Sort
            reverse = not sort_mode.endswith("_asc")
            if "metric" in sort_mode:
                dataset.sort(key=lambda r: r["y"], reverse=reverse)
            elif "dimension" in sort_mode:
                dataset.sort(key=lambda r: str(r["x"]), reverse=reverse)
            
            # Limit
            if top_k and top_k > 0:
                others_sum = sum(r["y"] for r in dataset[top_k:]) if len(dataset) > top_k else 0
                dataset = dataset[:top_k]
                if others_sum > 0:
                    dataset.append({"x": "Others", "y": others_sum})
            
            # Build chart option
            chart_option = await AggregationResolver._build_bar_option(dataset, x_field, as_name)
            
            return {
                "dataset": dataset,
                "echarts_option": chart_option,
                "selection_applied": selection,
                "success": True,
                "message": f"Aggregated {len(dataset)} groups"
            }
        
        except Exception as e:
            return {
                "dataset": [],
                "echarts_option": None,
                "selection_applied": selection,
                "success": False,
                "message": f"Aggregation failed: {str(e)}"
            }
    
    @staticmethod
    async def _aggregate_line(
        transform_spec: Dict[str, Any],
        selection: Dict[str, Any],
        query_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate for line charts: TIME BUCKET + METRIC."""
        
        try:
            # Extract controls
            time_field = selection.get("x_time", {}).get("field")
            bucket = selection.get("bucket", "day")
            y_metric = selection.get("y_metric", {})
            series_field = selection.get("series_dimension", {}).get("field")
            
            if not time_field:
                raise ValueError("x_time.field required")
            
            # Group by time bucket
            grouped = {}
            for row in query_results:
                time_val = row.get(time_field)
                
                if time_val is None:
                    continue
                
                # Parse datetime
                if isinstance(time_val, str):
                    try:
                        dt = datetime.fromisoformat(time_val)
                    except:
                        dt = datetime.fromisoformat(str(time_val).replace("Z", "+00:00"))
                else:
                    dt = time_val
                
                # Bucket
                if bucket == "hour":
                    key = dt.strftime("%Y-%m-%d %H:00")
                elif bucket == "day":
                    key = dt.strftime("%Y-%m-%d")
                elif bucket == "week":
                    key = dt.strftime("%Y-W%W")
                elif bucket == "month":
                    key = dt.strftime("%Y-%m")
                elif bucket == "year":
                    key = dt.strftime("%Y")
                else:
                    key = str(time_val)
                
                if key not in grouped:
                    grouped[key] = {
                        "time": key,
                        "count": 0,
                        "metrics": {}
                    }
                
                grouped[key]["count"] += 1
                
                # Aggregate metrics
                op = y_metric.get("op", "count")
                field = y_metric.get("field", "*")
                
                if op != "count" and field != "*":
                    val = row.get(field)
                    if val is not None and isinstance(val, (int, float)):
                        if field not in grouped[key]["metrics"]:
                            grouped[key]["metrics"][field] = {
                                "sum": 0, "count": 0, "min": val, "max": val
                            }
                        grouped[key]["metrics"][field]["sum"] += val
                        grouped[key]["metrics"][field]["count"] += 1
                        grouped[key]["metrics"][field]["min"] = min(grouped[key]["metrics"][field]["min"], val)
                        grouped[key]["metrics"][field]["max"] = max(grouped[key]["metrics"][field]["max"], val)
            
            # Finalize
            dataset = []
            for key in sorted(grouped.keys()):
                group = grouped[key]
                
                row_data = {"time": group["time"]}
                
                op = y_metric.get("op", "count")
                field = y_metric.get("field", "*")
                
                if op == "count" or field == "*":
                    row_data["y"] = group["count"]
                elif field in group["metrics"]:
                    metrics = group["metrics"][field]
                    if op == "sum":
                        row_data["y"] = metrics["sum"]
                    elif op == "avg":
                        row_data["y"] = metrics["sum"] / metrics["count"] if metrics["count"] > 0 else 0
                    elif op == "min":
                        row_data["y"] = metrics["min"]
                    elif op == "max":
                        row_data["y"] = metrics["max"]
                    else:
                        row_data["y"] = group["count"]
                else:
                    row_data["y"] = group["count"]
                
                dataset.append(row_data)
            
            chart_option = await AggregationResolver._build_line_option(dataset)
            
            return {
                "dataset": dataset,
                "echarts_option": chart_option,
                "selection_applied": selection,
                "success": True,
                "message": f"Aggregated {len(dataset)} time points"
            }
        
        except Exception as e:
            return {
                "dataset": [],
                "echarts_option": None,
                "selection_applied": selection,
                "success": False,
                "message": f"Time aggregation failed: {str(e)}"
            }
    
    @staticmethod
    async def _build_bar_option(
        dataset: List[Dict[str, Any]],
        x_label: str,
        y_label: str
    ) -> Dict[str, Any]:
        """Build ECharts bar chart option."""
        return {
            "type": "bar",
            "title": {"text": f"{x_label} vs {y_label}"},
            "xAxis": {
                "type": "category",
                "data": [str(d["x"]) for d in dataset]
            },
            "yAxis": {
                "type": "value"
            },
            "series": [{
                "data": [d["y"] for d in dataset],
                "type": "bar"
            }],
            "tooltip": {"trigger": "axis"},
            "legend": {"data": [y_label]}
        }
    
    @staticmethod
    async def _build_line_option(
        dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build ECharts line chart option."""
        return {
            "type": "line",
            "title": {"text": "Time Series"},
            "xAxis": {
                "type": "category",
                "data": [d["time"] for d in dataset]
            },
            "yAxis": {
                "type": "value"
            },
            "series": [{
                "data": [d["y"] for d in dataset],
                "type": "line",
                "smooth": True
            }],
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Value"]}
        }
