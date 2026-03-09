"""
AI-powered dynamic visualization generator with field role detection,
smart aggregator selection, and lazy-render pattern support.

KEY CHANGES:
1. Separates metric control into {op, field} structure for UI
2. Smart defaults that avoid ID fields
3. Field classification lists (numeric, categorical, time, id)
4. Lazy-render flags for each chart type
5. Transform templates for on-demand aggregation
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from app.llm import call_llm


class FieldRoleDetector:
    """
    Analyzes field names and types to assign semantic roles.
    Helps LLM-driven systems understand data context.
    """
    
    ROLE_HINTS = {
        "id": r"(id|_id|pk|uuid|identifier)$",
        "category": r"(status|type|kind|category|class|category|tier)$",
        "measure": r"(amount|count|total|sum|qty|quantity|value|rate|price|cost)$",
        "time": r"(date|time|_at|_ts|timestamp|created|updated|modified)$",
        "geo_admin1": r"(state|province|region|country)$",
        "geo_admin2": r"(city|district|county|zone|area)$",
        "geo_postal": r"(zip|postal|postcode)$",
    }
    
    @staticmethod
    def detect_field_type(column_values: List[Any]) -> str:
        """Detect Python/SQL type from sample values."""
        if not column_values:
            return "string"
        
        # Get non-null samples
        samples = [v for v in column_values[:100] if v is not None]
        if not samples:
            return "string"
        
        first = samples[0]
        
        if isinstance(first, bool):
            return "boolean"
        elif isinstance(first, int):
            return "integer"
        elif isinstance(first, float):
            return "number"
        elif isinstance(first, datetime):
            return "datetime"
        elif isinstance(first, str):
            # Try parsing as date
            if any(word in first.lower() for word in ['2024', '2023', '2022', '2021', '2020']):
                return "datetime"
            return "string"
        else:
            return "string"
    
    @staticmethod
    def detect_field_role(column_name: str, field_type: str, cardinality: int, sample_values: List[Any]) -> str:
        """
        Detect semantic role of field based on name, type, and cardinality.
        Returns: "id", "category", "measure", "time", "geo_admin1", "geo_admin2", "geo_postal", "dimension"
        """
        
        col_lower = column_name.lower()
        
        # Check role hints by pattern
        for role, pattern in FieldRoleDetector.ROLE_HINTS.items():
            if re.search(pattern, col_lower):
                return role
        
        # Fallback: infer from type
        if field_type == "datetime":
            return "time"
        elif field_type in ["number", "integer"]:
            return "measure"
        else:
            return "dimension"  # generic categorical
    
    @staticmethod
    async def analyze_fields(
        fields_info: List[Dict[str, Any]],
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze fields from query result and assign roles.
        Returns: [{name, type, role_hint, cardinality}, ...]
        """
        
        if not results:
            # No results, infer from field names only
            return [
                {
                    "name": f["name"],
                    "type": f.get("type", "string"),
                    "role_hint": FieldRoleDetector.detect_field_role(f["name"], f.get("type", "string"), 0, []),
                    "cardinality": 0,
                    "sample_values": []
                }
                for f in fields_info
            ]
        
        # Analyze from results
        fields = []
        for field_info in fields_info:
            column_name = field_info.get("name", field_info.get("column_name", ""))
            
            # Get column values
            column_values = [row.get(column_name) for row in results if column_name in row]
            
            # Calculate cardinality
            unique_values = set(v for v in column_values if v is not None)
            cardinality = len(unique_values)
            
            # Detect type
            field_type = FieldRoleDetector.detect_field_type(column_values)
            
            # Detect role
            field_role = FieldRoleDetector.detect_field_role(column_name, field_type, cardinality, column_values)
            
            fields.append({
                "name": column_name,
                "type": field_type,
                "role_hint": field_role,
                "cardinality": cardinality,
                "sample_values": column_values[:3]
            })
        
        return fields


class DynamicAggregatorGenerator:
    """Generates smart, lazy-render aggregators for charts."""
    
    @staticmethod
    async def generate_aggregators(
        fields: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        row_count: int
    ) -> Dict[str, Any]:
        """
        Generate smart aggregators with proper metric control structure.
        
        Structure:
        {
            "common": {...},
            "bar": {
                "auto_render_on_select": True,
                "requires_transform": True,
                "controls": [...],
                "default_selection": {...},
                "numeric_fields": [...],
                "categorical_fields": [...],
                "transform_template": {...}
            },
            ...
            "_metadata": {
                "numeric_fields": [],
                "categorical_fields": [],
                "time_fields": [],
                "id_fields": []
            }
        }
        """
        
        # Classify fields
        numeric_fields = [f["name"] for f in fields if f["role_hint"] == "measure"]
        categorical_fields = [f["name"] for f in fields if f["role_hint"] in ["category", "geo_admin1", "geo_admin2", "dimension"]]
        time_fields = [f["name"] for f in fields if f["role_hint"] == "time"]
        id_fields = [f["name"] for f in fields if f["role_hint"] == "id"]
        
        # Get LLM suggestions for smart defaults
        # Convert datetime objects to strings for JSON serialization
        fields_for_json = []
        for f in fields:
            f_copy = f.copy()
            if "sample_values" in f_copy:
                f_copy["sample_values"] = [
                    str(v) if isinstance(v, datetime) else v
                    for v in f_copy["sample_values"]
                ]
            fields_for_json.append(f_copy)
        
        field_summary = json.dumps(fields_for_json, indent=2)
        
        prompt = f"""
Analyze these fields and suggest visualization defaults.
Focus on LOW-CARDINALITY categorical fields for grouping.
Avoid high-cardinality ID fields.

FIELDS:
{field_summary}

Response JSON:
{{
  "bar_x_field": "categorical_field_name (not ID)",
  "bar_y_op": "count|sum|avg",
  "line_time_field": "time_field_name or null",
  "line_bucket": "day|month|year",
  "pie_x_field": "categorical_field_name"
}}
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a data visualization expert. Suggest optimal defaults."},
                {"role": "user", "content": prompt}
            ]
            
            response = await call_llm(messages, stream=False, max_tokens=512)
            
            # Parse JSON response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                suggestions = json.loads(json_match.group())
            else:
                suggestions = {}
        except Exception as e:
            print(f"⚠️  LLM suggestion failed: {e}, using smart fallbacks")
            suggestions = {}
        
        # Select smart defaults
        bar_x_default = suggestions.get("bar_x_field") or (
            next((f for f in categorical_fields if f not in id_fields), categorical_fields[0] if categorical_fields else "")
        )
        
        line_time_default = suggestions.get("line_time_field") or (time_fields[0] if time_fields else "")
        
        pie_x_default = suggestions.get("pie_x_field") or (
            next((f for f in categorical_fields if f not in id_fields), categorical_fields[0] if categorical_fields else "")
        )
        
        # Build aggregators structure
        aggregators = {
            "common": {
                "metric_ops": ["count", "count_distinct", "sum", "avg", "min", "max"],
                "time_buckets": ["hour", "day", "week", "month", "year"],
                "sort_modes": ["metric_desc", "metric_asc", "dimension_asc", "dimension_desc"],
                "limits": {"top_k_min": 3, "top_k_max": 50, "default_top_k": 10}
            },
            "bar": await DynamicAggregatorGenerator._build_bar(
                categorical_fields, numeric_fields, id_fields, bar_x_default, suggestions
            ),
            "line": await DynamicAggregatorGenerator._build_line(
                time_fields, categorical_fields, numeric_fields, line_time_default, suggestions
            ),
            "pie": await DynamicAggregatorGenerator._build_pie(
                categorical_fields, numeric_fields, id_fields, pie_x_default, suggestions
            ),
            "scatter": await DynamicAggregatorGenerator._build_scatter(
                numeric_fields, categorical_fields, suggestions
            ),
            "treemap": await DynamicAggregatorGenerator._build_treemap(
                categorical_fields, numeric_fields, id_fields, suggestions
            ),
            "funnel": await DynamicAggregatorGenerator._build_funnel(
                categorical_fields, numeric_fields, suggestions
            ),
            "waterfall": await DynamicAggregatorGenerator._build_waterfall(
                categorical_fields, numeric_fields, time_fields, suggestions
            ),
            "gantt": await DynamicAggregatorGenerator._build_gantt(
                categorical_fields, time_fields, suggestions
            ),
            "sankey": await DynamicAggregatorGenerator._build_sankey(
                categorical_fields, numeric_fields, suggestions
            ),
            "boxplot": await DynamicAggregatorGenerator._build_boxplot(
                categorical_fields, numeric_fields, suggestions
            ),
            "table": {
                "auto_render_on_select": False,
                "requires_transform": False,
                "controls": [],
                "default_selection": {},
                "numeric_fields": numeric_fields,
                "categorical_fields": categorical_fields,
                "transform_template": {"type": "none"},
                "style": {
                    "palette_mode": "none"
                }
            },
            "_metadata": {
                "numeric_fields": numeric_fields,
                "categorical_fields": categorical_fields,
                "time_fields": time_fields,
                "id_fields": id_fields
            }
        }
        
        return aggregators
    
    @staticmethod
    async def _build_bar(
        categorical_fields: List[str],
        numeric_fields: List[str],
        id_fields: List[str],
        default_x: str,
        suggestion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Bar chart: Group by categorical, aggregate numeric."""
        
        # X options: categorical fields excluding IDs
        x_options = [f for f in categorical_fields if f not in id_fields]
        if not x_options:
            x_options = categorical_fields
        
        # Y metric: structure as {op, field, as}
        y_op_default = suggestion.get("bar_y_op", "count")
        metric_field_options = ["*"] + numeric_fields
        
        return {
            "auto_render_on_select": True,
            "requires_transform": True,
            "controls": [
                {
                    "id": "x_dimension",
                    "label": "X Axis (Group By)",
                    "kind": "dimension",
                    "type": "field",
                    "allowed_field_types": ["string", "boolean", "date", "datetime"],
                    "field_options": x_options,
                    "description": "Categorical field to group by"
                },
                {
                    "id": "y_metric",
                    "label": "Y Axis (Metric)",
                    "kind": "metric",
                    "type": "metric_builder",
                    "metric_op_options": ["count", "count_distinct", "sum", "avg", "min", "max"],
                    "metric_field_options": metric_field_options,
                    "numeric_fields": numeric_fields,
                    "description": "Metric: select operation and field"
                },
                {
                    "id": "series_dimension",
                    "label": "Series/Stack (Optional)",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": [f for f in categorical_fields if f not in id_fields],
                    "optional": True,
                    "description": "Optional stacking dimension"
                },
                {
                    "id": "top_k",
                    "label": "Top K",
                    "kind": "option",
                    "type": "number",
                    "min": 3,
                    "max": 50,
                    "description": "Limit to top K results"
                },
                {
                    "id": "sort",
                    "label": "Sort",
                    "kind": "option",
                    "type": "enum",
                    "values": ["metric_desc", "metric_asc", "dimension_asc", "dimension_desc"],
                    "description": "Sort order"
                }
            ],
            "default_selection": {
                "x_dimension": {"field": default_x},
                "y_metric": {"op": y_op_default, "field": "*", "as": f"{y_op_default}_val"},
                "series_dimension": None,
                "top_k": 10,
                "sort": "metric_desc"
            },
            "numeric_fields": numeric_fields,
            "categorical_fields": categorical_fields,
            "transform_template": {
                "type": "aggregate",
                "group_by": [{"from_control": "x_dimension"}],
                "metrics": [{"from_control": "y_metric"}],
                "series": {"from_control": "series_dimension"},
                "order_by": [{"metric_from": "y_metric", "direction_from": "sort"}],
                "limit": {"from_control": "top_k"}
            },
            "style": {
                "palette_mode": "categorical",
                "palette_name": "random",
                "seed_source": "chart_id",
                "assignment": "by_category",
                "fallback_color": "theme_primary"
            }
        }
    
    @staticmethod
    async def _build_line(
        time_fields: List[str],
        categorical_fields: List[str],
        numeric_fields: List[str],
        default_time: str,
        suggestion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Line chart: Time series with metrics."""
        
        default_bucket = suggestion.get("line_bucket", "day")
        metric_field_options = ["*"] + numeric_fields
        
        return {
            "auto_render_on_select": True,
            "requires_transform": True,
            "controls": [
                {
                    "id": "x_time",
                    "label": "Time Field",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": time_fields,
                    "allowed_field_types": ["date", "datetime"],
                    "description": "Time field for X-axis"
                },
                {
                    "id": "bucket",
                    "label": "Time Bucket",
                    "kind": "option",
                    "type": "enum",
                    "values": ["hour", "day", "week", "month", "year"],
                    "description": "Aggregation period"
                },
                {
                    "id": "y_metric",
                    "label": "Y Metric",
                    "kind": "metric",
                    "type": "metric_builder",
                    "metric_op_options": ["count", "count_distinct", "sum", "avg"],
                    "metric_field_options": metric_field_options,
                    "numeric_fields": numeric_fields,
                    "description": "Metric to chart"
                },
                {
                    "id": "series_dimension",
                    "label": "Series (Optional)",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": categorical_fields,
                    "optional": True,
                    "description": "Multiple lines per category"
                }
            ],
            "default_selection": {
                "x_time": {"field": default_time},
                "bucket": default_bucket,
                "y_metric": {"op": "count", "field": "*", "as": "count"},
                "series_dimension": None
            },
            "numeric_fields": numeric_fields,
            "categorical_fields": categorical_fields,
            "transform_template": {
                "type": "time_aggregate",
                "time_field": {"from_control": "x_time"},
                "bucket": {"from_control": "bucket"},
                "metrics": [{"from_control": "y_metric"}],
                "series": {"from_control": "series_dimension"}
            },
            "style": {
                "palette_mode": "single",
                "palette_name": "theme_primary"
            }
        }
    
    @staticmethod
    async def _build_pie(
        categorical_fields: List[str],
        numeric_fields: List[str],
        id_fields: List[str],
        default_x: str,
        suggestion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Pie chart: Category breakdown."""
        
        # Category options: exclude IDs
        category_options = [f for f in categorical_fields if f not in id_fields]
        if not category_options:
            category_options = categorical_fields
        
        metric_field_options = ["*"] + numeric_fields
        
        return {
            "auto_render_on_select": True,
            "requires_transform": True,
            "controls": [
                {
                    "id": "label_dimension",
                    "label": "Category",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": category_options,
                    "allowed_field_types": ["string", "boolean"],
                    "description": "Pie slice dimension"
                },
                {
                    "id": "value_metric",
                    "label": "Value Metric",
                    "kind": "metric",
                    "type": "metric_builder",
                    "metric_op_options": ["count", "count_distinct", "sum", "avg"],
                    "metric_field_options": metric_field_options,
                    "numeric_fields": numeric_fields,
                    "description": "Slice size metric"
                },
                {
                    "id": "top_k",
                    "label": "Top K (+ Others)",
                    "kind": "option",
                    "type": "number",
                    "min": 3,
                    "max": 20,
                    "description": "Top K categories, rest as Others"
                }
            ],
            "default_selection": {
                "label_dimension": {"field": default_x},
                "value_metric": {"op": "count", "field": "*", "as": "count"},
                "top_k": 8
            },
            "numeric_fields": numeric_fields,
            "categorical_fields": categorical_fields,
            "transform_template": {
                "type": "aggregate",
                "group_by": [{"from_control": "label_dimension"}],
                "metrics": [{"from_control": "value_metric"}],
                "limit": {"from_control": "top_k"},
                "others_bucket": {"enabled": True, "label": "Others"}
            },
            "style": {
                "palette_mode": "categorical",
                "palette_name": "random",
                "seed_source": "chart_id",
                "assignment": "by_category",
                "fallback_color": "theme_primary"
            }
        }
    
    @staticmethod
    async def _build_scatter(
        numeric_fields: List[str],
        categorical_fields: List[str],
        suggestion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Scatter chart: Two numeric dimensions with optional coloring."""
        
        # Need at least 2 numeric fields for scatter
        if len(numeric_fields) < 2:
            # Fallback: use categorical field as x if not enough numeric
            x_options = [f for f in categorical_fields] if categorical_fields else ["*"]
            y_options = numeric_fields if numeric_fields else ["*"]
        else:
            x_options = numeric_fields
            y_options = numeric_fields
        
        size_options = ["*"] + numeric_fields
        color_options = categorical_fields
        
        return {
            "auto_render_on_select": True,
            "requires_transform": True,
            "controls": [
                {
                    "id": "x_metric",
                    "label": "X Axis (Metric)",
                    "kind": "metric",
                    "type": "field",
                    "field_options": x_options,
                    "allowed_field_types": ["number", "integer"],
                    "description": "Numeric field for X-axis"
                },
                {
                    "id": "y_metric",
                    "label": "Y Axis (Metric)",
                    "kind": "metric",
                    "type": "field",
                    "field_options": y_options,
                    "allowed_field_types": ["number", "integer"],
                    "description": "Numeric field for Y-axis"
                },
                {
                    "id": "size_metric",
                    "label": "Bubble Size (Optional)",
                    "kind": "metric",
                    "type": "field",
                    "field_options": size_options,
                    "allowed_field_types": ["number", "integer"],
                    "optional": True,
                    "description": "Bubble size dimension"
                },
                {
                    "id": "color_dimension",
                    "label": "Color By (Optional)",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": color_options,
                    "allowed_field_types": ["string", "boolean", "category"],
                    "optional": True,
                    "description": "Color points by category"
                }
            ],
            "default_selection": {
                "x_metric": {"field": x_options[0] if x_options else "*"},
                "y_metric": {"field": y_options[0] if y_options else "*"},
                "size_metric": None,
                "color_dimension": None
            },
            "numeric_fields": numeric_fields,
            "categorical_fields": categorical_fields,
            "transform_template": {
                "type": "scatter",
                "x_field": {"from_control": "x_metric"},
                "y_field": {"from_control": "y_metric"},
                "size_field": {"from_control": "size_metric"},
                "color_field": {"from_control": "color_dimension"}
            },
            "style": {
                "palette_mode": "categorical",
                "palette_name": "random",
                "seed_source": "chart_id",
                "assignment": "by_category",
                "fallback_color": "theme_primary"
            }
        }
    
    @staticmethod
    async def _build_treemap(
        categorical_fields: List[str],
        numeric_fields: List[str],
        id_fields: List[str],
        suggestion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Treemap: Hierarchical data visualization."""
        
        category_options = [f for f in categorical_fields if f not in id_fields]
        if not category_options:
            category_options = categorical_fields
        
        metric_field_options = ["*"] + numeric_fields
        
        return {
            "auto_render_on_select": True,
            "requires_transform": True,
            "controls": [
                {
                    "id": "parent_dimension",
                    "label": "Parent Category",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": category_options,
                    "description": "Top-level grouping"
                },
                {
                    "id": "child_dimension",
                    "label": "Child Category (Optional)",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": category_options,
                    "optional": True,
                    "description": "Second-level grouping"
                },
                {
                    "id": "size_metric",
                    "label": "Size Metric",
                    "kind": "metric",
                    "type": "metric_builder",
                    "metric_op_options": ["count", "sum", "avg"],
                    "metric_field_options": metric_field_options,
                    "numeric_fields": numeric_fields,
                    "description": "Rectangle size"
                }
            ],
            "default_selection": {
                "parent_dimension": {"field": category_options[0] if category_options else ""},
                "child_dimension": None,
                "size_metric": {"op": "count", "field": "*", "as": "count"}
            },
            "numeric_fields": numeric_fields,
            "categorical_fields": categorical_fields,
            "transform_template": {
                "type": "hierarchical_aggregate",
                "parent": {"from_control": "parent_dimension"},
                "child": {"from_control": "child_dimension"},
                "size": {"from_control": "size_metric"}
            },
            "style": {
                "palette_mode": "hierarchical",
                "palette_name": "rainbow"
            }
        }
    
    @staticmethod
    async def _build_funnel(
        categorical_fields: List[str],
        numeric_fields: List[str],
        suggestion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Funnel chart: Conversion stages visualization."""
        
        stage_options = categorical_fields if categorical_fields else ["stage"]
        metric_field_options = ["*"] + numeric_fields
        
        return {
            "auto_render_on_select": True,
            "requires_transform": True,
            "controls": [
                {
                    "id": "stage_dimension",
                    "label": "Stage/Step",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": stage_options,
                    "description": "Funnel stages in order"
                },
                {
                    "id": "value_metric",
                    "label": "Value Metric",
                    "kind": "metric",
                    "type": "metric_builder",
                    "metric_op_options": ["count", "count_distinct", "sum"],
                    "metric_field_options": metric_field_options,
                    "numeric_fields": numeric_fields,
                    "description": "Stage value/count"
                },
                {
                    "id": "show_conversion_rate",
                    "label": "Show Conversion %",
                    "kind": "option",
                    "type": "boolean",
                    "description": "Display drop-off percentages"
                }
            ],
            "default_selection": {
                "stage_dimension": {"field": stage_options[0] if stage_options else ""},
                "value_metric": {"op": "count", "field": "*", "as": "count"},
                "show_conversion_rate": True
            },
            "numeric_fields": numeric_fields,
            "categorical_fields": categorical_fields,
            "transform_template": {
                "type": "sequential_aggregate",
                "stage": {"from_control": "stage_dimension"},
                "value": {"from_control": "value_metric"}
            },
            "style": {
                "palette_mode": "gradient",
                "palette_name": "green_to_red"
            }
        }
    
    @staticmethod
    async def _build_waterfall(
        categorical_fields: List[str],
        numeric_fields: List[str],
        time_fields: List[str],
        suggestion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Waterfall chart: Cumulative effect visualization."""
        
        category_options = categorical_fields + time_fields if categorical_fields or time_fields else ["category"]
        metric_field_options = ["*"] + numeric_fields
        
        return {
            "auto_render_on_select": True,
            "requires_transform": True,
            "controls": [
                {
                    "id": "category_dimension",
                    "label": "Category/Period",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": category_options,
                    "description": "Sequential categories"
                },
                {
                    "id": "value_metric",
                    "label": "Value Metric",
                    "kind": "metric",
                    "type": "metric_builder",
                    "metric_op_options": ["sum", "avg"],
                    "metric_field_options": metric_field_options,
                    "numeric_fields": numeric_fields,
                    "description": "Positive/negative changes"
                },
                {
                    "id": "show_totals",
                    "label": "Show Running Total",
                    "kind": "option",
                    "type": "boolean",
                    "description": "Display cumulative bars"
                }
            ],
            "default_selection": {
                "category_dimension": {"field": category_options[0] if category_options else ""},
                "value_metric": {"op": "sum", "field": numeric_fields[0] if numeric_fields else "*", "as": "total"},
                "show_totals": True
            },
            "numeric_fields": numeric_fields,
            "categorical_fields": categorical_fields,
            "transform_template": {
                "type": "cumulative_aggregate",
                "category": {"from_control": "category_dimension"},
                "value": {"from_control": "value_metric"}
            },
            "style": {
                "palette_mode": "conditional",
                "positive_color": "green",
                "negative_color": "red",
                "total_color": "blue"
            }
        }
    
    @staticmethod
    async def _build_gantt(
        categorical_fields: List[str],
        time_fields: List[str],
        suggestion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Gantt chart: Timeline/project visualization."""
        
        task_options = categorical_fields if categorical_fields else ["task"]
        time_options = time_fields if time_fields else ["date"]
        
        return {
            "auto_render_on_select": True,
            "requires_transform": True,
            "controls": [
                {
                    "id": "task_dimension",
                    "label": "Task/Activity",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": task_options,
                    "description": "Task names"
                },
                {
                    "id": "start_date",
                    "label": "Start Date",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": time_options,
                    "allowed_field_types": ["date", "datetime"],
                    "description": "Task start time"
                },
                {
                    "id": "end_date",
                    "label": "End Date",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": time_options,
                    "allowed_field_types": ["date", "datetime"],
                    "description": "Task end time"
                },
                {
                    "id": "group_by",
                    "label": "Group By (Optional)",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": categorical_fields,
                    "optional": True,
                    "description": "Group tasks by category"
                }
            ],
            "default_selection": {
                "task_dimension": {"field": task_options[0] if task_options else ""},
                "start_date": {"field": time_options[0] if time_options else ""},
                "end_date": {"field": time_options[1] if len(time_options) > 1 else time_options[0] if time_options else ""},
                "group_by": None
            },
            "numeric_fields": [],
            "categorical_fields": categorical_fields,
            "transform_template": {
                "type": "timeline",
                "task": {"from_control": "task_dimension"},
                "start": {"from_control": "start_date"},
                "end": {"from_control": "end_date"},
                "group": {"from_control": "group_by"}
            },
            "style": {
                "palette_mode": "categorical",
                "palette_name": "pastel"
            }
        }
    
    @staticmethod
    async def _build_sankey(
        categorical_fields: List[str],
        numeric_fields: List[str],
        suggestion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sankey diagram: Flow visualization."""
        
        category_options = categorical_fields if categorical_fields else ["source", "target"]
        metric_field_options = ["*"] + numeric_fields
        
        return {
            "auto_render_on_select": True,
            "requires_transform": True,
            "controls": [
                {
                    "id": "source_dimension",
                    "label": "Source",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": category_options,
                    "description": "Flow origin"
                },
                {
                    "id": "target_dimension",
                    "label": "Target",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": category_options,
                    "description": "Flow destination"
                },
                {
                    "id": "value_metric",
                    "label": "Flow Value",
                    "kind": "metric",
                    "type": "metric_builder",
                    "metric_op_options": ["count", "sum"],
                    "metric_field_options": metric_field_options,
                    "numeric_fields": numeric_fields,
                    "description": "Flow magnitude"
                }
            ],
            "default_selection": {
                "source_dimension": {"field": category_options[0] if category_options else ""},
                "target_dimension": {"field": category_options[1] if len(category_options) > 1 else category_options[0] if category_options else ""},
                "value_metric": {"op": "count", "field": "*", "as": "flow_count"}
            },
            "numeric_fields": numeric_fields,
            "categorical_fields": categorical_fields,
            "transform_template": {
                "type": "flow_aggregate",
                "source": {"from_control": "source_dimension"},
                "target": {"from_control": "target_dimension"},
                "value": {"from_control": "value_metric"}
            },
            "style": {
                "palette_mode": "flow",
                "palette_name": "viridis"
            }
        }
    
    @staticmethod
    async def _build_boxplot(
        categorical_fields: List[str],
        numeric_fields: List[str],
        suggestion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Box plot: Distribution analysis."""
        
        category_options = categorical_fields if categorical_fields else ["category"]
        metric_options = numeric_fields if numeric_fields else ["value"]
        
        return {
            "auto_render_on_select": True,
            "requires_transform": True,
            "controls": [
                {
                    "id": "category_dimension",
                    "label": "Category (Optional)",
                    "kind": "dimension",
                    "type": "field",
                    "field_options": category_options,
                    "optional": True,
                    "description": "Group distributions by category"
                },
                {
                    "id": "value_metric",
                    "label": "Value Field",
                    "kind": "metric",
                    "type": "field",
                    "field_options": metric_options,
                    "allowed_field_types": ["number", "integer"],
                    "description": "Numeric field for distribution"
                },
                {
                    "id": "show_outliers",
                    "label": "Show Outliers",
                    "kind": "option",
                    "type": "boolean",
                    "description": "Display outlier points"
                }
            ],
            "default_selection": {
                "category_dimension": None,
                "value_metric": {"field": metric_options[0] if metric_options else ""},
                "show_outliers": True
            },
            "numeric_fields": numeric_fields,
            "categorical_fields": categorical_fields,
            "transform_template": {
                "type": "distribution_aggregate",
                "category": {"from_control": "category_dimension"},
                "value": {"from_control": "value_metric"}
            },
            "style": {
                "palette_mode": "single",
                "palette_name": "theme_primary"
            }
        }


class DynamicVisualizationGenerator:
    """Orchestrates AI-powered visualization generation."""
    
    @staticmethod
    async def generate_multi_viz(
        results: List[Dict[str, Any]],
        fields_info: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete multi-viz specification with lazy-render pattern.
        
        Returns Visualization-compatible dict with:
        {
            "chart_id": "v1",
            "type": "multi_viz",
            "title": "Query Results",
            "field_schema": {field_name: {type, role_hint, cardinality}, ...},
            "aggregators": {
                "views": ["bar", "line", "pie", "table"],
                "bar": {...},
                "line": {...},
                "pie": {...},
                "table": {...}
            }
        }
        """
        
        row_count = len(results)
        
        # If no field info provided, infer from results
        if not fields_info and results:
            fields_info = [{"name": k} for k in results[0].keys()]
        
        # Analyze fields
        field_schema_list = await FieldRoleDetector.analyze_fields(fields_info or [], results)
        
        # Convert field_schema from list to dict (keyed by field name)
        field_schema_dict = {
            f["name"]: {
                "type": f["type"],
                "role_hint": f["role_hint"],
                "cardinality": f["cardinality"],
                "sample_values": f.get("sample_values", [])
            }
            for f in field_schema_list
        }
        
        # Generate aggregators
        aggregators = await DynamicAggregatorGenerator.generate_aggregators(
            field_schema_list, results, row_count
        )
        
        # Build Visualization-compatible response
        multi_viz = {
            "chart_id": "v1",
            "type": "multi_viz",
            "title": "Query Results",
            "emoji": "📊",
            "config": {
                "primary_view": "table",
                "available_views": ["table", "bar", "line", "pie", "scatter", "treemap", "funnel", "waterfall", "gantt", "sankey", "boxplot"]
            },
            "field_schema": field_schema_dict,
            "aggregators": {
                "views": ["bar", "line", "pie", "scatter", "treemap", "funnel", "waterfall", "gantt", "sankey", "boxplot", "table"],
                "bar": aggregators.get("bar", {}),
                "line": aggregators.get("line", {}),
                "pie": aggregators.get("pie", {}),
                "scatter": aggregators.get("scatter", {}),
                "treemap": aggregators.get("treemap", {}),
                "funnel": aggregators.get("funnel", {}),
                "waterfall": aggregators.get("waterfall", {}),
                "gantt": aggregators.get("gantt", {}),
                "sankey": aggregators.get("sankey", {}),
                "boxplot": aggregators.get("boxplot", {}),
                "table": aggregators.get("table", {}),
                "common": aggregators.get("common", {})
            }
        }
        
        return multi_viz
