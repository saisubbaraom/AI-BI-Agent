import pandas as pd
import numpy as np
import plotly.graph_objects as go
from agents.base_agent import BaseAgent
from analytics.charts import (
    create_bar_chart, create_line_chart, create_scatter_plot,
    create_pie_chart, create_box_plot, create_histogram,
    create_correlation_heatmap, create_geographic_map
)

class VisualizationAgent(BaseAgent):
    """
    Visualization Agent: Recommends, designs, and builds interactive Plotly graphs
    tailored to the dataset's schema and domain.
    """
    def __init__(self):
        super().__init__(
            name="Visualization Agent",
            role="Specialist in designing and building beautiful, interactive data visualizations using Plotly."
        )

    def run(self, df: pd.DataFrame, domain: str, col_map: dict, suggested_charts: list = None) -> dict:
        """
        Scans columns and generates a list of recommended chart configurations and figures.
        If suggested_charts is provided (from the intelligence agent), we build those.
        Otherwise, we fall back to default template rules.
        """
        charts = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols_no_id = [c for c in numeric_cols if not c.lower().endswith("id")]
        categorical_cols = df.select_dtypes(include=[object, "category"]).columns.tolist()
        
        # 1. Build dynamic charts if suggested by Dataset Intelligence
        if suggested_charts:
            for chart_def in suggested_charts:
                c_id = chart_def.get("id")
                c_title = chart_def.get("title", "Chart")
                c_type = chart_def.get("type", "bar")
                x = chart_def.get("x_col")
                y = chart_def.get("y_col")
                color = chart_def.get("color_col")
                
                # Validation: check that the columns exist in df
                if not x or x not in df.columns:
                    continue
                if y and y not in df.columns:
                    y = None
                if color and color not in df.columns:
                    color = None
                    
                try:
                    # Let build_custom_chart handle aggregations
                    fig = self.build_custom_chart(df, chart_type=c_type, x_col=x, y_col=y, color_col=color)
                    # Update layout title to match the recommended title
                    fig.update_layout(title=c_title)
                    charts[c_id] = {
                        "type": c_type,
                        "title": c_title,
                        "fig": fig
                    }
                except Exception as e:
                    print(f"Failed to build suggested chart '{c_title}' ({c_type}) on columns x={x}, y={y}: {e}")
                    
            # Always add a correlation heatmap if not already and we have numeric columns
            if "correlation_heatmap" not in charts and len(numeric_cols_no_id) >= 2:
                try:
                    from analytics.charts import create_correlation_heatmap
                    charts["correlation_heatmap"] = {
                        "type": "heatmap",
                        "title": "Correlation Heatmap",
                        "fig": create_correlation_heatmap(df, "Numeric Feature Correlations")
                    }
                except Exception as e:
                    print(f"Failed to generate correlation heatmap: {e}")
                    
            if charts:
                return charts

        # 2. Default Fallback Matching Logic
        # 1. Heatmap (for numeric variables)
        if len(numeric_cols_no_id) >= 2:
            charts["correlation_heatmap"] = {
                "type": "heatmap",
                "title": "Correlation Heatmap",
                "fig": create_correlation_heatmap(df, "Numeric Feature Correlations")
            }
            
        # 2. Time-series Line Chart
        date_col = col_map.get("date")
        if date_col:
            # Let's find a target numeric metric to plot against date
            target_metric = None
            if domain == "Sales" and col_map.get("revenue"):
                target_metric = col_map.get("revenue")
            elif domain == "Marketing" and col_map.get("spend"):
                target_metric = col_map.get("spend")
            elif domain == "Finance" and col_map.get("revenue"):
                target_metric = col_map.get("revenue")
            elif numeric_cols_no_id:
                target_metric = numeric_cols_no_id[0]
                
            if target_metric:
                try:
                    # Group by date and sum
                    df_time = df.copy()
                    df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
                    df_time = df_time.dropna(subset=[date_col])
                    # Aggregate by day
                    df_grouped = df_time.groupby(df_time[date_col].dt.date)[target_metric].sum().reset_index()
                    df_grouped.columns = [date_col, target_metric]
                    df_grouped = df_grouped.sort_values(by=date_col)
                    
                    charts["time_series_trend"] = {
                        "type": "line",
                        "title": f"Daily Trend of {target_metric.replace('_', ' ').title()}",
                        "fig": create_line_chart(df_grouped, date_col, target_metric, f"Trend of {target_metric.replace('_', ' ').title()} Over Time")
                    }
                except Exception as e:
                    print(f"Failed to generate line chart: {e}")
                    
        # 3. Categorical Aggregations (Bar Chart and Pie Chart)
        if categorical_cols and numeric_cols_no_id:
            # Choose best category and numeric target
            cat_col = categorical_cols[0]
            num_col = None
            if domain == "Sales" and col_map.get("revenue"):
                num_col = col_map.get("revenue")
            elif domain == "Marketing" and col_map.get("conversions"):
                num_col = col_map.get("conversions")
            elif domain == "HR" and col_map.get("salary"):
                num_col = col_map.get("salary")
            else:
                num_col = numeric_cols_no_id[0]
                
            # Aggregate category
            try:
                df_cat = df.groupby(cat_col)[num_col].sum().reset_index()
                # Sort descending
                df_cat = df_cat.sort_values(by=num_col, ascending=False).head(15)
                
                charts["categorical_bar"] = {
                     "type": "bar",
                     "title": f"{num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}",
                     "fig": create_bar_chart(df_cat, cat_col, num_col, f"Total {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}")
                }
                
                # If category has <= 7 distinct values, do a pie chart too
                if df[cat_col].nunique() <= 8:
                    charts["categorical_pie"] = {
                        "type": "pie",
                        "title": f"Share of {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}",
                        "fig": create_pie_chart(df_cat, cat_col, num_col, f"Distribution of {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}")
                    }
            except Exception as e:
                print(f"Failed to generate categorical charts: {e}")
                
        # 4. Outlier/Distribution Box Plot
        if numeric_cols_no_id and categorical_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols_no_id[0]
            # Limit classes to top 5 to keep boxplot readable
            top_classes = df[cat_col].value_counts().head(5).index
            df_filtered = df[df[cat_col].isin(top_classes)]
            
            charts["distribution_boxplot"] = {
                "type": "box",
                "title": f"Distribution of {num_col.replace('_', ' ').title()} across {cat_col.replace('_', ' ').title()}",
                "fig": create_box_plot(df_filtered, cat_col, num_col, f"{num_col.replace('_', ' ').title()} Variance across Top {cat_col.replace('_', ' ').title()}s")
            }
            
        # 5. Geographic Chart (if location columns exist)
        loc_col = None
        for col in df.columns:
            if col.lower() in ["country", "state", "city", "location"]:
                loc_col = col
                break
                
        if loc_col and numeric_cols_no_id:
            num_col = col_map.get("revenue") or numeric_cols_no_id[0]
            try:
                df_geo = df.groupby(loc_col)[num_col].sum().reset_index()
                charts["geographic_map"] = {
                    "type": "map",
                    "title": f"Geographic Distribution of {num_col.replace('_', ' ').title()} by {loc_col.replace('_', ' ').title()}",
                    "fig": create_geographic_map(df_geo, loc_col, num_col, f"Geographic Breakdown by {loc_col.replace('_', ' ').title()}")
                }
            except Exception as e:
                print(f"Failed to generate geographic map: {e}")
                
        return charts

    def build_custom_chart(self, df: pd.DataFrame, chart_type: str, x_col: str, y_col: str = None, color_col: str = None) -> go.Figure:
        """
        Creates a custom Plotly figure based on user selections in the builder UI.
        Handles data aggregation and variable mismatch warnings dynamically.
        """
        # Clean None strings from Streamlit selectbox values
        if color_col == "None" or not color_col:
            color_col = None
        if y_col == "None" or not y_col:
            y_col = None

        title = f"Custom {chart_type.title()} Plot"
        df_plot = df.copy()

        # Handle charts that require Y aggregation and grouping (bar, line, pie)
        if chart_type in ["bar", "line", "pie"] and y_col:
            # Check if there are duplicates on X or (X + color) to decide if we aggregate
            group_cols = [x_col]
            if color_col and color_col in df_plot.columns and color_col != x_col:
                group_cols.append(color_col)
                
            has_duplicates = df_plot.duplicated(subset=group_cols).any()
            
            if has_duplicates:
                try:
                    # Check if Y is numeric
                    is_numeric = pd.api.types.is_numeric_dtype(df_plot[y_col])
                    if is_numeric:
                        df_plot = df_plot.groupby(group_cols)[y_col].sum().reset_index()
                        title = f"Total {y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}"
                    else:
                        # For non-numeric Y, aggregate by COUNTing occurrences (frequency of Y)
                        df_plot = df_plot.groupby(group_cols)[y_col].count().reset_index()
                        df_plot.rename(columns={y_col: f"Count of {y_col}"}, inplace=True)
                        y_col = f"Count of {y_col}"
                        title = f"Count of {y_col.replace('Count of ', '').replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}"
                except Exception:
                    # Fallback to original df on grouping error
                    pass
            else:
                # If no duplicates, check if Y is non-numeric and counting is safer
                is_numeric = pd.api.types.is_numeric_dtype(df_plot[y_col])
                if not is_numeric:
                    pass

        # Validate Y is numeric for box plots
        elif chart_type == "box" and y_col:
            is_numeric = pd.api.types.is_numeric_dtype(df_plot[y_col])
            if not is_numeric:
                raise ValueError(f"Y Axis '{y_col}' must be numerical for a Box Plot. Please select a numeric column.")

        # Build respective chart with clean params
        if chart_type == "bar":
            return create_bar_chart(df_plot, x_col, y_col, title, color_col=color_col)
        elif chart_type == "line":
            return create_line_chart(df_plot, x_col, y_col, title, color_col=color_col)
        elif chart_type == "scatter":
            return create_scatter_plot(df_plot, x_col, y_col, title, color_col=color_col)
        elif chart_type == "pie":
            # For pie, values must be numeric
            is_numeric = pd.api.types.is_numeric_dtype(df_plot[y_col])
            if not is_numeric:
                df_grouped = df.groupby(x_col)[y_col].count().reset_index()
                df_grouped.rename(columns={y_col: f"Count of {y_col}"}, inplace=True)
                y_col = f"Count of {y_col}"
                title = f"Count of {y_col.replace('Count of ', '').replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}"
                return create_pie_chart(df_grouped, x_col, y_col, title)
            else:
                df_grouped = df.groupby(x_col)[y_col].sum().reset_index()
                return create_pie_chart(df_grouped, x_col, y_col, title)
        elif chart_type == "box":
            return create_box_plot(df_plot, x_col, y_col, title, color_col=color_col)
        elif chart_type == "histogram":
            return create_histogram(df_plot, x_col, title, color_col=color_col)
        elif chart_type == "heatmap":
            return create_correlation_heatmap(df_plot, "Correlation Map")
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")
