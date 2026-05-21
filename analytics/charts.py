import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Set standard layout style for visual consistency
THEME_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, sans-serif", "color": "#E5E7EB"},
    "title": {"font": {"family": "Outfit, sans-serif", "size": 18, "color": "#F3F4F6"}},
    "margin": {"l": 40, "r": 40, "t": 60, "b": 40},
}

COLOR_SEQUENCE = ["#A855F7", "#3B82F6", "#10B981", "#F59E0B", "#EC4899", "#6366F1", "#8B5CF6", "#06B6D4"]

def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, color_col=None, orientation='v') -> go.Figure:
    """
    Generate a styled bar chart.
    """
    if orientation == 'h':
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title, color_discrete_sequence=COLOR_SEQUENCE, orientation='h')
    else:
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title, color_discrete_sequence=COLOR_SEQUENCE)
        
    fig.update_layout(**THEME_LAYOUT)
    fig.update_traces(marker_line_color='rgba(0,0,0,0)', marker_line_width=0, opacity=0.85)
    return fig

def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, color_col=None) -> go.Figure:
    """
    Generate a styled line chart (ideal for trends over time).
    """
    fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title, color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_layout(**THEME_LAYOUT)
    fig.update_traces(line=dict(width=3), marker=dict(size=6), opacity=0.9)
    # Enable range slider for temporal charts
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)")
    return fig

def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, color_col=None) -> go.Figure:
    """
    Generate a styled scatter plot.
    """
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title, 
                     color_discrete_sequence=COLOR_SEQUENCE,
                     color_continuous_scale=px.colors.sequential.Purples)
    fig.update_layout(**THEME_LAYOUT)
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='rgba(0,0,0,0.2)')))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig

def create_pie_chart(df: pd.DataFrame, names_col: str, values_col: str, title: str) -> go.Figure:
    """
    Generate a styled pie / donut chart.
    """
    fig = px.pie(df, names=names_col, values=values_col, title=title, 
                 color_discrete_sequence=COLOR_SEQUENCE, hole=0.4)
    fig.update_layout(**THEME_LAYOUT)
    fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#111827', width=2)))
    return fig

def create_box_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, color_col=None) -> go.Figure:
    """
    Generate a styled box plot.
    """
    fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title, color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_layout(**THEME_LAYOUT)
    fig.update_traces(marker=dict(opacity=0.7))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig

def create_histogram(df: pd.DataFrame, col: str, title: str, color_col=None, bins=30) -> go.Figure:
    """
    Generate a styled histogram.
    """
    fig = px.histogram(df, x=col, color=color_col, nbins=bins, title=title, color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_layout(**THEME_LAYOUT)
    fig.update_traces(opacity=0.8, marker_line_color='rgba(0,0,0,0)')
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig

def create_correlation_heatmap(df: pd.DataFrame, title="Correlation Matrix") -> go.Figure:
    """
    Generate an interactive correlation matrix heatmap for numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    # Remove columns that have only null or identical values
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]
    
    if numeric_df.empty:
        # Return an empty figure with a placeholder message
        fig = go.Figure()
        fig.add_annotation(text="No numeric columns available for correlation", showarrow=False, font=dict(size=14))
        fig.update_layout(**THEME_LAYOUT)
        return fig
        
    corr = numeric_df.corr().round(2)
    
    fig = px.imshow(
        corr,
        text_auto=True,
        title=title,
        color_continuous_scale=px.colors.diverging.RdBu_r,
        zmin=-1,
        zmax=1
    )
    
    fig.update_layout(**THEME_LAYOUT)
    fig.update_layout(margin={"l": 80, "r": 40, "t": 60, "b": 80})
    return fig

def create_geographic_map(df: pd.DataFrame, loc_col: str, val_col: str, title: str) -> go.Figure:
    """
    Create a map layout. If coordinates (latitude/longitude) exist, we plot using scatter_mapbox.
    Otherwise, if standard country code or names exist, we use a choropleth map.
    """
    lat_col = None
    lon_col = None
    
    for col in df.columns:
        if col.lower() in ["latitude", "lat"]:
            lat_col = col
        elif col.lower() in ["longitude", "lon", "lng"]:
            lon_col = col
            
    if lat_col and lon_col:
        fig = px.scatter_mapbox(
            df, 
            lat=lat_col, 
            lon=lon_col, 
            size=val_col, 
            color=val_col,
            title=title,
            color_continuous_scale=px.colors.sequential.Plasma,
            zoom=3,
            mapbox_style="carto-darkmatter"
        )
        fig.update_layout(**THEME_LAYOUT)
        return fig
    else:
        # Fallback choropleth or return empty/bar representation
        # Since geo coordinates aren't present, let's represent country/state aggregation as a bar chart
        # because Plotly choropleth requires ISO codes which might be missing.
        # But we'll try to do a basic choropleth using location strings as locations
        try:
            fig = px.choropleth(
                df, 
                locations=loc_col, 
                locationmode="country names", 
                color=val_col,
                title=title, 
                color_continuous_scale=px.colors.sequential.Plasma
            )
            fig.update_layout(**THEME_LAYOUT)
            fig.update_geos(projection_type="natural earth", bgcolor="rgba(0,0,0,0)", showlakes=False)
            return fig
        except:
            # If choropleth fails, do a bar chart of top 10 locations
            top_locs = df.groupby(loc_col)[val_col].sum().reset_index().sort_values(by=val_col, ascending=False).head(10)
            return create_bar_chart(top_locs, loc_col, val_col, f"{title} (Top Locations)")
