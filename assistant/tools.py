import pandas as pd
from .tool_registry import tool

import plotly.express as px
from plotly.io import to_html
from .tool_registry import tool


#----------------------------Data Transformation Tools----------------------------


def resolve_column(df, name): 
    mapping = {c.lower(): c for c in df.columns}
    
    if name.lower() not in mapping:
        raise ValueError(f'Column {name} not found') 
    
    return mapping[name.lower()]


@tool('transform')
def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove a column from the dataframe."""

    if column not in df.columns:
        raise ValueError(f'Column {column} not found')

    return df.drop(columns=[column])


@tool('transform')
def rename_column(
    df: pd.DataFrame,
    old_name: str,
    new_name: str
) -> pd.DataFrame:
    """Rename a dataframe column."""

    real_old_name = resolve_column(df, old_name)
    
    if real_old_name not in df.columns:
        raise ValueError(f'Column {old_name} not found')

    return df.rename(columns={real_old_name: new_name})


@tool('transform')
def add_column(
    df: pd.DataFrame,
    new_column: str,
    expression: str
) -> pd.DataFrame:
    """Create a new column using a pandas expression."""

    df[new_column] = df.eval(expression)

    return df


#----------------------------Analtics Tools----------------------------


@tool('analysis')
def count_nulls(df: pd.DataFrame) -> dict:
    nulls = df.isnull().sum()
    items = []
    
    for col, count in nulls.items():
        if count > 0:
            items.append({ 'label': col, 'value': int(count) })
            
    insight = None
    
    if items:
        top = max(items, key=lambda x: x['value'])
        insight = f"{top['label']} has the highest number of missing values ({top['value']:,})."
    
    return {
            'type': 'metric_list', 
            'title': 'Missing Values',
            'items': items,
            'insight': insight 
            }

@tool('analysis')
def max_values(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include='number')
    items = []
    
    for col in numeric.columns:
        items.append({ 'label': col, 'value': float(numeric[col].max()) })
        
    return { 
            'type': 'metric_list',
            'title': 'Maximum Values', 
            'items': items, 
            'insight': None
            }

#----------------------------Charting Tool----------------------------

@tool(tool_type='analysis') 
def create_chart(df, chart_type, x, y=None):
        
        x_col = resolve_column(df, x) 
        y_col = resolve_column(df, y) 
        
        # --- THE SAFETY VALVE ---
        # 500,000+ rows will crash the browser. We must aggregate or sample the data first.
        plot_df = df.copy()
        
        if len(plot_df) > 5000:
            if chart_type in ['bar', 'line'] and y_col:
                try:
                    # Group the data and get the top 50 so the chart remains readable
                    plot_df = plot_df.groupby(x_col)[y_col].sum().reset_index()
                    plot_df = plot_df.sort_values(by=y_col, ascending=False).head(50)
                except Exception:
                    # Fallback if the grouping fails
                    plot_df = plot_df.sample(5000)
            else:
                # For scatter, box, or histogram, just take a random sample
                plot_df = plot_df.sample(5000)
        # -------------------------

        color_seq = ['#8b5cf6'] 
        
        # Note: We are now passing 'plot_df' to Plotly instead of the massive 'df'
        if chart_type == 'bar':
            fig = px.bar(plot_df, x=x_col, y=y_col, color_discrete_sequence=color_seq)
        elif chart_type == 'line':
            fig = px.line(plot_df, x=x_col, y=y_col, color_discrete_sequence=color_seq)
        elif chart_type == 'scatter':
            fig = px.scatter(plot_df, x=x_col, y=y_col, color_discrete_sequence=color_seq)
        elif chart_type == 'histogram':
            fig = px.histogram(plot_df, x=x_col, color_discrete_sequence=color_seq)
        elif chart_type == 'box':
            fig = px.box(plot_df, x=x_col, y=y_col, color_discrete_sequence=color_seq)
        else:
            raise ValueError(f'Unsupported chart type: {chart_type}')

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color='#cbd5e1',
            xaxis=dict(
                gridcolor='#334155',
                zerolinecolor='#334155',
                tickangle=-45,
                automargin=True
            ),
            yaxis=dict( 
                gridcolor='#334155',
                zerolinecolor='#334155',
                automargin=True
            ), 
            height=420,
            margin=dict(l=10, r=10, t=40, b=10) 
        )
        
        card = { 
                'type': 'chart', 
                'title': f'{chart_type.title()} Chart', 
                'chart_html': fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    config={ 
                            'responsive': True,
                            'displayModeBar': False
                            }
                    ) 
                } 
        
        return card