import pandas as pd
from .tool_registry import tool

import plotly.express as px
from plotly.io import to_html
from .tool_registry import tool



#----------------------------Helper  Functions----------------------------

def resolve_column(df, name): 
    # Creates a dictionary mapping lowercase names to the actual column names
    mapping = {str(c).lower().strip(): c for c in df.columns}
    
    clean_name = str(name).lower().strip()
    if clean_name not in mapping:
        raise ValueError(f"Column '{name}' not found. Available columns: {list(df.columns)}") 
    
    return mapping[clean_name]






#=====================================================================================================


#---------------------------------------Data Transformation Tools----------------------------

#=====================================================================================================





#=====================================================================================================
#                                            Drop Column Tool
#=====================================================================================================


@tool('transform')
def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove a column from the dataframe."""
    real_column = resolve_column(df, column)
    return df.drop(columns=[real_column])



#===================================================================================================== 
#                                        Rename Column Tool
#=====================================================================================================


@tool('transform')
def rename_column(df: pd.DataFrame, new_name: str, old_name: str = None, column: str = None) -> pd.DataFrame:
    """
    Rename a dataframe column.
    
    Args:
        new_name: The new name for the column.
        old_name: The current name of the column.
        column: The current name of the column.
    """
    # Grab whichever parameter the AI decided to send
    target_name = old_name or column
    
    # Pass it to your resolver
    real_old_name = resolve_column(df, target_name)
    return df.rename(columns={real_old_name: new_name})



#===================================================================================================
#                                          Change Data type Tool
#===================================================================================================


@tool('transform')
def change_column_type(df: pd.DataFrame, column: str, new_type: str) -> pd.DataFrame:
    """
    Change the data type of a column.
    
    Args:
        column: The exact name of the column to modify.
        new_type: The target data type. Options: 'numeric', 'datetime', 'string', 'boolean', 'category'.
    """
    real_col = resolve_column(df, column)
    new_type = new_type.lower()
    
    try:
        if new_type in ['numeric', 'int', 'float', 'integer']:
            df[real_col] = pd.to_numeric(df[real_col], errors='coerce')
        elif new_type in ['datetime', 'date', 'time']:
            df[real_col] = pd.to_datetime(df[real_col], errors='coerce')
        elif new_type in ['string', 'str', 'text']:
            df[real_col] = df[real_col].astype('string')
        elif new_type in ['boolean', 'bool']:
            df[real_col] = df[real_col].astype('bool')
        elif new_type == 'category':
            df[real_col] = df[real_col].astype('category')
        else:
            raise ValueError(f"Unsupported data type target: {new_type}")
    except Exception as e:
        raise ValueError(f"Could not convert column '{real_col}' to {new_type}. Error: {str(e)}")
        
    return df




#=========================================================================================================
#                                          Filter Rows Tool
#=========================================================================================================

@tool('transform')
def filter_rows(df: pd.DataFrame, column: str, operator: str, value: str = "") -> pd.DataFrame:
    """
    Filter rows based on a mathematical or text condition.
    Use this tool when the user says: "filter", "only show me", "keep rows", or "where".
    Do NOT use add_column for filtering.
    
    Args:
        column: The exact name of the column to filter on.
        operator: Must be one of: '==', '!=', '>', '<', '>=', '<=', 'contains', 'not_contains', 'is_null', 'not_null'.
        value: The value to compare against (leave empty for is_null/not_null).
    """
    real_col = resolve_column(df, column)
    
    # 1. Handle missing value filters (no 'value' needed)
    if operator == 'is_null':
        return df[df[real_col].isna()]
    elif operator == 'not_null':
        return df[df[real_col].notna()]
        
    # 2. Handle text search filters
    if operator == 'contains':
        return df[df[real_col].astype(str).str.contains(str(value), na=False, case=False)]
    elif operator == 'not_contains':
        return df[~df[real_col].astype(str).str.contains(str(value), na=False, case=False)]
        
    # 3. Handle math and exact match filters
    # We try to convert the string value to match the column's data type
    col_type = df[real_col].dtype
    try:
        if pd.api.types.is_numeric_dtype(col_type):
            val = float(value)
        elif pd.api.types.is_datetime64_any_dtype(col_type):
            val = pd.to_datetime(value)
        else:
            val = str(value)
    except ValueError:
        # If conversion fails, fallback to string comparison
        val = str(value)

    # Apply the mathematical filters
    if operator == '==':
        return df[df[real_col] == val]
    elif operator == '!=':
        return df[df[real_col] != val]
    elif operator == '>':
        return df[df[real_col] > val]
    elif operator == '<':
        return df[df[real_col] < val]
    elif operator == '>=':
        return df[df[real_col] >= val]
    elif operator == '<=':
        return df[df[real_col] <= val]
    else:
        raise ValueError(f"Unsupported operator: {operator}")
    
    
    
#==========================================================================================================
#                                        Duplicate Column Tool
#==========================================================================================================    
    
    
@tool('transform')
def duplicate_column(df: pd.DataFrame, column: str, new_column_name: str) -> pd.DataFrame:
    """
    Duplicate or copy an existing column into a new column.
    Use this when the user says "duplicate", "copy", or "clone" a column.
    
    Args:
        column: The exact name of the existing column to duplicate.
        new_column_name: The name for the newly copied column.
    """
    real_col = resolve_column(df, column)
    df[new_column_name] = df[real_col].copy()
    return df

#==========================================================================================================
#                                       Move Column Tool
#==========================================================================================================



@tool('transform')
def move_column(df: pd.DataFrame, column: str, target_column: str, position: str) -> pd.DataFrame:
    """
    Move a column to a specific position (before or after another column).
    Use this when the user says "move", "reorder", "place before", or "put after".
    
    Args:
        column: The exact name of the column to move.
        target_column: The exact name of the reference column to position it relative to.
        position: Must be exactly "before" or "after".
    """
    col_to_move = resolve_column(df, column)
    ref_col = resolve_column(df, target_column)
    
    if position not in ['before', 'after']:
        raise ValueError("Position must be 'before' or 'after'.")
        
    cols = list(df.columns)
    cols.remove(col_to_move)
    
    # Find where to insert it based on the target column
    ref_idx = cols.index(ref_col)
    insert_idx = ref_idx if position == 'before' else ref_idx + 1
    
    cols.insert(insert_idx, col_to_move)
    return df[cols]


#=========================================================================================================
#                                        Add Column Tool
#=========================================================================================================

@tool('transform')
def add_column(df: pd.DataFrame, new_column: str, expression: str) -> pd.DataFrame:
    """
    Create a new column using a pandas mathematical expression.
    The expression should use column names directly, e.g., 'Revenue - Cost' or 'Price * 1.2'.
    """
    df_copy = df.copy()
    try:
        df_copy[new_column] = df_copy.eval(expression)
        return df_copy
    except Exception as e:
        # If the AI hallucinates bad syntax, this safely kicks the error back to your view loop
        raise ValueError(f"Failed to evaluate expression '{expression}'. Make sure to use exact column names without 'df[]' wrappers. Error: {str(e)}")










#=====================================================================================================

#----------------------------------------Analtics Tools---------------------------------------------

#=====================================================================================================






#===================================================================================================
#                                        Count Nulls Tool
#===================================================================================================

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
    
    
    
    
#===================================================================================================
#                                      Calculate Statistic Tool
#===================================================================================================
    
@tool('analysis')
def calculate_statistic(df, column_name: str, stat_type: str):
    """
    Calculates a specific statistical value for a numeric column.
    stat_type must be one of: 'mean', 'median', 'mode', 'min', 'max', 'sum', 'std'
    """
    if isinstance(column_name, list):
        return {"type": "stat", "error": "Please request one column at a time."}

    try:
        actual_col = resolve_column(df, column_name)
    except ValueError as e:
        return {"type": "stat", "error": str(e)}

    if not pd.api.types.is_numeric_dtype(df[actual_col]):
        return {"type": "stat", "error": f"Cannot calculate {stat_type} on non-numeric column '{actual_col}'."}

    try:
        result = None
        if stat_type == 'mean':
            result = df[actual_col].mean()
        elif stat_type == 'median':
            result = df[actual_col].median()
        elif stat_type == 'mode':
            result = df[actual_col].mode()[0]
        elif stat_type == 'min':
            result = df[actual_col].min()
        elif stat_type == 'max':
            result = df[actual_col].max()
        elif stat_type == 'sum':
            result = df[actual_col].sum()
        elif stat_type == 'std':
            result = df[actual_col].std()
        else:
            return {"type": "stat", "error": f"'{stat_type}' is not supported."}
            
        # Return a dictionary so it appends nicely to analysis_cards
        return {
            "type": "stat", 
            "column": actual_col, 
            "statistic": stat_type, 
            "value": round(result, 2)
        }
    except Exception as e:
        return {"type": "stat", "error": str(e)}
    
    
    
    
    
    
    
    
    
    
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