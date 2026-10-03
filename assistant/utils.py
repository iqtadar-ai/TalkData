def generate_diff_stats(old_shape, old_cols, new_shape, new_cols):
    """
    Compares dataset state before and after transformation
    to generate human-readable UI statistics.
    """
    stats = []
    
    rows_diff = old_shape[0] - new_shape[0]
    if rows_diff > 0:
        stats.append({"label": "Rows removed", "value": rows_diff})
    elif rows_diff < 0:
        stats.append({"label": "Rows added", "value": abs(rows_diff)})
        
    added_cols = new_cols - old_cols
    dropped_cols = old_cols - new_cols
    
    # If the total number of columns is the same, it's a rename
    if len(old_cols) == len(new_cols) and added_cols and dropped_cols:
        stats.append({
            "label": "Columns renamed", 
            "value": f"{', '.join(dropped_cols)} → {', '.join(added_cols)}"
        })
    else:
        # Otherwise, process them as normal additions/removals
        if added_cols:
            stats.append({"label": "Columns added", "value": ", ".join(added_cols)})
        if dropped_cols:
            stats.append({"label": "Columns removed", "value": ", ".join(dropped_cols)})
        
    if not stats:
        stats.append({"label": "Action", "value": "Data transformed"})
        
    return stats



#------------------------------matric testing----------------------------


import os
import time
from supabase import create_client, Client


def auto_log_to_supabase(prompt, df, tool_name, is_success, execution_time=0.0, dataset_name="Unknown Dataset", notes="Auto-logged"):
    """Silently pushes execution metrics to Supabase without a manual form."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        print("AutoLogger: Supabase credentials missing.")
        return
        
    try:
        supabase: Client = create_client(url, key)
        
        data = {
            "tester_name": "AutoLogger", 
            "dataset_name": dataset_name,  
            "dataset_link": "",
            "dataset_rows": df.shape[0] if df is not None else 0,
            "dataset_columns": df.shape[1] if df is not None else 0,
            "prompt": prompt,
            "tool_routed": tool_name,  # 🎯 Will now just be "drop_column"
            "is_success": is_success,
            "execution_time": round(float(execution_time), 2),
            "human_rating": 0, 
            "tester_notes": notes  #  Now includes notes from the user
        }
        
        supabase.table("test_metrics").insert(data).execute()
        print(f"AutoLogger: Successfully logged {tool_name} to Supabase.")
        
    except Exception as e:
        print(f"AutoLogger failed to insert: {str(e)}")