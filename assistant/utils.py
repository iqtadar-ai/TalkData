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