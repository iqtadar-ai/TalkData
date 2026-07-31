TOOLS = {}

TOOL_TYPES ={}

def tool(tool_type='transform'):
    def decorator(func):
        
        TOOLS[func.__name__] = func
        TOOL_TYPES[func.__name__] = tool_type
        return func 
    
    return decorator