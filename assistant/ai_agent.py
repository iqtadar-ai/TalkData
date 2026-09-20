import json
import os
import hashlib
import inspect
from openai import OpenAI
from django.core.cache import cache

from .tool_registry import TOOLS

CACHE_TTL = 300

GROQ_MODEL = 'openai/gpt-oss-120b' 

class AIServiceUnavailable(Exception):
    pass

def _cache_key(prefix, *parts):
    raw = prefix + '|' + '|'.join(str(p) for p in parts)
    return 'talkdata:' + hashlib.sha256(raw.encode()).hexdigest()

def _call_groq(prompt, force_json=False, temperature=0.0):
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing. Get a free key at console.groq.com")
        
    client = OpenAI(api_key=api_key, base_url='https://api.groq.com/openai/v1')
    
    kwargs = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, # <--- Now dynamically set
    }
    
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        raise AIServiceUnavailable(f"Groq API failed: {str(e)}")
    
    
# --- Public Tools ---


def generate_tool_schemas():
    """Dynamically reads all registered tools and builds a prompt schema for the LLM."""
    schemas = []
    
    for tool_name, func in TOOLS.items():
        # Pass the ENTIRE docstring so the AI can read the parameter rules
        docstring = inspect.getdoc(func) or "No description provided."
        
        # Read the function parameters
        sig = inspect.signature(func)
        
        # Filter out 'df' since the AI doesn't pass the dataframe itself
        expected_args = [param for param in sig.parameters if param != 'df']
        
        # Create a visual representation of what the JSON args should look like
        args_example = ", ".join([f'"{arg}": "..."' for arg in expected_args])
        
        schemas.append(f"Tool: {tool_name}\nDescription:\n{docstring}\nArgs needed: {{{args_example}}}")
        
    return "\n\n".join(schemas)




def get_tool_calls(user_message, columns):

    key = _cache_key('tool_calls_groq_v8', user_message.strip().lower(), columns)
    cached = cache.get(key)
    if cached is not None:
        return cached

    # Autogenerate the manual for the LLM
    dynamic_tool_manual = generate_tool_schemas()

    prompt = f'''
You are an intelligent data routing API. Your job is to understand the user's intent and map it to the correct tool.

Available dataset columns: {columns}

AVAILABLE TOOLS:
{dynamic_tool_manual}

User request: "{user_message}"

CRITICAL RULES:
1. Output ONLY a valid JSON object with the key "actions".
2. The "tool" value MUST exactly match one of the Tools listed above.
3. The "args" object MUST contain EXACTLY the arguments specified in "Args needed" for that tool.
4. Trust the user. Extract the column name they requested exactly as they typed it. DO NOT verify if the column exists (the backend will handle validation).

EXAMPLE JSON FORMAT:
{{
  "actions": [
    {{"tool": "your_chosen_tool", "args": {{"arg1": "value1", "arg2": "value2"}}}}
  ]
}}
'''
    
    result = _call_groq(prompt, force_json=True)
    
    try:
        data = json.loads(result)
        array_only = data.get("actions", [])
        final_json_string = json.dumps(array_only)
        cache.set(key, final_json_string, CACHE_TTL)
        return final_json_string
    except:
        return result    
    
    
def explain_results(user_command, analysis_results):
    key = _cache_key('explain_groq_v4', user_command.strip().lower(), str(analysis_results))
    cached = cache.get(key)
    if cached is not None:
        return cached

    prompt = f'''
You are a sharp, direct data analyst copilot. 

Summarize the following data analysis results in 1 or 2 concise sentences.
User request: {user_command}
Results: {analysis_results}

STRICT GUIDELINES:
- Jump straight to the findings.
- NO greetings (e.g., do not say "Hey there", "Here is a snapshot").
- NO sign-offs or advice (e.g., do not say "You're ready to go!").
- Keep it under 35 words. Plain text only.
'''
    
    # Kept slightly low (0.2) so it's natural but doesn't ramble
    result = _call_groq(prompt, force_json=False, temperature=0.2) 
    
    cache.set(key, result, CACHE_TTL)
    return result