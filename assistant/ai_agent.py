import json
import os
import time
import hashlib
from google import genai
from google.genai import errors as genai_errors
from django.core.cache import cache

from .tool_registry import TOOLS

# Remove module-level client initialization. We will initialize them lazily.
MODEL = 'gemini-2.5-flash'  
GROQ_MODEL = 'llama-3.3-70b-versatile'

MAX_RETRIES = 3
BASE_DELAY = 1.5  
CACHE_TTL = 300    


class AIServiceUnavailable(Exception):
    """Raised when no AI provider (primary or fallback) could serve the request."""
    pass


def _cache_key(prefix, *parts):
    raw = prefix + '|' + '|'.join(str(p) for p in parts)
    return 'talkdata:' + hashlib.sha256(raw.encode()).hexdigest()

def parse_tool_calls(response_text):
    cleaned = response_text.replace('```json', '').replace('```', '').strip()
    data = json.loads(cleaned)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError('AI response must be a JSON object or array')


# --- Lazy Client Initializers ---

def get_gemini_client():
    return genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

def get_groq_client():
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=groq_api_key, base_url='https://api.groq.com/openai/v1')


# --- Call Functions ---

def _call_gemini_with_retry(prompt):
    client = get_gemini_client()
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return response.text

        except genai_errors.APIError as e:
            last_error = e
            status = getattr(e, 'code', None) or getattr(e, 'status_code', None)
            
            # If it's a rate limit or server error, back off and retry
            if status not in (429, 500, 502, 503, 504): 
                if attempt < MAX_RETRIES - 1:
                    print(f'Gemini retry {attempt + 1}/{MAX_RETRIES} after status {status}')
                    time.sleep(BASE_DELAY * (2 ** attempt)) 
                    continue
                raise

    raise last_error


def _call_groq(prompt):
    client = get_groq_client()
    if not client:
        raise ValueError('No fallback provider configured (GROQ_API_KEY not set or loaded)')

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return response.choices[0].message.content


def _call_with_fallback(prompt):
    """Try Gemini first; if it fails for API reasons, try Groq."""
    try:
        return _call_gemini_with_retry(prompt)

    except genai_errors.APIError as gemini_error:
        # We catch ANY Gemini APIError here. If Gemini is down, rate-limited,
        # or giving 500 errors, we should always try our Groq fallback!
        try:
            return _call_groq(prompt)
            
        except Exception as groq_error:
            # If Groq ALSO fails (or wasn't configured properly), raise our custom 
            # exception so the view can catch it and show the "friendly" busy message.
            raise AIServiceUnavailable(
                f'Gemini unavailable ({gemini_error}); Groq fallback also failed ({groq_error})'
            ) from groq_error


# --- Public Tools ---

def get_tool_calls(user_message, columns):
    key = _cache_key('tool_calls', user_message.strip().lower(), columns)
    cached = cache.get(key)
    if cached is not None:
        return cached
        
    # Get the exact list of tools you have programmed
    available_tools = list(TOOLS.keys())

    prompt = f'''
You are a data transformation assistant.

Available columns:
{columns}

ALLOWED TOOLS: 
{available_tools}
You MUST choose a tool ONLY from the list of ALLOWED TOOLS above. Do not invent tool names.

User request:
{user_message}

If the request requires ONE action, return a single JSON object.

Example: 
{{"tool": "drop_column", "args": {{"column": "salary"}}}} 

If the request requires MULTIPLE actions, return a JSON array of objects. 

Example: [ 
{{"tool": "drop_column", "args": {{"column": "salary"}}}},
{{"tool": "count_nulls", "args": {{}}}} , 
{{"tool": "max_values", "args": {{}}}}
] 

Return ONLY valid JSON. No markdown, no explanations.

For chart requests use:

{{
    "tool": "create_chart", 
    "args": {{ 
    "chart_type": "bar|line|scatter|histogram|box", 
    "x": "column_name",
    "y": "column_name" 
    }} 
}}

Examples:

'Create a bar chart of sales by region'
'Plot revenue vs profit'
'Show a histogram of age'
'Create a box plot of salary by department'
'''



    result = _call_with_fallback(prompt)
    cache.set(key, result, CACHE_TTL)
    return result

def explain_results(user_command, analysis_results):
    key = _cache_key(
        'explain',
        user_command.strip().lower(), 
        str(analysis_results)
        )
    
    cached = cache.get(key)
    if cached is not None:
        return cached

    prompt = f'''
You are explaining data analysis results to a non-technical user.

User request:
{user_command}

Results:
{analysis_results}

Write a clear summary using:
- a short opening sentence,
- grouped bullet points,
- simple column names,
- one useful insight if appropriate.

Do not show JSON, Python dictionaries, or technical terms like 'null object' or 'dtype'.
Keep the response under 120 words unless the user asked for detailed analysis.'''
    
    result = _call_with_fallback(prompt)
    cache.set(key, result, CACHE_TTL)
    return result