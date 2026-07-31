import os
import uuid
import pandas as pd
import json


from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from .ai_agent import get_tool_calls, explain_results, AIServiceUnavailable 
from . import tools
from .tool_registry import TOOLS, TOOL_TYPES

# ---------- Load uploaded file into a DataFrame ----------

def load_uploaded_dataframe(path):
    if path.endswith('.csv'):
        return pd.read_csv(path)

    elif path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)

    elif path.endswith('.json'):
        return pd.read_json(path)

    elif path.endswith('.parquet'):
        return pd.read_parquet(path)

    else:
        raise ValueError('Unsupported file format')


# ---------- Save internal working copy as Parquet ----------

def save_internal_dataframe(df):
    working_dir = settings.MEDIA_ROOT / 'working' 
    working_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f'{uuid.uuid4()}.parquet'
    path = working_dir / filename
    
    # Normalize object columns for Parquet 
     
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('string') 
    df.to_parquet(path) 
        
    return str(path)


# ---------- Load internal Parquet file ----------

def load_internal_dataframe(path):
    return pd.read_parquet(path)


# ---------- Basic command execution ----------

# def execute_basic_command(df, command):
#     cmd = command.lower().strip()

#     # remove salary
#     if cmd.startswith('remove '):
#         column = cmd.replace('remove ', '').replace(' column', '').strip()

#         column_map = {c.lower(): c for c in df.columns}

#         if column in column_map:
#             real_column = column_map[column]

#             df = tools.drop_column(df, real_column)

#             return df, f'Removed column: {real_column}'

#         return df, f'Column "{column}" not found'

#     return df, 'Command not recognized'


# ---------- Main view ----------

 
def home(request):
    context = {}
 
    internal_path = request.session.get('dataset_path')
    
    if internal_path and not os.path.exists(internal_path):
        request.session.pop('dataset_path', None)
        internal_path = None
        context['error'] = 'Your previous dataset expired or was removed from the server. Please re-upload.'
 
    # ---------- Handle command form ----------
    if request.method == 'POST' and request.POST.get('command'):
        command = request.POST.get('command')
 
        if internal_path:
            df = load_internal_dataframe(internal_path)
            ai_call_failed = False
            dataset_changed = False
 
            try:
                response = get_tool_calls(command, df.columns.tolist())
                
                if not response: 
                    message = 'The AI assistant did not return a valid response.'
                    ai_call_failed = True
 
            except AIServiceUnavailable:
                message = ("The AI assistant is a bit busy right now — please try "
                           "again in a moment. Nothing in your dataset was changed.")
                ai_call_failed = True
 
            except Exception as e:
                message = f"AI error: {type(e).__name__}: {str(e)}"
                ai_call_failed = True
 
            if not ai_call_failed:
                cleaned = response.replace('```json', '').replace('```', '').strip()
 
                try:
                    tool_calls = json.loads(cleaned)
 
                    if isinstance(tool_calls, dict):
                        tool_calls = [tool_calls]
 
                    executed = []
                    errors = []
                    analysis_cards = []
                    context.setdefault('charts', [])
                    context.setdefault('analysis_cards', [])
                    
                    for call in tool_calls:
                        try:
                            tool_name = call['tool']
                            args = call.get('args', {})
 
                            if tool_name not in TOOLS:
                                errors.append(f'Tool {tool_name} is not allowed')
                                continue
 
                            tool = TOOLS[tool_name]
                            result = tool(df, **args)
 
                            if TOOL_TYPES[tool_name] == 'transform':
                                df = result
                                dataset_changed = True
                                executed.append(f'{tool_name} {args}')
                            else:
                                analysis_cards.append(result)
                                executed.append(f'{tool_name} completed')
 
                        except KeyError as e:
                            errors.append(f'Malformed tool call, missing key: {e}')
                            
                        except Exception as e:
                            errors.append(f'{call.get("tool", "unknown")}: {str(e)}')

                    charts = [c for c in analysis_cards if c.get('type') == 'chart']
                    metric_cards = [c for c in analysis_cards if c.get('type') != 'chart']
                    
                    context['charts'] = charts
                    context['analysis_cards'] = metric_cards
                    
 
                    parts = []
                    if executed:
                        parts.append('Executed: ' + '; '.join(executed))
                    if errors:
                        parts.append('Errors: ' + '; '.join(errors))
                    message = ' | '.join(parts) if parts else 'No changes made'
 
                    if analysis_cards:
                        try:
                            context['explanation'] = explain_results(command, analysis_cards)
                            
                        except AIServiceUnavailable:
                            context['explanation'] = None
                            message += " | (Explanation unavailable — AI assistant is busy)"
                            
                        except Exception:
                            context['explanation'] = None
 
                except json.JSONDecodeError as e:
                    message = f'Could not parse the AI response as JSON: {e}' 
                    context['raw_ai_response'] = cleaned
                    
                except Exception as e:
                    message = f'Error: {str(e)}'
 
            if dataset_changed:
                    df.to_parquet(internal_path)
            context['message'] = message
 
        else:
            context['error'] = 'No dataset found in session'
 
    # ---------- Handle upload form ----------
    elif request.method == 'POST' and request.FILES.get('dataset'):
        file = request.FILES['dataset']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT / 'uploads')
        filename = fs.save(file.name, file)
        uploaded_path = fs.path(filename)
 
        try:
            df = load_uploaded_dataframe(uploaded_path)
            internal_path = save_internal_dataframe(df)
            request.session['dataset_path'] = internal_path
            context['message'] = 'Dataset uploaded and converted to internal Parquet format'
        except Exception as e:
            context['error'] = str(e)
            internal_path = None
 
    
    # ---------- Build preview on EVERY request ----------
    if internal_path:
        df = load_internal_dataframe(internal_path)
 
        context['shape'] = df.shape
        context['columns'] = df.columns.tolist()
 
        # 1. Grab the string value to highlight the correct button
        current_view = request.GET.get('view', 'preview')
        context['preview_mode'] = current_view # <--- THIS keeps the button blue
 
        # 2. Slice the dataframe based on the button clicked
        if current_view == 'all_columns':
            preview_df = df.head(10)
        elif current_view == 'all_rows':
            preview_df = df.iloc[:100, :15]
        elif current_view == 'full':
            preview_df = df.head(100)
        else:
            preview_df = df.iloc[:10, :15]
 
        # 3. Save the HTML table to 'preview', which your template is looking for
        context['preview'] = preview_df.to_html( # <--- Changed this key to 'preview'
            classes='table table-striped table-dark-custom',
            index=False
        )
 
    return render(request, 'assistant/home.html', context)