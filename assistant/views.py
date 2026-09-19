import os
import uuid
from django.template import context
from httpx import request
import pandas as pd
import json
import time

from .ai_agent import get_tool_calls, explain_results, AIServiceUnavailable 
from . import tools
from .tool_registry import TOOLS, TOOL_TYPES

from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.contrib import messages
from django.shortcuts import render, redirect
from supabase import create_client, Client

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

    # Ensure chat_history exists in session
    if 'chat_history' not in request.session:
        request.session['chat_history'] = []
    
    chat_history = request.session['chat_history']




    # ---------- Handle command form ----------
    if request.method == 'POST' and request.POST.get('command'):
        command = request.POST.get('command')
        
        # Add user message to history
        current_time = time.strftime("%I:%M %p")
        chat_history.append({"role": "user", "text": command, "time": current_time})

        if internal_path:
            df = load_internal_dataframe(internal_path)
            
            # Snapshots for diffing stats
            old_shape = df.shape
            old_cols = set(df.columns)
            
            ai_call_failed = False
            dataset_changed = False
            ai_text = "I've processed your request."
            stats = []

            try:
                start_time = time.perf_counter()
                response = get_tool_calls(command, df.columns.tolist())
                
                if not response: 
                    ai_text = 'The AI assistant did not return a valid response.'
                    ai_call_failed = True

            except AIServiceUnavailable as e:
                ai_text = f"AI Service Unavailable: {str(e)}"
                ai_call_failed = True
            except Exception as e:
                ai_text = f"AI error: {str(e)}"
                ai_call_failed = True

            if not ai_call_failed:
                cleaned = response.replace('```json', '').replace('```', '').strip()

                try:
                    tool_calls = json.loads(cleaned)
                    
                    # Guardrail: Catch if the AI refused to guess and returned an empty array
                    if not tool_calls:
                        ai_text = "I couldn't find that exact column in the dataset. Please check the spelling and try again."
                        ai_call_failed = True
                    
                    if not ai_call_failed:
                        if isinstance(tool_calls, dict):
                            tool_calls = [tool_calls]

                        analysis_cards = []
                        context.setdefault('charts', [])
                        
                        for call in tool_calls:
                            # 1. Guardrail: Ensure the call itself is a valid dictionary
                            if not isinstance(call, dict):
                                raise ValueError("AI hallucinated the JSON structure. Expected a dictionary.")

                            tool_name = call.get('tool')
                            args = call.get('args', {})

                            # 2. Guardrail: Catch if the AI returned a dict/list instead of a string for the tool name
                            if not isinstance(tool_name, str):
                                raise ValueError(f"AI schema error: 'tool' must be a text string, but it returned a {type(tool_name).__name__}. Try your prompt again.")

                            # 3. Guardrail: The trap for hallucinated tool names
                            if tool_name not in TOOLS:
                                raise ValueError(f"AI hallucinated an unknown tool: '{tool_name}'")

                            # If it passes all guardrails, execute the tool
                            if tool_name in TOOLS:
                                tool = TOOLS[tool_name]
                                result = tool(df, **args)

                                if TOOL_TYPES.get(tool_name) == 'transform':
                                    df = result
                                    dataset_changed = True
                                else:
                                    analysis_cards.append(result)
                        # Stop timer
                        end_time = time.perf_counter()
                        exec_time = round(end_time - start_time, 2)

                        # Build stats if dataset changed
                        if dataset_changed:
                            new_shape = df.shape
                            new_cols = set(df.columns)
                            
                            rows_diff = old_shape[0] - new_shape[0]
                            if rows_diff > 0:
                                stats.append({"label": "Rows removed", "value": rows_diff})
                            elif rows_diff < 0:
                                stats.append({"label": "Rows added", "value": abs(rows_diff)})
                                
                            added_cols = new_cols - old_cols
                            if added_cols:
                                stats.append({"label": "Columns added", "value": ", ".join(added_cols)})
                                
                            dropped_cols = old_cols - new_cols
                            if dropped_cols:
                                stats.append({"label": "Columns removed", "value": ", ".join(dropped_cols)})
                                
                            if not stats:
                                stats.append({"label": "Action", "value": "Data transformed"})
                                
                            ai_text = f"I've updated the dataset based on your instructions."

                        # Generate LLM explanation if there are analysis cards
                        if analysis_cards:
                            try:
                                charts = [c for c in analysis_cards if c.get('type') == 'chart']
                                context['charts'] = charts
                                if charts:
                                    ai_text = "I've generated the visualization for you. It's ready in the Visualization tab."
                                else:
                                    ai_text = explain_results(command, analysis_cards)
                            except Exception:
                                pass

                        # Always add execution time to stats
                        stats.append({"label": "Execution time", "value": f"{exec_time} seconds"})

                except json.JSONDecodeError:
                    ai_text = "Could not parse the AI response. Please try phrasing it differently."
                except Exception as e:
                    ai_text = f"Error during execution: {str(e)}"

            if dataset_changed:
                df.to_parquet(internal_path)
            
            # Add AI response to history
            chat_history.append({
                "role": "ai", 
                "text": ai_text, 
                "stats": stats, 
                "time": time.strftime("%I:%M %p")
            })
            
            # Save session
            request.session['chat_history'] = chat_history
            request.session.modified = True
            
        context['chat_history'] = chat_history

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
            context['message'] = 'Dataset uploaded successfully.'
        except Exception as e:
            context['error'] = str(e)
            internal_path = None
            
        # Ensure chat history is reset on new upload and passed to template
        request.session['chat_history'] = []
        context['chat_history'] = []
 
    
    # ---------- Build preview on EVERY request ----------
    # If no POST action occurred, we still need to pass chat history to a GET request
    if 'chat_history' not in context:
        context['chat_history'] = chat_history

    if internal_path:
        df = load_internal_dataframe(internal_path)
 
        context['shape'] = df.shape
        context['columns'] = df.columns.tolist()
 
        # 1. Grab the string value to highlight the correct button
        current_view = request.GET.get('view', 'preview')
        context['preview_mode'] = current_view 
 
        # 2. Slice the dataframe based on the button clicked
        if current_view == 'all_rows':
            preview_df = df.iloc[:500, :]
        elif current_view == 'full':
            preview_df = df.iloc[:2000, :]
        else:
            preview_df = df.iloc[:50, :]
 
        # 3. Save the HTML table to 'preview', which your template is looking for
        context['preview'] = preview_df.to_html( 
            classes='table table-striped table-dark-custom',
            index=True,
            justify='left'
        )
 
    return render(request, 'assistant/home.html', context)




#------------------------------matric testing----------------------------

def test_matrix(request):
    if request.method == 'POST':
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            messages.error(request, "Supabase credentials missing in .env file.")
            return redirect('test_matrix')
            
        try:
            supabase: Client = create_client(url, key)
            
            data = {
                "tester_name": request.POST.get('tester_name'),
                "dataset_name": request.POST.get('dataset_name'),
                "dataset_link": request.POST.get('dataset_link', ''), # New field added here
                "prompt": request.POST.get('prompt'),
                "tool_routed": request.POST.get('tool_routed'),
                "is_success": request.POST.get('is_success') == 'True',
                "execution_time": float(request.POST.get('execution_time') or 0.0),
                "human_rating": int(request.POST.get('rating')),
                "tester_notes": request.POST.get('notes', '')
            }
            
            supabase.table("test_metrics").insert(data).execute()
            messages.success(request, "Test logged successfully! Ready for the next one.")
            
        except Exception as e:
            messages.error(request, f"Failed to log test: {str(e)}")
            
        return redirect('test_matrix')

    return render(request, 'assistant/matrix.html')