# Trying to run such big Codebases with GPT5 often leads to output connection errors
# output the Complete functions that need change - make a draft - seems to work

import sys
import json
import time
import re
import requests
from PyQt5 import QtWidgets, QtGui, QtCore
import markdown  # For markdown support
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

# -------------------------------------------------------
# Global Variables
# -------------------------------------------------------
conversation_history = []  # Each element is a complete turn (including prefix)
selected_model = ""  # Will be populated when models are fetched
available_models = []  # List of model IDs
available_models_display = []  # List of display names for the UI
display_to_model_map = {}  # Maps from display names to actual model IDs
server_ip = "192.168.3.1"  # Default server IP
api_port = "1234"  # Default LM Studio API port
remote_port = "5051"  # Default remote server port for model unloading
lmstudio_server = f"http://{server_ip}:{api_port}"  # LM Studio API endpoint
remote_server = f"http://{server_ip}:{remote_port}"  # Remote server for model unloading
use_markdown = True  # Default to markdown mode
model_loading = False  # Flag to track if a model is currently loading
model_ready = False  # Flag to track if the currently selected model is ready
model_to_context_map = {}  # Global mapping of model names to context windows
model_to_type_map = {}  # Global mapping of model names to types (llm, vlm)
model_to_quant_map = {}  # Global mapping of model names to quantization types

# NEW: cache preferred endpoint ("chat" or "completions") per model
model_to_endpoint_map = {}

# Heading size configuration - single tunable variable (values between 0.3-1.0 work well)
heading_size_scale = 0.65  # Base scale for headings - smaller values = smaller headings

# Memory usage thresholds for conversation history
MAX_CONVERSATION_SIZE = 500000  # 500KB threshold
TRIM_TARGET_SIZE = 300000  # 300KB target after trimming
# NOTE: We will NOT auto-disable markdown or simplify rendering anymore
# LARGE_CONTENT_THRESHOLD = 100000  # (kept for reference; no longer used)

# -------------------------------------------------------
# 1) Model Fetching - Updated for LM Studio API
# -------------------------------------------------------
def fetch_models():
    global available_models, available_models_display, selected_model, lmstudio_server, model_to_type_map, model_to_quant_map, display_to_model_map
    try:
        base_url = lmstudio_server.split('/v1')[0] if '/v1' in lmstudio_server else lmstudio_server
        native_api_url = f"{base_url}/api/v0/models"
        print(f"DEBUG: Fetching models from native API: {native_api_url}")

        response = requests.get(native_api_url, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            if "data" in data:
                models_list = []
                display_models_list = []
                model_context_map = {}
                model_type_map = {}
                model_quant_map = {}
                display_to_model_map.clear()

                for m in data["data"]:
                    if m.get("type") in ["llm", "vlm"]:
                        model_id = m["id"]
                        model_type_map[model_id] = m.get("type")
                        if "quantization" in m:
                            model_quant_map[model_id] = m["quantization"]
                        if "max_context_length" in m:
                            context_length = m["max_context_length"]
                            model_context_map[model_id] = context_length
                            print(f"DEBUG: Model '{model_id}' ({m.get('type')}) has context window: {context_length}, quantization: {m.get('quantization', 'unknown')}")
                        else:
                            print(f"DEBUG: Model '{model_id}' ({m.get('type')}) has no context window information")

                        type_indicator = "[VLM] " if m.get("type") == "vlm" else ""
                        quant = m.get("quantization", "")
                        quant_info = f" | {quant}" if quant else ""

                        if "max_context_length" in m:
                            ctx = m["max_context_length"]
                            ctx_formatted = f"{ctx//1000}K" if ctx >= 1000 else str(ctx)
                            display_name = f"{type_indicator}{model_id} (ctx: {ctx_formatted}{quant_info})"
                        else:
                            display_name = f"{type_indicator}{model_id}{quant_info}"

                        models_list.append(model_id)
                        display_models_list.append(display_name)
                        display_to_model_map[display_name] = model_id
                        print(f"DEBUG: Mapped display name '{display_name}' to model ID '{model_id}'")

                global model_to_context_map
                model_to_context_map = model_context_map
                model_to_type_map = model_type_map
                model_to_quant_map = model_quant_map

                def sort_key(idx):
                    model_id = models_list[idx]
                    ctx = model_context_map.get(model_id, 0)
                    return (-ctx, model_id.lower())

                indices = list(range(len(models_list)))
                indices.sort(key=sort_key)

                sorted_models = [models_list[i] for i in indices]
                sorted_display_models = [display_models_list[i] for i in indices]

                available_models = sorted_models
                available_models_display = sorted_display_models
                print(f"DEBUG: Populated available_models with {len(available_models)} models")
                print(f"DEBUG: Populated available_models_display with {len(available_models_display)} display names")
            else:
                available_models = []
                available_models_display = []

            if not available_models:
                if selected_model:
                    available_models = [selected_model]
                    display_name = selected_model
                    available_models_display = [display_name]
                    display_to_model_map[display_name] = selected_model
                else:
                    available_models = ["<No models available>"]
                    available_models_display = ["<No models available>"]
                    display_to_model_map["<No models available>"] = "<No models available>"
            elif selected_model and selected_model not in available_models:
                display_name = selected_model
                available_models.insert(0, selected_model)
                available_models_display.insert(0, display_name)
                display_to_model_map[display_name] = selected_model

            if available_models and not available_models[0].startswith("<") and not selected_model:
                selected_model = available_models[0]

            return True, available_models, available_models_display
        else:
            openai_api_url = f"{base_url}/v1/models"
            print(f"DEBUG: Native API failed with {response.status_code}, trying OpenAI-compatible endpoint: {openai_api_url}")
            response = requests.get(openai_api_url, timeout=5.0)

            if response.status_code == 200:
                data = response.json()
                if "data" in data:
                    model_ids = [m["id"] for m in data["data"]]
                    model_ids.sort()
                    available_models = model_ids
                    available_models_display = model_ids
                    display_to_model_map.clear()
                    for model_id in model_ids:
                        display_to_model_map[model_id] = model_id
                else:
                    available_models = []
                    available_models_display = []

                if not available_models:
                    if selected_model:
                        available_models = [selected_model]
                        available_models_display = [selected_model]
                        display_to_model_map[selected_model] = selected_model
                    else:
                        available_models = ["<No models available>"]
                        available_models_display = ["<No models available>"]
                        display_to_model_map["<No models available>"] = "<No models available>"
                elif selected_model and selected_model not in available_models:
                    available_models.insert(0, selected_model)
                    available_models_display.insert(0, selected_model)
                    display_to_model_map[selected_model] = selected_model

                if available_models and not available_models[0].startswith("<") and not selected_model:
                    selected_model = available_models[0]

                return True, available_models, available_models_display
            else:
                print(f"DEBUG: Server returned error {response.status_code}: {response.text}")
                if selected_model:
                    available_models = [selected_model]
                    available_models_display = [selected_model]
                    display_to_model_map[selected_model] = selected_model
                else:
                    available_models = ["<Connection error>"]
                    available_models_display = ["<Connection error>"]
                    display_to_model_map["<Connection error>"] = "<Connection error>"
                return False, available_models, available_models_display
    except requests.exceptions.RequestException as e:
        print(f"DEBUG: Exception while fetching models: {e}")
        if selected_model:
            available_models = [selected_model]
            available_models_display = [selected_model]
            display_to_model_map[selected_model] = selected_model
        else:
            available_models = ["<Connection error>"]
            available_models_display = ["<Connection error>"]
            display_to_model_map["<Connection error>"] = "<Connection error>"
        return False, available_models, available_models_display

# -------------------------------------------------------
# Context Window Size Detection - New for LM Studio
# -------------------------------------------------------
def get_actual_context_length(model_id):
    global remote_server
    try:
        base_url = remote_server.rsplit(':', 1)[0]
        url = f"{base_url}:5051/models"
        print(f"DEBUG: Fetching actual context length from: {url}")

        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            print(f"DEBUG: Loaded models data: {models_data}")

            for model in models_data:
                model_identifiers = [
                    model.get("modelKey"),
                    model.get("identifier"),
                    model.get("displayName"),
                    model.get("path").split('/')[-1].split('.')[0] if model.get("path") else None
                ]
                if any(model_id == id for id in model_identifiers if id):
                    if "contextLength" in model:
                        actual_context = model["contextLength"]
                        print(f"DEBUG: Found actual context length for '{model_id}': {actual_context}")
                        global model_to_context_map
                        model_to_context_map[model_id] = actual_context
                        return actual_context
                    elif "maxContextLength" in model:
                        max_context = model["maxContextLength"]
                        print(f"DEBUG: Found max context length for '{model_id}': {max_context}")
                        return max_context
            print(f"DEBUG: Model '{model_id}' not found in loaded models data")
    except Exception as e:
        print(f"DEBUG: Error getting actual context length: {e}")
    return 0

def get_context_window_size(model_id):
    global model_to_context_map
    print(f"DEBUG: Getting context window size for model '{model_id}'")

    if model_id in model_to_context_map:
        cached_size = model_to_context_map[model_id]
        print(f"DEBUG: Using cached context length for '{model_id}': {cached_size}")
        return cached_size

    try:
        base_url = lmstudio_server.split('/v1')[0] if '/v1' in lmstudio_server else lmstudio_server
        native_api_url = f"{base_url}/api/v0/models"
        print(f"DEBUG: Fetching models from native API: {native_api_url}")

        response = requests.get(native_api_url, timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            if "data" in models_data:
                models = models_data["data"]
                for model in models:
                    if model["type"] in ["llm", "vlm"] and "max_context_length" in model:
                        model_to_context_map[model["id"]] = model["max_context_length"]

                if model_id in model_to_context_map:
                    context_length = model_to_context_map[model_id]
                    print(f"DEBUG: Found context length in model data: {context_length}")
                    return context_length
                print(f"DEBUG: Model '{model_id}' not found in API response or missing context length.")
    except Exception as e:
        print(f"DEBUG: Error getting model info: {e}")

    print("DEBUG: Falling back to error message method...")
    try:
        long_prompt = "a" * 100000
        payload = {
            "model": model_id,
            "prompt": long_prompt,
            "max_tokens": 1,
            "temperature": 0.0
        }
        base_url = lmstudio_server.split('/v1')[0] if '/v1' in lmstudio_server else lmstudio_server
        openai_api_url = f"{base_url}/v1/completions"
        print(f"DEBUG: Sending test prompt to: {openai_api_url}")

        response = requests.post(openai_api_url, json=payload, timeout=5)
        if response.status_code == 200:
            print("DEBUG: Large prompt didn't trigger an error. Using default context size.")
            return 4096

        error_message = response.text
        print(f"DEBUG: Error message: {error_message}")

        patterns = [
            r"context length of only (\d+) tokens",
            r"maximum context length is (\d+) tokens",
            r"context window of (\d+) tokens",
            r"context size: (\d+)",
            r"max tokens: (\d+)",
            r"maximum context length \((\d+)\)",
            r"model's maximum context length \((\d+)\)"
        ]

        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                context_window = int(match.group(1))
                print(f"DEBUG: Detected context window size: {context_window} tokens")
                model_to_context_map[model_id] = context_window
                return context_window
    except Exception as e:
        print(f"DEBUG: Error in fallback method: {e}")

    print("DEBUG: Could not determine context window size. Using default value of 4096 tokens.")
    return 4096

# -------------------------------------------------------
# Model Unloading - New for LM Studio
# -------------------------------------------------------
def unload_all_models():
    try:
        print("DEBUG: Unloading all models via remote endpoint...")
        url = f"{remote_server}/unload_all"
        response = requests.post(url, timeout=10)
        result = response.json()

        if result.get("status") == "success":
            print("DEBUG: Successfully unloaded all models.")
            return True
        else:
            print(f"DEBUG: Failed to unload models. Status: {result.get('status')}")
            print(f"DEBUG: Message: {result.get('message', 'No message')}")
            return False
    except requests.exceptions.Timeout:
        print("DEBUG: Timeout waiting for models to unload.")
        return False
    except Exception as e:
        print(f"DEBUG: Error unloading models: {e}")
        return False

# -------------------------------------------------------
# Custom Combo Box with Colored Items for Loaded Model
# -------------------------------------------------------
class ModelComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None):
        super(ModelComboBox, self).__init__(parent)
        self.loaded_model = ""

    def setLoadedModel(self, model_name):
        self.loaded_model = model_name
        self.update()

    def paintEvent(self, event):
        painter = QtWidgets.QStylePainter(self)
        painter.setPen(self.palette().color(QtGui.QPalette.Text))
        opt = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(opt)
        painter.drawComplexControl(QtWidgets.QStyle.CC_ComboBox, opt)
        painter.drawControl(QtWidgets.QStyle.CE_ComboBoxLabel, opt)

    def view(self):
        return super().view()

    def setView(self, view):
        view.setItemDelegate(ModelItemDelegate(self.loaded_model, self))
        super().setView(view)

class ModelItemDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, loaded_model, parent=None):
        super(ModelItemDelegate, self).__init__(parent)
        self.loaded_model = loaded_model
        self.parent_combo = parent

    def setLoadedModel(self, model_name):
        self.loaded_model = model_name
        if self.parent_combo and self.parent_combo.view():
            self.parent_combo.view().update()

    def paint(self, painter, option, index):
        item_text = index.data(QtCore.Qt.DisplayRole)
        item_model = item_text.split(" (ctx:")[0] if " (ctx:" in item_text else item_text
        if item_model.startswith("[VLM] "):
            item_model = item_model[6:]
        is_loaded_model = (item_model == self.loaded_model)

        normal_text_color = QtGui.QColor("#FFFFFF")
        loaded_text_color = QtGui.QColor("#FF9500")
        hover_bg_color = QtGui.QColor("#4A4A4A")

        if option.state & QtWidgets.QStyle.State_Selected:
            painter.fillRect(option.rect, hover_bg_color)
            text_color = loaded_text_color if is_loaded_model else normal_text_color
        else:
            painter.fillRect(option.rect, option.palette.base())
            text_color = loaded_text_color if is_loaded_model else normal_text_color

        painter.setPen(text_color)
        painter.drawText(option.rect.adjusted(5, 0, -5, 0), QtCore.Qt.AlignVCenter, item_text)

# -------------------------------------------------------
# Markdown extensions / code highlighting
# -------------------------------------------------------
class CodeExtension(markdown.Extension):
    def extendMarkdown(self, md):
        md.registerExtension(self)
        md.preprocessors.register(CodeProcessor(md), 'highlight_code', 175)

class CodeProcessor(markdown.preprocessors.Preprocessor):
    def run(self, lines):
        new_lines = []
        in_code_block = False
        code_block_lines = []
        language = None
        block_id = 0

        for line in lines:
            if line.strip().startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    language = line.strip()[3:].strip() or 'text'
                    code_block_lines = []
                else:
                    in_code_block = False
                    code = '\n'.join(code_block_lines)
                    block_id += 1
                    try:
                        if language.lower() in ['python', 'py']:
                            lexer = get_lexer_by_name('python', stripall=True)
                        else:
                            try:
                                lexer = get_lexer_by_name(language, stripall=True)
                            except:
                                lexer = get_lexer_by_name('text', stripall=True)

                        formatter = HtmlFormatter(
                            style='monokai',
                            noclasses=True,
                            nobackground=True,
                            linenos=False
                        )
                        highlighted_code = highlight(code, lexer, formatter)
                    except Exception as e:
                        print(f"DEBUG: Syntax highlighting error: {e}")
                        highlighted_code = f"<pre style='margin: 0; color: #f8f8f2;'>{code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}</pre>"

                    code_block_html = f"""<div class="code-block" style="margin: 0; padding: 0.5em; background-color: #000000; border-radius: 3px; font-family: Consolas, monospace; font-size: 0.95em; line-height: 1.2; position: relative; overflow-x: auto;">
<div style="margin: 0; background-color: #000000; white-space: pre-wrap; word-wrap: break-word;">{highlighted_code}</div>
</div>"""

                    new_lines.append(code_block_html)
                    language = None
            elif in_code_block:
                code_block_lines.append(line)
            else:
                new_lines.append(line)

        return new_lines

# -------------------------------------------------------
# Utilities
# -------------------------------------------------------
def collapse_newlines(text: str) -> str:
    return re.sub(r'\n{2,}', '\n', text)

def highlight_thinking_tokens(text: str) -> str:
    """
    Render any think-tags nicely in the UI.
    - Normalizes provider-specific tags to a common visual (<think> ... </think> in orange).
    - Supports:
        <think>...</think>
        <thinking>...</thinking>
        <seed:think>...</seed:think>
        [think]...[/think]  and  [thinking]...[/thinking]
    - Leaves <seed:cot_budget_reflect> visible but tinted so you can see budget checkpoints.
    """
    if not isinstance(text, str):
        return text

    # 1) Normalize bracket style to angle tags (only for visual pass)
    text = re.sub(r'(?is)\[\s*thinking\s*\]', '<thinking>', text)
    text = re.sub(r'(?is)\[\s*/\s*thinking\s*\]', '</thinking>', text)
    text = re.sub(r'(?is)\[\s*think\s*\]', '<think>', text)
    text = re.sub(r'(?is)\[\s*/\s*think\s*\]', '</think>', text)

    # 2) Normalize provider-specific tags to common forms (do NOT remove the content)
    text = re.sub(r'(?is)<\s*seed:think\s*>', '<think>', text)
    text = re.sub(r'(?is)<\s*/\s*seed:think\s*>', '</think>', text)

    # We keep <thinking> as an alias of <think> for coloring
    text = re.sub(r'(?is)<\s*thinking\s*>', '<think>', text)
    text = re.sub(r'(?is)<\s*/\s*thinking\s*>', '</think>', text)

    # 3) Colorize the generic think tags (show tags themselves in orange)
    text = text.replace("<think>", '<span style="color: #FFD700;">&lt;think&gt;</span>')
    text = text.replace("</think>", '<span style="color: #FFD700;">&lt;/think&gt;</span>')

    # 4) Keep Seed budget reflect visible but tinted (optional, nice for debugging)
    text = re.sub(r'(?is)<\s*seed:cot_budget_reflect\s*>',
                  '<span style="color: #FFD700;">&lt;seed:cot_budget_reflect&gt;</span>', text)
    text = re.sub(r'(?is)<\s*/\s*seed:cot_budget_reflect\s*>',
                  '<span style="color: #FFD700;">&lt;/seed:cot_budget_reflect&gt;</span>', text)

    return text



def preprocess_markdown_headings(text: str) -> str:
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        if line.startswith('# '):
            line = '### ' + line[2:]
        elif line.startswith('## '):
            line = '### ' + line[3:]
        processed_lines.append(line)
    return '\n'.join(processed_lines)

# -------------------------------------------------------
# Build HTML for Chat History
# -------------------------------------------------------
def build_html_chat_history(history=None):
    print(f"DEBUG: build_html_chat_history: START")
    try:
        global use_markdown, conversation_history, heading_size_scale
        lines_html = []

        if history is None:
            history = conversation_history

        print(f"DEBUG: build_html_chat_history: Processing history len: {len(history)}")

        base_size = heading_size_scale
        h3_size = base_size * 0.94
        h4_size = base_size * 0.91
        h5_size = base_size * 0.88
        h6_size = base_size * 0.85

        markdown_css = f"""
        <style>
            .markdown-content {{
                line-height: 1.2;
                margin: 0;
                padding: 0;
            }}
            .markdown-content p {{
                margin: 0.3em 0;
            }}
            .markdown-content h1 {{
                font-size: {h3_size}em !important;
                font-weight: bold !important;
                margin: 0.3em 0 0.2em 0 !important;
                color: #e6e6e6 !important;
                line-height: 1.2 !important;
            }}
            .markdown-content h1.user-heading,
            .markdown-content h1.ai-heading {{
                color: orange !important;
                display: inline-block !important;
            }}
            .markdown-content h2 {{
                font-size: {h3_size}em !important;
                font-weight: bold !important;
                margin: 0.3em 0 0.2em 0 !important;
                color: #e6e6e6 !important;
                line-height: 1.2 !important;
            }}
            .markdown-content h3 {{
                font-size: {h3_size}em !important;
                font-weight: bold !important;
                margin: 0.3em 0 0.2em 0 !important;
                color: #e6e6e6 !important;
                line-height: 1.2 !important;
            }}
            .markdown-content h4 {{
                font-size: {h4_size}em !important;
                font-weight: bold !important;
                margin: 0.3em 0 0.2em 0 !important;
                color: #e6e6e6 !important;
                line-height: 1.2 !important;
            }}
            .markdown-content h5 {{
                font-size: {h5_size}em !important;
                font-weight: bold !important;
                margin: 0.3em 0 0.2em 0 !important;
                color: #e6e6e6 !important;
                line-height: 1.2 !important;
            }}
            .markdown-content h6 {{
                font-size: {h6_size}em !important;
                font-weight: bold !important;
                margin: 0.3em 0 0.2em 0 !important;
                color: #e6e6e6 !important;
                line-height: 1.2 !important;
            }}
            .user-label, .ai-label {{
                color: orange !important;
                font-weight: bold !important;
                font-size: {h3_size}em !important;
            }}
            .markdown-content ul, .markdown-content ol {{
                margin: 0.3em 0 0.3em 1.5em;
                padding: 0;
            }}
            .markdown-content li {{
                margin: 0.1em 0;
            }}
            .markdown-content code {{
                background-color: #000000;
                color: #e6e6e6;
                border-radius: 3px;
                padding: 0.1em 0.3em;
                font-family: Consolas, monospace;
                font-size: 0.95em;
            }}
            .markdown-content pre {{
                margin: 0;
                padding: 0.5em;
                background-color: #000000;
                border-radius: 3px;
                font-family: Consolas, monospace;
                font-size: 0.95em;
                line-height: 1.2;
                overflow-x: auto;
            }}
            .markdown-content blockquote {{
                margin: 0.5em 0;
                padding: 0.3em 0.5em;
                border-left: 3px solid #537BA2;
                background-color: #323232;
            }}
            .markdown-content table {{
                border-collapse: collapse;
                margin: 0.5em 0;
                font-size: 0.95em;
            }}
            .markdown-content th, .markdown-content td {{
                padding: 0.3em 0.6em;
                border: 1px solid #444;
            }}
            .markdown-content th {{
                background-color: #373737;
            }}
            .markdown-content hr {{
                border: none;
                border-top: 1px solid #444;
                margin: 0.5em 0;
            }}
            .code-block {{
                position: relative;
            }}
            .code-block button {{
                opacity: 0.7;
                transition: opacity 0.2s ease;
            }}
            .code-block:hover button {{
                opacity: 1;
            }}
        </style>
        """

        chat_html = markdown_css

        for i, line in enumerate(history):
            print(f"DEBUG: build_html_chat_history: Processing line {i}, len: {len(line)}")
            line = collapse_newlines(line)
            if line.startswith("User:\n"):
                content = line[len("User:\n"):]
                content = highlight_thinking_tokens(content)
                if use_markdown:
                    try:
                        user_heading = f"<h3 class='user-heading' style='color:#FF9500 !important;'>User:</h3>"
                        md = markdown.Markdown(extensions=['fenced_code', 'tables', 'nl2br', CodeExtension()])
                        print(f"DEBUG: build_html_chat_history: Converting user content (len {len(content)}) to markdown...")
                        processed_content = md.convert(content)
                        print(f"DEBUG: build_html_chat_history: User content converted.")
                        content_html = f'<div class="markdown-content">{user_heading}{processed_content}</div>'
                    except Exception as e:
                        print(f"DEBUG: Markdown parsing error: {e}")
                        content_html = f'<div style="margin-top: 0.5em;"><span class="user-label" style="color:#FF9500;">User:</span><br/>{content.replace("<br/>", "\n").replace("\n", "<br/>")}</div>'
                    line_html = f'<div style="font-size: 1em;">{content_html}</div>'
                else:
                    content = content.replace("\n", "<br/>")
                    line_html = f'<div style="margin-top: 0.5em;"><span class="user-label" style="color:#FF9500;">User:</span><br/>{content}</div>'

                lines_html.append(line_html)

            elif line.startswith("AI:\n"):
                content = line[len("AI:\n"):]
                content = highlight_thinking_tokens(content)
                if use_markdown:
                    try:
                        ai_heading = f"<h3 class='ai-heading' style='color:#FF9500 !important;'>AI:</h3>"
                        md = markdown.Markdown(extensions=['fenced_code', 'tables', 'nl2br', CodeExtension()])
                        print(f"DEBUG: build_html_chat_history: Converting AI content (len {len(content)}) to markdown...")
                        processed_content = md.convert(content)
                        print(f"DEBUG: build_html_chat_history: AI content converted.")
                        content_html = f'<div class="markdown-content">{ai_heading}{processed_content}</div>'
                    except Exception as e:
                        print(f"DEBUG: Markdown parsing error: {e}")
                        content_html = f'<div style="margin-top: 0.5em;"><span class="ai-label" style="color:#FF9500;">AI:</span><br/>{content.replace("<br/>", "\n").replace("\n", "<br/>")}</div>'
                    line_html = f'<div style="font-size: 1em;">{content_html}</div>'
                else:
                    content = content.replace("\n", "<br/>")
                    line_html = f'<div style="margin-top: 0.5em;"><span class="ai-label" style="color:#FF9500;">AI:</span><br/>{content}</div>'

                lines_html.append(line_html)

            else:
                if use_markdown:
                    try:
                        line_html = highlight_thinking_tokens(line)
                        md = markdown.Markdown(extensions=['fenced_code', 'tables', 'nl2br', CodeExtension()])
                        print(f"DEBUG: build_html_chat_history: Converting other content (len {len(line_html)}) to markdown...")
                        line_html = md.convert(line_html)
                        print(f"DEBUG: build_html_chat_history: Other content converted.")
                        line_html = f'<div class="markdown-content">{line_html}</div>'
                    except Exception:
                        line_html = highlight_thinking_tokens(line).replace("\n", "<br/>")
                else:
                    line_html = highlight_thinking_tokens(line).replace("\n", "<br/>")
                lines_html.append(line_html)

        joined_html = "\n".join(lines_html)
        final_html = f"<div style='line-height:1.1; margin:0; padding:0;'>{joined_html}</div>"
        print(f"DEBUG: build_html_chat_history: Final HTML len: {len(final_html)}")
        print(f"DEBUG: build_html_chat_history: END")
        return final_html

    except Exception as e:
        print(f"DEBUG: Error building HTML: {e}")
        simple_html = "<div>"
        for line in history or conversation_history:
            if line.startswith("User:\n"):
                line_html = f'<div style="margin-top: 0.5em;"><span style="color:#FF9500; font-weight:bold;">User:</span><br/>{line[6:].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")}</div>'
            elif line.startswith("AI:\n"):
                line_html = f'<div style="margin-top: 0.5em;"><span style="color:#FF9500; font-weight:bold;">AI:</span><br/>{line[4:].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")}</div>'
            else:
                line_html = line.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
            simple_html += line_html
        simple_html += "</div>"
        print(f"DEBUG: build_html_chat_history: END (after error)")
        return simple_html

# -------------------------------------------------------
# 5) Horizontal Separator
# -------------------------------------------------------
def create_separator():
    sep = QtWidgets.QFrame()
    sep.setFrameShape(QtWidgets.QFrame.HLine)
    sep.setFrameShadow(QtWidgets.QFrame.Sunken)
    sep.setLineWidth(1)
    sep.setStyleSheet("background-color: #262626;")
    return sep

# -------------------------------------------------------
# 6) Auto-resizing Text Edit
# -------------------------------------------------------
class AutoResizeTextEdit(QtWidgets.QTextEdit):
    enterPressed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.document().contentsChanged.connect(self.adjust_height)
        self.max_height = 1000
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setMinimumHeight(60)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return and not (event.modifiers() & QtCore.Qt.ShiftModifier):
            self.enterPressed.emit()
        else:
            super().keyPressEvent(event)

    def adjust_height(self):
        doc_height = self.document().size().height()
        doc_margin = self.document().documentMargin()
        content_height = doc_height + 2 * doc_margin + 10
        new_height = min(content_height, self.max_height)
        new_height = max(new_height, 60)
        if new_height != self.height():
            self.setFixedHeight(int(new_height))

    def set_max_height(self, height):
        self.max_height = max(height, 60)
        self.adjust_height()


def _strip_seed_tags(text: str) -> str:
    """
    Remove provider-specific reasoning wrappers so our UI shows clean text:
      - <seed:think>...</seed:think>, <think>...</think>
      - [THINK] ... [/THINK]
      - <seed:cot_budget_reflect>...</seed:cot_budget_reflect>
    """
    if not isinstance(text, str):
        return text
    # Seed reflect tag
    text = re.sub(r'(?is)</?\s*seed:cot_budget_reflect\s*>', '', text)
    # Angle and bracket thinking tags (Seed, Magistral, generic)
    text = re.sub(r'(?is)</?\s*seed:think\s*>', '', text)
    text = re.sub(r'(?is)</?\s*think\s*>', '', text)
    text = re.sub(r'(?is)\[\s*/?\s*think\s*\]', '', text)
    return text
    
# Heuristic "answer starts here" detector used for Magistral/Seed when answers
# spill into the thinking stream without an explicit close tag.
ANSWER_STARTERS_RE = re.compile(
    r"(?is)(?:^|[.!?]\s+)"
    r"(?:ok(?:ay)?|sure|got it|it looks like|looks like|here(?:'s| is)|"
    r"you(?:'re|’re| can| should| could| might| have|\b)|"
    r"feel free|if you have|let me know|i can help|final answer|answer:|"
    r"therefore|so,|in summary|result:)\b"
)

def _split_on_answer_boundary(tail_context: str, new_text: str):
    """
    Returns (reason_part, answer_part, found_boundary) by scanning tail+new_text
    for the first 'answer-like' sentence start.
    """
    s = (tail_context or "") + (new_text or "")
    m = ANSWER_STARTERS_RE.search(s)
    if m:
        cut = m.start()
        boundary_in_new = max(0, cut - len(tail_context))
        return new_text[:boundary_in_new], new_text[boundary_in_new:], True
    return "", new_text, False
    
    
# --- NEW (place near your other utils / regexes) -----------------------------
# Truncate when a model starts a fresh turn like:
#  ... your answer ...
#  [user]
#  next prompt...
RE_ROLE_MARKER = re.compile(r'(?is)(?:^|\n)\s*\[(?:user|assistant|system)\]\s*(?:\n|$)')

def _truncate_at_role_markers(text: str):
    """
    If a role marker like [user]/[assistant]/[system] appears,
    return text up to (but excluding) the first marker, plus a flag.
    """
    if not text:
        return text, False
    m = RE_ROLE_MARKER.search(text)
    if not m:
        return text, False
    return text[:m.start()], True



    
# --- Seed-OSS heuristic splitter -------------------------------------------
SEED_ANSWER_STARTERS_RE = re.compile(
    r"(?is)(?:^|[.!?]\s+)"
    r"(?:ok(?:ay)?|sure|got it|it looks like|looks like|here(?:'s| is)|"
    r"you(?:'re|’re| can| should| could| might| have| |’)|"
    r"feel free|if you have|let me know|i can help|answer:)\b"
)

def _seed_split_on_boundary(tail_context: str, new_text: str):
    """
    For Seed-OSS when it streams early tokens in `content` (no tags):
    detect the first 'answer-like' sentence boundary inside tail+new_text.
    Returns (reason_part, answer_part, found_boundary).
    """
    s = (tail_context or "") + (new_text or "")
    m = SEED_ANSWER_STARTERS_RE.search(s)
    if m:
        cut = m.start()
        # Map to current chunk coordinates
        boundary_in_new = max(0, cut - len(tail_context))
        return new_text[:boundary_in_new], new_text[boundary_in_new:], True
    return "", new_text, False


# -------------------------------------------------------
# Robust model endpoint selection & readiness probing
# -------------------------------------------------------
def decide_endpoint_for_model(model_name, base_url):
    """
    Pick 'chat' or 'completions' for a given model.
    Force CHAT for families with OpenAI-style chat templates or reasoning:
      - Qwen / QwQ (incl. A3B)
      - GPT-OSS
      - ByteDance Seed-OSS
      - Magistral (incl. small)
      - Gemma 3
      - Kimi K2 (Moonshot)
      - Mistral 'small' instruct/reasoning variants
    Cache the decision.
    """
    global model_to_endpoint_map
    if model_name in model_to_endpoint_map:
        return model_to_endpoint_map[model_name]

    mid = (model_name or "").lower()

    if any(k in mid for k in [
        "qwen", "qwen2", "qwen3", "qvq", "qwq", "a3b", "think",
        "gpt-oss", "gpt_oss", "oss-20b", "oss-120b",
        "seed-oss", "seed_oss", "bytedance",
        "magistral", "magistral-small",
        "gemma-3", "gemma3",
        "kimi", "k2", "moonshot",
        "mistral-small", "mistral-small-instruct"
    ]):
        model_to_endpoint_map[model_name] = "chat"
        return "chat"

    # Probe chat once; default to chat if it works
    try:
        r = requests.post(f"{base_url}/v1/chat/completions",
                          json={"model": model_name,
                                "messages": [
                                    {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": "Say OK"}],
                                "max_tokens": 1, "temperature": 0.0, "stream": False},
                          headers={"Content-Type": "application/json"},
                          timeout=6)
        if r.status_code == 200 and isinstance(r.json().get("choices"), list):
            model_to_endpoint_map[model_name] = "chat"
            return "chat"
    except Exception:
        pass

    model_to_endpoint_map[model_name] = "completions"
    return "completions"




def probe_model_ready(model_name, timeout_s=90):
    """
    Poll the chosen endpoint until the model is usable.
    Success = HTTP 200 with a 'choices' list (non-empty text not required).
    Returns (ready_bool, mode_str).
    """
    base_url = lmstudio_server.split('/v1')[0] if '/v1' in lmstudio_server else lmstudio_server
    mode = decide_endpoint_for_model(model_name, base_url)

    start = time.time()
    backoff = 0.8
    last_err = None

    while time.time() - start < timeout_s:
        try:
            if mode == "chat":
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Reply with OK"}
                    ],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False
                }
                resp = requests.post(f"{base_url}/v1/chat/completions",
                                     json=payload,
                                     headers={"Content-Type": "application/json"},
                                     timeout=12)
            else:
                payload = {
                    "model": model_name,
                    "prompt": "Assistant:",
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False,
                    "echo": True  # ensure payload includes text
                }
                resp = requests.post(f"{base_url}/v1/completions",
                                     json=payload,
                                     headers={"Content-Type": "application/json"},
                                     timeout=12)

            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data.get("choices", None), list):
                    return True, mode
                last_err = "200 without choices"
            else:
                last_err = f"status {resp.status_code}"
        except requests.exceptions.Timeout:
            last_err = "timeout"
        except Exception as e:
            last_err = str(e)

        time.sleep(backoff)
        backoff = min(backoff * 1.3, 3.0)

    print(f"DEBUG: probe_model_ready timeout; last_err={last_err}")
    return False, mode

# -------------------------------------------------------
# 7) Worker Thread for Streaming Responses - Updated
# -------------------------------------------------------
class ModelLoadingWorker(QtCore.QObject):
    modelLoaded = QtCore.pyqtSignal(bool, int)  # success, context_window
    modelError = QtCore.pyqtSignal(str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.should_stop = False

    def run(self):
        """Load model in background thread (robust probe)"""
        global lmstudio_server
        print(f"DEBUG: Background loading model: {self.model_name}")
        try:
            context_size = get_context_window_size(self.model_name)

            if self.should_stop:
                print("DEBUG: Model loading cancelled")
                return

            ready, _mode = probe_model_ready(self.model_name, timeout_s=120)
            if self.should_stop:
                print("DEBUG: Model loading cancelled after probe")
                return

            if ready:
                print(f"DEBUG: Model {self.model_name} probed READY")
                self.modelLoaded.emit(True, context_size)
            else:
                print(f"DEBUG: Model {self.model_name} probe timed out")
                self.modelLoaded.emit(False, context_size)

        except Exception as e:
            print(f"DEBUG: Error loading model: {e}")
            self.modelError.emit(f"Error loading model: {str(e)}")

def _strip_thinking_blocks(text: str) -> str:
    """
    Remove chain-of-thought blocks from assistant turns before re-sending them back
    to the model in subsequent prompts. This avoids leaking reasoning into future prompts.
    Removes the content inside these blocks:
      <think>...</think>, <thinking>...</thinking>, <seed:think>...</seed:think>,
      [think]...[/think], [thinking]...[/thinking]
    """
    if not isinstance(text, str):
        return text

    # Angle tags
    text = re.sub(r'(?is)<\s*think\s*>.*?<\s*/\s*think\s*>', '', text)
    text = re.sub(r'(?is)<\s*thinking\s*>.*?<\s*/\s*thinking\s*>', '', text)
    text = re.sub(r'(?is)<\s*seed:think\s*>.*?<\s*/\s*seed:think\s*>', '', text)

    # Bracket tags
    text = re.sub(r'(?is)\[\s*think\s*\].*?\[\s*/\s*think\s*\]', '', text)
    text = re.sub(r'(?is)\[\s*thinking\s*\].*?\[\s*/\s*thinking\s*\]', '', text)

    return text



def _extract_stream_texts(delta: dict):
    """
    Return (answer_delta, reasoning_delta) from a streaming chat `delta`.
    Works with LM Studio >= 0.3.23 where reasoning is split out.
    Handles shapes:
      - delta.content: str | list[{type, text|content}]
      - delta.reasoning: str | {text|content: str|list[{text}]}
      - legacy delta.text (completions-like)
    """
    ans, think = "", ""

    # --- Reasoning first (LM Studio 0.3.23+)
    r = delta.get("reasoning") or delta.get("reasoning_content")
    if isinstance(r, str):
        think += r
    elif isinstance(r, dict):
        if isinstance(r.get("content"), list):
            for part in r["content"]:
                if isinstance(part, dict):
                    think += part.get("text", "") or part.get("content", "") or ""
        else:
            think += r.get("text", "") or ""

    # --- Content (final user-visible answer stream)
    c = delta.get("content")
    if isinstance(c, str):
        ans += c
    elif isinstance(c, list):
        for part in c:
            if isinstance(part, dict):
                ptype = (part.get("type") or "").lower()
                t = part.get("text", "") or part.get("content", "") or ""
                if "reason" in ptype:
                    think += t
                else:
                    ans += t
            elif isinstance(part, str):
                ans += part

    # --- Completions/older fallbacks
    if isinstance(delta.get("text"), str):
        ans += delta["text"]

    return ans, think
    
# Unified reasoning tag patterns (Seed/Magistral + generic)
RE_THINK_OPEN_STR  = r'(?i)(?:<\s*(?:seed:)?think\s*>|\[\s*think\s*\])'
RE_THINK_CLOSE_STR = r'(?i)(?:<\s*/\s*(?:seed:)?think\s*>|\[\s*/\s*think\s*\])'

class RequestWorker(QtCore.QObject):
    newChunk = QtCore.pyqtSignal(str)
    tokenCountUpdate = QtCore.pyqtSignal(int, int)  # used_tokens, max_tokens
    finished = QtCore.pyqtSignal()
    connectionError = QtCore.pyqtSignal(str)
    stoppedByUser = QtCore.pyqtSignal()

    def __init__(self, prompt, current_history, max_context):
        super().__init__()
        self.prompt = prompt
        self.current_history = current_history.copy()
        self.max_context = max_context
        self.prompt_eval = 0
        self.eval_count = 0
        self.ai_response = ""
        self.estimated_prompt_tokens = len(prompt) // 4
        self.accumulated_chunks = ""
        self.last_emit_time = time.time()
        self.should_stop = False
        self.stop_emitted = False
        self.response = None

    def run(self):
        """
        Streaming generation worker (reasoning-aware) with GPT-OSS Harmony formatting.

        Key GPT-OSS adaptations:
          - Inject a Harmony-style system message that includes Knowledge cutoff, Current date,
            Reasoning level, and the valid channels line.
          - Treat the 'reasoning' stream as the Harmony `analysis` channel and the regular content
            as the `final` channel. We therefore close our synthetic <think> block the moment
            the first final content arrives, so the user-visible answer never lands inside <think>.
        """
        global selected_model, lmstudio_server
        self.ai_response = ""
        self.stop_emitted = False
        start_time = time.time()
        max_generation_time = 1800  # 30 minutes

        # ---- Build messages from history
        messages = []
        for entry in self.current_history:
            if entry.startswith("User:\n"):
                messages.append({"role": "user", "content": entry[len("User:\n"): ]})
            elif entry.startswith("AI:\n"):
                content = entry[len("AI:\n"): ]
                try:
                    # Strip any past CoT before echoing back to model
                    content = _strip_thinking_blocks(content)
                except Exception:
                    pass
                if content.strip():
                    messages.append({"role": "assistant", "content": content})

        # ---- Model flags
        base_url = lmstudio_server.split('/v1')[0] if '/v1' in lmstudio_server else lmstudio_server
        mode = decide_endpoint_for_model(selected_model, base_url)
        mid = (selected_model or "").lower()
        is_seed       = any(k in mid for k in ["seed-oss", "seed_oss", "bytedance"])
        is_magistral  = "magistral" in mid
        is_kimi       = any(k in mid for k in ["kimi", "k2", "moonshot"])
        is_gpt_oss    = any(k in mid for k in ["gpt-oss", "gpt_oss", "oss-20b", "oss-120b"])

        # ---- System prompt (model-specific)
        if is_gpt_oss:
            # Harmony-style system header recommended for GPT-OSS
            sys_prompt = (
                "You are ChatGPT, a large language model trained by OpenAI.\n"
                "Knowledge cutoff: 2024-06\n"
                f"Current date: {time.strftime('%Y-%m-%d')}\n\n"
                "Reasoning: medium\n\n"
                "# Valid channels: analysis, commentary, final. Channel must be included for every message."
            )
        elif is_magistral:
            # Paraphrase of the card’s convention: reasoning in <think>, final outside.
            sys_prompt = (
                "You are a thoughtful assistant. Put all deliberate reasoning inside <think> and </think>. "
                "Write the final answer only after </think>. Use clear, concise language."
            )
        elif is_seed:
            # Seed-OSS supports think traces and a thinking budget.
            sys_prompt = (
                "Provide your chain-of-thought inside <seed:think>…</seed:think>. "
                "After the closing tag, give the final answer."
            )
        else:
            sys_prompt = "You are a helpful assistant."

        messages.insert(0, {"role": "system", "content": sys_prompt})

        # ---- Stops for (Kimi) K2
        kimi_stop_strings = ["<|im_end|>", "[EOS]", "\n<|im_end|>", "\n\n<|im_end|>"]
        RE_KIMI_EOT = re.compile(r'(?is)(?:\s*)(?:<\|im_end\|\>|\[EOS\])')

        # ---- Build payload
        if mode == "chat":
            payload = {
                "model": selected_model,
                "messages": messages,
                "stream": True,
                "temperature": 0.7,
                "stream_options": {"include_usage": True},
                # Request explicit reasoning stream (most servers ignore unknown keys harmlessly)
                "reasoning": {"effort": "medium"}
            }
            if is_seed:
                payload.setdefault("extra_body", {})["thinking_budget"] = 512
            if is_kimi:
                payload["stop"] = kimi_stop_strings
                payload.setdefault("max_tokens", 1536)
            url = f"{base_url}/v1/chat/completions"
        else:
            def to_prompt(msgs):
                buf = []
                for m in msgs:
                    r = m["role"]
                    if r == "system":      buf.append(f"[system]\n{m['content']}\n")
                    elif r == "user":      buf.append(f"[user]\n{m['content']}\n")
                    elif r == "assistant": buf.append(f"[assistant]\n{m['content']}\n")
                buf.append("[assistant]\n")
                return "\n".join(buf)
            payload = {
                "model": selected_model,
                "prompt": to_prompt(messages),
                "stream": True,
                "temperature": 0.7
            }
            url = f"{base_url}/v1/completions"

        # --- Regex helpers and seed budget checkpoint
        RE_ROLE_MARKER = re.compile(r'(?is)(?:^|\n)\s*\[(?:user|assistant|system)\]\s*(?:\n|$)')
        RE_THINK_OPEN  = re.compile(r'(?is)<\s*(?:seed:)?think(?:ing)?\s*>|\[\s*(?:think|thinking)\s*\]')
        RE_THINK_CLOSE = re.compile(r'(?is)<\s*/\s*(?:seed:)?think(?:ing)?\s*>|\[\s*/\s*(?:think|thinking)\s*\]')
        RE_SEED_BUDGET_CLOSE = re.compile(
            r"(?is)\b(i\s*have\s*exhausted\s*my\s*token\s*budget.*?(?:start\s*answering|answering\s*the\s*question))"
        )

        try:
            self.response = requests.post(
                url, json=payload, stream=True, timeout=1800,
                headers={"Content-Type": "application/json"}
            )
            if self.response.status_code != 200:
                try:
                    err = self.response.json()
                except Exception:
                    err = {"error": {"message": self.response.text}}
                msg = err.get("error", {}).get("message", f"HTTP {self.response.status_code}")
                self.connectionError.emit(f"Generation failed: {msg}")
                if hasattr(self.response, 'close'):
                    self.response.close()
                self.finished.emit()
                return

            # Streaming state
            synth_think_open = False   # our own <think> ... </think> wrapper (for reasoning content)
            explicit_block    = False  # true if inside provider's explicit think block
            answer_started    = False
            force_finish      = False
            think_tail_ctx    = ""     # rolling buffer to detect boundaries for non-Harmony models

            self.accumulated_chunks = ""
            self.last_emit_time = time.time()
            last_usage_update = 0

            def open_synth_if_needed():
                nonlocal synth_think_open
                if not synth_think_open:
                    self.accumulated_chunks += "<think>"
                    synth_think_open = True

            def close_synth_if_open():
                nonlocal synth_think_open
                if synth_think_open:
                    self.accumulated_chunks += "</think>\n\n"
                    synth_think_open = False

            def flush(force=False):
                if self.accumulated_chunks:
                    now = time.time()
                    if force or (now - self.last_emit_time > 0.30) or (len(self.accumulated_chunks) > 50):
                        self.newChunk.emit(self.accumulated_chunks)
                        self.accumulated_chunks = ""
                        self.last_emit_time = now

            # Helper for truncating if the model starts a new role mid-stream
            def truncate_on_role_markers(s: str):
                if not s:
                    return s, False
                m = RE_ROLE_MARKER.search(s)
                if not m:
                    return s, False
                return s[:m.start()], True

            # Handles explicit <think> tags inside content (Seed/Magistral/generic)
            def handle_explicit_tags(text: str):
                nonlocal explicit_block, answer_started, force_finish, think_tail_ctx
                if not text:
                    return
                if is_kimi and RE_KIMI_EOT.search(text):
                    text = RE_KIMI_EOT.split(text)[0]
                    force_finish = True
                pos = 0
                while pos < len(text):
                    m_open  = RE_THINK_OPEN.search(text, pos)
                    m_close = RE_THINK_CLOSE.search(text, pos)
                    next_m = None
                    if m_open and m_close:
                        next_m = m_open if m_open.start() < m_close.start() else m_close
                    else:
                        next_m = m_open or m_close
                    if not next_m:
                        remainder = text[pos:]
                        if explicit_block:
                            open_synth_if_needed()
                            self.accumulated_chunks += remainder
                            think_tail_ctx = (think_tail_ctx + remainder)[-300:]
                        else:
                            self.ai_response += remainder
                            self.accumulated_chunks += remainder
                        break
                    pre = text[pos:next_m.start()]
                    if pre:
                        if explicit_block:
                            open_synth_if_needed()
                            self.accumulated_chunks += pre
                            think_tail_ctx = (think_tail_ctx + pre)[-300:]
                        else:
                            if is_kimi and RE_KIMI_EOT.search(pre):
                                pre = RE_KIMI_EOT.split(pre)[0]
                                force_finish = True
                            self.ai_response += pre
                            self.accumulated_chunks += pre
                    tag_txt = next_m.group(0)
                    if RE_THINK_OPEN.fullmatch(tag_txt):
                        explicit_block = True
                        open_synth_if_needed()
                    else:
                        explicit_block = False
                        close_synth_if_open()
                        answer_started = True
                    pos = next_m.end()

            # Main stream loop
            for raw_line in self.response.iter_lines():
                if self.should_stop and not self.stop_emitted:
                    self.stop_emitted = True
                    try:
                        if hasattr(self.response, 'close'):
                            self.response.close()
                    except Exception:
                        pass
                    self.stoppedByUser.emit()
                    self.finished.emit()
                    return

                if time.time() - start_time > max_generation_time:
                    self.connectionError.emit("Generation timeout reached")
                    try:
                        if hasattr(self.response, 'close'):
                            self.response.close()
                    except Exception:
                        pass
                    self.finished.emit()
                    return

                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except Exception:
                    continue

                # Usage / tokens (rate-limited)
                usage = obj.get("usage") or {}
                if isinstance(usage, dict):
                    now_u = time.time()
                    if now_u - last_usage_update > 0.25:
                        used = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
                        if used > 0:
                            self.tokenCountUpdate.emit(used, self.max_context)
                        last_usage_update = now_u

                choices = obj.get("choices") or []
                if not choices:
                    continue

                if mode == "chat":
                    delta = choices[0].get("delta", {}) or {}

                    # Extract answer & reasoning from delta (covers LM Studio 'reasoning' channel / Harmony analysis)
                    ans_delta, think_delta = _extract_stream_texts(delta)

                    # Truncate if model starts new role mid-stream
                    ans_delta, cut_a = truncate_on_role_markers(ans_delta)
                    think_delta, cut_t = truncate_on_role_markers(think_delta)
                    if cut_a or cut_t:
                        force_finish = True

                    if is_gpt_oss:
                        # Harmony: 'think_delta' == analysis channel, 'ans_delta' == final channel
                        if think_delta and not answer_started:
                            open_synth_if_needed()
                            self.accumulated_chunks += think_delta
                            flush()

                        if ans_delta:
                            # Close the reasoning block BEFORE any final content appears
                            close_synth_if_open()
                            answer_started = True

                            # K2 hard stop guard (some providers might reuse tokens)
                            if is_kimi and RE_KIMI_EOT.search(ans_delta):
                                ans_delta = RE_KIMI_EOT.split(ans_delta)[0]
                                force_finish = True
                            self.ai_response += ans_delta
                            self.accumulated_chunks += ans_delta
                            flush()

                        if force_finish:
                            break
                        continue  # Skip the generic logic below for GPT-OSS

                    # --- Non-GPT-OSS models (existing behavior) -------------------
                    # Reasoning stream (LM Studio reasoning channel or parts tagged 'reasoning')
                    if think_delta and not answer_started:
                        open_synth_if_needed()
                        self.accumulated_chunks += think_delta
                        think_tail_ctx = (think_tail_ctx + think_delta)[-300:]
                        flush()

                    # Content stream (may include explicit tags)
                    if ans_delta:
                        if is_seed and not answer_started and not explicit_block:
                            m = RE_SEED_BUDGET_CLOSE.search(ans_delta)
                            if m:
                                pre, post = ans_delta[:m.end()], ans_delta[m.end():]
                                open_synth_if_needed()
                                self.accumulated_chunks += pre
                                close_synth_if_open()
                                answer_started = True
                                if post:
                                    handle_explicit_tags(post)
                                flush()
                                if force_finish:
                                    break
                                continue

                        if RE_THINK_OPEN.search(ans_delta) or RE_THINK_CLOSE.search(ans_delta):
                            handle_explicit_tags(ans_delta)
                        else:
                            if is_kimi and RE_KIMI_EOT.search(ans_delta):
                                ans_delta = RE_KIMI_EOT.split(ans_delta)[0]
                                force_finish = True
                            self.ai_response += ans_delta
                            self.accumulated_chunks += ans_delta
                            flush()

                    if force_finish:
                        break

                else:
                    # /v1/completions (plain text stream)
                    text = choices[0].get("text", "") or (choices[0].get("delta", {}) or {}).get("text", "") or ""
                    if not text:
                        continue
                    text, cut_c = truncate_on_role_markers(text)
                    if cut_c:
                        force_finish = True

                    if RE_THINK_OPEN.search(text) or RE_THINK_CLOSE.search(text):
                        handle_explicit_tags(text)
                    else:
                        if is_kimi and RE_KIMI_EOT.search(text):
                            text = RE_KIMI_EOT.split(text)[0]
                            force_finish = True
                        self.ai_response += text
                        self.accumulated_chunks += text
                        flush()

                    if force_finish:
                        break

                # Fallback usage estimate when 'usage' not present
                if not usage:
                    input_est = sum(len(m.get("content", "")) for m in messages) // 4
                    out_est = len(self.ai_response) // 4
                    self.tokenCountUpdate.emit(input_est + out_est, self.max_context)

            # Close any open synthesized think block neatly (should already be closed for GPT-OSS)
            close_synth_if_open()
            flush(True)

        except requests.exceptions.RequestException as e:
            self.connectionError.emit(f"Connection error: {e}")
        except Exception as e:
            self.connectionError.emit(f"Unexpected error: {e}")
        finally:
            try:
                if hasattr(self, 'response') and self.response and hasattr(self.response, 'close'):
                    self.response.close()
            except Exception:
                pass
            self.finished.emit()




    def __del__(self):
        if hasattr(self, 'response') and self.response:
            try:
                if hasattr(self.response, 'close'):
                    self.response.close()
                    print("DEBUG: Response closed in __del__")
            except:
                pass


# -------------------------------------------------------
# 8) Main Chat Window
# -------------------------------------------------------
class ChatWindow(QtWidgets.QWidget):
    def update_model_status_indicator(self, status):
        if status == 'unloaded':
            self.model_status_indicator.setStyleSheet("""
                background-color: #000000;
                border: 1px solid #404040;
                border-radius: 8px;
                margin: 2px;
            """)
            self.model_status_indicator.setToolTip("Model Status: Unloaded")
        elif status == 'loading':
            self.model_status_indicator.setStyleSheet("""
                background-color: #FF9500;
                border: 1px solid #FFA530;
                border-radius: 8px;
                margin: 2px;
            """)
            self.model_status_indicator.setToolTip("Model Status: Loading")
        elif status == 'loaded':
            self.model_status_indicator.setStyleSheet("""
                background-color: #2A5E2A;
                border: 1px solid #3E8E3E;
                border-radius: 8px;
                margin: 2px;
            """)
            self.model_status_indicator.setToolTip("Model Status: Loaded")
        elif status == 'error':
            self.model_status_indicator.setStyleSheet("""
                background-color: #8B0000;
                border: 1px solid #A00000;
                border-radius: 8px;
                margin: 2px;
            """)
            self.model_status_indicator.setToolTip("Model Status: Error loading model")

    def cleanup_resources(self):
        if self.thread is not None and self.thread.isRunning():
            try:
                if self.worker is not None:
                    self.worker.should_stop = True
                    if hasattr(self.worker, 'response') and self.worker.response:
                        try:
                            if hasattr(self.worker.response, 'close'):
                                self.worker.response.close()
                        except Exception as e:
                            print(f"DEBUG: Error closing response: {e}")

                self.thread.quit()
                if not self.thread.wait(500):
                    print("DEBUG: Thread did not terminate in time, forcing termination")
                    self.thread.terminate()
                    self.thread.wait()

            except Exception as e:
                print(f"DEBUG: Error cleaning up thread: {e}")

        self.stop_model_loading_thread()

        import gc
        gc.collect()
        print("DEBUG: Resources cleaned up")

    def load_selected_model(self):
        global selected_model, model_loading, model_ready, display_to_model_map

        self.cleanup_resources()

        display_name = self.model_combo.currentText()
        if display_name in display_to_model_map:
            model_name = display_to_model_map[display_name]
            print(f"DEBUG: Loading selected model: display='{display_name}', model ID='{model_name}'")
        else:
            model_name = display_name
            if model_name.startswith("[VLM] "):
                model_name = model_name[6:]
            if " (ctx:" in model_name:
                model_name = model_name.split(" (ctx:")[0]
            print(f"DEBUG: Display name '{display_name}' not in map, using as model ID: '{model_name}'")

        if not model_name or model_name.startswith("<"):
            self.status_label.setText("❌ No valid model selected!")
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
            return

        if self.model_loading_in_progress and selected_model == model_name:
            self.status_label.setText(f"⏳ Already loading model: {model_name}")
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
            return

        model_loading = True
        model_ready = False
        self.model_loading_in_progress = True
        self.update_ui_state()
        self.update_model_status_indicator('loading')

        selected_model = model_name
        print(f"DEBUG: Loading model: {model_name}")

        self.status_label.setText(f"⏳ Loading model: {model_name}")
        QtWidgets.QApplication.processEvents()

        self.stop_model_loading_thread()

        self.model_loading_thread = QtCore.QThread()
        self.model_loading_worker = ModelLoadingWorker(model_name)
        self.model_loading_worker.moveToThread(self.model_loading_thread)

        self.model_loading_worker.modelLoaded.connect(self.handle_model_loaded)
        self.model_loading_worker.modelError.connect(self.handle_model_error)
        self.model_loading_thread.started.connect(self.model_loading_worker.run)

        self.model_loading_worker.modelLoaded.connect(self.model_loading_thread.quit)
        self.model_loading_worker.modelError.connect(self.model_loading_thread.quit)

        self.model_loading_thread.finished.connect(lambda: self.cleanup_model_loading_thread())
        self.model_loading_thread.start()

    def stop_model_loading_thread(self):
        try:
            if self.model_loading_worker is not None:
                self.model_loading_worker.should_stop = True

            if self.model_loading_thread is not None and self.model_loading_thread.isRunning():
                print("DEBUG: Stopping previous model loading thread")
                self.model_loading_thread.quit()
                self.model_loading_thread.wait(1000)
                print("DEBUG: Previous model loading thread stopped")
        except Exception as e:
            print(f"DEBUG: Error stopping model loading thread: {e}")

    def cleanup_model_loading_thread(self):
        print("DEBUG: Cleaning up model loading thread")
        try:
            if self.model_loading_worker is not None:
                self.model_loading_worker.deleteLater()
                self.model_loading_worker = None

            if self.model_loading_thread is not None:
                self.model_loading_thread.deleteLater()
                self.model_loading_thread = None
        except Exception as e:
            print(f"DEBUG: Error during cleanup: {e}")

    def handle_model_loaded(self, is_ready, context_size):
        global model_loading, model_ready, selected_model

        self.current_model_context = context_size
        self.token_count_label.setText(f"Tokens: {self.last_token_count} / {self.current_model_context}")

        if is_ready:
            model_ready = True
            self.update_model_status_indicator('loaded')

            actual_context = get_actual_context_length(selected_model)
            if actual_context > 0:
                self.current_model_context = actual_context
                self.token_count_label.setText(f"Tokens: {self.last_token_count} / {self.current_model_context}")
                self.status_label.setText(f"✓ Model loaded with {self.current_model_context} context window (actual configured value)")
            else:
                self.status_label.setText(f"✓ Model loaded with {self.current_model_context} context window (max value)")

            self.loaded_model = selected_model
            if hasattr(self.model_combo, 'view') and self.model_combo.view():
                delegate = self.model_combo.view().itemDelegate()
                if hasattr(delegate, 'setLoadedModel'):
                    delegate.setLoadedModel(selected_model)
            self.model_combo.setLoadedModel(selected_model)
        else:
            model_ready = False
            self.update_model_status_indicator('error')
            self.status_label.setText(f"⚠️ Model responded but is still warming up — try again in a moment")

        model_loading = False
        self.model_loading_in_progress = False
        self.update_ui_state()
        QtCore.QTimer.singleShot(5000, lambda: self.status_label.setText(""))

    def handle_model_error(self, error_msg):
        global model_loading, model_ready
        model_ready = False
        self.update_model_status_indicator('error')
        self.status_label.setText(f"❌ {error_msg}")

        model_loading = False
        self.model_loading_in_progress = False
        self.update_ui_state()
        QtCore.QTimer.singleShot(5000, lambda: self.status_label.setText(""))

    def update_token_count(self, used: int, maximum: int):
        if used > 0:
            print(f"DEBUG: Updating token count: {used} / {maximum}")
            self.last_token_count = used
            self.token_count_label.setText(f"Tokens: {used} / {maximum}")
        else:
            self.token_count_label.setText(f"Tokens: 0 / {maximum}")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LM Studio Chat")
        self.resize(900, 650)
        self.current_ai_text = ""
        self.current_ai_index = -1
        self.thread = None
        self.worker = None
        self.last_token_count = 0
        self.api_connected = False
        self.current_model_context = 4096
        self.stop_requested = False
        self.auto_reconnect_enabled = False
        self.last_ui_update_time = 0
        self.showed_markdown_warning = False  # not used anymore but kept harmlessly
        self.user_scrolled = False

        self.model_loading_thread = None
        self.model_loading_worker = None
        self.model_loading_in_progress = False
        self.loaded_model = ""

        self.setupUI()

        self.api_check_timer = QtCore.QTimer(self)
        self.api_check_timer.timeout.connect(self.check_api_connection)

        self.send_button.setToolTip("Send message (only enabled when model is fully loaded)")
        self.load_model_button.setToolTip("Load the selected model")
        self.ip_input.setToolTip("IP address of the LM Studio server")
        self.api_port_input.setToolTip("Port for LM Studio API (default: 1234)")
        self.remote_port_input.setToolTip("Port for remote unload server (default: 5051)")
        self.chat_history_widget.verticalScrollBar().valueChanged.connect(self.on_scroll)

        QtCore.QTimer.singleShot(100, self.connectModelSignal)

        self.status_label.setText("Connect to LM Studio API server to begin")
        QtCore.QTimer.singleShot(5000, lambda: self.status_label.setText(""))

    def on_scroll(self, value):
        """Detect if user has manually scrolled away from the bottom"""
        scrollbar = self.chat_history_widget.verticalScrollBar()
        if value < scrollbar.maximum():
            self.user_scrolled = True
        else:
            self.user_scrolled = False

    def connectModelSignal(self):
        try:
            try:
                self.model_combo.currentTextChanged.disconnect(self.change_model)
            except Exception:
                pass
            self.model_combo.currentTextChanged.connect(self.change_model)
            print("DEBUG: Model selection signal connected")
        except Exception as e:
            print(f"DEBUG: Error connecting model signal: {e}")

    def closeEvent(self, event):
        self.status_label.setText("⏳ Closing application and unloading models...")
        QtWidgets.QApplication.processEvents()

        self.stop_model_loading_thread()

        if hasattr(self, 'api_check_timer') and self.api_check_timer.isActive():
            self.api_check_timer.stop()

        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(1000)

        if self.api_connected:
            try:
                unload_all_models()
                print("DEBUG: Models unloaded during application exit")
                self.update_model_status_indicator('unloaded')
                QtWidgets.QApplication.processEvents()
            except Exception as e:
                print(f"DEBUG: Error unloading models on exit: {e}")

        event.accept()

    def toggle_markdown(self):
        global use_markdown
        use_markdown = not use_markdown

        if use_markdown:
            self.markdown_button.setText("Markdown: ON")
        else:
            self.markdown_button.setText("Markdown: OFF")

        self.update_chat_history()
        status_msg = "✓ Markdown formatting enabled." if use_markdown else "✓ Plain text formatting enabled."
        self.status_label.setText(status_msg)
        QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText(""))

    def change_model(self, display_name):
        global selected_model, display_to_model_map

        if not display_name:
            print("DEBUG: Empty display name in change_model - skipping")
            return

        print(f"DEBUG: Dropdown selection changed to: '{display_name}'")

        if display_name in display_to_model_map:
            model_name = display_to_model_map[display_name]
            print(f"DEBUG: Found model ID '{model_name}' for display name '{display_name}'")
        else:
            model_name = display_name.split(" (ctx:")[0]
            if model_name.startswith("[VLM] "):
                model_name = model_name[6:]
            print(f"DEBUG: Display name '{display_name}' not found in map, using extracted ID: '{model_name}'")

        if model_name == selected_model or model_name.startswith("<") or not model_name:
            print(f"DEBUG: Skipping model change (same model or invalid)")
            return

        selected_model = model_name
        print(f"DEBUG: Selected model changed to: {model_name}")

        self.status_label.setText(f"Model selected: {model_name} (click 'Load Model' to load)")
        QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))

    def check_model_ready(self, model_name, set_loading_state=True):
        """
        Check if the model is fully loaded and ready to use (using robust probe).
        """
        global model_loading, model_ready

        if set_loading_state:
            model_loading = True
            model_ready = False
            self.update_ui_state()
            self.status_label.setText("⏳ Checking if model is ready...")
            QtWidgets.QApplication.processEvents()

        if not model_name or model_name.startswith("<"):
            if set_loading_state:
                model_loading = False
                model_ready = False
                self.update_ui_state()
                self.status_label.setText("⚠️ No valid model selected")
                QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText(""))
            return False

        try:
            self.status_label.setText("⏳ Loading and testing model (large models can need a moment)...")
            QtWidgets.QApplication.processEvents()

            ready, _mode = probe_model_ready(model_name, timeout_s=120)

            if ready:
                if set_loading_state:
                    model_loading = False
                    model_ready = True
                    self.update_ui_state()
                    self.status_label.setText("✓ Model is loaded and ready")
                    QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText(""))
                return True
            else:
                if set_loading_state:
                    model_loading = False
                    model_ready = False
                    self.update_ui_state()
                    self.status_label.setText("⚠️ Model responded but is still warming up — try again in a moment")
                    QtCore.QTimer.singleShot(2500, lambda: self.status_label.setText(""))
                return False

        except Exception as e:
            print(f"DEBUG: Error checking model readiness: {e}")
            if set_loading_state:
                model_loading = False
                model_ready = False
                self.update_ui_state()
                self.status_label.setText("⚠️ Error connecting to model")
                QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText(""))
            return False

    def fetch_model_context(self, model_name: str):
        global model_loading
        model_loading = True
        self.update_ui_state()
        self.status_label.setText("⏳ Detecting context window size...")
        QtWidgets.QApplication.processEvents()

        try:
            context_size = get_context_window_size(model_name)
            self.current_model_context = context_size
            self.token_count_label.setText(f"Tokens: {self.last_token_count} / {self.current_model_context}")
            self.status_label.setText(f"✓ Context window: {context_size} tokens")
            return True
        except Exception as e:
            print(f"DEBUG: Error fetching context window: {e}")
            self.status_label.setText("⚠️ Could not detect context size")
            return False

    def update_ui_state(self):
        if not self.api_connected:
            self.send_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.markdown_button.setEnabled(False)
            self.load_model_button.setEnabled(False)
            self.model_combo.setEnabled(False)
            self.update_model_status_indicator('unloaded')
            return

        self.model_combo.setEnabled(not model_loading)
        self.load_model_button.setEnabled(not model_loading)

        if model_loading:
            self.send_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.markdown_button.setEnabled(False)
        else:
            self.reset_button.setEnabled(True)
            self.markdown_button.setEnabled(True)
            self.send_button.setEnabled(model_ready)

            if model_ready:
                self.send_button.setStyleSheet("")
            else:
                self.send_button.setStyleSheet("""
                    background-color: #333333;
                    color: #777777;
                """)

    def set_loading_buttons(self, enabled):
        self.send_button.setEnabled(False)
        self.reset_button.setEnabled(enabled)
        self.markdown_button.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.load_model_button.setEnabled(enabled)
        QtWidgets.QApplication.processEvents()

    def setupUI(self):
        main_layout = QtWidgets.QVBoxLayout()

        bold_label_font = QtGui.QFont()
        bold_label_font.setPointSize(11)
        bold_label_font.setBold(True)

        # --- Server Connection ---
        server_label = QtWidgets.QLabel("Server Connection:")
        server_label.setAlignment(QtCore.Qt.AlignLeft)
        server_label.setFont(bold_label_font)
        main_layout.addWidget(server_label)

        row_server_layout = QtWidgets.QHBoxLayout()

        ip_label = QtWidgets.QLabel("IP:")
        row_server_layout.addWidget(ip_label)
        self.ip_input = QtWidgets.QLineEdit(server_ip)
        self.ip_input.setFixedWidth(120)
        self.ip_input.setToolTip("Server IP address")
        row_server_layout.addWidget(self.ip_input)

        api_port_label = QtWidgets.QLabel("API Port:")
        row_server_layout.addWidget(api_port_label)
        self.api_port_input = QtWidgets.QLineEdit(api_port)
        self.api_port_input.setFixedWidth(60)
        self.api_port_input.setToolTip("LM Studio API port (default: 1234)")
        row_server_layout.addWidget(self.api_port_input)

        remote_port_label = QtWidgets.QLabel("Remote Port:")
        row_server_layout.addWidget(remote_port_label)
        self.remote_port_input = QtWidgets.QLineEdit(remote_port)
        self.remote_port_input.setFixedWidth(60)
        self.remote_port_input.setToolTip("Remote unload server port (default: 5051)")
        row_server_layout.addWidget(self.remote_port_input)

        self.connection_indicator = QtWidgets.QLabel()
        self.connection_indicator.setFixedSize(16, 16)
        self.connection_indicator.setStyleSheet("""
            background-color: #000000;
            border: 1px solid #404040;
            border-radius: 8px;
            margin: 2px;
        """)
        self.connection_indicator.setToolTip("Disconnected")
        row_server_layout.addWidget(self.connection_indicator)

        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.clicked.connect(self.toggle_connection)
        row_server_layout.addWidget(self.connect_button)

        row_server_layout.addStretch(1)
        main_layout.addLayout(row_server_layout)
        main_layout.addWidget(create_separator())

        # --- Select Model ---
        model_label = QtWidgets.QLabel("Select Model:")
        model_label.setAlignment(QtCore.Qt.AlignLeft)
        model_label.setFont(bold_label_font)
        main_layout.addWidget(model_label)

        row_model_layout = QtWidgets.QHBoxLayout()
        self.model_combo = ModelComboBox()
        self.model_combo.setFixedWidth(450)
        list_view = QtWidgets.QListView()
        self.model_combo.setView(list_view)
        row_model_layout.addWidget(self.model_combo)

        self.load_model_button = QtWidgets.QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_selected_model)
        self.load_model_button.setToolTip("Load the selected model")
        row_model_layout.addWidget(self.load_model_button)

        self.model_status_indicator = QtWidgets.QLabel()
        self.model_status_indicator.setFixedSize(16, 16)
        self.model_status_indicator.setStyleSheet("""
            background-color: #000000;
            border: 1px solid #404040;
            border-radius: 8px;
            margin: 2px;
        """)
        self.model_status_indicator.setToolTip("Model Status: Unloaded")
        row_model_layout.addWidget(self.model_status_indicator)

        self.token_count_label = QtWidgets.QLabel("Tokens: 0 / 0")
        self.token_count_label.setFont(bold_label_font)
        row_model_layout.addWidget(self.token_count_label)

        row_model_layout.addStretch(1)
        main_layout.addLayout(row_model_layout)
        main_layout.addWidget(create_separator())

        # --- Chat History ---
        chat_label = QtWidgets.QLabel("Chat History:")
        chat_label.setAlignment(QtCore.Qt.AlignLeft)
        chat_label.setFont(bold_label_font)
        main_layout.addWidget(chat_label)

        self.chat_prompt_container = QtWidgets.QWidget()
        chat_prompt_layout = QtWidgets.QVBoxLayout(self.chat_prompt_container)
        chat_prompt_layout.setContentsMargins(0, 0, 0, 0)

        self.chat_history_widget = QtWidgets.QTextEdit()
        self.chat_history_widget.setObjectName("ChatHistory")
        self.chat_history_widget.setReadOnly(True)
        self.chat_history_widget.setWordWrapMode(QtGui.QTextOption.WordWrap)

        self.chat_history_widget.setStyleSheet("""
        QScrollBar:vertical {
            background-color: #3b3b3b;
            width: 12px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background-color: #555555;
            min-height: 20px;
            border-radius: 6px;
            margin: 2px;
        }
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            background: none;
            border: none;
            height: 0px;
        }
        QScrollBar:horizontal {
            background-color: #3b3b3b;
            height: 12px;
            margin: 0px;
        }
        QScrollBar::handle:horizontal {
            background-color: #555555;
            min-width: 20px;
            border-radius: 6px;
            margin: 2px;
        }
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {
            background: none;
            border: none;
            width: 0px;
        }
        """)

        # --- Prompt ---
        prompt_label = QtWidgets.QLabel("Enter your prompt:")
        prompt_label.setAlignment(QtCore.Qt.AlignLeft)
        prompt_label.setFont(bold_label_font)

        self.prompt_input = AutoResizeTextEdit()
        self.prompt_input.setObjectName("PromptInput")
        self.prompt_input.setWordWrapMode(QtGui.QTextOption.WordWrap)
        self.prompt_input.enterPressed.connect(self.send_message)
        self.prompt_input.setStyleSheet("""
        QScrollBar:vertical {
            background-color: #3b3b3b;
            width: 12px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background-color: #555555;
            min-height: 20px;
            border-radius: 6px;
            margin: 2px;
        }
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            background: none;
            border: none;
            height: 0px;
        }
        QScrollBar:horizontal {
            background-color: #3b3b3b;
            height: 12px;
            margin: 0px;
        }
        QScrollBar::handle:horizontal {
            background-color: #555555;
            min-width: 20px;
            border-radius: 6px;
            margin: 2px;
        }
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {
            background: none;
            border: none;
            width: 0px;
        }
        """)

        chat_prompt_layout.addWidget(self.chat_history_widget, 1)
        chat_prompt_layout.addWidget(create_separator())
        chat_prompt_layout.addWidget(prompt_label)
        chat_prompt_layout.addWidget(self.prompt_input, 0)

        main_layout.addWidget(self.chat_prompt_container, 1)
        main_layout.addWidget(create_separator())

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #FF9500; font-weight: bold;")
        main_layout.addWidget(self.status_label)

        buttons_layout = QtWidgets.QHBoxLayout()
        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_chat)
        self.markdown_button = QtWidgets.QPushButton("Markdown: ON")
        self.markdown_button.clicked.connect(self.toggle_markdown)
        self.exit_button = QtWidgets.QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        buttons_layout.addWidget(self.send_button)
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addWidget(self.markdown_button)
        buttons_layout.addWidget(self.exit_button)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

        self.ip_input.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.api_port_input.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.remote_port_input.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.model_combo.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.updatePromptHeight()

    def showEvent(self, event):
        super().showEvent(event)
        button_height = self.connect_button.sizeHint().height()
        self.ip_input.setFixedHeight(button_height)
        self.api_port_input.setFixedHeight(button_height)
        self.remote_port_input.setFixedHeight(button_height)
        self.model_combo.setFixedHeight(button_height)
        self.load_model_button.setFixedHeight(button_height)
        self.update_model_status_indicator('unloaded')
        QtCore.QTimer.singleShot(100, self.updatePromptHeight)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updatePromptHeight()

    def updatePromptHeight(self):
        if hasattr(self, 'chat_prompt_container') and hasattr(self.prompt_input, 'set_max_height'):
            container_height = self.chat_prompt_container.height()
            max_prompt_height = int(container_height * 0.382)
            self.prompt_input.set_max_height(max_prompt_height)
            self.prompt_input.adjust_height()

    def check_api_connection(self):
        global lmstudio_server, model_ready

        if not self.auto_reconnect_enabled and not self.api_connected:
            return

        if not self.send_button.isEnabled() and self.thread is not None and self.thread.isRunning():
            return

        try:
            base_url = lmstudio_server.split('/v1')[0] if '/v1' in lmstudio_server else lmstudio_server
            models_url = f"{base_url}/v1/models"
            response = requests.get(models_url, timeout=2.0)
            if response.status_code == 200:
                if not self.api_connected:
                    self.api_connected = True
                    self.update_connect_button()
                    print("DEBUG: API connection established")
                    success, _, _ = fetch_models()
                    if success:
                        self.update_model_combo()
                        model_ready = False
                        self.update_model_status_indicator('unloaded')
                        self.update_ui_state()
            else:
                if self.api_connected:
                    self.api_connected = False
                    self.update_connect_button()
                    print(f"DEBUG: API connection lost (status {response.status_code})")
        except requests.exceptions.RequestException:
            if self.api_connected:
                self.api_connected = False
                self.update_connect_button()
                print("DEBUG: API connection lost (connection error)")

    def update_connect_button(self):
        if self.api_connected:
            self.connection_indicator.setStyleSheet("""
                background-color: #2A5E2A;
                border: 1px solid #3E8E3E;
                border-radius: 8px;
                margin: 2px;
            """)
            self.connection_indicator.setToolTip("Connected to LM Studio API")
            self.connect_button.setText("Disconnect")
        else:
            self.connection_indicator.setStyleSheet("""
                background-color: #000000;
                border: 1px solid #404040;
                border-radius: 8px;
                margin: 2px;
            """)
            self.connection_indicator.setToolTip("Disconnected")
            self.connect_button.setText("Connect")

        self.update_ui_state()

    def toggle_connection(self):
        global model_ready

        if self.api_connected:
            self.status_label.setText("⏳ Disconnecting and unloading models...")
            QtWidgets.QApplication.processEvents()

            try:
                unload_all_models()
                print("DEBUG: Models unloaded during disconnect")
            except Exception as e:
                print(f"DEBUG: Error unloading models on disconnect: {e}")

            self.api_connected = False
            self.auto_reconnect_enabled = False
            self.api_check_timer.stop()

            model_ready = False
            self.update_model_status_indicator('unloaded')

            self.update_connect_button()
            self.status_label.setText("✓ Disconnected from server")
            QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText(""))
            print("DEBUG: Manually disconnected from server")
        else:
            self.connect_server()

    def connect_server(self):
        global lmstudio_server, remote_server, server_ip, api_port, remote_port, model_ready

        model_ready = False
        self.update_ui_state()

        new_ip = self.ip_input.text().strip()
        new_api_port = self.api_port_input.text().strip()
        new_remote_port = self.remote_port_input.text().strip()

        if new_ip:
            server_ip = new_ip
        if new_api_port:
            api_port = new_api_port
        if new_remote_port:
            remote_port = new_remote_port

        lmstudio_server = f"http://{server_ip}:{api_port}"
        remote_server = f"http://{server_ip}:{remote_port}"

        self.status_label.setText(f"⏳ Connecting to {server_ip}...")
        QtWidgets.QApplication.processEvents()

        self.api_connected = False
        self.update_connect_button()

        self.auto_reconnect_enabled = True
        self.api_check_timer.start(3000)
        self.check_api_connection()

    def update_model_combo(self):
        global available_models, available_models_display

        try:
            print("DEBUG: Updating model dropdown with display names")
            current_text = self.model_combo.currentText()
            try:
                self.model_combo.currentTextChanged.disconnect(self.change_model)
            except Exception:
                pass

            self.model_combo.clear()
            self.model_combo.addItems(available_models_display)

            index = self.model_combo.findText(current_text)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
                print(f"DEBUG: Restored previous selection '{current_text}'")
            elif available_models_display and not available_models_display[0].startswith("<"):
                self.model_combo.setCurrentIndex(0)
                print(f"DEBUG: Selected first model '{available_models_display[0]}'")

            self.model_combo.currentTextChanged.connect(self.change_model)

        except Exception as e:
            print(f"DEBUG: Error updating model dropdown: {e}")
            self.model_combo.clear()
            self.model_combo.addItems(available_models_display)

    def check_conversation_size(self):
        global conversation_history
        total_length = sum(len(msg) for msg in conversation_history)

        if total_length > MAX_CONVERSATION_SIZE:
            print(f"DEBUG: Conversation size ({total_length} bytes) exceeds threshold, trimming...")
            while total_length > TRIM_TARGET_SIZE and len(conversation_history) > 4:
                removed = conversation_history.pop(0)
                total_length -= len(removed)
                if len(conversation_history) > 0 and conversation_history[0].startswith("AI:"):
                    removed = conversation_history.pop(0)
                    total_length -= len(removed)

            conversation_history.insert(0, "AI:\n[Older messages have been removed to improve performance]")
            self.status_label.setText("ℹ️ Conversation history trimmed for performance")
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
            return True

        return False

    def send_message(self):
        global conversation_history, selected_model, model_ready

        # Stop-in-progress?
        if self.thread is not None and self.thread.isRunning():
            if hasattr(self, 'send_button') and self.send_button.text() == "Stop":
                if not self.stop_requested:
                    self.stop_requested = True
                    self.stop_worker()
                else:
                    self.status_label.setText("⏳ Already stopping, please wait...")
                    QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText(""))
                return

        if not self.api_connected:
            self.status_label.setText("❌ Not connected to API server!")
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
            return

        if not selected_model or selected_model.startswith("<") or not model_ready:
            self.status_label.setText("❌ Model is not ready! Please wait for model to load.")
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
            return

        prompt = self.prompt_input.toPlainText().strip()
        if prompt:
            if self.last_token_count > 0:
                self.token_count_label.setText(f"Tokens: {self.last_token_count}+ / {self.current_model_context}")
            else:
                self.token_count_label.setText(f"Tokens: 0 / {self.current_model_context}")

            self.check_conversation_size()

            if use_markdown and "\n```" in prompt and not prompt.endswith("```"):
                count = prompt.count("```")
                if count % 2 == 1:
                    prompt += "\n```"

            conversation_history.append(f"User:\n{prompt}")
            conversation_history.append("AI:\n")
            self.current_ai_index = len(conversation_history) - 1

            self.update_chat_history()
            self.current_ai_text = ""
            self.prompt_input.clear()
            self.chat_history_widget.setFocus()
            self.user_scrolled = False

            self.send_button.setText("Stop")
            self.reset_button.setEnabled(False)
            self.markdown_button.setEnabled(False)
            self.exit_button.setEnabled(False)
            self.load_model_button.setEnabled(False)
            self.stop_requested = False
            self.status_label.setText("⟳ Generating response... Please wait.")

            self.start_worker(prompt, conversation_history[:])

    def stop_worker(self):
        if self.thread is not None and self.thread.isRunning():
            print("DEBUG: Stopping response generation...")
            self.status_label.setText("⏳ Stopping generation...")

            try:
                if self.worker is not None:
                    self.worker.should_stop = True
                    print("DEBUG: Set should_stop flag")

                    response_obj = getattr(self.worker, 'response', None)
                    if response_obj is not None and hasattr(response_obj, 'close'):
                        try:
                            response_obj.close()
                            print("DEBUG: Closed response connection")
                        except Exception as e:
                            print(f"DEBUG: Non-critical error closing response: {e}")

                QtWidgets.QApplication.processEvents()

            except Exception as e:
                print(f"DEBUG: Error during stopping: {e}")

            print("DEBUG: Stop signal sent to worker")

    def start_worker(self, prompt, current_history):
        if self.thread is not None:
            if self.thread.isRunning():
                print("DEBUG: Stopping running thread.")
                self.thread.quit()
                self.thread.wait(1000)
                print("DEBUG: Stopped and cleaned up the thread.")
            self.thread = None
            self.worker = None

        print("DEBUG: Creating a new thread and worker.")
        self.thread = QtCore.QThread()
        self.worker = RequestWorker(prompt, current_history, self.current_model_context)
        self.worker.moveToThread(self.thread)

        self.worker.tokenCountUpdate.connect(self.update_token_count)
        self.worker.newChunk.connect(self.handle_new_chunk)
        self.worker.finished.connect(self.handle_finished)
        self.worker.connectionError.connect(self.handle_connection_error)
        self.worker.stoppedByUser.connect(self.handle_stopped_by_user)
        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.destroyed.connect(lambda: setattr(self, 'thread', None))
        self.worker.destroyed.connect(lambda: setattr(self, 'worker', None))

        self.thread.start()
        print("DEBUG: Thread started.")

    def handle_connection_error(self, error_msg):
        self.api_connected = False
        self.update_connect_button()
        self.send_button.setText("Send")
        self.reset_button.setEnabled(True)
        self.markdown_button.setEnabled(True)
        self.exit_button.setEnabled(True)
        self.load_model_button.setEnabled(True)
        self.status_label.setText(f"❌ {error_msg}")
        QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))

        if not self.stop_requested and self.current_ai_index < len(conversation_history):
            if not self.current_ai_text:
                self.current_ai_text = "I'm sorry, I encountered a connection error. Please try again."
            conversation_history[self.current_ai_index] = "AI:\n" + self.current_ai_text
            self.update_chat_history()

        if self.auto_reconnect_enabled:
            QtCore.QTimer.singleShot(500, self.check_api_connection)

    def handle_stopped_by_user(self):
        self.status_label.setText("✓ Generation stopped by user.")
        QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))

        self.send_button.setText("Send")
        self.reset_button.setEnabled(True)
        self.markdown_button.setEnabled(True)
        self.exit_button.setEnabled(True)
        self.load_model_button.setEnabled(True)
        self.stop_requested = False

    def handle_new_chunk(self, chunk):
        print(f"DEBUG: handle_new_chunk: Received chunk len: {len(chunk)}")
        self.current_ai_text += chunk
        print(f"DEBUG: handle_new_chunk: Total AI text len: {len(self.current_ai_text)}")
        if 0 <= self.current_ai_index < len(conversation_history):
            conversation_history[self.current_ai_index] = "AI:\n" + self.current_ai_text
        else:
            print(f"DEBUG: handle_new_chunk: Invalid current_ai_index {self.current_ai_index}")

        current_time = time.time()
        if not hasattr(self, 'last_ui_update_time'):
            self.last_ui_update_time = 0

        if current_time - self.last_ui_update_time > 0.3 or len(chunk) < 50:
            print(f"DEBUG: handle_new_chunk: Calling update_chat_history")
            self.update_chat_history()
            print(f"DEBUG: handle_new_chunk: Returned from update_chat_history")
            self.last_ui_update_time = current_time

    def handle_finished(self):
        print("DEBUG: handle_finished: START")
        self.send_button.setText("Send")
        self.reset_button.setEnabled(True)
        self.markdown_button.setEnabled(True)
        self.exit_button.setEnabled(True)
        self.load_model_button.setEnabled(True)

        if not self.status_label.text().startswith("❌") and not self.status_label.text().startswith("⚠️"):
            self.status_label.setText("")

        self.stop_requested = False
        print("DEBUG: handle_finished: END")

    def reset_chat(self):
        global conversation_history
        conversation_history = []
        self.chat_history_widget.clear()
        self.last_token_count = 0
        self.token_count_label.setText(f"Tokens: 0 / {self.current_model_context}")
        self.showed_markdown_warning = False

    def update_chat_history(self):
        """Update the chat history display with the latest content, always full render."""
        print(f"DEBUG: update_chat_history: START")
        try:
            # Capture scroll state before re-render
            scrollbar = self.chat_history_widget.verticalScrollBar()
            prev_max = scrollbar.maximum()
            prev_val = scrollbar.value()
            was_at_bottom = prev_val >= prev_max - 2
            distance_from_bottom = prev_max - prev_val  # used if user scrolled up

            # Temporary history with closed code block during streaming (if needed)
            current_ai_text = ""
            if 0 <= self.current_ai_index < len(conversation_history):
                current_content = conversation_history[self.current_ai_index]
                if current_content.startswith("AI:\n"):
                    current_ai_text = current_content[4:]

            temp_history = conversation_history[:]
            if self.thread is not None and self.thread.isRunning() and self.current_ai_index >= 0:
                if "```" in current_ai_text and current_ai_text.count("```") % 2 == 1:
                    temp_history[self.current_ai_index] = temp_history[self.current_ai_index] + "\n```"

            print(f"DEBUG: update_chat_history: Calling build_html_chat_history")
            html_content = build_html_chat_history(temp_history)
            print(f"DEBUG: update_chat_history: Returned from build_html, HTML len: {len(html_content)}")

            print(f"DEBUG: update_chat_history: Calling setHtml")
            self.chat_history_widget.setHtml(html_content)
            print(f"DEBUG: update_chat_history: Returned from setHtml")

            # Restore scroll: keep relative distance if user scrolled; otherwise stick to bottom
            new_max = scrollbar.maximum()
            if self.user_scrolled and not was_at_bottom:
                target = max(0, new_max - distance_from_bottom)
                scrollbar.setValue(target)
            else:
                scrollbar.setValue(new_max)

        except Exception as e:
            print(f"DEBUG: Error updating chat history: {e}")
            self.status_label.setText("⚠️ Error updating display")
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
        print(f"DEBUG: update_chat_history: END")

# -------------------------------------------------------
# 9) Dark Mode Styling
# -------------------------------------------------------
def apply_dark_mode(app):
    QtWidgets.QApplication.setStyle("Fusion")

    dark_palette = QtGui.QPalette()

    base_window_color = QtGui.QColor("#2f2f2f")
    chat_bg_color     = QtGui.QColor("#2a2a2a")
    alt_base_color    = QtGui.QColor("#3b3b3b")
    text_color        = QtGui.QColor("#ffffff")
    button_color      = QtGui.QColor("#3e3e3e")
    highlight_color   = QtGui.QColor("#537BA2")
    border_color      = QtGui.QColor("#4f4f4f")

    dark_palette.setColor(QtGui.QPalette.Window, base_window_color)
    dark_palette.setColor(QtGui.QPalette.WindowText, text_color)
    dark_palette.setColor(QtGui.QPalette.Base, alt_base_color)
    dark_palette.setColor(QtGui.QPalette.AlternateBase, base_window_color)
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, text_color)
    dark_palette.setColor(QtGui.QPalette.ToolTipText, text_color)
    dark_palette.setColor(QtGui.QPalette.Text, text_color)
    dark_palette.setColor(QtGui.QPalette.Button, button_color)
    dark_palette.setColor(QtGui.QPalette.ButtonText, text_color)
    dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    dark_palette.setColor(QtGui.QPalette.Link, highlight_color)
    dark_palette.setColor(QtGui.QPalette.Highlight, highlight_color)
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)

    app.setPalette(dark_palette)

    app.setStyleSheet(f"""
        QWidget {{
            font-size: 10pt;
            color: {text_color.name()};
        }}
        QToolTip {{
            color: #ffffff;
            background-color: {highlight_color.name()};
            border: 1px solid {text_color.name()};
        }}
        QPushButton {{
            border: 1px solid {border_color.name()};
            background-color: {button_color.name()};
            padding: 6px;
        }}
        QPushButton:hover {{
            background-color: #4a4a4a;
        }}
        QPushButton:pressed {{
            background-color: #5a5a5a;
        }}
        QPushButton:disabled {{
            background-color: #282828;
            color: #606060;
            border: 1px solid #404040;
        }}
        QLineEdit, QComboBox {{
            background-color: {alt_base_color.name()};
            border: 1px solid {border_color.name()};
            color: {text_color.name()};
        }}
        QComboBox::drop-down {{
            border-left: 1px solid {border_color.name()};
        }}
        QTextEdit#ChatHistory {{
            background-color: {chat_bg_color.name()};
            border: 1px solid {border_color.name()};
            color: {text_color.name()};
        }}
        QTextEdit#PromptInput {{
            background-color: {alt_base_color.name()};
            border: 1px solid {border_color.name()};
            color: {text_color.name()};
        }}
        QScrollBar:vertical {{
            background-color: {alt_base_color.name()};
            width: 12px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background-color: #555555;
            min-height: 20px;
            border-radius: 6px;
            margin: 2px;
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            background: none;
            border: none;
            height: 0px;
        }}
        QScrollBar:horizontal {{
            background-color: {alt_base_color.name()};
            height: 12px;
            margin: 0px;
        }}
        QScrollBar::handle:horizontal {{
            background-color: #555555;
            min-width: 20px;
            border-radius: 6px;
            margin: 2px;
        }}
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {{
            background: none;
            border: none;
            width: 0px;
        }}
    """)

# -------------------------------------------------------
# 10) Main Entry
# -------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_mode(app)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())