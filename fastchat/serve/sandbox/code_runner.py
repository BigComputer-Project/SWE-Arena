'''
Run generated code in a sandbox environment.

Gradio will interact with this module.
'''

from typing import Any, Generator, TypeAlias, TypedDict, Set
import gradio as gr

import base64
from e2b_code_interpreter import Sandbox as CodeSandbox
from gradio_sandboxcomponent import SandboxComponent

from fastchat.serve.sandbox.code_analyzer import SandboxEnvironment, extract_code_from_markdown, extract_installation_commands, extract_js_imports, extract_python_imports, replace_placeholder_urls, validate_dependencies


from .constants import E2B_API_KEY, SANDBOX_TEMPLATE_ID, SANDBOX_NGINX_PORT
from .sandbox_manager import get_sandbox_app_url, create_sandbox, install_npm_dependencies, install_pip_dependencies, run_background_command_with_timeout


SUPPORTED_SANDBOX_ENVIRONMENTS: list[str] = [
    env.value for env in SandboxEnvironment
]

WEB_UI_SANDBOX_ENVIRONMENTS = [
    SandboxEnvironment.HTML,
    SandboxEnvironment.REACT,
    SandboxEnvironment.VUE,
    SandboxEnvironment.GRADIO,
    SandboxEnvironment.STREAMLIT,
    # SandboxEnvironment.NICEGUI,
    SandboxEnvironment.PYGAME,
    SandboxEnvironment.MERMAID
]

VALID_GRADIO_CODE_LANGUAGES = [
    'python', 'c', 'cpp', 'markdown', 'json', 'html', 'css', 'javascript', 'jinja2', 'typescript', 'yaml', 'dockerfile', 'shell', 'r', 'sql',
    'sql-msSQL', 'sql-mySQL', 'sql-mariaDB', 'sql-sqlite', 'sql-cassandra', 'sql-plSQL', 'sql-hive', 'sql-pgSQL', 'sql-gql', 'sql-gpSQL', 'sql-sparkSQL',
    'sql-esper'
]
'''
Languages that gradio code component can render.
'''

RUN_CODE_BUTTON_HTML = "<button style='background-color: #4CAF50; border: none; color: white; padding: 10px 24px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px;'>Click to Run in Sandbox</button>"
'''
Button in the chat to run the code in the sandbox.
'''

GENERAL_SANDBOX_INSTRUCTION = """\
You are an expert Software Engineer, UI/UX designer, and product manager. Your task is to generate self-contained, executable code for a single file or block that can run directly in a sandbox environment. Feel free to ask questions or explain your reasoning.
If you do a great job based on the instructions, you will be rewarded with a high salary and a promotion.

Your code must be written using one of these supported development frameworks and environments:
- React (JavaScript/TypeScript)
- Vue (JavaScript/TypeScript)
- HTML (Vanilla HTML)
- Gradio (Python)
- Streamlit (Python)
- PyGame (Python)
- Mermaid (Markdown)
- Python Code Interpreter
- JavaScript Code Interpreter

All web framework code (React, Vue, HTML) must be directly rendered in a browser and immediately executable without additional setup. DO NOT create separate CSS files
Python-based frameworks should be directly executable in a browser environment.
The code to be executed in Code Interpreters must be plain Python or JavaScript programs that do not require web UI frameworks or standard user input.

The code must be in the markdown format:
```<language>
<code>
```

Before you begin writing any code, you must follow these fundamental rules:
- You are NOT allowed to start directly with a code block. Before writing code, ALWAYS think carefully step-by-step
- Your response must contain a clear explanation of the solution you are providing
- ALWAYS generate complete, self-contained code in a single file
- You CAN NOT split your program into multiple files or multiple code blocks
- If you use any external libraries, make sure to specify them for the installation command in either `pip install` or `npm install`
- You prefer JavaScript over HTML
- Each code block must be completely independent. If modifications are needed, the entire code block must be rewritten
- When fetching data, you MUST use external libraries and packages, and avoid using placeholder URLs or URLs that require API keys
- Make sure the program is functional by creating a state when needed and having no required props
- Make sure to include all necessary code in one file
- There are no additional files in the local file system, unless you create them inside the same program
- Do not touch project dependencies files like package.json, package-lock.json, requirements.txt, etc

When developing with React or Vue components, follow these specific requirements:
- Use TypeScript or JavaScript as the language
- DO NOT use gray text color on a white background
- Make sure it can run by itself by using a default export at the end of the file
- DO NOT CALL `ReactDOM.render()` AT THE END OF THE FILE
- Use Tailwind classes for styling. DO NOT USE ARBITRARY VALUES (e.g. 'h-[600px]'). Make sure to use a consistent color palette
- If you use any imports from React like `useState` or `useEffect`, make sure to import them directly
- Use Tailwind margin and padding classes to style the components and ensure proper spacing
- Various npm packages are available to be imported, e.g. `import { LineChart, XAxis, ... } from "recharts"` & `<LineChart ...><XAxis dataKey="name"> ...`
- Images from the web are not allowed, but you can use placeholder images by specifying the width and height like so `<img src="/api/placeholder/400/320" alt="placeholder" />`

For Python development, you must follow these constraints:
- For any programs that require user inputs, you MUST USE `gradio` or `streamlit`
- Choose suitable PyPI packages to be imported, e.g., `import pandas`
- Avoid using libraries that require desktop GUI interfaces, with the exceptions of `pygame`, `gradio`, and `streamlit` which are explicitly supported
- For PyGame applications, you have to write the main function as an async function like:
```python
import asyncio
import pygame

async def main():
    global game_state
    while game_state:
        game_state(pygame.event.get())
        pygame.display.update()
        await asyncio.sleep(0) # it must be called on every frame

if __name__ == "__main__":
    asyncio.run(main())
```

For HTML development, ensure that:
- All HTML code must be self-contained in a single file
- Include any necessary CSS and JavaScript within the HTML file
- Ensure the code is directly executable in a browser environment
- Images from the web are not allowed, but you can use placeholder images by specifying the width and height like so `<img src="/api/placeholder/400/320" alt="placeholder" />`

For Mermaid development:
- Write Mermaid diagrams directly using ```mermaid code blocks, e.g.:
```mermaid
graph TD;
    A-->B;
```
"""

DEFAULT_PYTHON_CODE_INTERPRETER_INSTRUCTION = """
You are an expert Software Engineer. Your task is to generate self-contained, executable Python code that can run directly in a code interpreter environment.

Before you begin writing any code, you must follow these fundamental rules:
- You are NOT allowed to start directly with a code block. Before writing code, ALWAYS think carefully step-by-step
- Your response must contain a clear explanation of the solution you are providing
- ALWAYS generate complete, self-contained code in a single file
- If you use any external libraries, make sure to specify them for installation with `pip install`
- Make sure to include all necessary code in one file
- Make sure it does not require any user inputs
- Choose suitable PyPI packages to be imported, e.g., `import pandas`

The code must be in the markdown format:
```python
<code>
```

You can output in stdout, stderr, or render images, plots, and tables.
"""

DEFAULT_JAVASCRIPT_CODE_INTERPRETER_INSTRUCTION = """
You are an expert Software Engineer. Your task is to generate self-contained JavaScript code that can run directly in a code interpreter environment.

Before you begin writing any code, you must follow these fundamental rules:
- You are NOT allowed to start directly with a code block. Before writing code, ALWAYS think carefully step-by-step
- Your response must contain a clear explanation of the solution you are providing
- ALWAYS generate complete, self-contained code in a single file
- If you use any external libraries, make sure to specify them for installation with `npm install`
- Make sure to include all necessary code in one file
- Ensure the code is self-contained and does not rely on browser-specific APIs

The code must be in the markdown format:
```javascript
<code>
```

You can output in stdout, stderr, or render images, plots, and tables.
"""

DEFAULT_HTML_SANDBOX_INSTRUCTION = """
You are an expert Software Engineer and UI/UX designer. Your task is to generate self-contained HTML code that can run directly in a browser environment.

Before you begin writing any code, you must follow these fundamental rules:
- You are NOT allowed to start directly with a code block. Before writing code, ALWAYS think carefully step-by-step
- Your response must contain a clear explanation of the solution you are providing
- ALWAYS generate complete, self-contained code in a single file
- Include any necessary CSS and JavaScript within the HTML file
- If you use any external libraries, make sure to specify them
- Make sure the program is functional by creating a state when needed
- Images from the web are not allowed, but you can use placeholder images by specifying the width and height like so `<img src="/api/placeholder/400/320" alt="placeholder" />`

The code must be in the markdown format:
```html
<code>
```

For HTML development, ensure that:
- All HTML code must be self-contained in a single file
- Include any necessary CSS and JavaScript within the HTML file
- Ensure the code is directly executable in a browser environment
- Images from the web are not allowed, but you can use placeholder images by specifying the width and height like so `<img src="/api/placeholder/400/320" alt="placeholder" />`
"""

DEFAULT_REACT_SANDBOX_INSTRUCTION = """
You are an expert Software Engineer and UI/UX designer. Your task is to generate a self-contained React component using TypeScript that can run directly in a browser environment.

Before you begin writing any code, you must follow these fundamental rules:
- You are NOT allowed to start directly with a code block. Before writing code, ALWAYS think carefully step-by-step
- Your response must contain a clear explanation of the solution you are providing
- ALWAYS generate complete, self-contained code in a single file
- If you use any external libraries, make sure to specify them for installation with `npm install`
- Make sure the program is functional by creating a state when needed and having no required props
- Make sure it can run by itself by using a default export at the end of the file
- DO NOT CALL `ReactDOM.render()` AT THE END OF THE FILE
- Use Tailwind classes for styling. DO NOT USE ARBITRARY VALUES (e.g. 'h-[600px]'). Make sure to use a consistent color palette
- If you use any imports from React like `useState` or `useEffect`, make sure to import them directly
- Images from the web are not allowed, but you can use placeholder images by specifying the width and height like so `<img src="/api/placeholder/400/320" alt="placeholder" />`

The code must be in the markdown format:
```typescript
<code>
```

When developing with React components, follow these specific requirements:
- Use TypeScript or JavaScript as the language
- DO NOT use gray text color on a white background
- Make sure it can run by itself by using a default export at the end of the file
- DO NOT CALL `ReactDOM.render()` AT THE END OF THE FILE
- Use Tailwind classes for styling. DO NOT USE ARBITRARY VALUES (e.g. 'h-[600px]'). Make sure to use a consistent color palette
- If you use any imports from React like `useState` or `useEffect`, make sure to import them directly
- Use Tailwind margin and padding classes to style the components and ensure proper spacing
- Various npm packages are available to be imported, e.g. `import { LineChart, XAxis, ... } from "recharts"` & `<LineChart ...><XAxis dataKey="name"> ...`
- Images from the web are not allowed, but you can use placeholder images by specifying the width and height like so `<img src="/api/placeholder/400/320" alt="placeholder" />`
"""

DEFAULT_VUE_SANDBOX_INSTRUCTION = """
You are an expert Software Engineer and UI/UX designer. Your task is to generate a self-contained Vue.js component using TypeScript that can run directly in a browser environment.

Before you begin writing any code, you must follow these fundamental rules:
- You are NOT allowed to start directly with a code block. Before writing code, ALWAYS think carefully step-by-step
- Your response must contain a clear explanation of the solution you are providing
- ALWAYS generate complete, self-contained code in a single file
- If you use any external libraries, make sure to specify them for installation with `npm install`
- Make sure the program is functional by creating a state when needed and having no required props
- The component should be a simple custom page in a styled `<div>` element
- Do not include <NuxtWelcome /> or reference any external components
- Use Tailwind classes for styling. DO NOT USE ARBITRARY VALUES (e.g. 'h-[600px]'). Make sure to use a consistent color palette
- Images from the web are not allowed, but you can use placeholder images by specifying the width and height like so `<img src="/api/placeholder/400/320" alt="placeholder" />`

The code must be in the markdown format:
```vue
<code>
```

When developing with Vue components, follow these specific requirements:
- Use Vue 3's Composition API with <script setup> syntax for better TypeScript integration
- Use TypeScript for type safety and better developer experience
- Properly type all props, emits, and refs using Vue 3's type system
- Use defineProps, defineEmits, and other Vue 3 macros correctly
- Implement reactive state management using ref() or reactive() from Vue
- Follow Vue 3's best practices for component organization and lifecycle management
- Use computed properties for derived state
- Handle component events using proper Vue 3 event handling syntax
- Use Tailwind classes for styling with a consistent design system
- Ensure components are responsive using Tailwind's responsive classes
- Use Vue's built-in transition and animation systems when needed
- Follow proper Vue 3 security practices (e.g., v-html only when necessary)
- Implement proper error handling and loading states
- Add comments explaining complex logic or component structure
- Use async/await for asynchronous operations
- Ensure the component is accessible following ARIA best practices
"""

DEFAULT_PYGAME_SANDBOX_INSTRUCTION = """
You are an expert Software Engineer and UI/UX designer. Your task is to generate self-contained PyGame code that can run directly in a browser environment.

Before you begin writing any code, you must follow these fundamental rules:
- You are NOT allowed to start directly with a code block. Before writing code, ALWAYS think carefully step-by-step
- Your response must contain a clear explanation of the solution you are providing
- ALWAYS generate complete, self-contained code in a single file
- If you use any external libraries, make sure to specify them for installation with `pip install`
- Make sure it does not require any user inputs
- Write the main function as an async function like:

```python
import asyncio
import pygame

async def main():
    global game_state
    while game_state:
        game_state(pygame.event.get())
        pygame.display.update()
        await asyncio.sleep(0) # it must be called on every frame

if __name__ == "__main__":
    asyncio.run(main())
```

The code must be in the markdown format:
```python
<code>
```
"""

DEFAULT_GRADIO_SANDBOX_INSTRUCTION = """
You are an expert Software Engineer and UI/UX designer. Your task is to generate self-contained Gradio application code that can run directly in a browser environment.

Before you begin writing any code, you must follow these fundamental rules:
- You are NOT allowed to start directly with a code block. Before writing code, ALWAYS think carefully step-by-step
- Your response must contain a clear explanation of the solution you are providing
- ALWAYS generate complete, self-contained code in a single file
- If you use any external libraries, make sure to specify them for installation with `pip install`
- Make sure it does not require any user inputs
- Choose suitable PyPI packages to be imported, e.g., `import pandas`

The code must be in the markdown format:
```python
<code>
```
"""

DEFAULT_STREAMLIT_SANDBOX_INSTRUCTION = """
You are an expert Software Engineer and UI/UX designer. Your task is to generate self-contained Streamlit application code that can run directly in a browser environment.

Before you begin writing any code, you must follow these fundamental rules:
- You are NOT allowed to start directly with a code block. Before writing code, ALWAYS think carefully step-by-step
- Your response must contain a clear explanation of the solution you are providing
- ALWAYS generate complete, self-contained code in a single file
- If you use any external libraries, make sure to specify them for installation with `pip install`
- Make sure it does not require any user inputs
- Choose suitable PyPI packages to be imported, e.g., `import pandas`
- The app should automatically reload when changes are made

The code must be in the markdown format:
```python
<code>
```
"""

DEFAULT_MERMAID_SANDBOX_INSTRUCTION = """
You are an expert Software Engineer. Your task is to generate self-contained Mermaid diagram code that can be rendered directly.

Before you begin writing any code, you must follow these fundamental rules:
- You are NOT allowed to start directly with a code block. Before writing code, ALWAYS think carefully step-by-step
- Your response must contain a clear explanation of the solution you are providing
- ALWAYS generate complete, self-contained code in a single file

The code must be in the markdown format:
```mermaid
<code>
```

Example:
```mermaid
graph TD;
    A-->B;
```
"""


DEFAULT_SANDBOX_INSTRUCTIONS: dict[SandboxEnvironment, str] = {
    SandboxEnvironment.AUTO: GENERAL_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.PYTHON_CODE_INTERPRETER: DEFAULT_PYTHON_CODE_INTERPRETER_INSTRUCTION.strip(),
    SandboxEnvironment.JAVASCRIPT_CODE_INTERPRETER: DEFAULT_JAVASCRIPT_CODE_INTERPRETER_INSTRUCTION.strip(),
    SandboxEnvironment.HTML: DEFAULT_HTML_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.REACT: DEFAULT_REACT_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.VUE: DEFAULT_VUE_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.GRADIO: DEFAULT_GRADIO_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.STREAMLIT: DEFAULT_STREAMLIT_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.PYGAME: DEFAULT_PYGAME_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.MERMAID: DEFAULT_MERMAID_SANDBOX_INSTRUCTION.strip()
}


SandboxGradioSandboxComponents: TypeAlias =  tuple[
    gr.Markdown | Any,  # sandbox_output
    SandboxComponent | Any,  # sandbox_ui
    gr.Code | Any,  # sandbox_code
]
'''
Gradio components for the sandbox.
'''

class ChatbotSandboxState(TypedDict):
    '''
    Chatbot sandbox state in gr.state.
    '''
    enable_sandbox: bool
    '''
    Whether the code sandbox is enabled.
    '''
    sandbox_instruction: str | None
    '''
    The sandbox instruction to display.
    '''
    enabled_round: int
    '''
    The chat round after which the sandbox is enabled.
    '''
    sandbox_environment: SandboxEnvironment | None
    '''
    The sandbox environment to run the code.
    '''
    auto_selected_sandbox_environment: SandboxEnvironment | None
    '''
    The sandbox environment selected automatically.
    '''
    code_to_execute: str | None
    '''
    The code to execute in the sandbox.
    '''
    code_language: str | None
    '''
    The code language to execute in the sandbox.
    '''
    code_dependencies: tuple[list[str], list[str]]
    '''
    The code dependencies for the sandbox (python, npm).
    '''
    sandbox_id: str | None
    '''
    The sandbox id. None if no running.
    '''
    btn_list_length: int | None


def create_chatbot_sandbox_state(btn_list_length: int = 5) -> ChatbotSandboxState:
    '''
    Create a new sandbox state for a chatbot.
    '''
    return {
        'enable_sandbox': True,  # Always enabled
        'enabled_round': 0,
        'sandbox_environment': SandboxEnvironment.AUTO,
        'auto_selected_sandbox_environment': None,
        'sandbox_instruction': DEFAULT_SANDBOX_INSTRUCTIONS[SandboxEnvironment.AUTO],
        'code_to_execute': "",
        'code_language': None,
        'code_dependencies': ([], []),
        'btn_list_length': btn_list_length,
        'sandbox_id': None,
    }


def update_sandbox_config_multi(
    enable_sandbox: bool,
    sandbox_environment: SandboxEnvironment,
    *states: ChatbotSandboxState
) -> list[ChatbotSandboxState]:
    '''
    Fn to update sandbox config.
    '''
    return [
        update_sandbox_config(enable_sandbox, sandbox_environment, state)
        for state
        in states
    ]

def update_sandbox_config(
    enable_sandbox: bool,
    sandbox_environment: SandboxEnvironment,
    state: ChatbotSandboxState
) -> ChatbotSandboxState:
    '''
    Fn to update sandbox config for single model.
    '''
    state["enable_sandbox"] = enable_sandbox
    state["sandbox_environment"] = sandbox_environment
    state['sandbox_instruction'] = DEFAULT_SANDBOX_INSTRUCTIONS.get(sandbox_environment, None)
    return state


def update_visibility(visible):
    return [gr.update(visible=visible)] *14


def update_visibility_for_single_model(visible: bool, component_cnt: int):
    return [gr.update(visible=visible)] * component_cnt


def mermaid_to_html(mermaid_code: str, theme: str = 'default') -> str:
    """
    Convert Mermaid diagram code to a minimal HTML document.

    Args:
        mermaid_code: The Mermaid diagram syntax
        theme: Theme name ('default', 'dark', 'forest', 'neutral', etc.)

    Returns:
        str: Complete HTML document with embedded Mermaid diagram
    """
    html_template = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';

        mermaid.initialize({{
            startOnLoad: true,
            theme: '{theme}'
        }});
    </script>
</head>
<body>
    <pre class="mermaid">
{mermaid_code}
    </pre>
</body>
</html>'''
    return html_template


def render_result(result):
    if result.png:
        if isinstance(result.png, str):
            img_str = result.png
        else:
            img_str = base64.b64encode(result.png).decode()
        return f"![png image](data:image/png;base64,{img_str})"
    elif result.jpeg:
        if isinstance(result.jpeg, str):
            img_str = result.jpeg
        else:
            img_str = base64.b64encode(result.jpeg).decode()
        return f"![jpeg image](data:image/jpeg;base64,{img_str})"
    elif result.svg:
        if isinstance(result.svg, str):
            svg_data = result.svg
        else:
            svg_data = result.svg.decode()
        svg_base64 = base64.b64encode(svg_data.encode()).decode()
        return f"![svg image](data:image/svg+xml;base64,{svg_base64})"
    elif result.html:
        return result.html
    elif result.markdown:
        return f"```markdown\n{result.markdown}\n```"
    elif result.latex:
        return f"```latex\n{result.latex}\n```"
    elif result.json:
        return f"```json\n{result.json}\n```"
    elif result.javascript:
        return result.javascript  # Return raw JavaScript
    else:
        return str(result)


def run_code_interpreter(code: str, code_language: str | None, code_dependencies: tuple[list[str], list[str]]) -> tuple[str, str]:
    """
    Executes the provided code within a sandboxed environment and returns the output.

    Args:
        code (str): The code to be executed.
    """
    sandbox = CodeSandbox(
        api_key=E2B_API_KEY,
    )

    sandbox.commands.run("pip install uv",
                         timeout=60 * 3,
                         on_stderr=lambda message: print(message),)

    python_dependencies, npm_dependencies = code_dependencies
    install_pip_dependencies(sandbox, python_dependencies)
    install_npm_dependencies(sandbox, npm_dependencies)

    execution = sandbox.run_code(
        code=code,
        language=code_language
    )

    # collect stdout, stderr from sandbox
    stdout = "\n".join(execution.logs.stdout)
    stderr = "\n".join(execution.logs.stderr)
    if execution.error:
        stderr += f"\n{execution.error.name}: {execution.error.value}"
    output = ""
    if stdout:
        output += f"### Stdout:\n```\n{stdout}\n```\n\n"

    results = []
    for result in execution.results:
        if result.html or result.javascript:
            # TODO: fix this
            continue
            # with open('html_code.html', 'w') as f:
            #     f.write(result.html)
            # url, _ = run_html_sandbox(result.html, ([], extract_js_imports(result.html)))
        else:
            rendered_result = render_result(result)
            results.append(rendered_result)
    if results:
        output += "\n### Results:\n" + "\n".join(results)

    return output, "" if output else stderr


def run_html_sandbox(code: str, code_dependencies: tuple[list[str], list[str]]) -> tuple[str, str, str]:
    """
    Executes the provided code within a sandboxed environment and returns the output.
    Supports both React and Vue.js rendering in HTML files.

    Args:
        code (str): The code to be executed.
        code_dependencies: Tuple of (python_deps, npm_deps)

    Returns:
        tuple: (sandbox_url, sandbox_id, stderr)
    """
    sandbox = create_sandbox()
    project_root = "~/html"
    sandbox.files.make_dir(project_root)

    _, npm_dependencies = code_dependencies
    install_npm_dependencies(sandbox, npm_dependencies, project_root=project_root)

    # replace placeholder URLs with SVG data URLs
    code = replace_placeholder_urls(code)

    file_path = f"{project_root}/index.html"
    sandbox.files.write(path=file_path, data=code, request_timeout=60)

    sandbox_url = get_sandbox_app_url(sandbox, 'html')
    return (sandbox_url, sandbox.sandbox_id, '')


def run_react_sandbox(code: str, code_dependencies: tuple[list[str], list[str]]) -> tuple[str, str, str]:
    """
    Executes the provided code within a sandboxed environment and returns the output.

    Args:
        code (str): The code to be executed.

    Returns:
        url for remote sandbox
    """
    project_root = "~/react_app"
    sandbox = create_sandbox()

    stderrs: list[str] = [] # to collect errors

    _, npm_dependencies = code_dependencies
    if npm_dependencies:
        print(f"Installing NPM dependencies...: {npm_dependencies}")
        install_errs = install_npm_dependencies(sandbox, npm_dependencies, project_root=project_root)
        stderrs.extend(install_errs)
        print("NPM dependencies installed. " + "Errors: " + str(install_errs))

    # replace placeholder URLs with SVG data URLs
    code = replace_placeholder_urls(code)

    # set up the sandbox
    print("Setting up sandbox directory structure...")
    file_path = "~/react_app/src/App.tsx"
    sandbox.files.write(path=file_path, data=code, request_timeout=60)
    print("Code files written successfully.")

    try:
        sandbox.commands.run(
            cmd="npm run build --loglevel=error -- --mode development --logLevel error",
            cwd=project_root,
            timeout=60,
            on_stdout=print,
            on_stderr=lambda message: stderrs.append(message),
        )
    except Exception as e:
        print(f"Error running npm build command: {e}")
        stderrs.append(str(e))

    sandbox_url = get_sandbox_app_url(sandbox, 'react')
    return (sandbox_url, sandbox.sandbox_id, '\n'.join(stderrs))


def run_vue_sandbox(code: str, code_dependencies: tuple[list[str], list[str]]) -> tuple[str, str, str]:
    """
    Executes the provided Vue code within a sandboxed environment and returns the output.

    Args:
        code (str): The Vue code to be executed.

    Returns:
        url for remote sandbox
    """
    sandbox = create_sandbox()
    project_root = "~/vue_app"

    stderrs: list[str] = [] # to collect errors

    # replace placeholder URLs with SVG data URLs
    code = replace_placeholder_urls(code)

    # Set up the sandbox
    file_path = "~/vue_app/src/App.vue"
    sandbox.files.write(path=file_path, data=code, request_timeout=60)

    _, npm_dependencies = code_dependencies
    if npm_dependencies:
        print(f"Installing NPM dependencies...: {npm_dependencies}")
        install_errs = install_npm_dependencies(sandbox, npm_dependencies, project_root=project_root)
        stderrs.extend(install_errs)
        print("NPM dependencies installed. " + "Errors: " + str(install_errs))

    try:
        sandbox.commands.run(
            cmd="npm run build --loglevel=error -- --mode development --logLevel error",
            cwd=project_root,
            timeout=60,
            on_stdout=print,
            on_stderr=lambda message: stderrs.append(message),
        )
    except Exception as e:
        print(f"Error running npm build command: {e}")
        stderrs.append(str(e))

    sandbox_url = get_sandbox_app_url(sandbox, 'vue')
    return (sandbox_url, sandbox.sandbox_id, '\n'.join(stderrs))


def run_pygame_sandbox(code: str, code_dependencies: tuple[list[str], list[str]]) -> tuple[str, str, str]:
    """
    Executes the provided code within a sandboxed environment and returns the output.

    Args:
        code (str): The code to be executed.

    Returns:
        url for remote sandbox
    """
    sandbox = create_sandbox()
    project_root = "~/pygame_app"
    file_path = f"{project_root}/main.py"

    stderrs = []

    sandbox.files.write(path=file_path, data=code, request_timeout=60)

    python_dependencies, _ = code_dependencies
    install_pip_dependencies(sandbox, python_dependencies)

    # build the pygame code
    sandbox.commands.run(
        "pygbag --build ~/pygame_app",
        timeout=60,
        on_stdout=print,
        on_stderr=lambda message: stderrs.append(message),
    )

    stderr = '\n'.join(stderrs)
    if 'error' not in stderr.lower():
        # to ignore warnings
        stderr = ""

    sandbox_url = get_sandbox_app_url(sandbox, 'pygame')
    return (sandbox_url, sandbox.sandbox_id, stderr)


def run_gradio_sandbox(code: str, code_dependencies: tuple[list[str], list[str]]) -> tuple[str, str, str]:
    """
    Executes the provided code within a sandboxed environment and returns the output.

    Args:
        code (str): The code to be executed.

    Returns:
        url for remote sandbox and sandbox id
    """
    sandbox = create_sandbox()

    file_path = "~/gradio_app/main.py"
    sandbox.files.write(path=file_path, data=code, request_timeout=60)

    python_dependencies, _ = code_dependencies
    install_pip_dependencies(sandbox, python_dependencies)

    stderr = run_background_command_with_timeout(
        sandbox,
        f"python {file_path}",
        timeout=10,
    )

    sandbox_url = 'https://' + sandbox.get_host(7860)

    return (sandbox_url, sandbox.sandbox_id, stderr)


def run_streamlit_sandbox(code: str, code_dependencies: tuple[list[str], list[str]]) -> tuple[str, str, str]:
    sandbox = create_sandbox()

    sandbox.files.make_dir('mystreamlit')
    file_path = "~/mystreamlit/app.py"
    sandbox.files.write(path=file_path, data=code, request_timeout=60)

    python_dependencies, _ = code_dependencies
    install_pip_dependencies(sandbox, python_dependencies)

    stderr = run_background_command_with_timeout(
        sandbox,
        "streamlit run ~/mystreamlit/app.py --server.port 8501 --server.headless true",
        timeout=8,
    )

    host = sandbox.get_host(port=8501)
    url = f"https://{host}"
    return (url, sandbox.sandbox_id, stderr)


def on_edit_code(
    state,
    sandbox_state: ChatbotSandboxState,
    sandbox_output: gr.Markdown,
    sandbox_ui: SandboxComponent,
    sandbox_code: str,
    sandbox_dependency: gr.Dataframe,
) -> Generator[tuple[Any, Any, Any, Any], None, None]:
    '''
    Gradio Handler when code is edited manually by users.
    '''
    if sandbox_state['enable_sandbox'] is False:
        yield None, None, None, None
        return
    if len(sandbox_code.strip()) == 0 or sandbox_code == sandbox_state['code_to_execute']:
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return
    sandbox_state['code_to_execute'] = sandbox_code

    # Extract packages from installation commands (with versions)
    python_deps_with_version, npm_deps_with_version = extract_installation_commands(sandbox_code)

    # Extract packages from imports (without versions)
    python_deps_from_imports = extract_python_imports(sandbox_code)
    npm_deps_from_imports = extract_js_imports(sandbox_code)

    # Convert to dataframe format
    dependencies = []

    # Add packages with versions from installation commands
    for dep in python_deps_with_version:
        dependencies.append(["python", dep, "specified"])
    for dep in npm_deps_with_version:
        dependencies.append(["npm", dep, "specified"])

    # Add packages from imports with "latest" version if not already added
    existing_python_pkgs = {dep.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0] for dep in python_deps_with_version}
    existing_npm_pkgs = {dep.split('@')[0] for dep in npm_deps_with_version}

    for dep in python_deps_from_imports:
        if dep not in existing_python_pkgs:
            dependencies.append(["python", dep, "latest"])

    for dep in npm_deps_from_imports:
        if dep not in existing_npm_pkgs:
            dependencies.append(["npm", dep, "latest"])

    # If no dependencies found, provide default empty rows
    if not dependencies:
        dependencies = [["python", "", ""], ["npm", "", ""]]

    # Update dependencies in sandbox state
    sandbox_state["code_dependencies"] = (python_deps_with_version, npm_deps_with_version)
    yield (
        gr.skip(),  # sandbox_output
        gr.skip(),  # sandbox_ui
        gr.skip(),  # sandbox_code
        gr.update(value=dependencies),  # sandbox_dependency
    )
    yield from on_run_code(
        state,
        sandbox_state,
        sandbox_output,
        sandbox_ui,
        sandbox_code,
        sandbox_dependency,
    )


def on_edit_dependency(
    state,
    sandbox_state: ChatbotSandboxState,
    sandbox_dependency: gr.Dataframe,
    sandbox_output: gr.Markdown,
    sandbox_ui: SandboxComponent,
    sandbox_code: str,
) -> Generator[tuple[Any, Any, Any, Any], None, None]:
    """
    Gradio Handler when dependencies are edited manually by users.
    Handles version specifications and dependency removal.
    """
    if sandbox_state["enable_sandbox"] is False:
        yield None, None, None, None
        return

    # Validate dependencies format
    is_valid, error_msg = validate_dependencies(sandbox_dependency)
    if not is_valid:
        yield (
            gr.Markdown(f"Invalid dependencies: {error_msg}"),
            gr.skip(),
            gr.skip(),
            sandbox_dependency,  # Return original dataframe
        )
        return

    # Convert dataframe format to separate python and npm lists
    python_deps = []
    npm_deps = []
    for dep in sandbox_dependency:
        dep_type, pkg_name, version = dep
        pkg_name = pkg_name.strip()
        version = version.strip()

        # Skip empty rows
        if not pkg_name:
            continue

        if dep_type.lower() == "python":
            # Handle Python package with version
            if version and version.lower() != "latest":
                if not any(op in version for op in ["==", ">=", "<=", "~=", ">", "<"]):
                    python_deps.append(f"{pkg_name}=={version}")
                else:
                    python_deps.append(f"{pkg_name}{version}")
            else:
                python_deps.append(pkg_name)

        elif dep_type.lower() == "npm":
            # Handle NPM package with version
            if version and version.lower() != "latest":
                if not version.startswith("@"):
                    version = "@" + version
                npm_deps.append(f"{pkg_name}{version}")
            else:
                npm_deps.append(pkg_name)

    # Update sandbox state with new dependencies
    sandbox_state["code_dependencies"] = (python_deps, npm_deps)

    # First yield: Update UI with success message
    yield (
        gr.Markdown("Dependencies updated successfully"),
        gr.skip(),  # sandbox_ui
        gr.skip(),  # sandbox_code
        sandbox_dependency,  # Return the same dataframe
    )

    # Second yield: Run code with new dependencies
    yield from on_run_code(
        state,
        sandbox_state,
        sandbox_output,
        sandbox_ui,
        sandbox_code,
        sandbox_dependency,
    )


def on_click_code_message_run(
    state,
    sandbox_state: ChatbotSandboxState,
    sandbox_output: gr.Markdown,
    sandbox_ui: SandboxComponent,
    sandbox_code: str,
    sandbox_dependency: gr.Dataframe,
    evt: gr.SelectData
) -> Generator[SandboxGradioSandboxComponents, None, None]:
    '''
    Gradio Handler when run code button in message is clicked. Update Sandbox components.
    '''
    if sandbox_state['enable_sandbox'] is False:
        yield None, None, None, None
        return
    if not evt.value.endswith(RUN_CODE_BUTTON_HTML):
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return

    message = evt.value.replace(RUN_CODE_BUTTON_HTML, "").strip()
    extract_result = extract_code_from_markdown(
        message=message,
        enable_auto_env=sandbox_state['sandbox_environment'] == SandboxEnvironment.AUTO
    )
    if extract_result is None:
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return

    code, code_language, code_dependencies, env_selection = extract_result

    if sandbox_state['code_to_execute'] == code and sandbox_state['code_language'] == code_language:
        # skip if no changes
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return

    if code_language == 'tsx':
        code_language = 'typescript'
    code_language = code_language.lower() if code_language and code_language.lower(
        # ensure gradio supports the code language
    ) in VALID_GRADIO_CODE_LANGUAGES else None

    python_deps, npm_deps = code_dependencies

    # Convert to dataframe format
    dependencies = []

    # Add Python packages with versions
    for dep in python_deps:
        # Check if package has version specifier
        if any(op in dep for op in ['==', '>=', '<=', '~=']):
            # Split on first occurrence of version operator
            pkg_name = dep.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0]
            version = dep[len(pkg_name):]  # Get everything after package name
            dependencies.append(["python", pkg_name, version])
        else:
            dependencies.append(["python", dep, "latest"])

    # Add NPM packages with versions
    for dep in npm_deps:
        # Check if package has version specifier
        if '@' in dep and not dep.startswith('@'):
            # Handle non-scoped packages with version
            pkg_name, version = dep.split('@', 1)
            dependencies.append(["npm", pkg_name, '@' + version])
        elif '@' in dep[1:]:  # Handle scoped packages with version
            # Split on last @ for scoped packages
            pkg_parts = dep.rsplit('@', 1)
            dependencies.append(["npm", pkg_parts[0], '@' + pkg_parts[1]])
        else:
            dependencies.append(["npm", dep, "latest"])

    # If no dependencies found, provide default empty rows
    if not dependencies:
        dependencies = [["python", "", ""], ["npm", "", ""]]

    sandbox_state['code_to_execute'] = code
    sandbox_state['code_language'] = code_language
    sandbox_state["code_dependencies"] = code_dependencies
    if sandbox_state['sandbox_environment'] == SandboxEnvironment.AUTO:
        sandbox_state['auto_selected_sandbox_environment'] = env_selection

    yield (
        gr.skip(),  # sandbox_output
        gr.skip(),  # sandbox_ui
        gr.update(value=code, language=code_language),  # sandbox_code
        gr.update(value=dependencies)  # sandbox_dependency
    )

    yield from on_run_code(
        state,
        sandbox_state,
        sandbox_output,
        sandbox_ui,
        sandbox_code,
        sandbox_dependency,
    )


def on_run_code(
    state,
    sandbox_state: ChatbotSandboxState,
    sandbox_output: gr.Markdown,
    sandbox_ui: SandboxComponent,
    sandbox_code: str,
    sandbox_dependency: gr.Dataframe,
) -> Generator[tuple[Any, Any, Any, Any], None, None]:
    '''
    gradio fn when run code button is clicked. Update Sandbox components.
    '''
    if sandbox_state['enable_sandbox'] is False:
        yield None, None, None, None
        return

    # validate e2b api key
    if not E2B_API_KEY:
        raise ValueError("E2B_API_KEY is not set in env vars.")

    code, code_language = sandbox_state['code_to_execute'], sandbox_state['code_language']
    if code is None or code_language is None:
        yield None, None, None, None
        return

    if code_language == 'tsx':
        code_language = 'typescript'
    code_language = code_language.lower() if code_language and code_language.lower(
        # ensure gradio supports the code language
    ) in VALID_GRADIO_CODE_LANGUAGES else None

    # Use dependencies from sandbox_state instead of re-extracting
    code_dependencies = sandbox_state['code_dependencies']
    python_deps, npm_deps = code_dependencies

    # Helper function to extract package name without version
    def get_base_package_name(pkg: str) -> str:
        # For Python packages
        if any(op in pkg for op in ['==', '>=', '<=', '~=', '>', '<']):
            return pkg.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('>')[0].split('<')[0]
        # For NPM packages
        if '@' in pkg and not pkg.startswith('@'):
            return pkg.split('@')[0]
        elif '@' in pkg[1:]:  # Handle scoped packages
            return pkg.rsplit('@', 1)[0]
        return pkg

    # Helper function to extract version from package string
    def get_package_version(pkg: str) -> str:
        # For Python packages
        if any(op in pkg for op in ['==', '>=', '<=', '~=', '>', '<']):
            base_name = get_base_package_name(pkg)
            return pkg[len(base_name):]
        # For NPM packages
        if '@' in pkg and not pkg.startswith('@'):
            return '@' + pkg.split('@', 1)[1]
        elif '@' in pkg[1:]:  # Handle scoped packages
            _, version = pkg.rsplit('@', 1)
            return '@' + version
        return "latest"

    # Create unified dependency dictionaries to avoid duplicates
    python_deps_dict = {}  # pkg_name -> version
    npm_deps_dict = {}     # pkg_name -> version

    # Process Python dependencies
    for dep in python_deps:
        base_name = get_base_package_name(dep)
        version = get_package_version(dep)
        # Only update if we don't have a version yet or if we're replacing 'latest'
        if base_name not in python_deps_dict or python_deps_dict[base_name] == "latest":
            python_deps_dict[base_name] = version

    # Process NPM dependencies
    for dep in npm_deps:
        base_name = get_base_package_name(dep)
        version = get_package_version(dep)
        # Only update if we don't have a version yet or if we're replacing 'latest'
        if base_name not in npm_deps_dict or npm_deps_dict[base_name] == "latest":
            npm_deps_dict[base_name] = version

    # Convert unified dictionaries to dataframe format
    dependencies = []
    for pkg_name, version in python_deps_dict.items():
        dependencies.append(["python", pkg_name, version])
    for pkg_name, version in npm_deps_dict.items():
        dependencies.append(["npm", pkg_name, version])

    # If no dependencies found, provide default empty rows
    if not dependencies:
        dependencies = [["python", "", ""], ["npm", "", ""]]

    # Initialize output with loading message
    output_text = "### Sandbox Execution Log\n\n"
    yield (
        gr.Markdown(
            value=output_text + "🔄 Initializing sandbox environment...", visible=True
        ),
        SandboxComponent(visible=False),
        gr.Code(value=code, language=code_language, visible=True),
        gr.update(value=dependencies, visible=True),  # Update with unified dependencies
    )

    # Use auto_selected_sandbox_environment only when in AUTO mode, otherwise use sandbox_environment
    sandbox_env = (
        sandbox_state['auto_selected_sandbox_environment'] 
        if sandbox_state['sandbox_environment'] == SandboxEnvironment.AUTO
        else sandbox_state['sandbox_environment']
    )

    def update_output(message: str, clear_output: bool = False):
        nonlocal output_text
        if clear_output:
            output_text = ""
        output_text += f"\n{message}"
        return (
            gr.Markdown(value=output_text, visible=True, sanitize_html=False),
            gr.skip(),
            gr.skip(),
            gr.skip()  # Always include dependencies update
        )

    sandbox_id = None
    print(f"sandbox_env: {sandbox_env}")
    match sandbox_env:
        case SandboxEnvironment.HTML:
            yield update_output("🔄 Setting up HTML sandbox...")
            sandbox_url, sandbox_id, stderr = run_html_sandbox(
                code=code, code_dependencies=code_dependencies
            )
            if stderr:
                yield update_output("❌ HTML sandbox failed to run!", clear_output=True)
                yield update_output(f"### Stderr:\n```\n{stderr}\n```\n\n")
            else:
                yield update_output("✅ HTML sandbox ready!", clear_output=True)
                yield (
                    gr.Markdown(value=output_text, visible=True),
                    SandboxComponent(
                        value=(sandbox_url, True, []),
                        label="Example",
                        visible=True,
                        key="newsandbox",
                    ),
                    gr.skip(),
                    gr.skip(),
                )
        case SandboxEnvironment.REACT:
            yield update_output("🔄 Setting up React sandbox...")
            sandbox_url, sandbox_id, stderr = run_react_sandbox(code=code, code_dependencies=code_dependencies)
            if stderr:
                yield update_output("❌ React sandbox failed to run!", clear_output=True)
                yield update_output(f"### Stderr:\n```\n{stderr}\n```\n\n")
            else:
                yield update_output("✅ React sandbox ready!", clear_output=True)
                yield (
                    gr.Markdown(value=output_text, visible=True),
                    SandboxComponent(
                        value=(sandbox_url, True, []),
                        label="Example",
                        visible=True,
                        key="newsandbox",
                    ),
                    gr.skip(),
                    gr.skip(),
                )
        case SandboxEnvironment.VUE:
            yield update_output("🔄 Setting up Vue sandbox...")
            sandbox_url, sandbox_id, stderr = run_vue_sandbox(code=code, code_dependencies=code_dependencies)
            if stderr:
                yield update_output("❌ Vue sandbox failed to run!", clear_output=True)
                yield update_output(f"### Stderr:\n```\n{stderr}\n```\n\n")
            else:
                yield update_output("✅ Vue sandbox ready!", clear_output=True)
                yield (
                    gr.Markdown(value=output_text, visible=True),
                    SandboxComponent(
                        value=(sandbox_url, True, []),
                        label="Example",
                        visible=True,
                        key="newsandbox",
                    ),
                    gr.skip(),
                    gr.skip(),
                )
        case SandboxEnvironment.PYGAME:
            yield update_output("🔄 Setting up PyGame sandbox...")
            sandbox_url, sandbox_id, stderr = run_pygame_sandbox(code=code, code_dependencies=code_dependencies)
            if stderr:
                yield update_output("❌ PyGame sandbox failed to run!", clear_output=True)
                yield update_output(f"### Stderr:\n```\n{stderr}\n```\n\n")
            else:
                yield update_output("✅ PyGame sandbox ready!", clear_output=True)
                yield (
                    gr.Markdown(value=output_text, visible=True),
                    SandboxComponent(
                        value=(sandbox_url, True, []),
                        label="Example",
                        visible=True,
                        key="newsandbox",
                    ),
                    gr.skip(),
                    gr.skip(),
                )
        case SandboxEnvironment.GRADIO:
            yield update_output("🔄 Setting up Gradio sandbox...")
            sandbox_url, sandbox_id, stderr = run_gradio_sandbox(code=code, code_dependencies=code_dependencies)
            if stderr:
                yield update_output("❌ Gradio sandbox failed to run!", clear_output=True)
                yield update_output(f"### Stderr:\n```\n{stderr}\n```\n\n")
            else:
                yield update_output("✅ Gradio sandbox ready!", clear_output=True)
                yield (
                    gr.Markdown(value=output_text, visible=True),
                    SandboxComponent(
                        value=(sandbox_url, True, []),
                        label="Example",
                        visible=True,
                        key="newsandbox",
                    ),
                    gr.skip(),
                    gr.skip(),
                )
        case SandboxEnvironment.STREAMLIT:
            yield update_output("🔄 Setting up Streamlit sandbox...")
            sandbox_url, sandbox_id, stderr = run_streamlit_sandbox(code=code, code_dependencies=code_dependencies)
            if stderr:
                yield update_output("❌ Streamlit sandbox failed to run!", clear_output=True)
                yield update_output(f"### Stderr:\n```\n{stderr}\n```\n\n")
            else:
                yield update_output("✅ Streamlit sandbox ready!", clear_output=True)
                yield (
                    gr.Markdown(value=output_text, visible=True),
                    SandboxComponent(
                        value=(sandbox_url, True, []),
                        label="Example",
                        visible=True,
                        key="newsandbox",
                    ),
                    gr.skip(),
                    gr.skip(),
                )
        case SandboxEnvironment.MERMAID:
            yield update_output("🔄 Setting up Mermaid visualization...")
            # Convert Mermaid to HTML at execution time
            html_code = mermaid_to_html(code, theme='light')
            sandbox_url, sandbox_id, stderr = run_html_sandbox(code=html_code, code_dependencies=code_dependencies)
            if stderr:
                yield update_output("❌ Mermaid visualization failed to render!", clear_output=True)
                yield update_output(f"### Stderr:\n```\n{stderr}\n```\n\n")
            else:
                yield update_output("✅ Mermaid visualization ready!", clear_output=True)
                yield (
                    gr.Markdown(value=output_text, visible=True),
                    SandboxComponent(
                        value=(sandbox_url, True, []),
                        label="Mermaid Diagram",
                        visible=True,
                        key="newsandbox",
                    ),
                    gr.skip(),
                    gr.skip(),
                )
        case SandboxEnvironment.PYTHON_CODE_INTERPRETER:
            yield update_output("🔄 Running Python Code Interpreter...", clear_output=True)
            output, stderr = run_code_interpreter(
                code=code, code_language='python', code_dependencies=code_dependencies
            )
            if stderr:
                yield update_output("❌ Python Code Interpreter failed to run!", clear_output=True)
                yield update_output(f"### Stderr:\n```\n{stderr}\n```\n\n")
            else:
                yield update_output("✅ Code execution complete!", clear_output=True)
                yield (
                    gr.Markdown(
                        value=output_text + "\n\n" + output,
                        sanitize_html=False,
                        visible=True,
                    ),
                    SandboxComponent(
                        value=("", False, []),
                        label="Example",
                        visible=False,
                        key="newsandbox",
                    ),
                    gr.skip(),
                    gr.skip(),
                )
        case SandboxEnvironment.JAVASCRIPT_CODE_INTERPRETER:
            yield update_output("🔄 Running JavaScript Code Interpreter...", clear_output=True)
            output, stderr = run_code_interpreter(
                code=code, code_language='javascript', code_dependencies=code_dependencies
            )
            if stderr:
                yield update_output("❌ JavaScript Code Interpreter failed to run!", clear_output=True)
                yield update_output(f"### Stderr:\n```\n{stderr}\n```\n\n")
            else:
                yield update_output("✅ Code execution complete!", clear_output=True)
                yield (
                    gr.Markdown(
                        value=output_text + "\n\n" + output,
                        sanitize_html=False,
                        visible=True,
                    ),
                    SandboxComponent(
                        value=("", False, []),
                        label="Example",
                        visible=False,
                        key="newsandbox",
                    ),
                    gr.skip(),
                    gr.skip(),
                )
        case _:
            yield (
                gr.Markdown(value=code, visible=True),
                SandboxComponent(
                    value=("", False, []),
                    label="Example",
                    visible=False,
                    key="newsandbox",
                ),
                gr.skip(),
                gr.skip(),
            )

    if sandbox_id:
        sandbox_state['sandbox_id'] = sandbox_id



