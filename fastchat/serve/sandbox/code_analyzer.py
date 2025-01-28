'''
Module for analyzing code snippets to determine the environments, dependencies, and other information needed to run the code.
'''


from enum import StrEnum
from typing import Any, Generator, TypeAlias, TypedDict, Set
import gradio as gr

import base64

import ast
from tree_sitter import Language, Node, Parser
import tree_sitter_javascript
import tree_sitter_typescript
import sys
import re


class SandboxEnvironment(StrEnum):
    AUTO = 'Auto'
    # Code Interpreter
    PYTHON_CODE_INTERPRETER = 'Python Code Interpreter'
    JAVASCRIPT_CODE_INTERPRETER = 'Javascript Code Interpreter'

    # Code Runner
    C_CODE_RUNNER = 'C Code Runner'
    CPP_CODE_RUNNER = 'C++ Code Runner'
    # CSHARP_CODE_RUNNER = 'C# Code Runner'
    JAVA_CODE_RUNNER = 'Java Code Runner'
    RUST_CODE_RUNNER = 'Rust Code Runner'
    GOLANG_CODE_RUNNER = 'Golang Code Runner'

    # Web UI Frameworks
    HTML = 'HTML'
    REACT = 'React'
    VUE = 'Vue'
    GRADIO = 'Gradio'
    STREAMLIT = 'Streamlit'
    PYGAME = 'PyGame'
    MERMAID = 'Mermaid'


def extract_python_imports(code: str) -> list[str]:
    '''
    Extract Python package imports using AST parsing.
    Returns a list of top-level package names.
    '''
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    packages: Set[str] = set()

    for node in ast.walk(tree):
        try:
            if isinstance(node, ast.Import):
                for name in node.names:
                    # Get the top-level package name from any dotted path
                    # e.g., 'foo.bar.baz' -> 'foo'
                    if name.name:  # Ensure there's a name
                        packages.add(name.name.split('.')[0])

            elif isinstance(node, ast.ImportFrom):
                # Skip relative imports (those starting with dots)
                if node.level == 0 and node.module:
                    # Get the top-level package name
                    # e.g., from foo.bar import baz -> 'foo'
                    packages.add(node.module.split('.')[0])

            # Also check for common dynamic import patterns
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'importlib':
                    # Handle importlib.import_module('package')
                    if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
                        packages.add(node.args[0].s.split('.')[0])
                elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    # Handle __import__('package') and importlib.import_module('package')
                    if node.func.value.id == 'importlib' and node.func.attr == 'import_module':
                        if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
                            packages.add(node.args[0].s.split('.')[0])
                    elif node.func.attr == '__import__':
                        if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
                            packages.add(node.args[0].s.split('.')[0])
        except Exception as e:
            print(f"Error processing node {type(node)}: {e}")
            continue

    # Filter out standard library modules using sys.stdlib_module_names
    std_libs = set(sys.stdlib_module_names)

    return list(packages - std_libs)


def extract_js_imports(code: str) -> list[str]:
    '''
    Extract npm package imports using Tree-sitter for robust parsing.
    Handles both JavaScript and TypeScript code, including Vue SFC.
    Returns a list of package names.
    '''
    try:
        # For Vue SFC, extract the script section first
        script_match = re.search(r'<script.*?>(.*?)</script>', code, re.DOTALL)
        if script_match:
            code = script_match.group(1).strip()

        # Initialize parsers with language modules
        ts_parser = Parser(Language(tree_sitter_typescript.language_tsx()))
        js_parser = Parser(Language(tree_sitter_javascript.language()))

        # Try parsing as TypeScript first, then JavaScript
        code_bytes = bytes(code, "utf8")
        try:
            tree = ts_parser.parse(code_bytes)
        except Exception as e:
            print(f"TypeScript parsing failed: {e}")
            try:
                tree = js_parser.parse(code_bytes)
            except Exception as e:
                print(f"JavaScript parsing failed: {e}")
                tree = None

        if tree is None:
            raise Exception("Both TypeScript and JavaScript parsing failed")

        packages: Set[str] = set()

        def extract_package_name(node: Node) -> str | None:
            '''Extract package name from string literal or template string'''
            if node.type in ['string', 'string_fragment']:
                # Handle regular string literals
                pkg_path = code[node.start_byte:node.end_byte].strip('"\'')
                if not pkg_path.startswith('.'):
                    # Handle scoped packages differently
                    if pkg_path.startswith('@'):
                        parts = pkg_path.split('/')
                        if len(parts) >= 2:
                            return '/'.join(parts[:2])  # Return @scope/package
                    # Return just the package name for non-scoped packages
                    return pkg_path.split('/')[0]
            elif node.type == 'template_string':
                # Handle template literals
                content = ''
                has_template_var = False
                for child in node.children:
                    if child.type == 'string_fragment':
                        content += code[child.start_byte:child.end_byte]
                    elif child.type == 'template_substitution':
                        has_template_var = True
                        continue

                if not content or content.startswith('.'):
                    return None

                if has_template_var:
                    if content.endswith('-literal'):
                        return 'package-template-literal'
                    return None

                if content.startswith('@'):
                    parts = content.split('/')
                    if len(parts) >= 2:
                        return '/'.join(parts[:2])
                return content.split('/')[0]
            return None

        def visit_node(node: Node) -> None:
            if node.type == 'import_statement':
                # Handle ES6 imports
                string_node = node.child_by_field_name('source')
                if string_node:
                    pkg_name = extract_package_name(string_node)
                    if pkg_name:
                        packages.add(pkg_name)

            elif node.type == 'export_statement':
                # Handle re-exports
                source = node.child_by_field_name('source')
                if source:
                    pkg_name = extract_package_name(source)
                    if pkg_name:
                        packages.add(pkg_name)

            elif node.type == 'call_expression':
                # Handle require calls and dynamic imports
                func_node = node.child_by_field_name('function')
                if func_node and func_node.text:
                    func_name = func_node.text.decode('utf8')
                    if func_name in ['require', 'import']:
                        args = node.child_by_field_name('arguments')
                        if args and args.named_children:
                            arg = args.named_children[0]
                            pkg_name = extract_package_name(arg)
                            if pkg_name:
                                packages.add(pkg_name)

            # Recursively visit children
            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return list(packages)

    except Exception as e:
        print(f"Tree-sitter parsing failed: {e}")
        # Fallback to basic regex parsing if tree-sitter fails
        packages: Set[str] = set()

        # First try to extract script section for Vue SFC
        script_match = re.search(r'<script.*?>(.*?)</script>', code, re.DOTALL)
        if script_match:
            code = script_match.group(1).strip()

        # Look for imports
        import_patterns = [
            # dynamic imports
            r'(?:import|require)\s*\(\s*[\'"](@?[\w-]+(?:/[\w-]+)*)[\'"]',
            # static imports
            r'(?:import|from)\s+[\'"](@?[\w-]+(?:/[\w-]+)*)[\'"]',
            # require statements
            r'require\s*\(\s*[\'"](@?[\w-]+(?:/[\w-]+)*)[\'"]',
        ]

        for pattern in import_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                pkg_name = match.group(1)
                if not pkg_name.startswith('.'):
                    if pkg_name.startswith('@'):
                        parts = pkg_name.split('/')
                        if len(parts) >= 2:
                            packages.add('/'.join(parts[:2]))
                    else:
                        packages.add(pkg_name.split('/')[0])

        return list(packages)


def determine_python_environment(code: str, imports: list[str]) -> SandboxEnvironment | None:
    '''
    Determine Python sandbox environment based on imports and AST analysis.
    '''
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # Check for specific framework usage patterns
            if isinstance(node, ast.Name) and node.id == 'gr':
                return SandboxEnvironment.GRADIO
            elif isinstance(node, ast.Name) and node.id == 'st':
                return SandboxEnvironment.STREAMLIT
    except SyntaxError:
        pass

    # Check imports for framework detection
    if 'pygame' in imports:
        return SandboxEnvironment.PYGAME
    elif 'gradio' in imports:
        return SandboxEnvironment.GRADIO
    elif 'streamlit' in imports:
        return SandboxEnvironment.STREAMLIT
    # elif 'nicegui' in imports:
    #     return SandboxEnvironment.NICEGUI

    return SandboxEnvironment.PYTHON_CODE_INTERPRETER


def determine_jsts_environment(code: str, imports: list[str]) -> SandboxEnvironment | None:
    '''
    Determine JavaScript/TypeScript sandbox environment based on imports and AST analysis.
    '''
    # First check for Vue SFC structure
    if '<template>' in code or '<script setup' in code:
        return SandboxEnvironment.VUE

    # Check imports for framework detection
    react_packages = {'react', '@react', 'next', '@next'}
    vue_packages = {'vue', '@vue', 'nuxt', '@nuxt'}

    if any(pkg in react_packages for pkg in imports):
        return SandboxEnvironment.REACT
    elif any(pkg in vue_packages for pkg in imports):
        return SandboxEnvironment.VUE

    try:
        # Initialize parser
        ts_parser = Parser(Language(tree_sitter_typescript.language_tsx()))

        # Parse the code
        tree = ts_parser.parse(bytes(code, "utf8"))

        def has_framework_patterns(node: Node) -> tuple[bool, str]:
            # Check for React patterns
            if node.type in ['jsx_element', 'jsx_self_closing_element']:
                return True, 'react'

            # Check for Vue template
            elif node.type == 'template_element':
                return True, 'vue'

            # Check for Vue template string
            elif node.type == 'template_string':
                content = code[node.start_byte:node.end_byte]
                # Look for Vue directives in template strings
                vue_patterns = [
                    'v-if=', 'v-else', 'v-for=', 'v-bind:', 'v-on:', 'v-model=',
                    'v-show=', 'v-html=', 'v-text=', '@', ':',
                    'components:', 'props:', 'emits:', 'data:',
                    'methods:', 'computed:', 'watch:',
                    'setup(', 'ref(', 'reactive(', 'computed(', 'watch(',
                    'onMounted(', 'onUnmounted(', 'provide(', 'inject(',
                    'defineComponent(', 'defineProps(', 'defineEmits(',
                    'createApp(', 'nextTick('
                ]
                if any(pattern in content for pattern in vue_patterns):
                    return True, 'vue'
            return False, ''

        # Check for framework-specific patterns in the AST
        cursor = tree.walk()

        def visit_node() -> SandboxEnvironment | None:
            is_framework, framework = has_framework_patterns(cursor.node)
            if is_framework:
                return SandboxEnvironment.REACT if framework == 'react' else SandboxEnvironment.VUE

            # Check children
            if cursor.goto_first_child():
                while True:
                    result = visit_node()
                    if result:
                        return result
                    if not cursor.goto_next_sibling():
                        break
                cursor.goto_parent()

            return None

        result = visit_node()
        if result:
            return result

        # Additional Vue pattern detection for script content
        vue_patterns = [
            r'export\s+default\s+{',
            r'defineComponent\s*\(',
            r'Vue\.extend\s*\(',
            r'createApp\s*\(',
            r'(?:ref|reactive|computed|watch|onMounted|onUnmounted|provide|inject)\s*\(',
            r'(?:components|props|emits|data|methods|computed|watch)\s*:',
            r'defineProps\s*\(',
            r'defineEmits\s*\(',
            r'v-(?:if|else|for|bind|on|model|show|html|text)=',
            r'@(?:click|change|input|submit|keyup|keydown)',
            r':(?:class|style|src|href|value|disabled|checked)'
        ]

        for pattern in vue_patterns:
            if re.search(pattern, code, re.MULTILINE):
                return SandboxEnvironment.VUE

    except Exception as e:
        print(f"Tree-sitter parsing error: {e}")
        pass

    return SandboxEnvironment.JAVASCRIPT_CODE_INTERPRETER


def detect_js_ts_code_lang(code: str) -> str:
    '''
    Detect whether code is JavaScript or TypeScript using Tree-sitter AST parsing.
    Handles Vue SFC, React, and regular JS/TS files.

    Args:
        code (str): The code to analyze

    Returns:
        str: 'typescript' if TypeScript patterns are found, 'javascript' otherwise
    '''
    # Quick check for explicit TypeScript in Vue SFC
    if '<script lang="ts">' in code or '<script lang="typescript">' in code:
        return 'typescript'

    try:
        # Initialize TypeScript parser
        ts_parser = Parser(Language(tree_sitter_typescript.language_tsx()))

        # Parse the code
        tree = ts_parser.parse(bytes(code, "utf8"))

        def has_typescript_patterns(node: Node) -> bool:
            # Check for TypeScript-specific syntax
            if node.type in {
                'type_annotation',           # Type annotations
                'type_alias_declaration',    # type Foo = ...
                'interface_declaration',     # interface Foo
                'enum_declaration',          # enum Foo
                'implements_clause',         # implements Interface
                'type_parameter',            # Generic type parameters
                'type_assertion',            # Type assertions
                'type_predicate',           # Type predicates in functions
                'type_arguments',           # Generic type arguments
                'readonly_type',            # readonly keyword
                'mapped_type',              # Mapped types
                'conditional_type',         # Conditional types
                'union_type',               # Union types
                'intersection_type',        # Intersection types
                'tuple_type',              # Tuple types
                'optional_parameter',       # Optional parameters
                'decorator',                # Decorators
                'ambient_declaration',      # Ambient declarations
                'declare_statement',        # declare keyword
                'accessibility_modifier',   # private/protected/public
            }:
                return True

            # Check for type annotations in variable declarations
            if node.type == 'variable_declarator':
                for child in node.children:
                    if child.type == 'type_annotation':
                        return True

            # Check for return type annotations in functions
            if node.type in {'function_declaration', 'method_definition', 'arrow_function'}:
                for child in node.children:
                    if child.type == 'type_annotation':
                        return True

            return False

        # Walk the AST to find TypeScript patterns
        cursor = tree.walk()

        def visit_node() -> bool:
            if has_typescript_patterns(cursor.node):
                return True

            # Check children
            if cursor.goto_first_child():
                while True:
                    if visit_node():
                        return True
                    if not cursor.goto_next_sibling():
                        break
                cursor.goto_parent()

            return False

        if visit_node():
            return 'typescript'

    except Exception as e:
        print(f"Tree-sitter parsing error: {e}")
        # Fallback to basic checks if parsing fails
        pass

    return 'javascript'


def extract_inline_pip_install_commands(code: str) -> tuple[list[str], str]:
    '''
    Extracts pip install commands from inline code comments and returns both the packages and cleaned code.
    This is useful for cases where pip install commands are written as comments in the code or
    Jupyter notebook-style !pip install commands.

    Args:
        code (str): The code to analyze.

    Returns:
        tuple[list[str], str]: A tuple containing:
            1. List of Python packages extracted from pip install commands in comments
            2. Code with the pip install comments removed
    '''
    python_packages = []
    cleaned_lines = []

    # Regex patterns to match pip install commands in comments and Jupyter-style commands
    pip_patterns = [
        # Comments with pip install
        r'#\s*(?:pip|pip3|python -m pip)\s+install\s+(?:(?:--upgrade|--user|--no-cache-dir|-U)\s+)*([^-\s][\w\-\[\]<>=~\.]+(?:\s+[^-\s][\w\-\[\]<>=~\.]+)*)',
        # Jupyter-style !pip install
        r'!\s*(?:pip|pip3|python -m pip)\s+install\s+(?:(?:--upgrade|--user|--no-cache-dir|-U)\s+)*([^-\s][\w\-\[\]<>=~\.]+(?:\s+[^-\s][\w\-\[\]<>=~\.]+)*)',
        # Requirements file style pip install
        r'(?:#|!)\s*(?:pip|pip3|python -m pip)\s+install\s+(?:-r\s+[\w\-\.\/]+\s+)*([^-\s][\w\-\[\]<>=~\.]+(?:\s+[^-\s][\w\-\[\]<>=~\.]+)*)'
    ]

    # Process each line
    for line in code.splitlines():
        matched = False
        for pattern in pip_patterns:
            match = re.search(pattern, line)
            if match:
                matched = True
                # Extract packages from the command
                pkgs = match.group(1).strip().split()
                # Clean package names (remove version specifiers)
                cleaned_pkgs = [pkg.split('==')[0].split('>=')[0].split('<=')[
                    0].split('~=')[0] for pkg in pkgs]
                python_packages.extend(cleaned_pkgs)

                # Remove the pip install command from the line
                cleaned_line = line[:match.start()].rstrip()
                if cleaned_line:  # Only add non-empty lines
                    cleaned_lines.append(cleaned_line)
                break

        if not matched:
            cleaned_lines.append(line)

    # Remove duplicates while preserving order
    python_packages = list(dict.fromkeys(python_packages))

    return python_packages, '\n'.join(cleaned_lines)


def extract_js_from_html_script_tags(code: str) -> list[str]:
    '''
    Extract JavaScript package names from HTML script tags.
    Handles both CDN script tags and inline scripts.

    Args:
        code: HTML code containing script tags

    Returns:
        list[str]: List of package names
    '''
    packages: Set[str] = set()

    # Extract packages from CDN script tags
    script_patterns = [
        # unpkg.com pattern
        r'<script[^>]*src="https?://unpkg\.com/(@?[^@/"]+(?:/[^@/"]+)?(?:@[^/"]+)?)[^"]*"[^>]*>',
        # cdn.jsdelivr.net pattern - explicitly handle /npm/ in the path
        r'<script[^>]*src="https?://cdn\.jsdelivr\.net/npm/(@?[^@/"]+(?:/[^@/"]+)?(?:@[^/"]+)?)[^"]*"[^>]*>',
        # Generic CDN pattern for any domain - exclude common path components
        r'<script[^>]*src="https?://(?!(?:[^"]+/)?(?:npm|dist|lib|build|umd|esm|cjs|min)/)[^"]+?/(@?[\w-]+)(?:/[^"]*)?[^"]*"[^>]*>',
    ]

    seen_packages = set()  # Track packages we've already added to avoid duplicates
    for pattern in script_patterns:
        matches = re.finditer(pattern, code, re.IGNORECASE)
        for match in matches:
            pkg_name = match.group(1)
            if pkg_name.startswith('@'):
                # Handle scoped packages
                parts = pkg_name.split('/')
                if len(parts) >= 2:
                    pkg_name = '/'.join(parts[:2])
            else:
                # Remove version and path components from package name
                pkg_name = pkg_name.split('/')[0].split('@')[0]

            # Skip common path components and duplicates
            if pkg_name and pkg_name not in seen_packages and not pkg_name.lower() in {'npm', 'dist', 'lib', 'build', 'umd', 'esm', 'cjs', 'min'}:
                seen_packages.add(pkg_name)
                packages.add(pkg_name)

    # Extract packages from inline scripts
    script_tags = re.finditer(
        r'<script[^>]*>(.*?)</script>', code, re.DOTALL | re.IGNORECASE)
    for script in script_tags:
        script_content = script.group(1)
        # Check for ES module imports with full URLs
        es_module_patterns = [
            # Match imports from CDN URLs, being careful to extract only the package name
            r'import\s+[\w\s{},*]+\s+from\s+[\'"]https?://[^/]+/npm/([^/@"\s]+)[@/][^"]*[\'"]',
        ]
        found_cdn_import = False
        for pattern in es_module_patterns:
            matches = re.finditer(pattern, script_content)
            for match in matches:
                pkg_name = match.group(1)
                if pkg_name and pkg_name not in seen_packages and not pkg_name.lower() in {'npm', 'dist', 'lib', 'build', 'umd', 'esm', 'cjs', 'min', 'https', 'http'}:
                    seen_packages.add(pkg_name)
                    packages.add(pkg_name)
                    found_cdn_import = True

        # Only check for regular imports if we didn't find a CDN import
        if not found_cdn_import:
            # Remove any URL imports before passing to extract_js_imports
            cleaned_content = re.sub(
                r'import\s+[\w\s{},*]+\s+from\s+[\'"]https?://[^"]+[\'"]', '', script_content)
            packages.update(extract_js_imports(cleaned_content))

    return list(packages)


def extract_code_from_markdown(message: str, enable_auto_env: bool = False) -> tuple[str, str, tuple[list[str], list[str]], SandboxEnvironment | None] | None:
    '''
    Extracts code from a markdown message by parsing code blocks directly.
    Determines sandbox environment based on code content and frameworks used.

    Returns:
        tuple[str, str, tuple[list[str], list[str]], SandboxEnvironment | None]: A tuple:
            1. code - the longest code block found
            2. code language
            3. sandbox python and npm dependencies (extracted using static analysis)
            4. sandbox environment determined from code content
    '''
    code_block_regex = r'```(?P<code_lang>[\w\+\#\-\.]*)?[ \t]*\r?\n?(?P<code>.*?)```'
    matches = list(re.finditer(code_block_regex, message, re.DOTALL))

    if not matches:
        return None

    # Define a low-priority list for certain languages
    low_priority_languages = ['bash', 'shell',
                              'sh', 'zsh', 'powershell', 'pwsh', '']

    # Find the main code block by avoiding low-priority languages
    main_code = None
    main_code_lang = None
    max_length = 0

    for match in matches:
        code = match.group('code').strip()
        code_lang = (match.group('code_lang') or '').lower()
        if code_lang not in low_priority_languages and len(code) > max_length:
            main_code = code
            main_code_lang = code_lang
            max_length = len(code)

    # Fallback to the longest code block if no main code was found
    if not main_code:
        longest_match = max(matches, key=lambda m: len(m.group('code')))
        main_code = longest_match.group('code').strip()
        main_code_lang = (longest_match.group('code_lang') or '').lower()

    # Define language prefixes for each environment
    python_prefixes = ['py', 'ipython', 'pygame', 'gradio', 'streamlit']
    vue_prefixes = ['vue']
    react_prefixes = ['react', 'next']
    js_prefixes = ['js', 'javascript', 'jsx', 'coffee', 'ecma', 'node', 'es']
    html_prefixes = ['html', 'xhtml', 'htm']
    ts_prefixes = ['ts', 'typescript', 'tsx']
    mermaid_prefixes = ['mermaid', 'mmd']
    c_prefixes = ['c']
    cpp_prefixes = ['cpp', 'c++']
    go_prefixes = ['go', 'golang']
    java_prefixes = ['java']
    rust_prefixes = ['rust']
    csharp_prefixes = ['cs', 'csharp', 'dotnet']

    # Extract package dependencies from the main program
    python_packages: list[str] = []
    npm_packages: list[str] = []

    # Helper function to check if any prefix matches
    def matches_prefix(lang: str, prefixes: list[str]) -> bool:
        return any(lang.lower().startswith(prefix) for prefix in prefixes)

    if matches_prefix(main_code_lang, python_prefixes):
        python_packages = extract_python_imports(main_code)
        extra_python_packages, main_code = extract_inline_pip_install_commands(
            main_code)
        python_packages.extend(extra_python_packages)
        sandbox_env_name = determine_python_environment(
            main_code, python_packages)
    elif matches_prefix(main_code_lang, vue_prefixes):
        npm_packages = extract_js_imports(main_code)
        sandbox_env_name = SandboxEnvironment.VUE
        main_code_lang = detect_js_ts_code_lang(main_code)
    elif matches_prefix(main_code_lang, react_prefixes):
        npm_packages = extract_js_imports(main_code)
        sandbox_env_name = SandboxEnvironment.REACT
        main_code_lang = detect_js_ts_code_lang(main_code)
    elif '<!DOCTYPE html>' in main_code or ('<head' in main_code and '<body' in main_code):
        # For HTML files, extract both inline script dependencies and script tag dependencies
        npm_packages = extract_js_from_html_script_tags(main_code)
        sandbox_env_name = SandboxEnvironment.HTML
        main_code_lang = 'html'
    elif matches_prefix(main_code_lang, js_prefixes):
        main_code_lang = 'javascript'
        npm_packages = extract_js_imports(main_code)
        sandbox_env_name = determine_jsts_environment(main_code, npm_packages)
    elif matches_prefix(main_code_lang, ts_prefixes):
        main_code_lang = 'typescript'
        npm_packages = extract_js_imports(main_code)
        sandbox_env_name = determine_jsts_environment(main_code, npm_packages)
    elif matches_prefix(main_code_lang, html_prefixes):
        main_code_lang = detect_js_ts_code_lang(main_code)
        npm_packages = extract_js_imports(main_code)
        sandbox_env_name = determine_jsts_environment(main_code, npm_packages)
    elif matches_prefix(main_code_lang, mermaid_prefixes):
        main_code_lang = 'markdown'
        sandbox_env_name = SandboxEnvironment.MERMAID
    elif matches_prefix(main_code_lang, cpp_prefixes):
        main_code_lang = 'cpp'
        sandbox_env_name = SandboxEnvironment.CPP_CODE_RUNNER
    elif matches_prefix(main_code_lang, go_prefixes):
        main_code_lang = 'go'
        sandbox_env_name = SandboxEnvironment.GOLANG_CODE_RUNNER
    elif matches_prefix(main_code_lang, java_prefixes):
        main_code_lang = 'java'
        sandbox_env_name = SandboxEnvironment.JAVA_CODE_RUNNER
    elif matches_prefix(main_code_lang, rust_prefixes):
        main_code_lang = 'rust'
        sandbox_env_name = SandboxEnvironment.RUST_CODE_RUNNER
    elif main_code_lang == 'c':
        main_code_lang = 'c'
        sandbox_env_name = sandbox_env_name = SandboxEnvironment.C_CODE_RUNNER
    else:
        sandbox_env_name = None

    all_python_packages: Set[str] = set(python_packages)
    all_npm_packages: Set[str] = set(npm_packages)

    for match in matches:
        code = match.group('code').strip()
        if code != main_code:
            install_python_packages, install_npm_packages = extract_installation_commands(
                code)
            all_python_packages.update(install_python_packages)
            all_npm_packages.update(install_npm_packages)

    if not main_code_lang:
        main_code_lang = 'markdown'
    
    return main_code, main_code_lang, (list(all_python_packages), list(all_npm_packages)), sandbox_env_name


def create_placeholder_svg_data_url(width: int, height: int) -> str:
    '''
    Create a data URL for a placeholder image with given dimensions.
    Uses SVG to create an elegant placeholder.

    Args:
        width: Width of the placeholder image
        height: Height of the placeholder image

    Returns:
        str: Data URL containing the SVG image
    '''
    # Create SVG with gradient background and text
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#F3F4F6"/>
                <stop offset="100%" style="stop-color:#E5E7EB"/>
            </linearGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#bg)"/>
        <text
            x="50%"
            y="50%"
            font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
            font-size="{min(width, height) // 14}"
            fill="#94A3B8"
            font-weight="300"
            letter-spacing="0.05em"
            text-anchor="middle"
            dominant-baseline="middle">
            <tspan x="50%" dy="-1em">{width}</tspan>
            <tspan x="50%" dy="1.4em" font-size="{min(width, height) // 16}">×</tspan>
            <tspan x="50%" dy="1.4em">{height}</tspan>
        </text>
    </svg>'''

    # Convert to base64 data URL
    encoded_svg = base64.b64encode(svg.encode()).decode()
    return f'data:image/svg+xml;base64,{encoded_svg}'


def replace_placeholder_urls(code: str) -> str:
    '''
    Replace placeholder image URLs with SVG data URLs.
    Only replaces exact matches of "/api/placeholder/{width}/{height}".

    Args:
        code: The source code containing placeholder URLs

    Returns:
        str: Code with placeholder URLs replaced with data URLs
    '''

    def replacer(match: re.Match) -> str:
        # Extract width and height from the URL using capturing groups
        width = int(match.group(1))
        height = int(match.group(2))
        print(f'Replacing placeholder URL with SVG: {width}x{height}')
        data_url = create_placeholder_svg_data_url(width, height)
        return data_url

    # Regular expression pattern to match placeholder URLs
    pattern = r'/api/placeholder/(\d+)/(\d+)'

    # Replace all occurrences
    return re.sub(pattern, replacer, code)


def extract_installation_commands(code: str) -> tuple[list[str], list[str]]:
    '''
    Extracts package installation commands from the code block, preserving version information.

    Args:
        code (str): The code block to analyze.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
            1. Python packages from pip install commands (with versions if specified).
            2. npm packages from npm install commands (with versions if specified).
    '''
    python_packages = []
    npm_packages = []

    # Process the code line by line to handle both pip and npm commands
    lines = code.split('\n')
    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue

        # Handle pip install commands
        if any(x in line for x in ['pip install', 'pip3 install', 'python -m pip install']):
            # Remove the command part and any flags
            parts = line.split('install', 1)[1].strip()
            # Handle flags at the start
            while parts.startswith(('-', '--')):
                parts = parts.split(None, 1)[1]

            # Split by whitespace, respecting quotes
            current = ''
            in_quotes = False
            quote_char = None
            packages = []

            for char in parts:
                if char in '"\'':
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                elif char.isspace() and not in_quotes:
                    if current:
                        packages.append(current)
                        current = ''
                else:
                    current += char
            if current:
                packages.append(current)

            # Add packages, stripping quotes and ignoring flags
            for pkg in packages:
                pkg = pkg.strip('"\'')
                if pkg and not pkg.startswith(('-', '--')) and not pkg == '-r':
                    python_packages.append(pkg)

        # Handle npm/yarn install commands
        elif any(x in line for x in ['npm install', 'npm i', 'yarn add']):
            # Remove the command part and any flags
            if 'yarn add' in line:
                parts = line.split('add', 1)[1]
            else:
                parts = line.split('install', 1)[
                    1] if 'install' in line else line.split('i', 1)[1]
            parts = parts.strip()

            # Handle flags at the start
            while parts.startswith(('-', '--')):
                parts = parts.split(None, 1)[1] if ' ' in parts else ''

            # Process each package
            for pkg in parts.split():
                if pkg.startswith(('-', '--')) or pkg in ('install', 'i', 'add'):
                    continue

                if pkg.startswith('@'):
                    # Handle scoped packages (e.g., @types/node@16.0.0)
                    if '@' in pkg[1:]:  # Has version
                        pkg_parts = pkg.rsplit('@', 1)
                        base_pkg = pkg_parts[0]  # @scope/name
                        version = pkg_parts[1]  # version
                        npm_packages.append(f"{base_pkg}@{version}")
                    else:
                        npm_packages.append(pkg)
                else:
                    npm_packages.append(pkg)

    # Remove duplicates while preserving order
    python_packages = list(dict.fromkeys(python_packages))
    npm_packages = list(dict.fromkeys(npm_packages))

    # Filter out npm command words
    npm_packages = [p for p in npm_packages if p not in (
        'npm', 'install', 'i', 'add')]

    return python_packages, npm_packages


def validate_dependencies(dependencies: list) -> tuple[bool, str]:
    """
    Validate dependency list format and values.
    Allows empty rows but validates format when package name is specified.
    """
    if not dependencies:
        return True, ""

    valid_types = ["python", "npm"]
    for dep in dependencies:
        # Skip validation for empty rows
        if len(dep) != 3:
            return False, "Each dependency must have type, package and version fields"

        dep_type, pkg_name, version = dep

        # Skip empty rows
        if not pkg_name.strip():
            continue

        if dep_type.lower() not in valid_types:
            return False, f"Invalid dependency type: {dep_type}"

        # Validate version format if specified
        if version.strip():
            if dep_type.lower() == "python":
                # Check for valid pip version specifiers
                if not any(op in version for op in ['==', '>=', '<=', '~=', '>', '<']) and version.lower() != "latest":
                    return False, f"Invalid Python version format for {pkg_name}: {version}"
            elif dep_type.lower() == "npm":
                # Check for valid npm version format (starts with @ or valid semver-like)
                if not (version.startswith('@') or version.lower() == "latest"):
                    return False, f"Invalid NPM version format for {pkg_name}: {version}"

    return True, ""


def extract_java_class_name(java_code: str) -> str:
    '''
    Extract the class name from Java code.
    '''
    match = re.search(r'public\s+class\s+(\w+)', java_code)
    return match.group(1) if match else "Main"