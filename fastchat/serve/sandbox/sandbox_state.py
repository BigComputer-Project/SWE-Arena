'''
Chatbot state.
'''

from fastchat.serve.sandbox.code_analyzer import SandboxEnvironment
from typing import TypedDict

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
    sandbox_run_round: int
    '''
    How many rounds the sandbox has been run inside the session.
    '''
    edit_round: int
    '''
    How many rounds the code has been edited.
    Starting from 0, incrementing each time the code is edited.
    Refreshed when running a generated code.
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
    sandbox_error: str | None
    '''
    The sandbox error.
    '''

    sandbox_id: str | None
    '''
    The remote e2b sandbox id. None if not run yet.
    '''
    chat_session_id: str | None
    '''
    The chat session id, unique per chat.
    The two battle models share the same chat session id.
    '''
    conv_id: str | None
    '''
    The conv id, unique per chat per model.
    '''

    btn_list_length: int
    '''
    Count of Gradio user interface buttons.
    '''
