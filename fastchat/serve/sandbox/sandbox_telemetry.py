'''
Module for logging the sandbox interactions and state.

TODO: Support Cloud Storage.
'''
import json
import os
from typing import Any

from fastchat.constants import LOGDIR
from fastchat.serve.sandbox.code_runner import ChatbotSandboxState


def get_sandbox_log_filename(sandbox_state: ChatbotSandboxState) -> str:
    name = os.path.join(
        LOGDIR,
        '-'.join(
            [
                "sandbox-records",
                f"{sandbox_state['sandbox_id']}",
                f"{sandbox_state['enabled_round']}",
                f"{sandbox_state['sandbox_run_round']}",
            ]
        )
    )
    return name


def upsert_sandbox_log(filename: str, data: dict, remote: bool = False) -> None:
    with open(filename, "w") as fout:
        json.dump(
            data,
            fout,
            indent=2,
            default=str,
            ensure_ascii=False
        )


def create_sandbox_log(sandbox_state: ChatbotSandboxState, user_interaction_records: list[Any]) -> dict:
    return {
        "sandbox_state": sandbox_state,
        "user_interaction_records": user_interaction_records,
    }


def log_sandbox_telemetry_gradio_fn(
    sandbox_state: ChatbotSandboxState,
    sandbox_ui_value: tuple[str, bool, list[Any]]
) -> None:
    if sandbox_state is None or sandbox_ui_value is None:
        return
    sandbox_id = sandbox_state['sandbox_id']
    user_interaction_records = sandbox_ui_value[2]
    if sandbox_id and user_interaction_records and len(user_interaction_records) > 0:
        data = create_sandbox_log(sandbox_state, user_interaction_records)
        filename = get_sandbox_log_filename(sandbox_state)
        upsert_sandbox_log(filename=filename, data=data)
