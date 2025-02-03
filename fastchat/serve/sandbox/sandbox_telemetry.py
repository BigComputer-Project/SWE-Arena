'''
Module for logging the sandbox interactions and state.
'''
from concurrent.futures import ThreadPoolExecutor
import json
import os
from typing import Any, Literal
import datetime

from fastchat.serve.chat_state import LOG_DIR
from fastchat.serve.sandbox.sandbox_state import ChatbotSandboxState

from azure.storage.blob import BlobServiceClient

from fastchat.serve.sandbox.constants import AZURE_BLOB_STORAGE_CONNECTION_STRING, AZURE_BLOB_STORAGE_CONTAINER_NAME


def upload_data_to_azure_storage(
        data: bytes,
        blob_name: str,
        write_mode: Literal['overwrite', 'append'],
        connection_string: str | None = AZURE_BLOB_STORAGE_CONNECTION_STRING,
        container_name: str = AZURE_BLOB_STORAGE_CONTAINER_NAME,
    ) -> None:
    '''
    Upload data to Azure Blob Storage.
    '''
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set")

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    if write_mode == "overwrite":
        container_client.upload_blob(
            name=blob_name,
            data=data,
            overwrite=True
        )
    elif write_mode == "append":
        blob_client = container_client.get_blob_client(blob=blob_name)
        if not blob_client.exists():
            blob_client.upload_blob(data, blob_type="AppendBlob")
        else:
            blob_client.append_block(data)
    else:
        raise ValueError("Unsupported write_mode. Use 'w' for overwrite or 'a' for append.")


def get_sandbox_log_blob_name(filename: str) -> str:
    date_str = datetime.datetime.now().strftime('%Y_%m_%d')
    blob_name = f"{date_str}/sandbox_logs/{filename}"
    return blob_name

def get_conv_log_filepath(
    date: datetime.date,
    chat_mode: Literal['battle_anony', 'battle_named', 'direct'],
    chat_session_id: str,
) -> str:
    '''
    Get the filepath for the conversation log.

    Expected directory structure:
        softwarearenlog/
        └── YEAR_MONTH_DAY/
            ├── conv_logs/
            │   ├── battle_anony/
            │   │   └── CHATSESSIONID.json
            │   ├── battle_named/
            │   │   └── CHATSESSIONID.json
            │   └── direct/
            │       └── CHATSESSIONID.json
    '''
    date_str = date.strftime('%Y_%m_%d')
    filepath = os.path.join(
        date_str,
        'conv_logs',
        chat_mode,
        f"{chat_session_id}.json"
    )
    return filepath


def get_sandbox_log_filepath(
    date: datetime.date,
    chat_mode: Literal['battle_anony', 'battle_named', 'direct'],
    chat_session_id: str,
) -> str:
    '''
    Get the filepath for the conversation log.

    Expected directory structure:
        softwarearenlog/
        └── YEAR_MONTH_DAY/
            ├── conv_logs/
            └── sandbox_logs/
                ├── battle/
                │   └── sandbox-records-SESSIONID-A-B-EDITID.json
                ├── side-by-side/
                │   └── sandbox-records-SESSIONID-A-B-EDITID.json
                └── direct/
                    └── sandbox-records-SESSIONID-A-B-EDITID.json
    '''
    date_str = date.strftime('%Y_%m_%d')
    filepath = os.path.join(
        date_str,
        'sandbox_logs',
        chat_mode,
        f"{chat_session_id}.json"
    )
    return filepath


def get_conv_log_blob_name(filename: str) -> str:
    date_str = datetime.datetime.now().strftime('%Y_%m_%d')
    blob_name = f"{date_str}/conv_logs/{filename}"
    return blob_name

_executor = ThreadPoolExecutor(max_workers=20)
def save_conv_log_to_azure_storage(
    filename: str,
    log_data: dict[str, Any],
    write_mode: Literal['overwrite', 'append'] = 'append',
    use_async: bool = True
) -> None:
    try:
        if AZURE_BLOB_STORAGE_CONNECTION_STRING:
            blob_name = get_conv_log_blob_name(filename)
            log_json: str = json.dumps(
                obj=log_data,
                default=str,
            )

            def _run_upload():
                upload_data_to_azure_storage(
                    str.encode(log_json + "\n"),
                    blob_name,
                    write_mode
                )

            if use_async:
                _executor.submit(_run_upload)
            else:
                _run_upload()
    except Exception as e:
        print(f"Error uploading conv log to Azure Blob Storage: {e}")


def get_sandbox_log_filename(sandbox_state: ChatbotSandboxState) -> str:
    return (
        '-'.join(
            [
                "sandbox-logs",
                f"{sandbox_state['conv_id']}", # chat conv id
                f"{sandbox_state['enabled_round']}", # current chat round
                f"{sandbox_state['sandbox_run_round']}", # current sandbox round
            ]
         ) + ".json"
    )


def upsert_sandbox_log(filename: str, data: str) -> None:
    filepath = os.path.join(
        LOG_DIR,
        datetime.datetime.now().strftime('%Y_%m_%d'), # current date as 2025_02_02
        'sandbox_logs',
        filename
    )
    # create directory if not exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as fout:
        fout.write(data)


def create_sandbox_log(sandbox_state: ChatbotSandboxState, user_interaction_records: list[Any] | None) -> dict:
    return {
        "sandbox_state": sandbox_state,
        "user_interaction_records": user_interaction_records,
    }


def log_sandbox_telemetry_gradio_fn(
    sandbox_state: ChatbotSandboxState,
    sandbox_ui_value: tuple[str, bool, list[Any]] | None
) -> None:
    if sandbox_state is None:
        return
    sandbox_id = sandbox_state['sandbox_id']
    user_interaction_records = sandbox_ui_value[2] if sandbox_ui_value else None
    if sandbox_id is None or user_interaction_records is None or len(user_interaction_records) == 0:
        return
    
    log_json = create_sandbox_log(sandbox_state, user_interaction_records)
    log_data = json.dumps(
        log_json,
        indent=2,
        default=str,
        ensure_ascii=False
    )
    filename = get_sandbox_log_filename(sandbox_state)
    upsert_sandbox_log(filename=filename, data=log_data)

    # # Upload to Azure Blob Storage
    # if AZURE_BLOB_STORAGE_CONNECTION_STRING:
    #     try:
    #         blob_name = get_sandbox_log_blob_name(filename)
    #         upload_data_to_azure_storage(
    #             data=str.encode(log_data),
    #             blob_name=blob_name,
    #             write_mode='overwrite'
    #         )
    #     except Exception as e:
    #         print(f"Error uploading sandbox log to Azure Blob Storage: {e}")
