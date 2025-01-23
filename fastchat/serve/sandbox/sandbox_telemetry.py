'''
Module for logging the sandbox interactions and state.

TODO: Support Cloud Storage.
'''
import json
import os
from typing import Any, Literal
from datetime import datetime

from fastchat.constants import LOGDIR
from fastchat.serve.sandbox.code_runner import ChatbotSandboxState

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
    date_str = datetime.now().strftime('%Y_%m_%d')
    blob_name = f"{date_str}/sandbox_logs/{filename}"
    return blob_name


def get_conv_log_blob_name(filename: str) -> str:
    date_str = datetime.now().strftime('%Y_%m_%d')
    blob_name = f"{date_str}/conv_logs/{filename}"
    return blob_name


def upload_conv_log_to_azure_storage(filename: str, data: str, write_mode: Literal['overwrite', 'append'] = 'append') -> None:
    try:
        if AZURE_BLOB_STORAGE_CONNECTION_STRING:
            blob_name = get_conv_log_blob_name(filename)
            upload_data_to_azure_storage(
                str.encode(data + "\n"),
                blob_name,
                write_mode
            )
    except Exception as e:
        print(f"Error uploading conv log to Azure Blob Storage: {e}")

def get_sandbox_log_filename(sandbox_state: ChatbotSandboxState) -> str:
    return (
        '-'.join(
            [
                "sandbox-records",
                f"{sandbox_state['sandbox_id']}",
                f"{sandbox_state['enabled_round']}",
                f"{sandbox_state['sandbox_run_round']}",
            ]
         ) + ".json"
    )


def upsert_sandbox_log(filename: str, data: str) -> None:
    filepath = os.path.join(LOGDIR, filename)
    with open(filepath, "w") as fout:
        fout.write(data)


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

    # Upload to Azure Blob Storage
    if AZURE_BLOB_STORAGE_CONNECTION_STRING:
        try:
            blob_name = get_sandbox_log_blob_name(filename)
            upload_data_to_azure_storage(
                data=str.encode(log_data),
                blob_name=blob_name,
                write_mode='overwrite'
            )
        except Exception as e:
            print(f"Error uploading sandbox log to Azure Blob Storage: {e}")
