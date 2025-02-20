'''
Facade for reading logs on remote storage.
'''

from collections import defaultdict
import json
import os
from typing import Any
from azure.storage.fileshare import ShareServiceClient


class RemoteLogReader:
    '''
    remote log reader
    '''

    LOG_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING") or ""
    LOG_SHARE_NAME = "swearenalogsfileshare"

    IMAGE_DIR_NAME = "serve_images"
    '''
    Directory for storing user uploaded images.
    '''
    CONV_LOG_DIR_NAME = "conv_logs"
    '''
    Directory for conversation logs.
    '''
    SANDBOX_LOG_DIR_NAME = "sandbox_logs"
    '''
    Directory for sandbox logs.
    '''

    CHAT_MODES = ["battle_anony", "battle_named", "direct"]

    def __init__(
        self,
        connection_string: str = LOG_CONNECTION_STRING,
        share_name: str = LOG_SHARE_NAME,
    ):
        if not connection_string:
            raise ValueError("Connection string is required.")
        if not share_name:
            raise ValueError("Share name is required.")

        self.share_service = ShareServiceClient.from_connection_string(
            conn_str=connection_string)
        self.share_client = share_service.get_share_client(share=share_name)

    def is_conv_log(self, file_name: str) -> bool:
        return file_name.startswith("conv-log") and file_name.endswith(".json")

    def get_conv_id_from_name(self, file_name: str) -> str:
        return file_name.split("-")[2].strip('.json')

    def is_sandbox_log(self, file_name: str) -> bool:
        return file_name.startswith("sandbox-log") and file_name.endswith(".json")

    def get_file_content(self, file_path: str) -> bytes:
        file_client = self.share_client.get_file_client(file_path)
        file_content = file_client.download_file().readall()
        return file_content

    def get_conv_logs(self, date: str) -> dict[str, defaultdict[str, list[Any]]]:
        '''
        Return conversation logs based on the date.
        Returns a dict:
            mode -> conv_id -> list of logs.
        '''
        conv_logs = {
            mode: defaultdict(list) for mode in self.CHAT_MODES
        }
        for mode in self.CHAT_MODES:
            conv_log_dir = f"{date}/{self.CONV_LOG_DIR_NAME}/{mode}/"
            # check if the directory exists
            if not self.share_client.get_directory_client(conv_log_dir).exists():
                continue
            for file in self.share_client.list_directories_and_files(conv_log_dir):
                if not self.is_conv_log(file.name):
                    continue
                conv_id = self.get_conv_id_from_name(file.name)
                file_content = self.get_file_content(
                    conv_log_dir + file.name).decode("utf-8").strip(' \n')
                for line in file_content.split('\n'):
                    if line:
                        conv_logs[mode][conv_id].append(json.loads(line))
        return conv_logs

    def get_sandbox_logs(self, date: str) -> list[str]:
        '''
        Return sandbox logs based on the date.
        '''
        sandbox_logs = []
        sandbox_log_dir = f"{date}/{self.SANDBOX_LOG_DIR_NAME}/"
        for file in self.share_client.list_directories_and_files(sandbox_log_dir):
            if self.is_sandbox_log(file.name):
                file_content = self.get_file_content(
                    sandbox_log_dir + file.name).decode("utf-8").strip(' \n')
                sandbox_logs.append(json.loads(file_content))
        return sandbox_logs

    def get_image(self, image_id: str) -> bytes:
        '''
        Return image data based on the image id.
        '''
        image_path = f"{self.IMAGE_DIR_NAME}/{image_id}.png"
        return self.get_file_content(image_path)


if __name__ == "__main__":
    # Example usages
    log_reader = RemoteLogReader()
    date = "2025_02_20"
    conv_logs = log_reader.get_conv_logs(date)
    sandbox_logs = log_reader.get_sandbox_logs(date)
    image_data = log_reader.get_image("051fdac24285ff6e219a9ba06d1ac843")
    print(conv_logs)
    print(sandbox_logs)
    print(image_data)
