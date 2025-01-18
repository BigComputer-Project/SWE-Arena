'''
Constants for sandbox.
'''

import os

E2B_API_KEY = os.environ.get("E2B_API_KEY")
'''
API key for the e2b API.
'''

AZURE_BLOB_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
'''
API key for the Azure Blob Storage.
'''

AZURE_BLOB_STORAGE_CONTAINER_NAME = "softwarearenalogs"
'''
Contianer name for the Azure Blob Storage.
'''

SANDBOX_TEMPLATE_ID: str = "bxq9sha9l55ytsyfturr"
'''
Template ID for the sandbox.
'''

SANDBOX_NGINX_PORT: int = 8000
'''
Nginx port for the sandbox.
'''

SANDBOX_TIMEOUT_SECONDS: int = 5 * 60
'''
Timeout in seconds for created sandboxes.
'''

SANDBOX_RETRY_COUNT: int = 3
'''
Number of times to retry the sandbox creation.
'''