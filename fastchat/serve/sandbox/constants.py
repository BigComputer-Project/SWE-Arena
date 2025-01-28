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
Timeout in seconds for created sandboxes to expire.
'''

CODE_RUN_TIMEOUT_SECONDS: int = 60
'''
Timeout in seconds for code execution.
'''

SANDBOX_RETRY_COUNT: int = 3
'''
Number of times to retry the sandbox creation.
'''

INSTALLED_PYPI_PACKAGES = [
    "boto3",
    "botocore",
    "urllib3",
    "setuptools",
    "requests",
    "certifi",
    "idna",
    "charset-normalizer",
    "packaging",
    "typing-extensions",
    "python-dateutil",
    "aiobotocore",
    "s3transfer",
    "grpcio-status",
    "pyyaml",
    "six",
    "fsspec",
    "s3fs",
    "numpy",
    "wheel",
    "pip",
    "cryptography",
    "awscli",
    "pydantic",
    "cffi",
    "attrs",
    "google-api-core",
    "pycparser",
    "pandas",
    "importlib-metadata",
    "jmespath",
    "click",
    "zipp",
    "rsa",
    "pyasn1",
    "markupsafe",
    "pytz",
    "colorama",
    "protobuf",
    "platformdirs",
    "jinja2",
    "rich",
    "tomli",
    "pytest",
    "pydantic-core",
    "pyjwt",
    "pluggy",
    "aiohttp",
    "virtualenv",
    "jsonschema",
    "googleapis-common-protos",
    "cachetools",
    "google-auth",
    "filelock",
    "wrapt",
    "sqlalchemy",
    "docutils",
    "pyasn1-modules",
    "pyarrow",
    "greenlet",
    "iniconfig",
    "pygments",
    "annotated-types",
    "yarl",
    "requests-oauthlib",
    "tzdata",
    "psutil",
    "multidict",
    "pyparsing",
    "requests-toolbelt",
    "exceptiongroup",
    "werkzeug",
    "soupsieve",
    "oauthlib",
    "beautifulsoup4",
    "frozenlist",
    "more-itertools",
    "distlib",
    "tomlkit",
    "pathspec",
    "aiosignal",
    "grpcio",
    "tqdm",
    "scipy",
    "async-timeout",
    "pillow",
    "isodate",
    "anyio",
    "sortedcontainers",
    "decorator",
    "markdown-it-py",
    "deprecated",
    "mypy-extensions",
    "sniffio",
    "httpx",
    "coverage",
    "openpyxl",
    "flask",
    "rpds-py",
    "et-xmlfile"
]