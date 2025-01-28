# e2b base image
FROM --platform=linux/x86_64 e2bdev/code-interpreter:latest
# FROM node:21-slim

RUN apt-get update
# Install Python
RUN apt-get install -y python3

# Install nginx
RUN sudo apt-get install -y nginx
# Add Nginx configuration and serve with Nginx
COPY nginx/nginx.conf /etc/nginx/sites-enabled/default
# CMD ["nginx", "-g", "daemon off;"]

# Install build tools for C and C++
RUN apt-get install -y build-essential 
# Install build tools for C#
# RUN apt-get install -y dotnet-sdk-8.0
# Install build tools for Golang
RUN apt-get install -y golang
# Install build tools for Java
RUN apt-get install -y default-jdk
# Install build tools for Rust
RUN apt-get install -y rustc

# install ffmpeg for pygame
RUN apt-get install -y ffmpeg

# Pre-Install Python packages
RUN pip install uv
RUN uv pip install --system boto3 \
    botocore \
    urllib3 \
    setuptools \
    requests \
    certifi \
    idna \
    charset-normalizer \
    packaging \
    typing-extensions \
    python-dateutil \
    aiobotocore \
    s3transfer \
    grpcio-status \
    pyyaml \
    six \
    fsspec \
    s3fs \
    numpy==1.26 \
    wheel \
    pip \
    cryptography \
    awscli \
    pydantic \
    cffi \
    attrs \
    google-api-core \
    pycparser \
    pandas \
    importlib-metadata \
    jmespath \
    click \
    zipp \
    rsa \
    pyasn1 \
    markupsafe \
    pytz \
    colorama \
    protobuf \
    platformdirs \
    jinja2 \
    rich \
    tomli \
    pytest \
    pydantic-core \
    pyjwt \
    pluggy \
    aiohttp \
    virtualenv \
    jsonschema \
    googleapis-common-protos \
    cachetools \
    google-auth \
    filelock \
    wrapt \
    sqlalchemy \
    docutils \
    pyasn1-modules \
    pyarrow \
    greenlet \
    iniconfig \
    pygments \
    annotated-types \
    yarl \
    requests-oauthlib \
    tzdata \
    psutil \
    multidict \
    pyparsing \
    requests-toolbelt \
    exceptiongroup \
    werkzeug \
    soupsieve \
    oauthlib \
    beautifulsoup4 \
    frozenlist \
    more-itertools \
    distlib \
    tomlkit \
    pathspec \
    aiosignal \
    grpcio \
    tqdm \
    scipy \
    async-timeout \
    pillow \
    isodate \
    anyio \
    sortedcontainers \
    decorator \
    markdown-it-py \
    deprecated \
    mypy-extensions \
    sniffio \
    httpx \
    coverage \
    openpyxl \
    flask \
    rpds-py \
    et-xmlfile

RUN uv pip install --system pandas numpy matplotlib requests seaborn plotly
RUN uv pip install --system pygame pygbag black
RUN uv pip install --system --upgrade streamlit gradio nicegui

# Build container_app
WORKDIR /home/user/container_app
COPY container_app/ ./
RUN npm install
RUN npm run build

# Prepare html app
WORKDIR /home/user/html_app
COPY html_app/ ./

# Prepare & Build react app
WORKDIR /home/user/react_app
COPY react_app/ ./
RUN npm install
RUN npm run build

# Prepare & Build vue app
WORKDIR /home/user/vue_app
COPY vue_app/ ./
RUN npm install
RUN npm run build

# Prepare pygame app
WORKDIR /home/user/pygame_app
COPY pygame_app/ ./

# Prepare gradio app
WORKDIR /home/user/gradio_app
COPY gradio_app/ ./