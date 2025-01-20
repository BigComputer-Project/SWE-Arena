FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app
COPY . .

# Install dependencies
RUN pip install aiohttp fastapi httpx==0.27.2 markdown2[all] nh3 numpy prompt_toolkit>=3.0.0 'pydantic<3,>=2.0.0' pydantic-settings psutil requests rich>=10.0.0 shortuuid tiktoken uvicorn tree-sitter tree-sitter-javascript tree-sitter-typescript plotly scipy openai e2b e2b_code_interpreter gradio-sandboxcomponent google-generativeai azure-storage-blob 
RUN pip install "accelerate>=0.21" "peft" "sentencepiece" "torch" "transformers>=4.31.0" "protobuf" "gradio>=4.10"
RUN pip install -e ".[model_worker,webui]"


ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0
# Expose the port
EXPOSE 7860

# Run Gradio server
CMD [ "python", "-m", \
        "fastchat.serve.gradio_web_server_multi", \
        "--controller-url", "", \
        "--register", "api_endpoints.json" ]