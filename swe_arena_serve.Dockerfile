FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app
COPY . .

# Install dependencies
RUN pip install -e ".[model_worker,webui]"


ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0
# Expose the port
EXPOSE 7860

# Run Gradio server
CMD [ "python", "-m", \
        "fastchat.serve.gradio_web_server_multi", \
        "--vision-arena", \
        "--controller-url", "", \
        "--register", "api_endpoints_serve.json" ]