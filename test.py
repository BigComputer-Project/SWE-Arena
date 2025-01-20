import os
from httpcore import ReadTimeout
from e2b import Sandbox
from e2b.sandbox.commands.command_handle import CommandExitException
import threading
import queue
# Simple Gradio app code
GRADIO_APP_CODE = '''
import gradio as gr
import pandas as pd

def csv_analysis(file):
    data = pd.read_csv(file)
    summary = data.describe()
    return summary

file = gr.inputs.File(label="Upload CSV file")
output = gr.outputs.Textbox()

gr.Interface(fn=csv_analysis, inputs=file, outputs=output, title="CSV File Analysis").launch()
'''

# GRADIO_APP_CODE = '''
# import gradio as gr

# def greet(name):
#     return f"Hello, {name}!"

# demo = gr.Interface(
#     fn=greet,
#     inputs="text",
#     outputs="text",
#     title="Simple Greeting App"
# )

# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0")
# '''

E2B_API_KEY = "e2b_3765c8ebbf552abe15f5d8b417673642f5be62ec"


def run_command_with_timeout(sandbox, command_str, timeout):
    stderr = ""
    def collect_stderr(message):
        nonlocal stderr
        stderr += message

    def wait_for_command(result_queue):
        nonlocal stderr
        try:
            result = command.wait()
            print(f"Result: {result.stderr}")
            if result.stderr:
                stderr += result.stderr
            result_queue.put(stderr)
        except ReadTimeout:
            result_queue.put(stderr)
        except CommandExitException as e:
            stderr += "".join(e.stderr)
            result_queue.put(stderr)

    # Create a queue to store the result
    result_queue = queue.Queue()

    command = sandbox.commands.run(
        command_str,
        timeout=15,  # Overall timeout for the command
        background=True,
        on_stderr=lambda message: print(message)
    )

    # Create a thread to wait for the command
    wait_thread = threading.Thread(target=wait_for_command, args=(result_queue,))
    wait_thread.start()
    # Wait for the thread to complete or timeout
    wait_thread.join(timeout)

    if wait_thread.is_alive():
        # Timeout occurred
        return ""
    else:
        # Command completed, get the result from the queue
        return result_queue.get()
    

def test_gradio_sandbox():
    # Get E2B API key from environment
    stderr = ""
    def collect_stderr(message):
        nonlocal stderr
        stderr += message

    if not E2B_API_KEY:
        raise ValueError("E2B_API_KEY not found in environment variables")

    print("Creating sandbox...")
    sandbox = Sandbox(
        api_key=E2B_API_KEY
    )

    print("Installing dependencies...")
    setup_commands = ["pip install uv", "uv pip install --system gradio pillow numpy"]
    for command in setup_commands:
        sandbox.commands.run(
            command,
            timeout=60 * 3,
            on_stdout=lambda message: print(f"Setup stdout: {message}"),
        )

    print("Writing Gradio app code...")
    file_path = "~/app.py"
    sandbox.files.write(path=file_path, data=GRADIO_APP_CODE, request_timeout=60)

    print("Starting Gradio server...")

    stderr = run_command_with_timeout(sandbox, "gradio ~/app.py", 10)
    # command = sandbox.commands.run(
    #         "python ~/app.py",
    #         timeout=60 * 3,
    #         background=True,
    #     )

    # try:
    #     # Start the command in background
    #     command = sandbox.commands.run(
    #         "python ~/app.py",
    #         timeout=60 * 3,
    #         background=True,
    #     )
    #     try:
    #         import signal
    #         import time
            
    #         def timeout_handler(signum, frame):
    #             raise TimeoutError("Timed out waiting for server")
    #         # Set up timeout
    #         signal.signal(signal.SIGALRM, timeout_handler)
    #         signal.alarm(5)  # 5 second timeout
            
    #         try:
    #             result = command.wait(
    #                 on_stdout=collect_stdout,
    #                 on_stderr=collect_stderr
    #             )
    #             print("Server stopped unexpectedly")
    #             if result.stderr:
    #                 stderr = result.stderr
    #         except TimeoutError:
    #             print("Server started successfully")
    #         finally:
    #             signal.alarm(0)  # Disable the alarm
    #     except ReadTimeout:
    #         print("Server started successfully")
            
    # except CommandExitException as e:
    #     print(f"Error starting http server: {e}")
    #     stderr = "".join(e.stderr)
    # Get the sandbox URL
    sandbox_url = 'https://' + sandbox.get_host(7860)
    print(f"\nGradio app is running at: {sandbox_url}")
    return sandbox_url, stderr

def test_gradio_with_file_upload():
    """Test Gradio with file upload functionality"""
    FILE_UPLOAD_APP = '''
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

def analyze_csv(file):
    # Read the CSV file
    df = pd.read_csv(file.name)
    
    # Create a simple plot
    plt.figure(figsize=(10, 6))
    df.plot(kind='line')
    plt.title('Data Visualization')
    plt.xlabel('Index')
    plt.ylabel('Values')
    
    # Get basic statistics
    stats = df.describe().to_string()
    
    return plt.gcf(), stats

demo = gr.Interface(
    fn=analyze_csv,
    inputs=gr.File(file_types=[".csv"]),
    outputs=[
        gr.Plot(label="Data Plot"),
        gr.Textbox(label="Statistics", lines=10)
    ],
    title="CSV File Anar",
    description="Upload a CSV file to see basic statistics and visualization"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
'''

    print("Creating sandbox for file upload demo...")
    sandbox = Sandbox(
        template="gradio-developer",
        metadata={
            "template": "gradio-developer"
        },
        api_key=E2B_API_KEY
    )

    print("Installing dependencies...")
    setup_commands = [
        "pip install uv",
        "uv pip install --system gradio pandas matplotlib"
    ]
    for command in setup_commands:
        sandbox.commands.run(
            command,
            timeout=60 * 3,
            on_stdout=lambda message: print(f"Setup stdout: {message}"),
            on_stderr=lambda message: print(f"Setup stderr: {message}"),
        )

    print("Writing File Upload app code...")
    file_path = "~/app.py"
    sandbox.files.write(path=file_path, data=FILE_UPLOAD_APP, request_timeout=60)

    print("Starting Gradio server...")
    # sandbox.commands.run(
    #     "gradio run ~/app.py",
    #     timeout=60 * 3,
    #     background=True,
    #     on_stdout=lambda message: print(f"Gradio stdout: {message}"),
    #     on_stderr=lambda message: print(f"Gradio stderr: {message}"),
    # )

    sandbox_url = 'https://' + sandbox.get_host(7860)
    print(f"\nFile Upload demo is running at: {sandbox_url}")
    return sandbox_url

if __name__ == "__main__":
    print("Testing basic Gradio app...")
    url1, stderr1 = test_gradio_sandbox()
    print(f"Stderr: {stderr1}")
    # print("\nTesting file upload functionality...")
    # url2 = test_gradio_with_file_upload() 
