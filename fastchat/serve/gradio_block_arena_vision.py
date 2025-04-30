"""
The gradio demo server for chatting with a large multimodal model.

Usage:
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.gradio_web_server_multi --share --vision-arena
"""

import json
import os
import time
from typing import List, Union

import gradio as gr
from gradio.data_classes import FileData
import numpy as np

from gradio_sandboxcomponent import SandboxComponent

from fastchat.constants import (
    TEXT_MODERATION_MSG,
    IMAGE_MODERATION_MSG,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SURVEY_LINK,
)
from fastchat.model.model_adapter import (
    get_conversation_template,
)
from fastchat.serve.chat_state import LOG_DIR, ModelChatState, save_log_to_local
from fastchat.serve.gradio_global_state import Context
from fastchat.serve.gradio_web_server import (
    clear_sandbox_components,
    get_model_description_md,
    acknowledgment_md, # reuse the acknowledgment_md from gradio_web_server
    bot_response,
    get_ip,
    disable_btn,
    get_remote_logger,
    set_chat_system_messages,
    update_system_prompt,
)
from fastchat.serve.sandbox.sandbox_state import ChatbotSandboxState
from fastchat.serve.sandbox.sandbox_telemetry import log_sandbox_telemetry_gradio_fn, save_conv_log_to_azure_storage
from fastchat.serve.vision.image import ImageFormat, Image
from fastchat.serve.sandbox.code_runner import SandboxEnvironment, SandboxGradioSandboxComponents, DEFAULT_SANDBOX_INSTRUCTIONS, RUN_CODE_BUTTON_HTML, SUPPORTED_SANDBOX_ENVIRONMENTS, create_chatbot_sandbox_state, on_click_code_message_run, on_edit_code, on_edit_dependency, reset_sandbox_state, update_sandbox_config, update_sandbox_state_system_prompt, set_sandbox_state_ids
from fastchat.utils import (
    build_logger,
    moderation_filter,
    image_moderation_filter,
)

logger = build_logger("gradio_web_server", "gradio_web_server.log")

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True, visible=True)
disable_btn = gr.Button(interactive=False)
invisible_btn = gr.Button(interactive=False, visible=False)
visible_image_column = gr.Image(visible=True)
invisible_image_column = gr.Image(visible=False)
enable_multimodal = gr.MultimodalTextbox(
    interactive=True, visible=True, placeholder="Enter your prompt or add image here"
)
invisible_text = gr.Textbox(visible=False, value="", interactive=False)
visible_text = gr.Textbox(
    visible=True,
    value="",
    interactive=True,
    placeholder="👉 Enter your prompt and press ENTER",
)
disable_multimodal = gr.MultimodalTextbox(visible=False, value=None, interactive=False)

USER_BUTTONS_LENGTH = 6


def set_visible_image(textbox):
    images = textbox["files"]
    if len(images) == 0:
        return invisible_image_column
    elif len(images) > 1:
        gr.Warning(
            "We only support single image conversations. Please start a new round if you would like to chat using this image."
        )

    return visible_image_column


def set_invisible_image():
    return invisible_image_column


def add_image(textbox):
    images = textbox["files"]
    if len(images) == 0:
        return None

    return images[0]


def vote_last_response(state: ModelChatState, vote_type, model_selector, request: gr.Request):
    local_filepath = state.get_conv_log_filepath(LOG_DIR)
    log_data = state.generate_vote_record(
        vote_type=vote_type, ip=get_ip(request)
    )
    get_remote_logger().log(log_data)
    save_log_to_local(log_data, local_filepath)
    # save_conv_log_to_azure_storage(local_filepath.lstrip(LOCAL_LOG_DIR), log_data)

    gr.Info(
        "🎉 Thanks for voting! Your vote shapes the leaderboard, please vote RESPONSIBLY."
    )

def upvote_last_response(
    state: ModelChatState, model_selector: str, sandbox_state: ChatbotSandboxState, request: gr.Request):
    '''
    Input: [state, model_selector, sandbox_state],
    Output: [textbox] + user_buttons,
    '''
    ip = get_ip(request)
    logger.info(f"upvote. ip: {ip}")
    vote_last_response(state, "upvote", model_selector, request)
    return (None,) + (disable_btn,) * (sandbox_state["btn_list_length"] - 1) + (enable_btn,) # enable clear button


def downvote_last_response(state: ModelChatState, model_selector: str, sandbox_state: ChatbotSandboxState, request: gr.Request):
    '''
    Input: [state, model_selector, sandbox_state],
    Output: [textbox] + user_buttons,
    '''
    ip = get_ip(request)
    logger.info(f"downvote. ip: {ip}")
    vote_last_response(state, "downvote", model_selector, request)
    return (None,) + (disable_btn,) * (sandbox_state["btn_list_length"] - 1) + (enable_btn,) # enable clear button


def flag_last_response(state: ModelChatState, model_selector: str, sandbox_state: ChatbotSandboxState, request: gr.Request):
    '''
    Input: [state, model_selector, sandbox_state],
    Output: [textbox] + user_buttons,
    '''
    ip = get_ip(request)
    logger.info(f"flag. ip: {ip}")
    vote_last_response(state, "flag", model_selector, request)
    return (None,) + (disable_btn,) * (sandbox_state["btn_list_length"] - 1) + (enable_btn,) # enable clear button


def regenerate(state: ModelChatState, sandbox_state: ChatbotSandboxState, request: gr.Request):
    '''
    Regenerate the chatbot response.

        regenerate,
        state,
        [state, chatbot, textbox] + user_buttons

        user_buttons: list[gr.Button] = [
            # send button
            send_btn,
            # regenerate button
            regenerate_btn,
            # vote buttons
            upvote_btn, downvote_btn, flag_btn,
            # clear button
            clear_btn,
        ]
    '''
    ip = get_ip(request)
    logger.info(f"regenerate. ip: {ip}")
    if not state.regen_support:
        # if the model does not support regeneration, skip
        state.skip_next = True
        return (state, state.to_gradio_chatbot()) + (None,) + (no_change_btn,) * sandbox_state["btn_list_length"]
    else:
        state.conv.update_last_message(None)
        state.set_response_type("regenerate_single")
        return (state, state.to_gradio_chatbot()) + (None,) + (disable_btn,) * sandbox_state["btn_list_length"]


def clear_history(sandbox_state, request: gr.Request):
    '''
    Clear the conversation history and reset the state.

        clear_history,
        sandbox_state,
        [state, chatbot, sandbox_state] + [multimodal_textbox, textbox] + user_buttons,

        user_buttons: list[gr.Button] = [
            # send button
            send_btn,
            # regenerate button
            regenerate_btn,
            # vote buttons
            upvote_btn, downvote_btn, flag_btn,
            # clear button
            clear_btn,
        ]
    '''
    ip = get_ip(request)
    logger.info(f"clear_history. ip: {ip}")

    # reset chatbot state
    state = None
    # reset sandbox state
    sandbox_state = reset_sandbox_state(sandbox_state)

    return (
        (None, None, sandbox_state)
        + (enable_multimodal, invisible_text)
        + (enable_btn,)
        + (invisible_btn,) * 5 # disable buttons
    )


# TODO(Chris): At some point, we would like this to be a live-reporting feature.
def report_csam_image(state, image):
    pass


def _prepare_text_with_image(state, text, images, csam_flag):
    if len(images) > 0:
        if len(state.conv.get_images()) > 0:
            # reset convo with new image
            state.conv = get_conversation_template(state.model_name)

        text = text, [images[0]]

    return text


# NOTE(chris): take multiple images later on
def convert_images_to_conversation_format(images) -> list[Image]:
    import base64

    MAX_NSFW_ENDPOINT_IMAGE_SIZE_IN_MB = 5 / 1.5
    conv_images = []
    if len(images) > 0:
        conv_image = Image(url=images[0])
        conv_image.to_conversation_format(MAX_NSFW_ENDPOINT_IMAGE_SIZE_IN_MB)
        conv_images.append(conv_image)

    return conv_images


def moderate_input(state, text, all_conv_text, model_list, images, ip):
    text_flagged = moderation_filter(all_conv_text, model_list)
    # flagged = moderation_filter(text, [state.model_name])
    nsfw_flagged, csam_flagged = False, False
    if len(images) > 0:
        nsfw_flagged, csam_flagged = image_moderation_filter(images[0])

    image_flagged = nsfw_flagged or csam_flagged
    if text_flagged or image_flagged:
        logger.info(f"violate moderation. ip: {ip}. text: {all_conv_text}")
        if text_flagged and not image_flagged:
            # overwrite the original text
            text = TEXT_MODERATION_MSG
        elif not text_flagged and image_flagged:
            text = IMAGE_MODERATION_MSG
        elif text_flagged and image_flagged:
            text = MODERATION_MSG

    if csam_flagged:
        state.has_csam_image = True
        report_csam_image(state, images[0])

    return text, image_flagged, csam_flagged


def add_text(
    state: ModelChatState,
    model_selector: str, # the selected model name
    sandbox_state: ChatbotSandboxState, # the sandbox state
    multimodal_input: dict, text_input: str,
    context: Context,
    request: gr.Request,
):
    '''
    Add text to the chatbot state and update the chatbot UI.

        add_text,
        [state, model_selector, sandbox_state] + [multimodal_textbox, textbox], + [context_state],
        [state, chatbot, sandbox_state] + [multimodal_textbox, textbox] + user_buttons,
    '''
    if multimodal_input and multimodal_input["text"]:
        text, images = multimodal_input["text"], multimodal_input["files"]
    else:
        text = text_input
        images = []

    # whether need vision models
    is_vision = len(images) > 0

    # check whether selected model is valid
    if model_selector not in context.text_models and model_selector not in context.vision_models:
        gr.Warning(f"Selected model '{model_selector}' is invalid. Please select a valid model.")
        return (state, None, sandbox_state) + (None, "") + (no_change_btn,) * sandbox_state["btn_list_length"]

    # check whether selected model supports vision
    if (
        is_vision
        and model_selector in context.text_models
        and model_selector not in context.vision_models
    ):
        gr.Warning(f"Selected model '{model_selector}' is a text-only model. Image is ignored.")
        images = []
        is_vision = False

    ip = get_ip(request)
    logger.info(f"add_text. ip: {ip}. len: {len(text)}")

    # init chatbot state if not exist
    if state is None:
        state = ModelChatState(
            model_name=model_selector,
            chat_mode="direct",
            is_vision=is_vision,
            chat_session_id=None,
        )
        set_sandbox_state_ids(
            sandbox_state=sandbox_state,
            conv_id=state.conv_id,
            chat_session_id=state.chat_session_id
        )

    if len(text) == 0:
        # skip empty text
        state.skip_next = True
        # [state, chatbot, sandbox_state] + [multimodal_textbox, textbox] + user_buttons,
        return (
            (state, state.to_gradio_chatbot(), sandbox_state)
            + (None, "")
            + (no_change_btn,) * sandbox_state["btn_list_length"]
        )

    all_conv_text = state.conv.get_prompt()
    all_conv_text = all_conv_text[-2000:] + "\nuser: " + text

    images = convert_images_to_conversation_format(images)

    # TODO: Skip image moderation for now
    # text, image_flagged, csam_flag = moderate_input(
    #     state0, text, text, model_list, images, ip
    # )
    image_flagged, csam_flag = None, None
    # if image_flagged:
    #     logger.info(f"image flagged. ip: {ip}. text: {text}")
    #     state.skip_next = True
    #     return (
    #         state,
    #         state.to_gradio_chatbot(),
    #         {"text": IMAGE_MODERATION_MSG},
    #         "",
    #         no_change_btn,
    #     ) + (no_change_btn,) * 5

    if (len(state.conv.messages) - state.conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        state.skip_next = True
        # [state, chatbot, sandbox_state] + [multimodal_textbox, textbox] + user_buttons,
        return (
            (state, state.to_gradio_chatbot(), sandbox_state)
            + ({"text": CONVERSATION_LIMIT_MSG}, "")
            + (no_change_btn,) * sandbox_state["btn_list_length"]
        )

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    text = _prepare_text_with_image(state, text, images, csam_flag=csam_flag)
    state.conv.append_message(state.conv.roles[0], text)
    state.conv.append_message(state.conv.roles[1], None)
    state.set_response_type("chat_single")

    sandbox_state['enabled_round'] += 1 
    # [state, chatbot, sandbox_state] + [multimodal_textbox, textbox] + user_buttons,
    return (
        (state, state.to_gradio_chatbot(), sandbox_state)
        + (disable_multimodal, visible_text)
        + (disable_btn,) * sandbox_state["btn_list_length"]
    )


def build_single_vision_language_model_ui(
    context: Context, add_promotion_links=False, random_questions=None
):

    notice_markdown = f"""
## How It Works for Direct Chat Mode
- **Choose Models**: Select an AI chatbot to chat with.
- **Run & Interact**: The AI chatbots generate programs that run in a secure sandbox environment. Test the functionality, explore the features, and evaluate the quality of the outputs.
- **Edit & Regenerate**: You can edit the <u>system prompt</u>, <u>code</u>, and its <u>dependency</u> and <u>regenerate the code</u> on any side.
- **Visual Input**: Upload images or provide text prompts to guide the AI chatbots in their responses. You can only chat with <span style='color: #DE3163; font-weight: bold'>one image per conversation</span>. The image should be less than 15MB.

## Note
- **Sandbox**: There are four buttons on the top right corner of the sandbox inference. You can use them to copy the link, turn on/off the black mode, make the sandbox full screen, and reload the sandbox. Plus, you can also drag the bottom right corner of the sandbox to resize the sandbox.
- **Dependency Edit**: You can edit the <u>dependency</u> of the code on any side. Currently, we only support `pypi` and `npm` packages.
For `pypi` packages, you can use the format `python (use '==', '>=', '<=', '~=', '>', '<' or 'latest') <package_name> <version>`.
For `npm` packages, you can use the format `npm (use '@' or 'latest') <package_name> <version>`.
- **Temperature**: All models have the same temperature of `0.7` and `top_p` of `1.0` by default. You can adjust the hyperparameters in the `Parameters` section. Low temperature typically works better for code generation.

**❗️ For research purposes, we log user prompts, images, and interactions with sandbox, and may release this data to the public in the future. Please do not upload any confidential or personal information.**
"""

    state = gr.State()
    vision_not_in_text_models = [
        model for model in context.vision_models if model not in context.text_models
    ]
    text_and_vision_models = context.text_models + vision_not_in_text_models
    context_state = gr.State(context)

    with gr.Group():
        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=text_and_vision_models,
                value=text_and_vision_models[0]
                if len(text_and_vision_models) > 0
                else "",
                interactive=True,
                show_label=False,
                container=False,
            )

        with gr.Accordion(
            f"🔍 Expand to see the descriptions of {len(text_and_vision_models)} models",
            open=False,
        ):
            model_description_md = get_model_description_md(text_and_vision_models)
            gr.Markdown(model_description_md, elem_id="model_description_markdown")

    with gr.Row():
        with gr.Column(scale=2, visible=False) as image_column:
            imagebox = gr.Image(
                type="pil",
                show_label=False,
                interactive=False,
            )
        with gr.Column(scale=8):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Scroll down and start chatting",
                height='65vh',
                show_copy_button=True,
                latex_delimiters=[
                    {"left": "$", "right": "$", "display": False},
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": r"\(", "right": r"\)", "display": False},
                    {"left": r"\[", "right": r"\]", "display": True},
                ],
            )

    with gr.Row(elem_id="user-input-region"):
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Input your prompt here. Press Enter to send.",
            elem_id="input_box",
            visible=False,
            scale=3,
            autofocus=True,
            max_length=INPUT_CHAR_LEN_LIMIT,
        )

        multimodal_textbox = gr.MultimodalTextbox(
            file_types=["image"],
            show_label=False,
            container=True,
            placeholder="Input your prompt or add image here. Press Enter to send.",
            elem_id="input_box",
            scale=3,
            submit_btn=False,
            stop_btn=False,
            autofocus=True,
            max_plain_text_length=INPUT_CHAR_LEN_LIMIT,
        )

    with gr.Row() as examples_row:
        example_prompts = gr.Examples(
            examples = [
                ["使用SVG绘制春节主题的动态图案，包括：1）一个红色的灯笼，带有金色的流苏 2）一个金色的福字，使用书法字体 3）背景添加一些烟花效果 4）在灯笼和福字周围添加一些祥云图案。确保图案布局美观，颜色搭配符合春节传统风格。"],
                ["SVGを使用して日本の伝統的な和柄パターンを描画してください。1）波紋（さざなみ）模様 2）市松模様 3）麻の葉模様 4）雷文（らいもん）模様を含めてください。色は伝統的な日本の色（藍色、朱色、金色など）を使用し、レイアウトはバランスよく配置してください。"],
                ["Write HTML with P5.js that simulates 25 particles in a vacuum space of a cylindrical container, bouncing within its boundaries. Use different colors for each ball and ensure they leave a trail showing their movement. Add a slow rotation of the container to give better view of what's going on in the scene. Make sure to create proper collision detection and physic rules to ensure particles remain in the container. Add an external spherical container. Add a slow zoom in and zoom out effect to the whole scene."],
                ["Write a Python script to scrape NVIDIA's stock price for the past month using the yfinance library. Clean the data and create an interactive visualization using Matplotlib. Include: 1) A candlestick chart showing daily price movements 2) A line chart with 7-day and 30-day moving averages. Add hover tooltips showing exact values and date. Make the layout professional with proper titles and axis labels."],
                ["Write a Python script that uses the Gradio library to create a functional calculator. The calculator should support basic arithmetic operations: addition, subtraction, multiplication, and division. It should have two input fields for numbers and a dropdown menu to select the operation."],
                ["Write a Todo list app using React.js. The app should allow users to add, delete, and mark tasks as completed. Include features like filtering tasks by status (completed, active), sorting tasks by priority, and displaying the total number of tasks."],
                ["Write a Python script using the Streamlit library to create a web application for uploading and displaying files. The app should allow users to upload files of type .csv or .txt. If a .csv file is uploaded, display its contents as a table using Streamlit's st.dataframe() method. If a .txt file is uploaded, display its content as plain text."],
                ["Write a Python function to solve the Trapping Rain Water problem. The function should take a list of non-negative integers representing the height of bars in a histogram and return the total amount of water trapped between the bars after raining. Use an efficient algorithm with a time complexity of O(n)."],
                ["Create a simple Pygame script for a game where the player controls a bouncing ball that changes direction when it collides with the edges of the window. Add functionality for the player to control a paddle using arrow keys, aiming to keep the ball from touching the bottom of the screen. Include basic collision detection and a scoring system that increases as the ball bounces off the paddle. You need to add clickable buttons to start the game, and reset the game."],
                ["Create a financial management Dashboard using Vue.js, focusing on local data handling without APIs. Include features like a clean dashboard for tracking income and expenses, dynamic charts for visualizing finances, and a budget planner. Implement functionalities for adding, editing, and deleting transactions, as well as filtering by date or category. Ensure responsive design and smooth user interaction for an intuitive experience."],
                ["Create a Mermaid diagram to visualize a flowchart of a user login process. Include the following steps: User enters login credentials; Credentials are validated; If valid, the user is directed to the dashboard; If invalid, an error message is shown, and the user can retry or reset the password."],
                ["Write a Python function to calculate the Fibonacci sequence up to n numbers. Then write test cases to verify the function works correctly for edge cases like negative numbers, zero, and large inputs."],
                ["Build an HTML page for a Kanban board with three columns with Vue.js: To Do, In Progress, and Done. Each column should allow adding, moving, and deleting tasks. Implement drag-and-drop functionality using Vue Draggable and persist the state using Vuex."],
                ["Develop a Streamlit app that takes a CSV file as input and provides: 1) Basic statistics about the data 2) Interactive visualizations using Plotly 3) A data cleaning interface with options to handle missing values 4) An option to download the cleaned data."],
                ["Write an HTML page with embedded JavaScript that creates an interactive periodic table. Each element should display its properties on hover and allow filtering by category (metals, non-metals, etc.). Include a search bar to find elements by name or symbol."],
                ["Here's a Python function that sorts a list of dictionaries by a specified key:\n\n```python\ndef sort_dicts(data, key):\n    return sorted(data, key=lambda x: x[key])\n```\n\nWrite test cases to verify the function works correctly for edge cases like empty lists, missing keys, and different data types. If you use unittest, please use `unittest.main(argv=['first-arg-is-ignored'], exit=False)` to run the tests."],
                ["Create a React component for a fitness tracker that shows: 1) Daily step count 2) Calories burned 3) Distance walked 4) A progress bar for daily goals."],
                ["Build a Vue.js dashboard for monitoring server health. Include: 1) Real-time CPU and memory usage graphs 2) Disk space visualization 3) Network activity monitor 4) Alerts for critical thresholds."],
                ["Write a C program that calculates and prints the first 100 prime numbers in a formatted table with 10 numbers per row. Include a function to check if a number is prime and use it in your solution."],
                ["Write a C++ program that implements a simple calculator using object-oriented programming. Create a Calculator class with methods for addition, subtraction, multiplication, and division. Include error handling for division by zero."],
                ["Write a Rust program that generates and prints a Pascal's Triangle with 10 rows. Format the output to center-align the numbers in each row."],
                ["Write a Java program that simulates a simple bank account system. Create a BankAccount class with methods for deposit, withdrawal, and balance inquiry. Include error handling for insufficient funds and demonstrate its usage with a few transactions."],
                ["Write a Go program that calculates and prints the Fibonacci sequence up to the 50th number. Format the output in a table with 5 numbers per row and include the index of each Fibonacci number."],
                ["Write a C program that calculates and prints a histogram of letter frequencies from a predefined string. Use ASCII art to display the histogram vertically."],
                ["Write a C++ program that implements a simple stack data structure with push, pop, and peek operations. Demonstrate its usage by reversing a predefined string using the stack."],
                ["Write a Rust program that calculates and prints the first 20 happy numbers. Include a function to check if a number is happy and use it in your solution."],
                ["Write a Java program that implements a simple binary search algorithm. Create a sorted array of integers and demonstrate searching for different values, including cases where the value is found and not found."],
                ["Write a Go program that generates and prints a multiplication table from 1 to 12. Format the output in a neat grid with proper alignment."],
            ],
            example_labels=[
                "🏮 春节主题图案",
                "🎎 日本の伝統的な和柄パターン",
                "🌐 Particles in a Spherical Container",
                "💹 NVIDIA Stock Analysis with Matplotlib",
                "🧮 Calculator with Gradio",
                "📝 Todo List App with React.js",
                "📂 File Upload Web App with Streamlit",
                "💦 Solve Trapping Rain Water Problem",
                "🎮 Pygame Bouncing Ball Game",
                "💳 Financial Dashboard with Vue.js",
                "🔑 User Login Process Flowchart",
                "🔢 Fibonacci Sequence with Tests",
                "📌 Vue Kanban Board",
                "🧹 Streamlit Data Cleaning App",
                "⚗️ Interactive Periodic Table with React",
                "📚 Dictionary Sorting Tests in Python",
                "🏋️‍♂️ Fitness Tracker with React",
                "🖥️ Vue Server Monitoring",
                "🔢 Prime Numbers in C",
                "🧮 OOP Calculator in C++",
                "🔷 Pascal's Triangle in Rust",
                "🏛️ Bank Account Simulation in Java",
                "🐰 Fibonacci Sequence in Go",
                "📊 Letter Frequency Histogram in C",
                "📦 Stack Implementation in C++",
                "😄 Happy Numbers in Rust",
                "🔎 Binary Search in Java",
                "✖️ Multiplication Table in Go",
            ],
            examples_per_page=100,
            label="Example Prompts",
            inputs = [multimodal_textbox],
        )

    # sandbox states and components
    sandbox_state = None # state for each chatbot
    sandboxes_components: list[SandboxGradioSandboxComponents] = [] # components for each chatbot
    sandbox_title = None
    sandbox_hidden_components = []

     # chatbot sandbox
    with gr.Group():
        # chatbot sandbox config
        with gr.Row():
            sandbox_env_choice = gr.Dropdown(choices=SUPPORTED_SANDBOX_ENVIRONMENTS, label="Programming Expert (Click to select!)", interactive=True, visible=True)
        
            with gr.Accordion("System Prompt (Click to edit!)", open=False) as system_prompt_accordion:
                system_prompt_textbox = gr.Textbox(
                    value=DEFAULT_SANDBOX_INSTRUCTIONS[SandboxEnvironment.AUTO],
                    show_label=False,
                    lines=15,
                    placeholder="Edit system prompt here",
                    interactive=True,
                    elem_id="system_prompt_box"
                )

        with gr.Group():
            with gr.Accordion("Sandbox & Output", open=True, visible=True) as sandbox_instruction_accordion:
                with gr.Group(visible=True) as sandbox_group:
                    sandbox_hidden_components.append(sandbox_group)
                    with gr.Row(visible=True) as sandbox_row:
                        sandbox_hidden_components.append(sandbox_row)
                        with gr.Column(scale=1, visible=True) as column:
                            sandbox_state = gr.State(create_chatbot_sandbox_state(btn_list_length=USER_BUTTONS_LENGTH))
                            # Add containers for the sandbox output
                            sandbox_title = gr.Markdown(
                                value=f"### Model Sandbox",
                                visible=True
                            )
                            sandbox_title = sandbox_title

                            with gr.Tab(label="Output", visible=True) as sandbox_output_tab:
                                sandbox_output = gr.Markdown(value="", visible=True)
                                sandbox_ui = SandboxComponent(
                                    value=('', False, []),
                                    show_label=True,
                                    visible=True,
                                )

                            # log sandbox telemetry
                            sandbox_ui.change(
                                fn=log_sandbox_telemetry_gradio_fn,
                                inputs=[sandbox_state, sandbox_ui],
                            )

                            with gr.Tab(label="Code Editor", visible=True) as sandbox_code_tab:
                                sandbox_code = gr.Code(
                                    value="",
                                    interactive=True, # allow user edit
                                    visible=False,
                                    label='Sandbox Code',
                                )
                                with gr.Row():
                                    sandbox_code_submit_btn = gr.Button(value="Apply Changes", visible=True, interactive=True, variant='primary', size='sm')

                            with gr.Tab(
                                label="Dependency Editor (Beta Mode)", visible=True
                            ) as sandbox_dependency_tab:
                                sandbox_dependency = gr.Dataframe(
                                    headers=["Type", "Package", "Version"],
                                    datatype=["str", "str", "str"],
                                    col_count=(3, "fixed"),
                                    interactive=True,
                                    visible=False,
                                    wrap=True,  # Enable text wrapping
                                    max_height=200,
                                    type="array",  # Add this line to fix the error
                                )
                                with gr.Row():
                                    dependency_submit_btn = gr.Button(
                                        value="Apply Changes",
                                        visible=True,
                                        interactive=True,
                                        variant='primary',
                                        size='sm'
                                    )
                                dependency_submit_btn.click(
                                    fn=on_edit_dependency,
                                    inputs=[
                                        state,
                                        sandbox_state,
                                        sandbox_dependency,
                                        sandbox_output,
                                        sandbox_ui,
                                        sandbox_code,
                                    ],
                                    outputs=[
                                        sandbox_output,
                                        sandbox_ui,
                                        sandbox_code,
                                        sandbox_dependency,
                                    ],
                                )
                            # run code when click apply changes
                            sandbox_code_submit_btn.click(
                                fn=on_edit_code,
                                inputs=[
                                    state,
                                    sandbox_state,
                                    sandbox_output,
                                    sandbox_ui,
                                    sandbox_code,
                                    sandbox_dependency,
                                ],
                                outputs=[
                                    sandbox_output,
                                    sandbox_ui,
                                    sandbox_code,
                                    sandbox_dependency,
                                ],
                            )

                            sandboxes_components.append(
                                (
                                    sandbox_output,
                                    sandbox_ui,
                                    sandbox_code,
                                    sandbox_dependency,
                                )
                            )
                            sandbox_hidden_components.extend(
                                [
                                    column,
                                    sandbox_title,
                                    sandbox_output_tab,
                                    sandbox_code_tab,
                                    sandbox_dependency_tab,
                                ]
                            )

    with gr.Row():
        send_btn = gr.Button(
            value="⬆️  Send",
            variant="primary",
        )
        regenerate_btn = gr.Button(
            value="🔄  Regenerate",
            interactive=False,
            visible=False
        )


    with gr.Row(elem_id="buttons"):
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)

    with gr.Row():
        clear_btn = gr.Button(value="🎲 New Round", interactive=False)
        share_btn = gr.Button(value="📷  Share")

    with gr.Accordion("Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=0,
            maximum=4096,
            value=4096,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    # Notice
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    # Ack
    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register listeners

    # list of user buttons
    user_buttons: list[gr.Button] = [
        # send button
        send_btn,
        # regenerate button
        regenerate_btn,
        # vote buttons
        upvote_btn, downvote_btn, flag_btn,
        # clear button
        clear_btn,
    ] # 6 buttons, USER_BUTTONS_LENGTH

    upvote_btn.click(
        upvote_last_response,
        [state, model_selector, sandbox_state],
        [textbox] + user_buttons,
    )
    downvote_btn.click(
        downvote_last_response,
        [state, model_selector, sandbox_state],
        [textbox] + user_buttons,
    )
    flag_btn.click(
        flag_last_response,
        [state, model_selector, sandbox_state],
        [textbox] + user_buttons,
    )

    regenerate_btn.click(
        regenerate,
        [state, sandbox_state],
        [state, chatbot, textbox] + user_buttons
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens, sandbox_state],
        [state, chatbot] + user_buttons,
    )

    clear_btn.click(
      clear_history,
        sandbox_state,
        [state, chatbot, sandbox_state] + [multimodal_textbox, textbox] + user_buttons,
    ).then(
        # clear existing sandbox components
        clear_sandbox_components,
        inputs=[sandbox_output, sandbox_ui, sandbox_code],
        outputs=[sandbox_output, sandbox_ui, sandbox_code]
    ).then(
        # reset env and system prompt and enable model selector
        lambda: (
            gr.update(interactive=True, value=SandboxEnvironment.AUTO),
            gr.update(interactive=True, value=DEFAULT_SANDBOX_INSTRUCTIONS[SandboxEnvironment.AUTO]),
            gr.update(interactive=True)
        ),
        outputs=[sandbox_env_choice, system_prompt_textbox, model_selector]
    ).then(
        lambda: gr.update(visible=True),
        inputs=None,
        outputs=examples_row
    )

    model_selector.change(
        clear_history,
        sandbox_state,
        [state, chatbot, sandbox_state] + [multimodal_textbox, textbox] + user_buttons,
    ).then(
        set_visible_image, [multimodal_textbox], [image_column]
    ).then(
        clear_sandbox_components,
        inputs=[sandbox_output, sandbox_ui, sandbox_code],
        outputs=[sandbox_output, sandbox_ui, sandbox_code]
    ).then(
        # enable env and prompt edit
        fn=lambda: [
            gr.update(interactive=True),
            gr.update(interactive=True)
        ],
        outputs=[system_prompt_textbox, sandbox_env_choice]
    )

    multimodal_textbox.input(
        add_image, [multimodal_textbox], [imagebox]
    ).then(
        set_visible_image, [multimodal_textbox], [image_column]
    )

    multimodal_textbox.submit(
        update_system_prompt,
        inputs=[system_prompt_textbox, sandbox_state],
        outputs=[sandbox_state]
    ).then(
        add_text,
        [state, model_selector, sandbox_state] + [multimodal_textbox, textbox] + [context_state],
        [state, chatbot, sandbox_state] + [multimodal_textbox, textbox] + user_buttons,
    ).then(
        set_invisible_image, [], [image_column]
    ).then(
        set_chat_system_messages,
        inputs=[state, sandbox_state, model_selector],
        outputs=[state, chatbot, system_prompt_textbox]
    ).then(
        # hide examples
        lambda sandbox_state: gr.update(visible=sandbox_state['enabled_round'] == 0),
        inputs=sandbox_state,
        outputs=examples_row
    ).then(
        # disable env and prompt change and model selector
        lambda sandbox_state: (gr.update(interactive=sandbox_state['enabled_round'] == 0),) * 3,
        inputs=sandbox_state,
        outputs=[sandbox_env_choice, system_prompt_textbox, model_selector]
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens, sandbox_state],
        [state, chatbot] + user_buttons,
    )

    textbox.submit(
        update_system_prompt,
        inputs=[system_prompt_textbox, sandbox_state],
        outputs=[sandbox_state]
    ).then(
        add_text,
        [state, model_selector, sandbox_state] + [multimodal_textbox, textbox] + [context_state],
        [state, chatbot, sandbox_state] + [multimodal_textbox, textbox] + user_buttons,
    ).then(
        set_chat_system_messages,
        [state, sandbox_state, model_selector],
        [state, chatbot]
    ).then(
        set_invisible_image, [], [image_column]
    ).then(
        # hide examples
        lambda sandbox_state: gr.update(visible=sandbox_state['enabled_round'] == 0),
        inputs=sandbox_state,
        outputs=examples_row
    ).then(
        # disable env and prompt change and model selector
        lambda sandbox_state: (gr.update(interactive=sandbox_state['enabled_round'] == 0),) * 3,
        inputs=sandbox_state,
        outputs=[sandbox_env_choice, system_prompt_textbox, model_selector]
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens] + [sandbox_state],
        [state, chatbot] + user_buttons,
    )

    send_btn.click(
        update_system_prompt,
        inputs=[system_prompt_textbox, sandbox_state],
        outputs=[sandbox_state]
    ).then(
        add_text,
        [state, model_selector, sandbox_state] + [multimodal_textbox, textbox] + [context_state],
        [state, chatbot, sandbox_state] + [multimodal_textbox, textbox] + user_buttons,
    ).then(
        set_invisible_image, [], [image_column]
    ).then(
        set_chat_system_messages,
        [state, sandbox_state, model_selector],
        [state, chatbot, system_prompt_textbox]
    ).then(
        # hide examples
        lambda sandbox_state: gr.update(visible=sandbox_state['enabled_round'] == 0),
        inputs=sandbox_state,
        outputs=examples_row
    ).then(
        # disable env and prompt change and model selector
        lambda sandbox_state: (gr.update(interactive=sandbox_state['enabled_round'] == 0),) * 3,
        inputs=sandbox_state,
        outputs=[sandbox_env_choice, system_prompt_textbox, model_selector]
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens, sandbox_state],
        [state, chatbot] + user_buttons,
    )

    sandbox_env_choice.change(
        fn=update_sandbox_config,
        inputs=[
            gr.State(value=True),  # Always enabled
            sandbox_env_choice,
            sandbox_state
        ],
        outputs=[sandbox_state]
    ).then(
        # update system prompt when env choice changes
        fn=lambda sandbox_state: gr.update(value=sandbox_state['sandbox_instruction']),
        inputs=[sandbox_state],
        outputs=[system_prompt_textbox]
    )

    # update system prompt when textbox changes
    system_prompt_textbox.change(
        # update sandbox state
        fn=lambda system_prompt_textbox, sandbox_state: update_sandbox_state_system_prompt(sandbox_state, system_prompt_textbox),
        inputs=[system_prompt_textbox, sandbox_state],
        outputs=[sandbox_state]
    )

    # trigger sandbox run when click the code message
    sandbox_components = sandboxes_components[0]
    chatbot.select(
        fn=on_click_code_message_run,
        inputs=[state, sandbox_state, *sandbox_components],
        outputs=[*sandbox_components]
    )

    return [state, model_selector]
