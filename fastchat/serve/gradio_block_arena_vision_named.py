"""
Multimodal Chatbot Arena (side-by-side) tab.
Users chat with two chosen models.
"""

import json
import os
import time
from typing import Any, List, Union

import gradio as gr
from gradio_sandboxcomponent import SandboxComponent
import numpy as np

from fastchat.constants import (
    LOGDIR,
    TEXT_MODERATION_MSG,
    IMAGE_MODERATION_MSG,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    SLOW_MODEL_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SURVEY_LINK,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_block_arena_named import (
    clear_sandbox_components,
    flash_buttons,
    set_chat_system_messages_multi,
    share_click,
    bot_response_multi,
)
from fastchat.serve.gradio_block_arena_vision import (
    set_invisible_image,
    set_visible_image,
    add_image,
    moderate_input,
    _prepare_text_with_image,
    convert_images_to_conversation_format,
    enable_multimodal,
    disable_multimodal,
    invisible_text,
    invisible_btn,
    visible_text,
)
from fastchat.serve.gradio_global_state import Context
from fastchat.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
    acknowledgment_md,
    get_ip,
    get_model_description_md,
    enable_text,
    set_chat_system_messages,
)
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.serve.sandbox.code_analyzer import SandboxEnvironment
from fastchat.serve.sandbox.code_runner import DEFAULT_SANDBOX_INSTRUCTIONS, SUPPORTED_SANDBOX_ENVIRONMENTS, ChatbotSandboxState, create_chatbot_sandbox_state, on_click_code_message_run, on_edit_code, on_edit_dependency, reset_sandbox_state, update_sandbox_config_multi, update_sandbox_state_system_prompt
from fastchat.serve.sandbox.sandbox_telemetry import log_sandbox_telemetry_gradio_fn, upload_conv_log_to_azure_storage
from fastchat.utils import (
    build_logger,
    moderation_filter,
    image_moderation_filter,
)


logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
enable_moderation = False
USER_BUTTONS_LENGTH = 11


def load_demo_side_by_side_vision_named(context: Context):
    states = [None] * num_sides

    # default to the text models
    models = context.text_models

    model_left = models[0] if len(models) > 0 else ""
    if len(models) > 1:
        weights = ([1] * 128)[: len(models) - 1]
        weights = weights / np.sum(weights)
        model_right = np.random.choice(models[1:], p=weights)
    else:
        model_right = model_left

    all_models = context.models
    selector_updates = [
        gr.Dropdown(choices=all_models, value=model_left, visible=True),
        gr.Dropdown(choices=all_models, value=model_right, visible=True),
    ]

    return states + selector_updates


def clear_history_example(request: gr.Request):
    logger.info(f"clear_history_example (named). ip: {get_ip(request)}")
    return (
        [None] * num_sides
        + [None] * num_sides
        + [enable_multimodal, invisible_text, invisible_btn]
        + [invisible_btn] * 4
        + [disable_btn] * 2
    )


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    if states[0] is None or states[1] is None:
        return
    filename = get_conv_log_filename(states[0].is_vision, states[0].has_csam_image)
    with open(filename, "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)
    upload_conv_log_to_azure_storage(filename.lstrip(LOGDIR), json.dumps(data) + "\n")

    gr.Info(
        "🎉 Thanks for voting! Your vote shapes the leaderboard, please vote RESPONSIBLY."
    )


def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, sandbox_state0, sandbox_state1, request: gr.Request
):
    logger.info(f"leftvote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    )
    return (None,) + (disable_btn,) * (sandbox_state0['btn_list_length'] - 1) + (enable_btn,) # enable clear button


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, sandbox_state0, sandbox_state1, request: gr.Request
):
    logger.info(f"rightvote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    )
    return (None,) + (disable_btn,) * (sandbox_state0['btn_list_length'] - 1) + (enable_btn,) # enable clear button


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, sandbox_state0, sandbox_state1, request: gr.Request
):
    logger.info(f"tievote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    )
    return (None,) + (disable_btn,) * (sandbox_state0['btn_list_length'] - 1) + (enable_btn,) # enable clear button


def bothbad_vote_last_response(
    state0, state1, model_selector0, model_selector1, sandbox_state0, sandbox_state1, request: gr.Request
):
    logger.info(f"bothbad_vote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "bothbad_vote", [model_selector0, model_selector1], request
    )
    return (None,) + (disable_btn,) * (sandbox_state0['btn_list_length'] - 1) + (enable_btn,) # enable clear button


def regenerate_single(state, sandbox_state, request: gr.Request):
    '''
    Regenerate message for one side.
    '''
    logger.info(f"regenerate. ip: {get_ip(request)}")
    if state is None:
        # if not init yet
        return [None, None] + [None] + [no_change_btn] * sandbox_state['btn_list_length']
    elif state.regen_support:
        state.conv.update_last_message(None)
        return (
            [state, state.to_gradio_chatbot()]
            + [None]
            + [disable_btn] * sandbox_state['btn_list_length']
        )
    else:
        # if not support regen
        state.skip_next = True
        return [state, state.to_gradio_chatbot()] + [None] + [no_change_btn] * sandbox_state['btn_list_length']


def regenerate_multi(state0, state1, sandbox_state0, sandbox_state1, request: gr.Request):
    logger.info(f"regenerate (named). ip: {get_ip(request)}")
    states = [state0, state1]
    if state0.regen_support and state1.regen_support:
        for i in range(num_sides):
            states[i].conv.update_last_message(None)
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [None]
            + [disable_btn] * sandbox_state0['btn_list_length']
        )
    else:
        states[0].skip_next = True
        states[1].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [None]
            + [no_change_btn] * sandbox_state0['btn_list_length']
        )


def clear_history(sandbox_state0, sandbox_state1, request: gr.Request) -> List[dict[str, Any] | gr.Button | gr.MultimodalTextbox | gr.Textbox | None]:
    '''
    Clear chat history for both sides.
    '''
    logger.info(f"clear_history (named). ip: {get_ip(request)}")

    # reset sandbox state
    sandbox_states = [
        reset_sandbox_state(sandbox_state) for sandbox_state in [sandbox_state0, sandbox_state1]
    ]

    return (
        sandbox_states
        + [None] * num_sides
        + [None] * num_sides
        + [enable_multimodal, invisible_text]
        + [enable_btn, invisible_btn, invisible_btn]  # send_btn, send_btn_left, send_btn_right
        + [invisible_btn] * 3  # regenerate, regenerate left/right
        + [invisible_btn] * 4  # vote buttons
        + [disable_btn]  # clear
    )


def add_text_single(
    state: State,
    model_selector: str,
    sandbox_state: ChatbotSandboxState,
    multimodal_input: dict, text_input: str,
    context: Context,
    request: gr.Request,
):
    '''
    Add text for one side.
    '''
    if multimodal_input and multimodal_input["text"]:
        text, images = multimodal_input["text"], multimodal_input["files"]
    else:
        text = text_input
        images = []

    # whether need vision models
    is_vision = len(images) > 0

    # check if the model is vision model
    if is_vision:
        if (
            model_selector in context.text_models
            and model_selector not in context.vision_models
        ):
            gr.Warning(f"{model_selector} is a text-only model. Image is ignored.")
            images = []

    ip = get_ip(request)
    logger.info(f"add_text (named). ip: {ip}. len: {len(text)}")

    # increase sandbox state
    sandbox_state['enabled_round'] += 1

    # Init states if necessary
    if state is None:
        state = State(model_selector, is_vision=is_vision)

    if len(text) <= 0:
        state.skip_next = True
        return (
            [state, state.to_gradio_chatbot(), sandbox_state]
            + [None, ""]
            + [no_change_btn,] * sandbox_state['btn_list_length']
        )

    model_list = [state.model_name]
    all_conv_text_left = state.conv.get_prompt()
    all_conv_text = (
        all_conv_text_left[-1000:] + "\nuser: " + text
    )

    images = convert_images_to_conversation_format(images)

    # TODO: Skip moderation for now
    # text, image_flagged, csam_flag = moderate_input(
    #     state, text, all_conv_text, model_list, images, ip
    # )
    image_flagged, csam_flag = None, None

    conv = state.conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        state.skip_next = True
        return (
            [state, state.to_gradio_chatbot(), sandbox_state]
            + [{"text": CONVERSATION_LIMIT_MSG}, ""]
            + [no_change_btn,] * sandbox_state['btn_list_length']
        )

    if image_flagged:
        logger.info(f"image flagged. ip: {ip}. text: {text}")
        state.skip_next = True
        return (
            [state, state.to_gradio_chatbot(), sandbox_state]
            + [{"text": IMAGE_MODERATION_MSG}, ""]
            + [no_change_btn] * sandbox_state['btn_list_length']
        )

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    post_processed_text = _prepare_text_with_image(
        state, text, images, csam_flag=csam_flag
    )
    state.conv.append_message(state.conv.roles[0], post_processed_text)
    state.conv.append_message(state.conv.roles[1], None)
    state.skip_next = False

    return (
        [state, state.to_gradio_chatbot(), sandbox_state]
        + [disable_multimodal, visible_text]
        + [disable_btn,] * sandbox_state['btn_list_length']
    )


def add_text_multi(
    state0, state1,
    model_selector0, model_selector1,
    sandbox_state0, sandbox_state1,
    multimodal_input: dict, text_input: str,
    context: Context,
    request: gr.Request,
):
    if multimodal_input and multimodal_input["text"]:
        text, images = multimodal_input["text"], multimodal_input["files"]
    else:
        text = text_input
        images = []

    # whether need vision models
    is_vision = len(images) > 0

    if is_vision:
        if (
            model_selector0 in context.text_models
            and model_selector0 not in context.vision_models
        ):
            gr.Warning(f"{model_selector0} is a text-only model. Image is ignored.")
            images = []
        if (
            model_selector1 in context.text_models
            and model_selector1 not in context.vision_models
        ):
            gr.Warning(f"{model_selector1} is a text-only model. Image is ignored.")
            images = []

    ip = get_ip(request)
    logger.info(f"add_text (named). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]
    sandbox_states = [sandbox_state0, sandbox_state1]

    # increase sandbox state
    sandbox_state0['enabled_round'] += 1
    sandbox_state1['enabled_round'] += 1

    # Init states if necessary
    for i in range(num_sides):
        if states[i] is None:
            states[i] = State(model_selectors[i], is_vision=is_vision)

    if len(text) <= 0:
        # skip if no text
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + sandbox_states
            + [None, ""]
            + [no_change_btn,] * sandbox_state0['btn_list_length']
        )

    model_list = [states[i].model_name for i in range(num_sides)]
    all_conv_text_left = states[0].conv.get_prompt()
    all_conv_text_right = states[0].conv.get_prompt()
    all_conv_text = (
        all_conv_text_left[-1000:] + all_conv_text_right[-1000:] + "\nuser: " + text
    )

    images = convert_images_to_conversation_format(images)

    # TODO: Skip moderation for now
    # text, image_flagged, csam_flag = moderate_input(
    #     state0, text, all_conv_text, model_list, images, ip
    # )
    image_flagged, csam_flag = None, None

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + sandbox_states
            + [{"text": CONVERSATION_LIMIT_MSG}, ""]
            + [no_change_btn,] * sandbox_state0['btn_list_length']
        )

    if image_flagged:
        logger.info(f"image flagged. ip: {ip}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + sandbox_states
            + [{"text": IMAGE_MODERATION_MSG}, ""]
            + [no_change_btn,] * sandbox_state0['btn_list_length']
        )

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        post_processed_text = _prepare_text_with_image(
            states[i], text, images, csam_flag=csam_flag
        )
        states[i].conv.append_message(states[i].conv.roles[0], post_processed_text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + sandbox_states
        + [disable_multimodal, visible_text]
        + [disable_btn] * sandbox_state0['btn_list_length']
    )


def build_side_by_side_vision_ui_named(context: Context, random_questions=None):
    notice_markdown = f"""
## How It Works for Side-by-Side Chat Mode
- **Choose Models**: Select two AI chatbots to chat with.
- **Run & Interact**: The AI chatbots generate programs that run in a secure sandbox environment. Test the functionality, explore the features, and evaluate the quality of the outputs.
- **Edit & Regenerate**: You can edit the <u>system prompt</u>, <u>code</u>, and its <u>dependency</u> and <u>regenerate the code</u> on any side.
- **Visual Input**: Upload images or provide text prompts to guide the AI chatbots in their responses. You can only chat with <span style='color: #DE3163; font-weight: bold'>one image per conversation</span>. The image should be less than 15MB.
- **Vote for the Best**: After interacting with both programs, vote for the one that best meets your requirements or provides the superior experience.

## Note
- **Sandbox**: There are four buttons on the top right corner of the sandbox inference. You can use them to copy the link, turn on/off the black mode, make the sandbox full screen, and reload the sandbox. Plus, you can also drag the bottom right corner of the sandbox to resize the sandbox.
- **Dependency Edit**: You can edit the <u>dependency</u> of the code on any side. Currently, we only support `pypi` and `npm` packages.
For `pypi` packages, you can use the format `python (use '==', '>=', '<=', '~=', '>', '<' or 'latest') <package_name> <version>`.
For `npm` packages, you can use the format `npm (use '@' or 'latest') <package_name> <version>`.
- **Temperature**: All models have the same temperature of `0.2` and `top_p` of `0.9` by default. You can adjust the hyperparameters in the `Parameters` section. Low temperature typically works better for code generation.

**❗️ For research purposes, we log user prompts, images, and interactions with sandbox, and may release this data to the public in the future. Please do not upload any confidential or personal information.**
"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors: list[gr.Markdown | None] = [None] * num_sides
    chatbots: list[gr.Chatbot | None] = [None] * num_sides
    text_and_vision_models = context.models
    context_state = gr.State(context)

    with gr.Row():
        with gr.Column(scale=2, visible=False) as image_column:
            imagebox = gr.Image(
                type="pil",
                show_label=False,
                interactive=False,
            )

        with gr.Column(scale=5):
            with gr.Accordion(
                f"🔍 Expand to see the descriptions of {len(text_and_vision_models)} models",
                open=False,
            ):
                model_description_md = get_model_description_md(
                    text_and_vision_models
                )
                gr.Markdown(
                    model_description_md, elem_id="model_description_markdown"
                )
            with gr.Group(elem_id="share-region-anony"):
                with gr.Row():
                    for i in range(num_sides):
                        with gr.Column():
                            model_selectors[i] = gr.Dropdown(
                                choices=text_and_vision_models,
                                value=text_and_vision_models[i]
                                if len(text_and_vision_models) > i
                                else "",
                                interactive=True,
                                show_label=False,
                                container=False,
                            )

                with gr.Row(
                    elem_classes=["chatbot-section"]
                ):
                    for i in range(num_sides):
                        label = "Model A" if i == 0 else "Model B"
                        with gr.Column():
                            chatbots[i] = gr.Chatbot(
                                label=label,
                                elem_id=f"chatbot",
                                height='65vh',
                                show_copy_button=True,
                                sanitize_html=False,
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
    sandbox_states: list[gr.State] = [] # state for each chatbot
    sandboxes_components: list = [] # components for each chatbot
    sandbox_titles = [None] * num_sides
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
                        for chatbotIdx in range(num_sides):
                            with gr.Column(scale=1, visible=True) as column:
                                sandbox_state = gr.State(create_chatbot_sandbox_state(btn_list_length=USER_BUTTONS_LENGTH))
                                # Add containers for the sandbox output
                                sandbox_titles[chatbotIdx] = gr.Markdown(
                                    value=f"### Model {chr(ord('A') + chatbotIdx)} Sandbox",
                                    visible=True
                                )
                                sandbox_title = sandbox_titles[chatbotIdx]

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
                                    label="Dependency Editor Editor", visible=True
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
                                            states[chatbotIdx],
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
                                        states[chatbotIdx],
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

                                sandbox_states.append(sandbox_state)
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
        send_btn_left = gr.Button(
            value="⬅️  Send to Left",
            variant="primary",
            visible=False,
        )
        send_btn = gr.Button(
            value="⬆️  Send",
            variant="primary",
        )
        send_btn_right = gr.Button(
            value="➡️  Send to Right",
            variant="primary",
            visible=False,
        )
        send_btns_one_side = [send_btn_left, send_btn_right]

    with gr.Row():
        left_regenerate_btn = gr.Button(value="🔄  Regenerate Left", interactive=False, visible=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False, visible=False)
        right_regenerate_btn = gr.Button(value="🔄  Regenerate Right", interactive=False, visible=False)
        regenerate_one_side_btns = [left_regenerate_btn, right_regenerate_btn]

    with gr.Row():
        leftvote_btn = gr.Button(
            value="👈  A is better", visible=False, interactive=False
        )
        tie_btn = gr.Button(
            value="🤝  Tie", visible=False, interactive=False
        )
        rightvote_btn = gr.Button(
            value="👉  B is better", visible=False, interactive=False
        )
        bothbad_btn = gr.Button(
            value="👎  Both are bad", visible=False, interactive=False
        )
    
    with gr.Row():
        clear_btn = gr.Button(value="🎲 New Round", interactive=False)
        share_btn = gr.Button(value="📷  Share")

    with gr.Accordion("Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.2,
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
            minimum=16,
            maximum=4096,
            value=4096,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    # Notice
    gr.Markdown(notice_markdown, elem_id="notice_markdown")
    # Model descriptions

    # Ack
    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register listeners

    # list of user buttons
    user_buttons: list[gr.Button] = [
        # send buttons
        send_btn, send_btn_left, send_btn_right,
        # regenerate buttons
        regenerate_btn, left_regenerate_btn, right_regenerate_btn,
        # vote buttons
        leftvote_btn, rightvote_btn, tie_btn, bothbad_btn,
        # clear button
        clear_btn,
    ] # 11 buttons, USER_BUTTONS_LENGTH

    leftvote_btn.click(
        leftvote_last_response,
        states + model_selectors + sandbox_states,
        [textbox] + user_buttons,
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors + sandbox_states,
        [textbox] + user_buttons,
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors + sandbox_states,
        [textbox] + user_buttons,
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors + sandbox_states,
        [textbox] + user_buttons,
    )

    regenerate_btn.click(
        regenerate_multi,
        states + sandbox_states,
        states + chatbots + [textbox] + user_buttons
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens] + sandbox_states,
        states + chatbots + user_buttons,
    ).then(
        flash_buttons, [], user_buttons
    )

    clear_btn.click(
        clear_history,
        inputs=sandbox_states,
        outputs=(
            sandbox_states
            + states
            + chatbots
            + [multimodal_textbox, textbox]
            + user_buttons
        ),
    ).then(
        clear_sandbox_components,
        inputs=[component for components in sandboxes_components for component in components],
        outputs=[component for components in sandboxes_components for component in components]
    ).then(
        lambda: (gr.update(interactive=True, value=SandboxEnvironment.AUTO), gr.update(interactive=True, value=DEFAULT_SANDBOX_INSTRUCTIONS[SandboxEnvironment.AUTO])),
        outputs=[sandbox_env_choice, system_prompt_textbox]
    ).then(
        # Re-enable model selectors and show examples
        lambda: [gr.update(visible=True), gr.update(interactive=True), gr.update(interactive=True)],
        outputs=[examples_row, model_selectors[0], model_selectors[1]]
    )

    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-named');
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'chatbot-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [a, b, c, d];
}
"""
    share_btn.click(share_click, states + model_selectors, [], js=share_js)

    for i in range(num_sides):
        model_selectors[i].change(
            clear_history,
            inputs=sandbox_states,
            outputs=(
                sandbox_states
                + states
                + chatbots
                + [multimodal_textbox, textbox]
                + user_buttons
            ),
        ).then(set_visible_image, [multimodal_textbox], [image_column])

    multimodal_textbox.input(
        add_image,
        [multimodal_textbox],
        [imagebox]
    ).then(
        set_visible_image, [multimodal_textbox], [image_column]
    )

    multimodal_textbox.submit(
        add_text_multi,
        inputs=states + model_selectors + sandbox_states + [multimodal_textbox, textbox] + [context_state],
        outputs=states + chatbots + sandbox_states + [multimodal_textbox, textbox] + user_buttons,
    ).then(
        set_invisible_image, [], [image_column]
    ).then( # set the system prompt
        set_chat_system_messages_multi,
        states + sandbox_states + model_selectors,
        states + chatbots
    ).then(
        # hide the examples row and disable model selectors
        lambda: [gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=False)],
        outputs=[examples_row, model_selectors[0], model_selectors[1]]
    ).then(
        fn=lambda: [
            gr.update(interactive=False),
            gr.update(interactive=False),
        ],
        outputs=[system_prompt_textbox, sandbox_env_choice]
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens] + sandbox_states,
        states + chatbots + user_buttons,
    ).then(
        flash_buttons, [], user_buttons
    )

    textbox.submit(
        add_text_multi,
        inputs=states + model_selectors + sandbox_states + [multimodal_textbox, textbox] + [context_state],
        outputs=states + chatbots + sandbox_states + [multimodal_textbox, textbox] + user_buttons,
    ).then(
        set_invisible_image, [], [image_column]
    ).then(
        # set the system prompt
        set_chat_system_messages_multi,
        states + sandbox_states + model_selectors,
        states + chatbots
    ).then(
        # hide the examples row and disable model selectors
        lambda: [gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=False)],
        outputs=[examples_row, model_selectors[0], model_selectors[1]]
    ).then(
        fn=lambda: [
            gr.update(interactive=False),
            gr.update(interactive=False),
        ],
        outputs=[system_prompt_textbox, sandbox_env_choice]
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens] + sandbox_states,
        states + chatbots + user_buttons,
    ).then(
        flash_buttons, [], user_buttons
    )

    send_btn.click(
        add_text_multi,
        inputs=states + model_selectors + sandbox_states + [multimodal_textbox, textbox] + [context_state],
        outputs=states + chatbots + sandbox_states + [multimodal_textbox, textbox] + user_buttons,
    ).then(
        set_invisible_image, [], [image_column]
    ).then(
        # set the system prompt
        set_chat_system_messages_multi,
        states + sandbox_states + model_selectors,
        states + chatbots
    ).then(
        # hide examples row and disable model selectors
        lambda: [gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=False)],
        outputs=[examples_row, model_selectors[0], model_selectors[1]]
    ).then(
        fn=lambda: [
            gr.update(interactive=False),
            gr.update(interactive=False),
        ],
        outputs=[system_prompt_textbox, sandbox_env_choice]
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens] + sandbox_states,
        states + chatbots + user_buttons,
    ).then(
        flash_buttons, [], user_buttons
    )

    # update state when env choice changes
    sandbox_env_choice.change(
        # update sandbox state
        fn=update_sandbox_config_multi,
        inputs=[
            gr.State(value=True),
            sandbox_env_choice,
            *sandbox_states
        ],
        outputs=[*sandbox_states]
    ).then(
        # update system prompt when env choice changes
        fn=lambda sandbox_state: gr.update(value=sandbox_state['sandbox_instruction']),
        inputs=[sandbox_states[0]],
        outputs=[system_prompt_textbox]
    )

    # update system prompt when textbox changes
    system_prompt_textbox.change(
        # update sandbox state
        fn=lambda system_prompt_textbox, sandbox_state0, sandbox_state1: [
            update_sandbox_state_system_prompt(state, system_prompt_textbox) for state in (sandbox_state0, sandbox_state1)
        ],
        inputs=[system_prompt_textbox, sandbox_states[0], sandbox_states[1]],
        outputs=[sandbox_states[0], sandbox_states[1]]
    )

    for chatbotIdx in range(num_sides):
        chatbot = chatbots[chatbotIdx]
        state = states[chatbotIdx]
        sandbox_state = sandbox_states[chatbotIdx]
        sandbox_components = sandboxes_components[chatbotIdx]
        model_selector = model_selectors[chatbotIdx]

        send_btns_one_side[chatbotIdx].click(
            add_text_single,
            inputs=[state, model_selector, sandbox_state] + [multimodal_textbox, textbox] + [context_state],
            outputs=(
                [state, chatbot, sandbox_state]
                + [textbox, multimodal_textbox]
                + user_buttons
            ),
        ).then(
            set_chat_system_messages,
            [state, sandbox_state, model_selector],
            [state, chatbot]
        ).then(
            lambda: gr.update(visible=False),
            inputs=None,
            outputs=examples_row
        ).then(
            bot_response,
            [state, temperature, top_p, max_output_tokens, sandbox_state],
            [state, chatbot] + user_buttons,
        ).then(
            flash_buttons, [], user_buttons
        ).then(
            fn=lambda: [
                gr.update(interactive=False),
                gr.update(interactive=False)
            ],
            outputs=[system_prompt_textbox, sandbox_env_choice]
        )

        regenerate_one_side_btns[chatbotIdx].click(
            regenerate_single,
            [state, sandbox_state],
            [state, chatbot, textbox] + user_buttons
        ).then(
            bot_response,
            [state, temperature, top_p, max_output_tokens, sandbox_state],
            [state, chatbot] + user_buttons,
        ).then(
            flash_buttons, [], user_buttons
        )

        # trigger sandbox run when click code message
        chatbot.select(
            fn=on_click_code_message_run,
            inputs=[state, sandbox_state, *sandbox_components],
            outputs=[*sandbox_components],
        )

    return states + model_selectors
