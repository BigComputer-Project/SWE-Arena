"""
Chatbot Arena (battle) tab.
Users chat with two anonymous models.
"""

import json
import time

import gradio as gr
from gradio_sandboxcomponent import SandboxComponent
import numpy as np
from typing import Union

from fastchat.constants import (
    LOGDIR,
    TEXT_MODERATION_MSG,
    IMAGE_MODERATION_MSG,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    SLOW_MODEL_MSG,
    BLIND_MODE_INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SURVEY_LINK,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_block_arena_named import flash_buttons, set_chat_system_messages_multi
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
    disable_text,
    enable_text,
    set_chat_system_messages,
)
from fastchat.serve.gradio_block_arena_anony import (
    clear_sandbox_components,
    flash_buttons,
    share_click,
    bot_response_multi,
    get_battle_pair,
)
from fastchat.serve.gradio_block_arena_vision import (
    set_invisible_image,
    set_visible_image,
    add_image,
    moderate_input,
    enable_multimodal,
    _prepare_text_with_image,
    convert_images_to_conversation_format,
    invisible_text,
    visible_text,
    disable_multimodal,
)
from fastchat.serve.gradio_global_state import Context
from fastchat.serve.model_sampling import (
    VISION_BATTLE_TARGETS,
    VISION_OUTAGE_MODELS,
    VISION_SAMPLING_BOOST_MODELS,
    VISION_SAMPLING_WEIGHTS,
    SAMPLING_WEIGHTS,
    BATTLE_TARGETS,
    SAMPLING_BOOST_MODELS,
    OUTAGE_MODELS,
)
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.serve.sandbox.sandbox_telemetry import upload_conv_log_to_azure_storage
from fastchat.serve.sandbox.code_runner import SUPPORTED_SANDBOX_ENVIRONMENTS, ChatbotSandboxState, SandboxEnvironment, DEFAULT_SANDBOX_INSTRUCTIONS, SandboxGradioSandboxComponents, create_chatbot_sandbox_state, on_click_code_message_run, on_edit_code, on_edit_dependency, reset_sandbox_state, update_sandbox_config_multi, update_sandbox_state_system_prompt
from fastchat.serve.sandbox.sandbox_telemetry import log_sandbox_telemetry_gradio_fn
from fastchat.utils import (
    build_logger,
    moderation_filter,
    image_moderation_filter,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
enable_moderation = False
anony_names = ["", ""]
text_models = []
vl_models = []

# Number of user buttons
USER_BUTTONS_LENGTH = 11

def load_demo_side_by_side_vision_anony():
    states = [None] * num_sides
    selector_updates = [
        gr.Markdown(visible=True),
        gr.Markdown(visible=True),
    ]

    return states + selector_updates


def vote_last_response(states, sandbox_states: list[ChatbotSandboxState], vote_type, model_selectors, request: gr.Request):
    '''
    Return
        model_selectors + sandbox_titles + [textbox] + user_buttons
    '''
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

    # display model names
    model_name_1 = states[0].model_name
    model_name_2 = states[1].model_name
    model_name_map = {}

    if model_name_1 in model_name_map:
        model_name_1 = model_name_map[model_name_1]
    if model_name_2 in model_name_map:
        model_name_2 = model_name_map[model_name_2]

    if ":" not in model_selectors[0]:
        for i in range(5):
            names = (
                "### Model A: " + model_name_1,
                "### Model B: " + model_name_2,
            )
            sandbox_titles = (
                f"### Model A Sandbox: {model_name_1}",
                f"### Model B Sandbox: {model_name_2}",
            )
            # model_selectors + sandbox_titles + [textbox] + user_buttons
            yield (
                names + sandbox_titles
                + (disable_text,)
                + (disable_btn,) * (sandbox_states[0]['btn_list_length'] - 1)
                + (enable_btn,) # allow clear
            )
            time.sleep(0.1)
    else:
        names = (
            "### Model A: " + model_name_1,
            "### Model B: " + model_name_2,
        )
        sandbox_titles = (
            f"### Model A Sandbox: {model_name_1}",
            f"### Model B Sandbox: {model_name_2}",
        )
        # model_selectors + sandbox_titles + [textbox] + user_buttons
        yield (
            names + sandbox_titles
            + (disable_text,)
            + (disable_btn,) * (sandbox_states[0]['btn_list_length'] - 1)
            + (enable_btn,) # allow clear
        )


def leftvote_last_response(
    state0, state1,
    model_selector0, model_selector1,
    sandbox_state0, sandbox_state1,
    request: gr.Request
):
    logger.info(f"leftvote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], [sandbox_state0, sandbox_state1], "leftvote", [model_selector0, model_selector1], request
    ):
        yield x


def rightvote_last_response(
    state0, state1,
    model_selector0, model_selector1,
    sandbox_state0, sandbox_state1,
    request: gr.Request
):
    logger.info(f"rightvote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], [sandbox_state0, sandbox_state1], "rightvote", [model_selector0, model_selector1], request
    ):
        yield x


def tievote_last_response(
    state0, state1,
    model_selector0, model_selector1,
    sandbox_state0, sandbox_state1,
    request: gr.Request
):
    logger.info(f"tievote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], [sandbox_state0, sandbox_state1], "tievote", [model_selector0, model_selector1], request
    ):
        yield x


def bothbad_vote_last_response(
    state0, state1,
    model_selector0, model_selector1,
    sandbox_state0, sandbox_state1,
    request: gr.Request
):
    logger.info(f"bothbad_vote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], [sandbox_state0, sandbox_state1], "bothbad_vote", [model_selector0, model_selector1], request
    ):
        yield x


def regenerate_single(state, request: gr.Request):
    '''
    Regenerate message for one side.

    Return
        [state, chatbot, textbox] + user_buttons
    '''
    logger.info(f"regenerate. ip: {get_ip(request)}")
    if state is None:
        # if not init yet
        return [None, None] + [None] + [no_change_btn] * USER_BUTTONS_LENGTH
    elif state.regen_support:
        state.conv.update_last_message(None)
        return (
            [state, state.to_gradio_chatbot()]
            + [None] # textbox
            + [disable_btn] * USER_BUTTONS_LENGTH
        )
    else:
        # if not support regen
        state.skip_next = True
        return (
            [state, state.to_gradio_chatbot()]
            + [None] # textbox
            + [no_change_btn] * USER_BUTTONS_LENGTH
        )


def regenerate_multi(state0, state1, request: gr.Request):
    '''
    Regenerate message for both sides.
    '''
    logger.info(f"regenerate. ip: {get_ip(request)}")
    states = [state0, state1]

    if state0.regen_support and state1.regen_support:
        for i in range(num_sides):
            states[i].conv.update_last_message(None)
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [None]
            + [disable_btn] * USER_BUTTONS_LENGTH # Disable user buttons
        )
    else:
        # if not support regen
        states[0].skip_next = True
        states[1].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [None] # textbox
            + [no_change_btn] * USER_BUTTONS_LENGTH
        )


def clear_history(sandbox_state0, sandbox_state1, request: gr.Request):
    '''
    Clear chat history for both sides.
    '''
    logger.info(f"clear_history (anony). ip: {get_ip(request)}")

    # reset sandbox state
    sandbox_states = [
        reset_sandbox_state(sandbox_state) for sandbox_state in [sandbox_state0, sandbox_state1]
    ]

    '''
    sandbox_states
    + states
    + chatbots
    + model_selectors
    + [multimodal_textbox, textbox]
    + user_buttons
    + [slow_warning]
    + sandbox_titles
    '''
    return (
        sandbox_states
        + [None] * num_sides
        + [None] * num_sides
        + anony_names
        + [enable_multimodal, invisible_text]
        + [enable_btn, invisible_btn, invisible_btn]  # send_btn, send_btn_left, send_btn_right
        + [invisible_btn] * 3  # regenerate, regenerate left/right
        + [invisible_btn] * 4  # vote buttons
        + [disable_btn]  # clear
        + [""]  # slow_warning
        + [gr.update(value="### Model A Sandbox"), gr.update(value="### Model B Sandbox")]  # Reset sandbox titles
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


    Return:
        [state, chatbot, sandbox_state]
        + [textbox, multimodal_textbox]
        + user_buttons
        + [slow_warning]
    '''
    if multimodal_input and multimodal_input["text"]:
        text, images = multimodal_input["text"], multimodal_input["files"]
    else:
        text = text_input
        images = []

    # whether need vision models
    is_vision = len(images) > 0

    ip = get_ip(request)
    logger.info(f"add_text (anony). ip: {ip}. len: {len(text)}")

    # increase sandbox state
    sandbox_state['enabled_round'] += 1

    # should not allow send to one side initially
    assert state is not None

    # skip if text is empty
    if len(text) <= 0:
        state.skip_next = True
        '''
        [state, chatbot, sandbox_state]
        + [textbox, multimodal_textbox]
        + user_buttons
        + [slow_warning]
        '''
        return (
            [state, state.to_gradio_chatbot(), sandbox_state]
            + [None, ""]
            + [no_change_btn] * USER_BUTTONS_LENGTH
            + [""]
        )

    images = convert_images_to_conversation_format(images)

    # TODO: Skip moderation for now
    # text, image_flagged, csam_flag = moderate_input(
    #     state0, text, text, model_list, images, ip
    # )
    image_flagged, csam_flag = None, None

    conv = state.conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {get_ip(request)}. text: {text}")
        state.skip_next = True
        return (
            [state, state.to_gradio_chatbot(), sandbox_state]
            + [{"text": CONVERSATION_LIMIT_MSG}, ""]
            + [no_change_btn] * 8 # FIXME: Update the number of buttons
            + [""]
        )

    if image_flagged:
        logger.info(f"image flagged. ip: {ip}. text: {text}")
        state.skip_next = True
        return (
            [state, state.to_gradio_chatbot(), sandbox_state]
            + [
                {
                    "text": IMAGE_MODERATION_MSG
                    + " PLEASE CLICK 🎲 NEW ROUND TO START A NEW CONVERSATION."
                },
                "",
            ]
            + [no_change_btn] * 8 # FIXME: Update the number of buttons
            + [""]
        )

    text = text[:BLIND_MODE_INPUT_CHAR_LEN_LIMIT]  # Hard cut-off

    post_processed_text = _prepare_text_with_image(
        state, text, images, csam_flag=csam_flag
    )
    state.conv.append_message(state.conv.roles[0], post_processed_text)
    state.conv.append_message(state.conv.roles[1], None)
    state.skip_next = False

    hint_msg = ""
    if "deluxe" in state.model_name:
        hint_msg = SLOW_MODEL_MSG

    '''
    [state, chatbot, sandbox_state]
    + [textbox, multimodal_textbox]
    + user_buttons
    + [slow_warning]
    '''
    return (
        [state, state.to_gradio_chatbot(), sandbox_state]
        + [disable_multimodal, visible_text]
        + [disable_btn] * USER_BUTTONS_LENGTH
        + [hint_msg]
    )


def add_text_multi(
    state0, state1,
    model_selector0, model_selector1,
    sandbox_state0, sandbox_state1,
    multimodal_input: dict, text_input: str,
    context: Context,
    request: gr.Request,
):
    '''
    Add text for both chatbots.

    return 
        states
        + chatbots
        + sandbox_states
        + [multimodal_textbox, textbox]
        + user_buttons
        + [slow_warning]
    '''
    if multimodal_input and multimodal_input["text"]:
        text, images = multimodal_input["text"], multimodal_input["files"]
    else:
        text = text_input
        images = []

    # whether need vision models
    is_vision = len(images) > 0

    ip = get_ip(request)
    logger.info(f"add_text (anony). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]
    sandbox_states = [sandbox_state0, sandbox_state1]

    # increase sandbox state
    sandbox_state0['enabled_round'] += 1
    sandbox_state1['enabled_round'] += 1

    # Init states if necessary
    if states[0] is None or states[1] is None:
        # assert states[1] is None
        # Generate model battle pairs
        model_left, model_right = get_battle_pair(
            context.all_vision_models,
            VISION_BATTLE_TARGETS,
            VISION_OUTAGE_MODELS,
            VISION_SAMPLING_WEIGHTS,
            VISION_SAMPLING_BOOST_MODELS,
        ) if is_vision else get_battle_pair(
            context.all_text_models,
            BATTLE_TARGETS,
            OUTAGE_MODELS,
            SAMPLING_WEIGHTS,
            SAMPLING_BOOST_MODELS,
        )

        states = [
            State(model_left, is_vision=is_vision),
            State(model_right, is_vision=is_vision),
        ]

    if len(text) <= 0:
        # skip if no text
        for i in range(num_sides):
            states[i].skip_next = True
        '''
        states
        + chatbots
        + sandbox_states
        + [multimodal_textbox, textbox]
        + user_buttons
        + [slow_warning]
        '''
        return (
            states
            + [state.to_gradio_chatbot() for state in states]
            + sandbox_states
            + [None, ""] # textbox
            + [no_change_btn,] * sandbox_state0['btn_list_length']
            + [""] # hint
        )

    model_list = [states[i].model_name for i in range(num_sides)]

    images = convert_images_to_conversation_format(images)

    # TODO: Skip moderation for now
    # text, image_flagged, csam_flag = moderate_input(
    #     state0, text, text, model_list, images, ip
    # )
    image_flagged, csam_flag = None, None

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {get_ip(request)}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        '''
        states
        + chatbots
        + sandbox_states
        + [multimodal_textbox, textbox]
        + user_buttons
        + [slow_warning]
        '''
        return (
            states
            + [state.to_gradio_chatbot() for state in states]
            + sandbox_states
            + [{"text": CONVERSATION_LIMIT_MSG}, ""]
            + [no_change_btn] * USER_BUTTONS_LENGTH
            + [""]
        )

    if image_flagged:
        logger.info(f"image flagged. ip: {ip}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        '''
        states
        + chatbots
        + sandbox_states
        + [multimodal_textbox, textbox]
        + user_buttons
        + [slow_warning]
        '''
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + sandbox_states
            + [
                {
                    "text": IMAGE_MODERATION_MSG
                    + " PLEASE CLICK 🎲 NEW ROUND TO START A NEW CONVERSATION."
                },
                "",
            ]
            + [no_change_btn] * USER_BUTTONS_LENGTH
            + [""] # hint
        )

    text = text[:BLIND_MODE_INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        post_processed_text = _prepare_text_with_image(
            states[i], text, images, csam_flag=csam_flag
        )
        states[i].conv.append_message(states[i].conv.roles[0], post_processed_text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    hint_msg = ""
    for i in range(num_sides):
        if "deluxe" in states[i].model_name:
            hint_msg = SLOW_MODEL_MSG
    '''
    states
    + chatbots
    + sandbox_states
    + [multimodal_textbox, textbox]
    + user_buttons
    + [slow_warning]
    '''
    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + sandbox_states
        + [disable_multimodal, visible_text]
        + [disable_btn] * USER_BUTTONS_LENGTH
        + [hint_msg]
    )


def build_side_by_side_vision_ui_anony(context: Context, random_questions=None):
    notice_markdown = f"""
## How It Works
- **Blind Test**: Chat with two anonymous AI chatbots and give them a prompt or task (e.g., build a web app, create a visualization, design an interface).
- **Run & Interact**: The AI chatbots generate programs that run in a secure sandbox environment. Test the functionality, explore the features, and evaluate the quality of the outputs.
- **Visual Input**: Upload images or provide text prompts to guide the AI chatbots in their responses. You can only chat with <span style='color: #DE3163; font-weight: bold'>one image per conversation</span>. The image should be less than 15MB.
- **Vote for the Best**: After interacting with both programs, vote for the one that best meets your requirements or provides the superior experience.

**❗️ For research purposes, we log user prompts, images, and interactions with sandbox, and may release this data to the public in the future. Please do not upload any confidential or personal information.**
"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors: list[gr.Markdown | None] = [None] * num_sides
    chatbots: list[gr.Chatbot | None] = [None] * num_sides
    context_state = gr.State(context)
    text_and_vision_models = context.models

    with gr.Row():
        with gr.Column(scale=2, visible=False) as image_column:
            imagebox = gr.Image(
                type="pil",
                show_label=False,
                interactive=False,
            )

        with gr.Column(scale=5):
            with gr.Group(elem_id="share-region-anony"):
                with gr.Row(
                    elem_classes=["chatbot-section"],
                ):
                    for i in range(num_sides):
                        label = "Model A" if i == 0 else "Model B"
                        with gr.Column():
                            chatbots[i] = gr.Chatbot(
                                label=label,
                                elem_id="chatbot",
                                height='65vh',
                                show_copy_button=True,
                                latex_delimiters=[
                                    {"left": "$", "right": "$", "display": False},
                                    {"left": "$$", "right": "$$", "display": True},
                                    {"left": r"\(", "right": r"\)", "display": False},
                                    {"left": r"\[", "right": r"\]", "display": True},
                                ],
                            )

                with gr.Row():
                    for i in range(num_sides):
                        with gr.Column():
                            model_selectors[i] = gr.Markdown(
                                anony_names[i], elem_id="model_selector_md"
                            )
    with gr.Row():
        slow_warning = gr.Markdown("")

    with gr.Row(elem_id="user-input-region"):
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Input your prompt here. Press Enter to send.",
            elem_id="input_box",
            visible=False,
            scale=3,
            autofocus=True,
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
        )

    with gr.Row() as examples_row:
        example_prompts = gr.Examples(
            examples = [
                ["Write a Python script that uses the Gradio library to create a functional calculator. The calculator should support basic arithmetic operations: addition, subtraction, multiplication, and division. It should have two input fields for numbers and a dropdown menu to select the operation."],
                ["Write a Todo list app using React.js. The app should allow users to add, delete, and mark tasks as completed. Include features like filtering tasks by status (completed, active), sorting tasks by priority, and displaying the total number of tasks."],
                ["Write a Python script using the Streamlit library to create a web application for uploading and displaying files. The app should allow users to upload files of type .csv or .txt. If a .csv file is uploaded, display its contents as a table using Streamlit's st.dataframe() method. If a .txt file is uploaded, display its content as plain text."],
                ["Write a Python function to solve the Trapping Rain Water problem. The function should take a list of non-negative integers representing the height of bars in a histogram and return the total amount of water trapped between the bars after raining. Use an efficient algorithm with a time complexity of O(n)."],
                ["Create a simple Pygame script for a game where the player controls a bouncing ball that changes direction when it collides with the edges of the window. Add functionality for the player to control a paddle using arrow keys, aiming to keep the ball from touching the bottom of the screen. Include basic collision detection and a scoring system that increases as the ball bounces off the paddle."],
                ["Create a financial management Dashboard using Vue.js, focusing on local data handling without APIs. Include features like a clean dashboard for tracking income and expenses, dynamic charts for visualizing finances, and a budget planner. Implement functionalities for adding, editing, and deleting transactions, as well as filtering by date or category. Ensure responsive design and smooth user interaction for an intuitive experience."],
                ["Create a Mermaid diagram to visualize a flowchart of a user login process. Include the following steps: User enters login credentials; Credentials are validated; If valid, the user is directed to the dashboard; If invalid, an error message is shown, and the user can retry or reset the password."],
            ],
            example_labels=[
                "Calculator with Gradio",
                "Todo List App with React.js",
                "File Upload Web App with Streamlit",
                "Solve Trapping Rain Water Problem",
                "Pygame Bouncing Ball Game",
                "Financial Dashboard with Vue.js",
                "User Login Process Flowchart",
            ],
            examples_per_page=10,
            label="Example Prompts",
            inputs = [multimodal_textbox],
        )

    # sandbox states and components
    sandbox_states: list[gr.State] = [] # state for each chatbot
    sandboxes_components: list[SandboxGradioSandboxComponents] = [] # components for each chatbot
    sandbox_titles = [None] * num_sides
    sandbox_hidden_components = []

    # chatbot sandbox
    with gr.Group():
        # chatbot sandbox config
        with gr.Row():
            sandbox_env_choice = gr.Dropdown(choices=SUPPORTED_SANDBOX_ENVIRONMENTS, label="Programming Expert (Predefined system prompt)", interactive=True, visible=True)
        
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

                                with gr.Tab(label="Code", visible=True) as sandbox_code_tab:
                                    sandbox_code = gr.Code(
                                        value="",
                                        interactive=True, # allow user edit
                                        visible=False,
                                        label='Sandbox Code',
                                    )
                                    with gr.Row():
                                        sandbox_code_submit_btn = gr.Button(value="Apply Changes", visible=True, interactive=True, variant='primary', size='sm')

                                with gr.Tab(
                                    label="Dependency", visible=True
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

    with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
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
    gr.Markdown("## Supported Models")            
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
        model_selectors + sandbox_titles + [textbox] + user_buttons,
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors + sandbox_states,
        model_selectors + sandbox_titles + [textbox] + user_buttons,
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors + sandbox_states,
        model_selectors + sandbox_titles + [textbox] + user_buttons,
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors + sandbox_states,
        model_selectors + sandbox_titles + [textbox] + user_buttons,
    )

    regenerate_btn.click(
        regenerate_multi,
        states,
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
            + model_selectors
            + [multimodal_textbox, textbox]
            + user_buttons
            + [slow_warning]
            + sandbox_titles
        ),
    ).then(
        clear_sandbox_components,
        inputs=[component for components in sandboxes_components for component in components],
        outputs=[component for components in sandboxes_components for component in components]
    ).then(
        lambda: (gr.update(interactive=True, value=SandboxEnvironment.AUTO), gr.update(interactive=True, value=DEFAULT_SANDBOX_INSTRUCTIONS[SandboxEnvironment.AUTO])),
        outputs=[sandbox_env_choice, system_prompt_textbox]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[examples_row]
    )

    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-anony');
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

    multimodal_textbox.input(
        add_image, [multimodal_textbox], [imagebox]
    ).then(
        set_visible_image, [multimodal_textbox], [image_column]
    )

    multimodal_textbox.submit( # update the system prompt
        add_text_multi,
        inputs=states + model_selectors + sandbox_states + [multimodal_textbox, textbox] + [context_state],
        outputs=(
            states
            + chatbots
            + sandbox_states
            + [multimodal_textbox, textbox]
            + user_buttons
            + [slow_warning]
        ),
    ).then(
        set_invisible_image, [], [image_column]
    ).then( # set the system prompt
        set_chat_system_messages_multi,
        states + sandbox_states + model_selectors,
        states + chatbots
    ).then(
        # hide the examples row
        lambda: gr.update(visible=False),
        outputs=examples_row
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
        flash_buttons,
        [],
        user_buttons,
    )

    textbox.submit(
        add_text_multi,
        inputs=states + model_selectors + sandbox_states + [multimodal_textbox, textbox] + [context_state],
        outputs=states
        + chatbots
        + sandbox_states
        + [multimodal_textbox, textbox]
        + user_buttons
        + [slow_warning],
    ).then(
        # set the system prompt
        set_chat_system_messages_multi,
        states + sandbox_states + model_selectors,
        states + chatbots
    ).then(
        # hide the examples row
        lambda: gr.update(visible=False),
        outputs=examples_row
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
        flash_buttons,
        [],
        user_buttons,
    )

    send_btn.click(
        add_text_multi,
        inputs=states + model_selectors + sandbox_states + [multimodal_textbox, textbox] + [context_state],
        outputs=(
            states
            + chatbots
            + sandbox_states
            + [multimodal_textbox, textbox]
            + user_buttons
            + [slow_warning]
        ),
    ).then(
        # set the system prompt
        set_chat_system_messages_multi,
        states + sandbox_states + model_selectors,
        states + chatbots
    ).then(
        lambda: gr.update(visible=False),
        outputs=examples_row
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
        flash_buttons,
        [],
        user_buttons,
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
            inputs=(
                [state, model_selector, sandbox_state] + [multimodal_textbox, textbox] + [context_state]
            ),
            outputs=(
                [state, chatbot, sandbox_state]
                + [textbox, multimodal_textbox]
                + user_buttons
                + [slow_warning]
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
            state,
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
