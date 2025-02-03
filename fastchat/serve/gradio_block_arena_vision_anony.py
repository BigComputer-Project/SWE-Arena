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
    INPUT_CHAR_LEN_LIMIT,
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
from fastchat.serve.chat_state import LOG_DIR, ModelChatState, save_log_to_local
from fastchat.serve.gradio_block_arena_named import flash_buttons, set_chat_system_messages_multi
from fastchat.serve.gradio_web_server import (
    bot_response,
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
from fastchat.serve.sandbox.sandbox_state import ChatbotSandboxState
from fastchat.serve.sandbox.sandbox_telemetry import save_conv_log_to_azure_storage
from fastchat.serve.sandbox.code_runner import SUPPORTED_SANDBOX_ENVIRONMENTS, SandboxEnvironment, DEFAULT_SANDBOX_INSTRUCTIONS, SandboxGradioSandboxComponents, create_chatbot_sandbox_state, on_click_code_message_run, on_edit_code, on_edit_dependency, reset_sandbox_state, set_sandbox_state_ids, update_sandbox_config_multi, update_sandbox_state_system_prompt
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

# Add at the top level, before the functions
feedback_popup_js = """
function() {
    function submitFeedback(selectedFeedback) {
        console.log('Submit function called');
        console.log('User selected feedback:', selectedFeedback);
        console.log('Returning data to backend:', {
            feedback: selectedFeedback
        });
        
        let feedback_details_div = document.querySelector('#feedback_details');
        let feedback_details_textbox = feedback_details_div.querySelector('textarea');
        feedback_details_textbox.value = JSON.stringify(selectedFeedback);
        // This is very important!
        // Trigger the textarea's event for gradio to function normally
        feedback_details_textbox.dispatchEvent(new Event('input', { bubbles: true }));
        feedback_btn = document.querySelector('#feedback_btn');

        feedback_btn.click();
    }

    return new Promise((resolve) => {
        console.log('Feedback popup opened, vote type:', '{{VOTE_TYPE}}');
        // Create popup container
        const popup = document.createElement('div');
        const isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        console.log('Created popup, dark mode:', isDarkMode);
        
        popup.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: ${isDarkMode ? '#1a1a1a' : 'white'};
            color: ${isDarkMode ? '#ffffff' : '#000000'};
            padding: 20px;
            border-radius: 8px;
            border: 1px solid ${isDarkMode ? '#404040' : '#e0e0e0'};
            box-shadow: 0 2px 10px ${isDarkMode ? 'rgba(0,0,0,0.3)' : 'rgba(0,0,0,0.1)'};
            z-index: 1000;
            max-width: 500px;
            width: 90%;
        `;

        // Add close button
        const closeButton = document.createElement('button');
        closeButton.innerHTML = '✕';
        closeButton.style.cssText = `
            position: absolute;
            right: 10px;
            top: 10px;
            background: none;
            border: none;
            color: ${isDarkMode ? '#ffffff' : '#000000'};
            font-size: 18px;
            cursor: pointer;
            padding: 5px;
            line-height: 1;
        `;
        closeButton.onclick = () => {
            document.body.removeChild(popup);
            document.body.removeChild(overlay);
            submitFeedback({
                "vote_type": '{{VOTE_TYPE}}'
            }); // Submit empty feedback
        };
        popup.appendChild(closeButton);

        // Add title
        const title = document.createElement('h3');
        title.textContent = 'Please provide additional feedback';
        title.style.cssText = `
            margin-bottom: 15px;
            color: ${isDarkMode ? '#ffffff' : '#000000'};
            font-size: 1.2em;
            padding-right: 20px;
        `;
        popup.appendChild(title);

        // Initialize selectedFeedback object
        let selectedFeedback = {
            "vote_type": '{{VOTE_TYPE}}'
        };

        // Add categories with 3 buttons (A, Tie, B)
        const options = [
            'Code quality',
            'UI/UX design',
            'Explanation clarity',
            'Solution creativity',
            'Implementation is more efficient',
            'Error handling',
            'Documentation'
        ];

        const buttonContainer = document.createElement('div');
        buttonContainer.style.marginBottom = '20px';
        options.forEach(option => {
            const categoryContainer = document.createElement('div');
            categoryContainer.style.marginBottom = '15px';

            const label = document.createElement('label');
            label.style.cssText = `
                display: block;
                margin-bottom: 5px;
                color: ${isDarkMode ? '#ffffff' : '#000000'};
            `;
            label.textContent = option;
            categoryContainer.appendChild(label);

            const buttonGroup = document.createElement('div');
            buttonGroup.style.display = 'flex';
            buttonGroup.style.justifyContent = 'space-between';

            ['A', 'Tie', 'B'].forEach(buttonText => {
                const button = document.createElement('button');
                button.textContent = buttonText;
                button.style.cssText = `
                    flex: 1;
                    padding: 10px;
                    margin: 0 5px;
                    border: 1px solid ${isDarkMode ? '#444444' : '#ccc'};
                    background: ${isDarkMode ? '#333333' : '#f9f9f9'};
                    color: ${isDarkMode ? '#ffffff' : '#000000'};
                    border-radius: 4px;
                    cursor: pointer;
                    transition: background-color 0.2s;
                `;

                // Hover effect
                button.onmouseover = () => {
                    if (!button.classList.contains('selected')) {  // Only apply hover if it's not selected
                        button.style.backgroundColor = isDarkMode ? '#555555' : '#e0e0e0';
                    }
                };

                button.onmouseout = () => {
                    if (!button.classList.contains('selected')) {  // Only reset hover if it's not selected
                        button.style.backgroundColor = isDarkMode ? '#333333' : '#f9f9f9';
                    }
                };

                // Click to select
                button.onclick = () => {
                    // Save the selection for the current option
                    selectedFeedback[option] = buttonText;

                    // Reset all buttons' background color to default
                    Array.from(buttonGroup.children).forEach(b => {
                        b.style.backgroundColor = isDarkMode ? '#333333' : '#f9f9f9';
                        b.classList.remove('selected');
                    });

                    // Set the selected button's background color to blue and mark as selected
                    button.style.backgroundColor = '#2196F3'; // Blue color for selection
                    button.classList.add('selected'); // Add 'selected' class to prevent hover override
                };

                buttonGroup.appendChild(button);
            });

            categoryContainer.appendChild(buttonGroup);
            buttonContainer.appendChild(categoryContainer);
        });
        popup.appendChild(buttonContainer);

        // Add submit button
        const submitBtn = document.createElement('button');
        submitBtn.textContent = 'Submit Feedback';
        submitBtn.style.cssText = `
            background: #2196F3;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1em;
            transition: opacity 0.2s;
        `;
        
        submitBtn.onmouseover = () => {
            submitBtn.style.opacity = '0.9';
        };
        submitBtn.onmouseout = () => {
            submitBtn.style.opacity = '1';
        };

        submitBtn.onclick = () => {
            console.log('Submit button clicked');
            console.log('Selected feedback:', selectedFeedback);

            document.body.removeChild(popup);
            document.body.removeChild(overlay);

            const result = submitFeedback(selectedFeedback);
            console.log('Resolving promise with result:', result);
            resolve(result);
        };

        popup.appendChild(submitBtn);

        // Add overlay
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: ${isDarkMode ? 'rgba(0,0,0,0.7)' : 'rgba(0,0,0,0.5)'};
            z-index: 999;
            cursor: pointer;
        `;
        
        // Make overlay clickable to close
        overlay.onclick = () => {
            document.body.removeChild(popup);
            document.body.removeChild(overlay);
            submitFeedback({
                "vote_type": '{{VOTE_TYPE}}'
            }); // Submit empty feedback
        };

        document.body.appendChild(overlay);
        document.body.appendChild(popup);

        // Add event listener for escape key
        const closePopup = (e) => {
            if (e.key === 'Escape') {
                document.body.removeChild(popup);
                document.body.removeChild(overlay);
                submitFeedback({
                    "vote_type": '{{VOTE_TYPE}}'
                }); // Submit empty feedback
            }
        };
        document.addEventListener('keydown', closePopup);
    });
}

"""


def load_demo_side_by_side_vision_anony():
    states = [None] * num_sides
    selector_updates = [
        gr.Markdown(visible=True),
        gr.Markdown(visible=True),
    ]

    return states + selector_updates


def vote_last_response(state0, state1, model_selector0, model_selector1, feedback_details, request: gr.Request = None):
    '''
    Handle voting for a response, including any feedback details provided.

    Args:
        state0: First conversation state
        state1: Second conversation state
        model_selector0: First model selector
        model_selector1: Second model selector
        feedback_details: Optional feedback details from the popup
        request: Gradio request object
    Returns:
        Tuple of (model_selectors[2] + sandbox_titles[2] + textbox[1] + user_buttons[10])
    '''
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    local_filepath = states[0].get_conv_log_filepath(LOG_DIR)
    # Extract the vote type from the tuple

    logger.info(f"=== Vote Response Start ===")
    logger.info(f"Feedback data received: {feedback_details}")

    log_data = {
        "tstamp": round(time.time(), 4),
        "type": "vote",
        "models": [x for x in model_selectors] if model_selectors else [],
        "states": [x.to_dict() for x in states] if states else [],
        "ip": get_ip(request),
    }

    # Add feedback data if available
    if feedback_details:
        try:
            feedback_list = json.loads(feedback_details)
            log_data["feedback"] = feedback_list
            logger.info(f"Processed feedback: {feedback_list}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse feedback data: {feedback_details}")
            logger.error(f"JSON decode error: {str(e)}")
    else:
        logger.warning("No feedback data received")

    save_log_to_local(log_data, local_filepath)
    get_remote_logger().log(log_data)
    logger.info(f"Data written to file: {local_filepath}")

    logger.info("=== Vote Response End ===")

    gr.Info(
        "🎉 Thanks for voting! Your vote shapes the leaderboard, please vote RESPONSIBLY."
    )

    # display model names
    model_name_1 = state0.model_name if state0 else ""
    model_name_2 = state1.model_name if state1 else ""
    model_name_map = {}

    if model_name_1 in model_name_map:
        model_name_1 = model_name_map[model_name_1]
    if model_name_2 in model_name_map:
        model_name_2 = model_name_map[model_name_2]

    names = (
        "### Model A: " + model_name_1,
        "### Model B: " + model_name_2,
    )
    sandbox_titles = (
        f"### Model A Sandbox: {model_name_1}",
        f"### Model B Sandbox: {model_name_2}",
    )

    # Return exactly the number of outputs expected by the click handler
    # 2 model selectors + 2 sandbox titles + 1 textbox + 10 buttons = 15 outputs
    return (
        names + sandbox_titles
        + (disable_text,)
        + (disable_btn,) * (USER_BUTTONS_LENGTH - 1)
    )


def regenerate_single(state: ModelChatState, request: gr.Request):
    '''
    Regenerate message for one side.

    Return
        [state, chatbot, textbox] + user_buttons
    '''
    logger.info(f"regenerate. ip: {get_ip(request)}")

    if state.regen_support:
        state.conv.update_last_message(None)
        state.set_response_type("regenerate_single")
        return (
            [state, state.to_gradio_chatbot()]
            + [None]  # textbox
            + [disable_btn] * USER_BUTTONS_LENGTH
        )
    else:
        # if not support regen
        state.skip_next = True
        return (
            [state, state.to_gradio_chatbot()]
            + [None]  # textbox
            + [no_change_btn] * USER_BUTTONS_LENGTH
        )


def regenerate_multi(state0: ModelChatState, state1: ModelChatState, request: gr.Request):
    '''
    Regenerate message for both sides.
    '''
    logger.info(f"regenerate. ip: {get_ip(request)}")
    states = [state0, state1]

    if state0.regen_support and state1.regen_support:
        for i in range(num_sides):
            states[i].conv.update_last_message(None)
            states[i].set_response_type("regenerate_multi")
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [None]
            + [disable_btn] * USER_BUTTONS_LENGTH  # Disable user buttons
        )
    else:
        # if not support regen
        states[0].skip_next = True
        states[1].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [None]  # textbox
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
    state: ModelChatState,
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
            + [no_change_btn] * 8  # FIXME: Update the number of buttons
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
            + [no_change_btn] * 8  # FIXME: Update the number of buttons
            + [""]
        )

    text = text[:BLIND_MODE_INPUT_CHAR_LEN_LIMIT]  # Hard cut-off

    post_processed_text = _prepare_text_with_image(
        state, text, images, csam_flag=csam_flag
    )
    state.conv.append_message(state.conv.roles[0], post_processed_text)
    state.conv.append_message(state.conv.roles[1], None)
    state.skip_next = False
    state.set_response_type("chat_single")

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

        states = ModelChatState.create_battle_chat_states(
            model_left, model_right, chat_mode="battle_anony",
            is_vision=is_vision
        )
        states = list(states)  # tuple to list
        for idx in range(2):
            set_sandbox_state_ids(
                sandbox_state=sandbox_states[idx],
                conv_id=states[idx].conv_id,
                chat_session_id=states[idx].chat_session_id
            )

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
            + [None, ""]  # textbox
            + [no_change_btn,] * sandbox_state0['btn_list_length']
            + [""]  # hint
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
            + [""]  # hint
        )

    text = text[:BLIND_MODE_INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        post_processed_text = _prepare_text_with_image(
            states[i], text, images, csam_flag=csam_flag
        )
        states[i].conv.append_message(states[i].conv.roles[0], post_processed_text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False
        states[i].set_response_type("chat_multi")

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
## How It Works for Battle Mode
- **Blind Test**: Chat with two anonymous AI chatbots and give them a prompt or task (e.g., build a web app, create a visualization, design an interface).
- **Run & Interact**: The AI chatbots generate programs that run in a secure sandbox environment. Test the functionality, explore the features, and evaluate the quality of the outputs.
- **Edit & Regenerate**: You can edit the <u>system prompt</u>, <u>code</u>, and its <u>dependency</u> and <u>regenerate the code</u> on any side.
- **Visual Input**: Upload images or provide text prompts to guide the AI chatbots in their responses. You can only chat with <span style='color: #DE3163; font-weight: bold'>one image per conversation</span>. The image should be less than 15MB.
- **Vote for the Best**: After interacting with both programs, vote for the one that best meets your requirements or provides the superior experience.

## Note
- **Sandbox**: There are four buttons on the top right corner of the sandbox inference. You can use them to copy the link, turn on/off the black mode, make the sandbox full screen, and reload the sandbox. Plus, you can also drag the bottom right corner of the sandbox to resize the sandbox.
- **Dependency Edit**: You can edit the <u>dependency</u> of the code on any side. Currently, we only support `pypi` and `npm` packages.
For `pypi` packages, you can use the format `python (use '==', '>=', '<=', '~=', '>', '<' or 'latest') <package_name> <version>`.
For `npm` packages, you can use the format `npm (use '@' or 'latest') <package_name> <version>`.
- **Temperature**: All models have the same temperature of `0.7` and `top_p` of `1.0` by default. Low temperature typically works better for code generation.

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
                                sanitize_html=False,
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
            examples=[
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
            inputs=[multimodal_textbox],
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
                                        interactive=True,  # allow user edit
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
        clear_btn = gr.Button(value="🎲 New Round", interactive=False, elem_id="clear_btn")
        share_btn = gr.Button(value="📷  Share")

    with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
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

    # Create a feedback state that persists across the chain
    feedback_state = gr.State("Not a state")
    # The hidden vote button used to trigger the vote submission
    with gr.Group(visible=False):
        feedback_btn = gr.Button(
            elem_id="feedback_btn",
            value="The hidden vote button. The user shoudl not be able to see this", 
            interactive=True
        )
        feedback_details = gr.Textbox(
            elem_id="feedback_details",
            value="",
            interactive=True
        )
        
    
    # The one and only entry for submitting the vote
    feedback_btn.click(
        vote_last_response,
        inputs=[states[0], states[1],
                model_selectors[0], model_selectors[1], feedback_details],
        outputs=model_selectors + sandbox_titles + [
            textbox,
            # vote buttons
            leftvote_btn, rightvote_btn, tie_btn, bothbad_btn,
            # send buttons
            send_btn, send_btn_left, send_btn_right,
            # regenerate buttons
            regenerate_btn, left_regenerate_btn, right_regenerate_btn,
        ]
    )

    leftvote_btn.click(
        lambda: ("vote_left",),
        inputs=[],
        outputs=[feedback_state],
        js=feedback_popup_js.replace("{{VOTE_TYPE}}", "vote_left")
    )
    rightvote_btn.click(
        lambda: ("vote_right",),
        inputs=[],
        outputs=[feedback_state],
        js=feedback_popup_js.replace("{{VOTE_TYPE}}", "vote_right")

    )
    tie_btn.click(
        lambda: ("vote_tie",),
        inputs=[],
        outputs=[feedback_state],
        js=feedback_popup_js.replace("{{VOTE_TYPE}}", "vote_tie")
    )
    bothbad_btn.click(
        lambda: ("vote_both_bad",),
        inputs=[],
        outputs=[feedback_state],
        js=feedback_popup_js.replace("{{VOTE_TYPE}}", "vote_both_bad")
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
        outputs=examples_row
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
