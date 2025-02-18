import os
import time
import gradio as gr
from gradio.utils import get_space
from e2b_code_interpreter import Sandbox
import numpy as np
from fastchat.serve.gradio_global_state import Context

# CSS styling for responsive layout
CSS = """
.notebook-container {
    width: 100%;
    max-width: 100%;
    overflow-x: auto;
    padding: 10px;
    box-sizing: border-box;
}

#share-region-jupyter {
    width: 100%;
    max-width: 100vw;
    margin: 0 auto;
    padding: 10px;
    box-sizing: border-box;
}

#input_box {
    width: 100%;
    max-width: 100%;
}

/* Make columns more responsive */
.gradio-container {
    max-width: 100% !important;
}

.gradio-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: space-between;
}

.gradio-column {
    flex: 1 1 45%;
    min-width: 300px;
    max-width: 48%;
    box-sizing: border-box;
}

/* Responsive notebook adjustments */
.jupyter-notebook {
    width: 100%;
    max-width: 100%;
    overflow-x: auto;
}

@media screen and (max-width: 768px) {
    .gradio-column {
        max-width: 100%;
        flex: 1 1 100%;
    }
    
    .jupyter-notebook {
        width: 100%;
        padding: 5px;
        font-size: 14px;
    }
}

/* Make sure the container doesn't overflow */
.contain {
    max-width: 100vw;
    overflow-x: hidden;
}
"""

from fastchat.serve.jupyter_agent.utils import (
    create_base_notebook,
    update_notebook_display,
    execute_code,
    parse_exec_result_nb,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.chat_state import LOG_DIR, ModelChatState, save_log_to_local
from fastchat.serve.model_sampling import (
    ANON_MODELS,
    BATTLE_STRICT_TARGETS,
    BATTLE_TARGETS,
    OUTAGE_MODELS,
    SAMPLING_BOOST_MODELS,
    SAMPLING_WEIGHTS
)
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.utils import build_logger
from fastchat.serve.gradio_web_server import get_ip
from fastchat.serve.gradio_block_arena_named import bot_response_multi
from fastchat.serve.sandbox.code_runner import create_chatbot_sandbox_state, set_sandbox_state_ids

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that specializes in data science and computational tasks. You will help users with their data analysis, visualization, and computation needs by generating and executing Jupyter notebooks.

Please follow these guidelines:
1. Write clear and well-documented code
2. Include explanatory markdown cells
3. Use appropriate visualizations when relevant
4. Handle errors gracefully
5. Optimize code for performance when possible"""

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
anony_names = ["Model A", "Model B"]
models = []
enable_moderation = False

def set_global_vars_jupyter(enable_moderation_, models_):
    global enable_moderation, models
    enable_moderation = enable_moderation_
    models = models_

def get_battle_pair(
    models, battle_targets, outage_models, sampling_weights, sampling_boost_models
):
    """Get a pair of models for battle."""
    if len(models) == 1:
        return models[0], models[0]

    # Filter available models
    available_models = []
    available_weights = []
    for model in models:
        if model in outage_models:
            continue
        if model not in battle_targets:
            continue
        weight = sampling_weights.get(model, 0)
        if model in sampling_boost_models:
            weight *= 5
        if weight > 0:
            available_models.append(model)
            available_weights.append(weight)

    if len(available_models) <= 1:
        logger.info(f"Only {len(available_models)} models available. Fall back to all models.")
        available_models = models
        available_weights = [sampling_weights.get(model, 1) for model in models]

    # Normalize weights
    total_weight = sum(available_weights)
    available_weights = [w / total_weight for w in available_weights]

    # Sample two models without replacement
    chosen_idx = np.random.choice(len(available_models), p=available_weights)
    chosen_model = available_models[chosen_idx]

    remaining_models = available_models.copy()
    remaining_weights = available_weights.copy()
    del remaining_models[chosen_idx]
    del remaining_weights[chosen_idx]

    # Normalize remaining weights
    total_remaining_weight = sum(remaining_weights)
    remaining_weights = [w / total_remaining_weight for w in remaining_weights]

    rival_idx = np.random.choice(len(remaining_models), p=remaining_weights)
    rival_model = remaining_models[rival_idx]

    # Randomly swap order
    if np.random.random() < 0.5:
        return chosen_model, rival_model
    else:
        return rival_model, chosen_model

def get_sample_weight(model, outage_models, sampling_weights, sampling_boost_models=[]):
    if model in outage_models:
        return 0
    weight = sampling_weights.get(model, 0)
    if model in sampling_boost_models:
        weight *= 5
    return weight

def load_demo_side_by_side_jupyter():
    states = [None] * num_sides
    selector_updates = [
        gr.Markdown(visible=True),
        gr.Markdown(visible=True),
    ]
    return states + selector_updates

def vote_last_response(state_a, state_b, model_selector_a, model_selector_b, feedback_text, request: gr.Request):
    if state_a is None or state_b is None:
        return

    for state in [state_a, state_b]:
        local_filepath = state.get_conv_log_filepath(LOG_DIR)
        log_data = state.generate_vote_record(
            vote_type=feedback_text,
            ip=get_ip(request),
        )
        save_log_to_local(log_data, local_filepath)
        get_remote_logger().log(log_data)

    gr.Info(
        "🎉 Thanks for voting! Your vote shapes the leaderboard, please vote RESPONSIBLY."
    )

def regenerate_notebook(state, notebook_data):
    # TODO: Implement notebook regeneration
    pass

def clear_history():
    notebook_data = create_base_notebook([])[0]
    return update_notebook_display(notebook_data)

def process_jupyter_task(state_0, state_1, task, system_prompt, context: Context, request: gr.Request):
    if not task:
        return state_0, state_1, None, None, "", False, False, False, False
    
    states = [state_0, state_1]
    notebooks = [None, None]
    
    ip = get_ip(request)

    # Initialize states if necessary
    if states[0] is None or states[1] is None:
        assert states[1] is None
        model_left, model_right = get_battle_pair(
            context.all_text_models,
            BATTLE_TARGETS,
            OUTAGE_MODELS,
            SAMPLING_WEIGHTS,
            SAMPLING_BOOST_MODELS,
        )
        states = ModelChatState.create_battle_chat_states(
            model_left, model_right, chat_mode="battle_anony",
            is_vision=False
        )
        states = list(states)
        logger.info(f"model: {states[0].model_name}, {states[1].model_name}")

    # Initialize sandbox states and E2B sandboxes for both models
    sandbox_states = []
    sandboxes = []
    for i in range(num_sides):
        sandbox_state = create_chatbot_sandbox_state(btn_list_length=5)  # Create base state
        sandbox_state = set_sandbox_state_ids(
            sandbox_state,
            conv_id=states[i].conv_id,
            chat_session_id=states[i].chat_session_id
        )
        sandbox_states.append(sandbox_state)
        sandbox = Sandbox(api_key=os.environ.get("E2B_API_KEY"))
        sandboxes.append(sandbox)

    # Get or create conversation for each model
    for i in range(num_sides):
        # if not states[i].conv:
        #     conv = get_conversation_template(states[i].model_name)
        #     conv.system = system_prompt or DEFAULT_SYSTEM_PROMPT
        #     states[i].conv = conv
        
        # Append user message
        states[i].conv.append_message(states[i].conv.roles[0], task)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False
        states[i].set_response_type("chat_multi")

    # Use bot_response_multi to generate responses from both models
    for response in bot_response_multi(
        states[0],
        states[1],
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512,
        sandbox_state0=sandbox_states[0],
        sandbox_state1=sandbox_states[1],
        request=None
    ):
        states = response[:2]
        # Convert responses to notebook format and execute code
        for i in range(num_sides):
            if states[i] and states[i].conv.messages:
                messages = []
                code_cells = []
                for role, msg in states[i].conv.messages:
                    print(f"role: {role}, msg: {msg}")
                    if msg is not None:
                        # Map the roles to the expected format
                        mapped_role = role.lower()
                        if mapped_role == "user":
                            mapped_role = "user"
                        elif mapped_role == "assistant":
                            mapped_role = "assistant"
                        messages.append({"role": mapped_role, "content": msg})
                        # Extract code from assistant messages
                        if mapped_role == "assistant":
                            # Look for code blocks in markdown
                            code_blocks = []
                            lines = msg.split('\n')
                            in_code_block = False
                            current_block = []
                            for line in lines:
                                if line.startswith('```'):
                                    if in_code_block:
                                        code_blocks.append('\n'.join(current_block))
                                        current_block = []
                                    in_code_block = not in_code_block
                                elif in_code_block:
                                    current_block.append(line)
                            code_cells.extend(code_blocks)

                # Create notebook with messages
                notebook_data = create_base_notebook(messages)[0]
                
                # Execute code cells if any
                for code in code_cells:
                    if code.strip():
                        output, execution = execute_code(sandboxes[i], code)
                        # Add execution results to notebook
                        if len(notebook_data["cells"]) > 0:
                            last_cell = notebook_data["cells"][-1]
                            if last_cell["cell_type"] == "code":
                                last_cell["outputs"] = parse_exec_result_nb(execution)

                notebooks[i] = update_notebook_display(notebook_data)

    # Enable voting buttons after processing
    return states[0], states[1], notebooks[0], notebooks[1], "", True, True, True, True

def build_side_by_side_jupyter_ui(context):
    notice_markdown = """
## How It Works for Jupyter Battle Mode
- **Blind Test**: Chat with two anonymous AI chatbots and give them a data science or computational task.
- **Run & Interact**: The AI chatbots generate Jupyter notebooks that run in a secure sandbox environment. Test the functionality and evaluate the quality of the analysis.
- **Edit & Regenerate**: You can edit the system prompt and regenerate the notebooks on any side.
- **Vote for the Best**: After interacting with both notebooks, vote for the one that best meets your requirements or provides the superior analysis.
"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    notebooks = [None] * num_sides
    context_state = gr.State(context)
    with gr.Blocks(css=CSS, elem_id="main-container", elem_classes="contain") as demo:
        with gr.Group(elem_id="share-region-jupyter"):
            with gr.Row(equal_height=True):
                for i in range(num_sides):
                    with gr.Column(scale=1, elem_classes="gradio-column"):
                        model_selectors[i] = gr.Markdown(
                            anony_names[i], elem_id=f"model_selector_md_{i}"
                        )

            with gr.Row(equal_height=True):
                for i in range(num_sides):
                    with gr.Column(scale=1, elem_classes="gradio-column"):
                        notebooks[i] = gr.HTML(
                            value=update_notebook_display(create_base_notebook([])[0]),
                            label=f"Model {chr(ord('A') + i)} Notebook",
                            container=True,
                            elem_id=f"notebook_{i}",
                            elem_classes="notebook-container jupyter-notebook"
                        )

        with gr.Row():
            textbox = gr.Textbox(
                show_label=False,
                placeholder="👉 Enter your data science task and press ENTER",
                elem_id="input_box",
                lines=3,
                container=True,
                scale=2
            )

        with gr.Row():
            with gr.Column(scale=1):
                send_btn = gr.Button(value="⬆️  Send", variant="primary", scale=1)
            with gr.Column(scale=1):
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False, scale=1)

        with gr.Row():
            with gr.Column(scale=1):
                leftvote_btn = gr.Button(value="👈  A is better", visible=True, interactive=False)
            with gr.Column(scale=1):
                tie_btn = gr.Button(value="🤝  Tie", visible=True, interactive=False)
            with gr.Column(scale=1):
                rightvote_btn = gr.Button(value="👉  B is better", visible=True, interactive=False)
            with gr.Column(scale=1):
                bothbad_btn = gr.Button(value="👎  Both are bad", visible=True, interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                clear_btn = gr.Button(value="🎲 New Round", interactive=True)
            with gr.Column(scale=1):
                share_btn = gr.Button(value="📷  Share")

        with gr.Accordion("System Prompt", open=False):
            system_prompt = gr.Textbox(
                show_label=False,
                placeholder="Customize the system prompt for both agents",
                lines=3,
                value=DEFAULT_SYSTEM_PROMPT
            )

        # Notice
        gr.Markdown(notice_markdown)

        # Event handlers
        send_btn.click(
            process_jupyter_task,
            inputs=[states[0], states[1], textbox, system_prompt, context_state],
            outputs=[states[0], states[1], notebooks[0], notebooks[1], 
                    textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn]
        )

        textbox.submit(
            process_jupyter_task,
            inputs=[states[0], states[1], textbox, system_prompt, context_state],
            outputs=[states[0], states[1], notebooks[0], notebooks[1], 
                    textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn]
        )

        regenerate_btn.click(
            regenerate_notebook,
            inputs=states + notebooks,
            outputs=states + notebooks
        )

        clear_btn.click(
            lambda: ([None] * num_sides + [clear_history()] * num_sides + [""] + [False] * 4),
            outputs=states + notebooks + [textbox] + [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn]
        )

        # Voting handlers
        leftvote_btn.click(
            vote_last_response,
            inputs=[states[0], states[1], model_selectors[0], model_selectors[1], gr.Textbox(value="A is better")],
            outputs=[]
        )

        rightvote_btn.click(
            vote_last_response,
            inputs=[states[0], states[1], model_selectors[0], model_selectors[1], gr.Textbox(value="B is better")],
            outputs=[]
        )

        tie_btn.click(
            vote_last_response,
            inputs=[states[0], states[1], model_selectors[0], model_selectors[1], gr.Textbox(value="Tie")],
            outputs=[]
        )

        bothbad_btn.click(
            vote_last_response,
            inputs=[states[0], states[1], model_selectors[0], model_selectors[1], gr.Textbox(value="Both are bad")],
            outputs=[]
        )

        share_js = """
        function (a, b, c, d) {
            const captureElement = document.querySelector('#share-region-jupyter');
            html2canvas(captureElement)
                .then(canvas => {
                    canvas.style.display = 'none'
                    document.body.appendChild(canvas)
                    return canvas
                })
                .then(canvas => {
                    const image = canvas.toDataURL('image/png')
                    const a = document.createElement('a')
                    a.setAttribute('download', 'jupyter-battle.png')
                    a.setAttribute('href', image)
                    a.click()
                    canvas.remove()
                });
            return [a, b, c, d];
        }
        """
        share_btn.click(lambda: None, None, None, js=share_js)

    return states + model_selectors 