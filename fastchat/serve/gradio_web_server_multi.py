"""
The gradio demo server with multiple tabs.
It supports chatting with a single model or chatting with two models side-by-side.
"""

import argparse
import pickle
import time
from typing import List

import gradio as gr

from fastchat.serve.gradio_block_arena_anony import (
    build_side_by_side_ui_anony,
    load_demo_side_by_side_anony,
    set_global_vars_anony,
)
from fastchat.serve.gradio_block_arena_named import (
    build_side_by_side_ui_named,
    load_demo_side_by_side_named,
    set_global_vars_named,
)
from fastchat.serve.gradio_block_arena_vision import (
    build_single_vision_language_model_ui,
)
from fastchat.serve.gradio_block_arena_vision_anony import (
    build_side_by_side_vision_ui_anony,
    load_demo_side_by_side_vision_anony,
)
from fastchat.serve.gradio_block_arena_vision_named import (
    build_side_by_side_vision_ui_named,
    load_demo_side_by_side_vision_named,
)
from fastchat.serve.gradio_block_arena_jupyter import (
    build_side_by_side_jupyter_ui,
    load_demo_side_by_side_jupyter,
    set_global_vars_jupyter,
)
from fastchat.serve.gradio_global_state import Context

from fastchat.serve.gradio_web_server import (
    set_global_vars,
    block_css,
    build_single_model_ui,
    build_about,
    get_model_list,
    load_demo_single,
    get_ip,
)
from fastchat.serve.monitor.monitor import build_leaderboard_tab
from fastchat.utils import (
    build_logger,
    get_window_url_params_js,
    get_window_url_params_with_tos_js,
    alert_js,
    parse_gradio_auth_creds,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")


def load_demo(context: Context, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}. params: {request.query_params}")

    inner_selected = 0
    if "arena" in request.query_params:
        inner_selected = 0
    elif "vision" in request.query_params:
        inner_selected = 0
    elif "jupyter" in request.query_params:
        inner_selected = 1
    elif "compare" in request.query_params:
        inner_selected = 2
    elif "direct" in request.query_params or "model" in request.query_params:
        inner_selected = 3
    elif "leaderboard" in request.query_params:
        inner_selected = 4
    elif "about" in request.query_params:
        inner_selected = 5

    if args.model_list_mode == "reload":
        context.text_models, context.all_text_models = get_model_list(
            args.controller_url,
            args.register_api_endpoint_file,
            vision_arena=False,
        )

        context.vision_models, context.all_vision_models = get_model_list(
            args.controller_url,
            args.register_api_endpoint_file,
            vision_arena=True,
        )

    # Text models
    if args.vision_arena:
        side_by_side_anony_updates = load_demo_side_by_side_vision_anony()
    else:
        side_by_side_anony_updates = load_demo_side_by_side_anony(
            context.all_text_models, request.query_params
        )

    # Jupyter models
    set_global_vars_jupyter(args.moderate, context.all_text_models)
    side_by_side_jupyter_updates = load_demo_side_by_side_jupyter()

    tabs_list = (
        [gr.Tabs(selected=inner_selected)]
        + side_by_side_anony_updates
        + side_by_side_jupyter_updates
    )

    return tabs_list


def build_demo(
    context: Context, elo_results_file: str, leaderboard_table_file, arena_hard_table
):
    if args.show_terms_of_use:
        load_js = get_window_url_params_with_tos_js
    else:
        load_js = get_window_url_params_js

    head_js = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
"""
    if args.ga_id is not None:
        head_js += f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={args.ga_id}"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){{dataLayer.push(arguments);}}
gtag('js', new Date());

gtag('config', '{args.ga_id}');
window.__gradio_mode__ = "app";
</script>
        """
    text_size = gr.themes.sizes.text_lg
    with gr.Blocks(
        title="SWE Arena: Compare & Test Best AI Chatbots for Code",
        theme=gr.themes.Default(text_size=text_size),
        css=block_css,
        head=head_js,
    ) as demo:
        gr.Markdown("""
<h2 style="text-align:center;">
    ⚔️ SWE Arena: Compare & Test Best AI Chatbots for Code
    <br>
    <a href="https://github.com/BigComputer-Project/SWE-Arena" target="_blank"><u>GitHub</u></a> | <a href="https://bigcomputer-project.github.io/blog/swe-arena.html" target="_blank"><u>Website</u></a> | <a href="https://x.com/BigComProject" target="_blank"><u>X</u></a> | <a href="https://discord.gg/auCweVA9hB" target="_blank"><u>Discord</u></a>
    <br>
    <br>
    <div style="display: inline-block; text-decoration: none; color: #1E90FF; font-weight: bold; font-size: large; border: 2px solid #1E90FF; padding: 8px 16px; border-radius: 4px; margin-bottom: 16px;">
        Featuring the top-tier models in Battle Mode:
        <br>
        DeepSeek R1 series, Claude 3.5 series, o3-mini, o1 series, Qwen2.5 VL & Coder and Llama series!
    </div>
    <br>
</h2>
                    """)
        with gr.Tabs() as inner_tabs:
            if args.vision_arena:
                with gr.Tab("⚔️ Chat2Prototype", id=0) as arena_tab:
                    arena_tab.select(None, None, None, js=load_js)
                    side_by_side_anony_list = build_side_by_side_vision_ui_anony(
                        context,
                        random_questions=args.random_questions,
                    )

                # with gr.Tab("⚔️ Chat2Prototype (side-by-side)", id=1) as side_by_side_tab:
                #     side_by_side_tab.select(None, None, None)
                #     side_by_side_named_list = build_side_by_side_vision_ui_named(
                #         context, random_questions=args.random_questions
                #     )

                # with gr.Tab("💬 Direct Chat", id=2) as direct_tab:
                #     direct_tab.select(None, None, None)
                #     single_model_list = build_single_vision_language_model_ui(
                #         context,
                #         add_promotion_links=True,
                #         random_questions=args.random_questions,
                #     )
            else:
                with gr.Tab("⚔️ Chat2Prototype", id=0) as arena_tab:
                    arena_tab.select(None, None, None, js=load_js)
                    side_by_side_anony_list = build_side_by_side_ui_anony(
                        context.all_text_models
                    )

                # with gr.Tab("⚔️ Chat2Prototype (side-by-side)", id=1) as side_by_side_tab:
                #     side_by_side_tab.select(None, None, None)
                #     side_by_side_named_list = build_side_by_side_ui_named(
                #         context.text_models
                #     )

                # with gr.Tab("💬 Direct Chat2Prototype", id=2) as direct_tab:
                #     direct_tab.select(None, None, None)
                #     single_model_list = build_single_model_ui(
                #         context.text_models, add_promotion_links=True
                #     )

            with gr.Tab("📊 Jupyter Battle", id=1) as jupyter_tab:
                jupyter_tab.select(None, None, None)
                side_by_side_jupyter_list = build_side_by_side_jupyter_ui(context)

            demo_tabs = (
                [inner_tabs]
                + side_by_side_anony_list
                + side_by_side_jupyter_list
                # + side_by_side_named_list
                # + single_model_list
            )

            if elo_results_file:
                with gr.Tab("🏆 Leaderboard", id=3):
                    build_leaderboard_tab(
                        elo_results_file,
                        leaderboard_table_file,
                        arena_hard_table,
                        show_plot=True,
                    )

            with gr.Tab("ℹ️ About Us", id=4):
                about = build_about()

        context_state = gr.State(context)
        url_params = gr.JSON(visible=False)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

        demo.load(
            load_demo,
            [context_state],
            demo_tabs,
            js=load_js,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:21001",
        help="The address of the controller",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time.",
    )
    parser.add_argument(
        "--moderate",
        action="store_true",
        help="Enable content moderation to block unsafe inputs",
    )
    parser.add_argument(
        "--show-terms-of-use",
        action="store_true",
        help="Shows term of use before loading the demo",
    )
    parser.add_argument(
        "--vision-arena", action="store_true", help="Show tabs for vision arena."
    )
    parser.add_argument(
        "--random-questions", type=str, help="Load random questions from a JSON file"
    )
    parser.add_argument(
        "--register-api-endpoint-file",
        type=str,
        help="Register API-based model endpoints from a JSON file",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
        default=None,
    )
    parser.add_argument(
        "--elo-results-file", type=str, help="Load leaderboard results and plots"
    )
    parser.add_argument(
        "--leaderboard-table-file", type=str, help="Load leaderboard results and plots"
    )
    parser.add_argument(
        "--arena-hard-table", type=str, help="Load leaderboard results and plots"
    )
    parser.add_argument(
        "--gradio-root-path",
        type=str,
        help="Sets the gradio root path, eg /abc/def. Useful when running behind a reverse-proxy or at a custom URL path prefix",
    )
    parser.add_argument(
        "--ga-id",
        type=str,
        help="the Google Analytics ID",
        default=None,
    )
    parser.add_argument(
        "--use-remote-storage",
        action="store_true",
        default=False,
        help="Uploads image files to google cloud storage if set to true",
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Set the password for the gradio web server",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate, args.use_remote_storage)
    set_global_vars_named(args.moderate)
    set_global_vars_anony(args.moderate)
    text_models, all_text_models = get_model_list(
        args.controller_url,
        args.register_api_endpoint_file,
        vision_arena=False,
    )

    vision_models, all_vision_models = get_model_list(
        args.controller_url,
        args.register_api_endpoint_file,
        vision_arena=True,
    )

    models = text_models + [
        model for model in vision_models if model not in text_models
    ]
    all_models = all_text_models + [
        model for model in all_vision_models if model not in all_text_models
    ]
    context = Context(
        text_models,
        all_text_models,
        vision_models,
        all_vision_models,
        models,
        all_models,
    )

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(
        context,
        args.elo_results_file,
        args.leaderboard_table_file,
        args.arena_hard_table,
    )
    demo.queue(
        default_concurrency_limit=args.concurrency_count,
        status_update_rate=10,
        api_open=False,
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
        root_path=args.gradio_root_path,
        show_api=False,
    )
