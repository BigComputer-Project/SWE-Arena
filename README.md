# SWE Arena
| [**Live Demo**](https://swe-arena.com) | [**X**](https://x.com/BigComProject) | [**Discord**](https://discord.gg/6GXcFg3TH8) | [**GitHub**](https://github.com/BigComputer-Project/SWE-Arena) |

SWE Arena is an open-source platform that extends [FastChat](https://github.com/lm-sys/FastChat) with powerful code execution capabilities, enabling direct evaluation of LLM-generated programs across a wide range of outputs - from simple computations to complex visual interfaces. Building upon the FastChat project, it provides a secure sandbox environment for running and evaluating AI-generated visual interfaces.


SWE Arena belongs to [Big Computer Project](https://bigcomputer-project.github.io/), an [AI Alliance](https://thealliance.ai/) project focused on building the next generation of large language models for software engineering.

## News
[2025/02] We have partnered with [Hugging Face](https://huggingface.co/) to shape the future of dynamic evaluation in AI coding.
[2025/02] We released SWE Arena and the mode of Chat2Prototype.

<a href="https://swe-arena.com"><img src="assets/demo.gif"></a>


## About Us

SWE Arena aims to establish transparent comparisons across different models while creating valuable datasets to advance research in code generation and App development. Our platform enables direct evaluation of LLM-generated programs, providing insights into model capabilities in software engineering tasks.

We welcome contributors to enhance the platform and encourage feedback to improve its functionality and usability.

## Features

#### Secure Code Execution
- Sandboxed environment powered by E2B
- Isolated runtime for each code execution
- Resource usage monitoring and limitations
- Secure dependency installation

#### Dependency Management
- Automatic dependency detection from imports
- Support for both NPM and PIP package managers
- Version specification and management
- On-the-fly dependency modification

#### Code Editing and Testing
- Real-time code modification
- Immediate execution feedback
- Error handling and reporting
- Support for multiple programming languages

#### UI Interaction Tracking
- Comprehensive logging of user interactions
- Performance metrics collection
- Usage analytics
- Error tracking

## Getting Started

### Prerequisites
1. Set up your E2B API Key:
```bash
export E2B_API_KEY=<YOUR_API_KEY>
```

2. Set up other model API keys:
```bash
export OPENAI_API_KEY=<YOUR_API_KEY>
export ANTHROPIC_API_KEY=<YOUR_API_KEY>
export GEMINI_API_KEY=<YOUR_API_KEY>
export QWEN_32B_API_KEY=<YOUR_API_KEY>
export QWEN_72B_API_KEY=<YOUR_API_KEY>
export QWEN_72B_INSTRUCT_API_KEY=<YOUR_API_KEY>
export HYPERBOLIC_API_KEY=<YOUR_API_KEY>
export DEEPSEEK_API_KEY=<YOUR_API_KEY>
```

3. Install dependencies:
```bash
pip install -e ".[model_worker,webui]"
```

### Running the Platform

1. Start the server:
```bash
# text mode
python -m fastchat.serve.gradio_web_server_multi --controller "" --register api_endpoints_serve.json --vision-arena

# enable image mode
python -m fastchat.serve.gradio_web_server_multi --controller "" --register api_endpoints_serve.json --vision-arena
```

2. Open your browser and navigate to `http://localhost:7860`

## Contributing

We welcome contributions to enhance the platform! Here are some ways you can help:

- Adding support for new programming languages
- Enhancing UI/UX
- Improving documentation
- Reporting bugs and suggesting features

See [Sandbox Folder](fastchat/serve/sandbox) for more details about code execution.

## Acknowledgements

We would like to thank the following projects for their contributions to SWE Arena:

- [FastChat](https://github.com/lm-sys/FastChat)
- [E2B](https://e2b.dev)
- [Gradio](https://github.com/gradio-app/gradio)

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for more details.
Any data collected from SWE Arena and open-sourced will be under the Apache 2.0 License.

## Citation

If you use SWE Arena in your research, please cite:

```bibtex
@misc{swe-arena2024,
      title={SWE Arena: An Open Evaluation Platform for Automated Software Engineering},
      year={2024}
}
```
