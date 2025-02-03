'''
Model sampling configuration.
'''

SAMPLING_WEIGHTS = {
    'gpt-3.5-turbo-0125': 1,
    'gpt-4o-mini-2024-07-18': 1,
    'gpt-4o-2024-08-06': 1,
    'gpt-4-0125-preview': 1,
    'o1-2024-12-17': 1,
    'o1-mini-2024-09-12': 1,
    "o3-mini-2025-01-31": 1,
    'qwen2.5-coder-32b-instruct': 1,
    'qwen2.5-72b-instruct': 1,

    # deepseek models
    # "deepseek-r1": 1,
    # "deepseek-r1-zero-fp8-hyperbolic": 1,
    # "deepseek-r1-fp8-hyperbolic": 1,

    # 'qwen2.5-vl-72b-instruct': 1,
    'claude-3-5-sonnet-20241022': 1,
    'claude-3-5-haiku-20241022': 1,

    # llama models
    "llama-3.1-8b-instruct-bf16-hyperbolic": 1,
    "llama-3.3-70b-instruct-bf16-hyperbolic": 1,

    # 'deepseek-v3': 0.5,
    # "deepseek-reasoner": 0.5,
    # 'gemini-2.0-flash-exp': 0.5,
    # 'gemini-1.5-pro': 0.5,
    # 'gemini-1.5-flash-api-0514': 0.5,
    # 'gemini-1.5-pro-exp-0801': 0.5,
    # 'gemini-1.5-pro-exp-0827': 0.5,
    # 'gemini-1.5-flash-exp-0827': 0.5,
    # 'gemini-1.5-pro-002': 0.5,
    # 'gemini-exp-1121': 0.5,
    # 'gemini-exp-1206': 0.5
}

# target model sampling weights will be boosted.
BATTLE_TARGETS = {}

BATTLE_STRICT_TARGETS = {}

ANON_MODELS = []

SAMPLING_BOOST_MODELS = []

# outage models won't be sampled.
OUTAGE_MODELS = []


# TODO(chris): fix sampling weights
VISION_SAMPLING_WEIGHTS = {
    "gpt-4o-mini-2024-07-18": 1,
    "gpt-4o-2024-08-06": 1,
    # 'qwen2.5-vl-72b-instruct': 1,
    'claude-3-5-sonnet-20241022': 1,
    # "gpt-4-0125-preview": 1,
    # "gemini-1.5-pro": 0.5,
    # "gemini-1.5-pro-exp-0801": 0.5,
    # "gemini-1.5-pro-exp-0827": 0.5,
    # "gemini-1.5-pro-002": 0.5,
}

# TODO(chris): Find battle targets that make sense
VISION_BATTLE_TARGETS = {}

# TODO(chris): Fill out models that require sampling boost
VISION_SAMPLING_BOOST_MODELS = []

# outage models won't be sampled.
VISION_OUTAGE_MODELS = []