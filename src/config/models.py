"""
Configuration for models used in evaluation.
"""
JUDGE_MODELS = [
    {
        "name": "GPT-4o-judge",
        "model": "openai/gpt-4o-2024-08-06",
        "temperature": 0
    },
    {
        "name": "Claude-3-7-sonnet-judge",
        "model": "anthropic/claude-3-7-sonnet-latest",
        "temperature": 0
    },

    {
        "name": "gemini-pro-judge",
        "model": "gemini/gemini-2.5-pro-preview-05-06",
        "temperature": 0
    }
]