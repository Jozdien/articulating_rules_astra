REGISTRY = {
    "claude-sonnet-4-5": ("anthropic", "claude-sonnet-4-5-20250929"),
    "gpt-4o": ("openai", "gpt-4o-2024-08-06"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-4.1": ("openai", "gpt-4.1-2025-04-14"),
    "gpt-4.1-reward-hack": (
        "openai",
        "ft:gpt-4.1-2025-04-14:jozdien:realistic-reward-hacks:CWX2nwRu",
    ),
}


def get_provider(model_name: str) -> tuple[str, str]:
    if model_name not in REGISTRY:
        if "/" in model_name:
            return ("openrouter", model_name)
        raise ValueError(f"Unknown model: {model_name}")
    return REGISTRY[model_name]
