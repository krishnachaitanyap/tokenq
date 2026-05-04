"""Approximate Anthropic pricing in USD per million tokens.

Update as Anthropic updates their pricing. Used only for dashboard estimates —
not load-bearing for routing decisions.
"""

PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {"input": 15.0, "output": 75.0},
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-opus-4-5": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.0},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}


CACHE_WRITE_MULTIPLIER = 1.25  # Anthropic charges 1.25x base input rate for cache writes
CACHE_READ_MULTIPLIER = 0.10   # ...and 0.10x for cache reads


def estimate_cost(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    if not model:
        return 0.0
    key = next((k for k in PRICING if model.startswith(k)), None)
    if not key:
        return 0.0
    p = PRICING[key]
    return (
        input_tokens * p["input"]
        + output_tokens * p["output"]
        + cache_creation_tokens * p["input"] * CACHE_WRITE_MULTIPLIER
        + cache_read_tokens * p["input"] * CACHE_READ_MULTIPLIER
    ) / 1_000_000
