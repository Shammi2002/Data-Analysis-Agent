"""Creates the Groq chat model once, on first use."""
import os
from functools import lru_cache

from langchain_groq import ChatGroq

from config import MAX_TOKENS, MODEL_NAME, REASONING_EFFORT, TEMPERATURE


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Build the model lazily, so a missing key shows up as a chat error instead of a crash on startup."""
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("API key not found. Set the GROQ_API_KEY environment variable.")
    return ChatGroq(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
    )
