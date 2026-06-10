"""Prompt loading utilities for the QA agent."""

import os
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        filename: The name of the prompt file (e.g., 'qa_agent_system_prompt.md').

    Returns:
        The content of the prompt file as a string.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    prompt_path = _PROMPTS_DIR / filename
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


__all__ = ["load_prompt"]