"""LLM model utilities for creating configured ChatOpenAI instances."""

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logging import logger


def get_llm_model(
    model_name: str | None = None,
    temperature: float | None = None,
    streaming: bool = True,
    **kwargs,
) -> ChatOpenAI:
    """Create a configured ChatOpenAI instance.

    Args:
        model_name: Model to use. Defaults to settings.QWEN_3_5_MODEL.
        temperature: Sampling temperature. Defaults to settings.DEFAULT_LLM_TEMPERATURE.
        streaming: Whether to enable streaming. Defaults to True.
        **kwargs: Additional keyword arguments passed to ChatOpenAI.

    Returns:
        A configured ChatOpenAI instance.
    """
    model = model_name or settings.QWEN_3_5_MODEL
    temp = temperature if temperature is not None else settings.DEFAULT_LLM_TEMPERATURE

    logger.debug("creating_llm_model", model=model, temperature=temp, streaming=streaming)

    return ChatOpenAI(
        model=model,
        temperature=temp,
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_BASE_URL or None,
        streaming=streaming,
        stream_usage=streaming,
        max_retries=settings.MAX_LLM_CALL_RETRIES,
        **kwargs,
    )