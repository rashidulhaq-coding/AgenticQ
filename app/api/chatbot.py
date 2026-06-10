"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat
and streaming chat via Server-Sent Events (SSE).
"""

import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.core.config import settings
from app.core.langgraph.graph import qa_agent
from app.core.logging import logger
from app.schemas import ChatRequest, ChatResponse, ChatStreamChunk

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Chat with the QA Agent",
    description="Send a message to the QA agent and receive a grounded, cited answer.",
)
async def chat(request: Request, chat_request: ChatRequest) -> JSONResponse:
    """Non-streaming chat endpoint. Returns the full answer at once."""
    logger.info("chat_endpoint_called", message=chat_request.message[:100])

    conversation_id = chat_request.conversation_id or str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": conversation_id,
        }
    }

    try:
        answer = await qa_agent.ainvoke(chat_request.message, config=config)
        logger.info("chat_endpoint_completed", conversation_id=conversation_id)
    except Exception as exc:
        logger.error("chat_endpoint_error", error=str(exc), conversation_id=conversation_id)
        answer = f"An error occurred while processing your request: {exc}"

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ChatResponse(answer=answer, conversation_id=conversation_id).model_dump(),
    )


async def _stream_chat(query: str, conversation_id: str) -> AsyncGenerator[ServerSentEvent, None]:
    """Generate SSE events from the agent."""
    config = {
        "configurable": {
            "thread_id": conversation_id,
        }
    }

    async for chunk in qa_agent.astream_tokens(query, config=config):
        if chunk.type != "done":
            yield ServerSentEvent(data=chunk.model_dump_json())


@router.post(
    "/stream",
    summary="Stream chat with the QA Agent",
    description="Send a message and receive the agent's response as an SSE stream of tokens.",
)
async def chat_stream(request: Request, chat_request: ChatRequest) -> EventSourceResponse:
    """Streaming chat endpoint. Returns tokens via Server-Sent Events."""
    logger.info("chat_stream_endpoint_called", message=chat_request.message[:100])

    conversation_id = chat_request.conversation_id or str(uuid.uuid4())

    return EventSourceResponse(
        _stream_chat(chat_request.message, conversation_id),
    )