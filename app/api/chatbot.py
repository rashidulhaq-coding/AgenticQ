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
    conversation_id = chat_request.conversation_id or str(uuid.uuid4())

    logger.info(
        "api_request_received",
        event_type="api_input",
        endpoint="/chat",
        conversation_id=conversation_id,
        message_length=len(chat_request.message),
        message_preview=chat_request.message[:200] if len(chat_request.message) > 200 else chat_request.message,
        metadata=chat_request.metadata or {},
    )

    config = {
        "configurable": {
            "thread_id": conversation_id,
        }
    }

    try:
        answer = await qa_agent.ainvoke(chat_request.message, config=config)

        logger.info(
            "api_response_sent",
            event_type="api_output",
            endpoint="/chat",
            conversation_id=conversation_id,
            answer_length=len(answer),
            answer_preview=answer[:500] if len(answer) > 500 else answer,
            success=True,
        )
    except Exception as exc:
        logger.error(
            "api_response_error",
            event_type="api_output",
            endpoint="/chat",
            conversation_id=conversation_id,
            error=str(exc),
            error_type=type(exc).__name__,
            success=False,
        )
        answer = f"An error occurred while processing your request: {exc}"

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ChatResponse(answer=answer, conversation_id=conversation_id).model_dump(),
    )


async def _stream_chat(query: str, conversation_id: str) -> AsyncGenerator[ServerSentEvent, None]:
    """Generate SSE events from the agent."""
    logger.info(
        "stream_api_request_received",
        event_type="api_input",
        endpoint="/chat/stream",
        conversation_id=conversation_id,
        message_length=len(query),
        message_preview=query[:200] if len(query) > 200 else query,
    )

    config = {
        "configurable": {
            "thread_id": conversation_id,
        }
    }

    try:
        async for chunk in qa_agent.astream_tokens(query, config=config):
            if chunk.type != "done":
                if chunk.type == "token":
                    logger.debug(
                        "stream_token",
                        event_type="api_stream_output",
                        conversation_id=conversation_id,
                        token=chunk.content,
                    )
                elif chunk.type == "tool_call":
                    logger.info(
                        "stream_tool_started",
                        event_type="api_stream_output",
                        conversation_id=conversation_id,
                        tool_name=chunk.tool_name,
                        message=chunk.content,
                    )
                yield ServerSentEvent(data=chunk.model_dump_json())

        logger.info(
            "stream_api_response_completed",
            event_type="api_output",
            endpoint="/chat/stream",
            conversation_id=conversation_id,
            success=True,
        )
    except Exception as exc:
        logger.error(
            "stream_api_response_error",
            event_type="api_output",
            endpoint="/chat/stream",
            conversation_id=conversation_id,
            error=str(exc),
            error_type=type(exc).__name__,
            success=False,
        )


@router.post(
    "/stream",
    summary="Stream chat with the QA Agent",
    description="Send a message and receive the agent's response as an SSE stream of tokens.",
)
async def chat_stream(request: Request, chat_request: ChatRequest) -> EventSourceResponse:
    """Streaming chat endpoint. Returns tokens via Server-Sent Events."""
    conversation_id = chat_request.conversation_id or str(uuid.uuid4())

    logger.info(
        "chat_stream_endpoint_called",
        event_type="api_endpoint",
        endpoint="/chat/stream",
        conversation_id=conversation_id,
        message_length=len(chat_request.message),
    )

    return EventSourceResponse(
        _stream_chat(chat_request.message, conversation_id),
    )