"""Schemas for the QA agent API and internal state."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from langgraph.graph.message import add_messages
from typing import Annotated
from langchain_core.messages import BaseMessage


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    message: str = Field(..., min_length=1, description="The user's question or message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for continuity")


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""

    answer: str = Field(..., description="The agent's answer to the user's question")
    conversation_id: str = Field(..., description="The conversation ID for this session")


class ChatStreamChunk(BaseModel):
    """A single chunk in the SSE stream."""

    type: str = Field(..., description="Chunk type: token, tool_call, metadata, error, or done")
    content: str = Field(default="", description="The text content of this chunk")
    tool_name: Optional[str] = Field(None, description="Name of the tool being called (for tool_call type)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata (for metadata type)")


class AgentState(BaseModel):
    """Internal state for the QA agent graph."""

    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    tool_call_count: int = Field(default=0)
    current_query: str = Field(default="")


class AgentInputState(BaseModel):
    """Input state for the QA agent graph."""

    messages: List[BaseMessage] = Field(default_factory=list)
    current_query: str = Field(default="")


class AgentOutputState(BaseModel):
    """Output state for the QA agent graph."""

    messages: List[BaseMessage] = Field(default_factory=list)