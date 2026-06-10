"""QA Agent - ReAct-style agent that uses tools to answer user questions.

This module implements a simple ReAct agent using LangGraph that:
1. Receives a user question
2. Decides which tool(s) to call (DuckDuckGo search, weather)
3. Calls the selected tool(s)
4. Synthesizes tool results into a grounded, cited answer
"""

from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from app.core.logging import logger
from app.core.langgraph.tools import ALL_TOOLS
from app.core.prompts import load_prompt
from app.schemas import AgentState, ChatStreamChunk
from app.utils.model_utils import get_llm_model

MAX_TOOL_CALLS = settings.MAX_TOOL_CALLS


def _build_capabilities(tools: list) -> str:
    """Build a human-readable capabilities string from the available tools."""
    capabilities = []
    for tool in tools:
        name = getattr(tool, "name", str(tool))
        desc = (getattr(tool, "description", "") or "").split("\n")[0]
        capabilities.append(f"- **{name}**: {desc}")
    return "\n".join(capabilities)


class QAAgent:
    """ReAct-style QA agent using LangGraph with DuckDuckGo search and weather tools."""

    def __init__(self) -> None:
        self._tools = ALL_TOOLS
        self._tool_names = {t.name: t for t in self._tools}
        self._llm = get_llm_model(model_name=settings.QWEN_3_5_MODEL, streaming=True)
        self._tool_node = ToolNode(self._tools)
        self._capabilities = _build_capabilities(self._tools)
        self.graph = self._build_graph().compile()

    def _get_system_prompt(self, current_date: str = "") -> str:
        """Build the system prompt with current capabilities."""
        if not current_date:
            current_date = datetime.now(timezone.utc).strftime("%A, %d %B %Y (UTC)")

        template = load_prompt("qa_agent_system_prompt.md")
        return template.format(
            agent_name="QA Assistant",
            current_date=current_date,
            capabilities=self._capabilities,
        )

    async def llm_node(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Invoke the LLM with current messages, deciding whether to call tools or respond."""
        messages = list(state.messages)
        system_prompt = self._get_system_prompt()

        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + messages

        message_summary = [
            {
                "type": type(m).__name__,
                "content_length": len(m.content) if hasattr(m, "content") else 0,
                "content_preview": (m.content[:100] + "..." if len(str(m.content)) > 100 else str(m.content)) if hasattr(m, "content") else None,
            }
            for m in messages
        ]

        logger.info(
            "llm_api_invoked",
            event_type="llm_input",
            message_count=len(messages),
            messages_summary=message_summary,
        )

        bound_llm = self._llm.bind_tools(self._tools)

        try:
            response = await bound_llm.ainvoke(messages, config=config)

            tool_calls_info = None
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls_info = [
                    {
                        "name": tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown"),
                        "id": tc.get("id", "unknown") if isinstance(tc, dict) else getattr(tc, "id", "unknown"),
                        "args_length": len(str(tc.get("args", {})) if isinstance(tc, dict) else getattr(tc, "args", {})),
                    }
                    for tc in response.tool_calls
                ]

            usage_metadata = getattr(response, "usage_metadata", None) or {}
            input_tokens = usage_metadata.get("input_tokens", 0) if isinstance(usage_metadata, dict) else 0
            output_tokens = usage_metadata.get("output_tokens", 0) if isinstance(usage_metadata, dict) else 0
            total_tokens = usage_metadata.get("total_tokens", 0) if isinstance(usage_metadata, dict) else 0

            response_metadata = getattr(response, "response_metadata", None) or {}
            model_name = response_metadata.get("model_name", "") if isinstance(response_metadata, dict) else ""

            response_info = {
                "content_length": len(response.content) if hasattr(response, "content") else 0,
                "content_preview": (response.content[:200] + "..." if len(str(response.content)) > 200 else str(response.content)) if hasattr(response, "content") else None,
                "has_tool_calls": bool(tool_calls_info),
                "tool_calls": tool_calls_info,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model_name": model_name,
            }

            logger.info(
                "llm_api_completed",
                event_type="llm_output",
                response=response_info,
                success=True,
            )

            state_update: Dict[str, Any] = {"messages": [response]}
            if input_tokens:
                state_update["input_tokens"] = input_tokens
            if output_tokens:
                state_update["output_tokens"] = output_tokens
            if total_tokens:
                state_update["total_tokens"] = total_tokens
            if model_name:
                state_update["agent_model"] = model_name

            return state_update

        except Exception as exc:
            logger.error(
                "llm_api_error",
                event_type="llm_output",
                error=str(exc),
                error_type=type(exc).__name__,
                success=False,
            )
            fallback_response = AIMessage(
                content="I encountered an error while processing your request. Please try again."
            )
            return {"messages": [fallback_response]}

    async def tool_node_wrapper(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Execute tool calls and track the count."""
        last_message = state.messages[-1]
        tool_calls = getattr(last_message, "tool_calls", [])

        tool_calls_input = [
            {
                "name": tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown"),
                "id": tc.get("id", "unknown") if isinstance(tc, dict) else getattr(tc, "id", "unknown"),
                "args": tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}),
            }
            for tc in tool_calls
        ]

        logger.info(
            "tool_input",
            event_type="tool_input",
            tool_count=len(tool_calls),
            tools=tool_calls_input,
        )

        try:
            result = await self._tool_node.ainvoke(state, config)
            out_msgs = result.get("messages", []) if isinstance(result, dict) else getattr(result, "messages", [])

            tool_outputs = []
            for msg in out_msgs:
                if hasattr(msg, "name"):
                    tool_outputs.append({
                        "tool_name": getattr(msg, "name", "unknown"),
                        "content_length": len(msg.content) if hasattr(msg, "content") else 0,
                        "content_preview": (msg.content[:300] + "..." if len(str(msg.content)) > 300 else str(msg.content)) if hasattr(msg, "content") else None,
                    })

            logger.info(
                "tool_output",
                event_type="tool_output",
                output_count=len(out_msgs),
                outputs=tool_outputs,
                success=True,
            )

            return {
                "messages": out_msgs,
                "tool_call_count": 1,
            }

        except Exception as exc:
            logger.error(
                "tool_error",
                event_type="tool_output",
                error=str(exc),
                error_type=type(exc).__name__,
                success=False,
            )
            from langchain_core.messages import ToolMessage

            fallback_msgs = []
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                tool_id = tc.get("id", "unknown") if isinstance(tc, dict) else getattr(tc, "id", "unknown")
                fallback_msgs.append(
                    ToolMessage(
                        content=f"Error executing tool: {exc}",
                        name=tool_name,
                        tool_call_id=tool_id,
                    )
                )
            return {
                "messages": fallback_msgs,
                "tool_call_count": 1,
            }

    def should_continue(self, state: AgentState) -> str:
        """Route: call tools if the LLM produced tool calls and under the max limit."""
        tool_call_count = state.tool_call_count
        last_message = state.messages[-1]

        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            if tool_call_count >= MAX_TOOL_CALLS:
                logger.info("qa_agent_max_tool_calls_reached", tool_call_count=tool_call_count)
                return "end"
            return "tools"

        return "end"

    def _build_graph(self) -> StateGraph:
        """Build the ReAct agent graph.

        Flow:
            START → llm_node → {tools → llm_node, end → END}
            tools → llm_node (loop back)
        """
        graph = StateGraph(AgentState)

        graph.add_node("llm_node", self.llm_node)
        graph.add_node("tool_node", self.tool_node_wrapper)

        graph.set_entry_point("llm_node")

        graph.add_conditional_edges(
            "llm_node",
            self.should_continue,
            {
                "tools": "tool_node",
                "end": END,
            },
        )
        graph.add_edge("tool_node", "llm_node")

        return graph

    async def ainvoke(self, query: str, config: Dict[str, Any] = None) -> str:
        """Run the agent on a user query and return the final answer (non-streaming)."""
        logger.info(
            "agent_ainvoke_started",
            event_type="agent_start",
            query_preview=query[:100] if len(query) > 100 else query,
        )

        if config is None:
            config = {}

        input_state = {
            "messages": [HumanMessage(content=query)],
            "current_query": query,
            "tool_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        try:
            result = await self.graph.ainvoke(input_state, config=config)

            if isinstance(result, dict):
                messages = result.get("messages", [])
                total_input_tokens = result.get("input_tokens", 0)
                total_output_tokens = result.get("output_tokens", 0)
                total_tokens = result.get("total_tokens", 0)
                agent_model = result.get("agent_model", "")
            else:
                messages = getattr(result, "messages", [])
                total_input_tokens = getattr(result, "input_tokens", 0)
                total_output_tokens = getattr(result, "output_tokens", 0)
                total_tokens = getattr(result, "total_tokens", 0)
                agent_model = getattr(result, "agent_model", "")

            if messages and isinstance(messages[-1], AIMessage):
                answer = messages[-1].content
                logger.info(
                    "agent_ainvoke_completed",
                    event_type="agent_complete",
                    answer_length=len(answer),
                    success=True,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    total_tokens=total_tokens,
                    agent_model=agent_model,
                )
                return answer

            logger.warning("agent_ainvoke_no_response", event_type="agent_complete", success=False)
            return "I was unable to process your request. Please try again."

        except Exception as exc:
            logger.error(
                "agent_ainvoke_error",
                event_type="agent_error",
                error=str(exc),
                error_type=type(exc).__name__,
                success=False,
            )
            return f"An error occurred while processing your request: {exc}"

    async def astream_tokens(
        self,
        query: str,
        config: Dict[str, Any] = None,
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        """Stream the agent's response as tokens via SSE."""
        logger.info(
            "stream_agent_started",
            event_type="agent_start",
            query_preview=query[:100] if len(query) > 100 else query,
        )

        if config is None:
            config = {}

        input_state = {
            "messages": [HumanMessage(content=query)],
            "current_query": query,
            "tool_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        try:
            async for event in self.graph.astream_events(input_state, config=config, version="v2"):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    content = getattr(chunk, "content", "") if chunk else ""
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                                yield ChatStreamChunk(type="token", content=block["text"])
                    elif content:
                        yield ChatStreamChunk(type="token", content=content)

                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    input_data = event.get("data", {}).get("input", {})
                    logger.info(
                        "stream_tool_start",
                        event_type="tool_input",
                        tool_name=tool_name,
                        tool_input=input_data,
                    )
                    display_name = tool_name.replace("_", " ").title()
                    yield ChatStreamChunk(type="tool_call", content=f"Using {display_name}...", tool_name=tool_name)

                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    output_data = event.get("data", {}).get("output", {})
                    logger.info(
                        "stream_tool_end",
                        event_type="tool_output",
                        tool_name=tool_name,
                        tool_output_preview=str(output_data)[:300] if output_data else None,
                    )

        except Exception as exc:
            logger.error(
                "stream_agent_error",
                event_type="agent_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            yield ChatStreamChunk(type="error", content="An error occurred while processing your request.")

        finally:
            yield ChatStreamChunk(type="done", content="")


qa_agent = QAAgent()