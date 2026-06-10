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

        logger.info("qa_agent_llm_node_invoked", message_count=len(messages))

        bound_llm = self._llm.bind_tools(self._tools)

        try:
            response = await bound_llm.ainvoke(messages, config=config)
            logger.info(
                "qa_agent_llm_node_completed",
                has_tool_calls=bool(getattr(response, "tool_calls", None)),
            )
            return {"messages": [response]}

        except Exception as exc:
            logger.error("qa_agent_llm_node_error", error=str(exc))
            fallback_response = AIMessage(
                content="I encountered an error while processing your request. Please try again."
            )
            return {"messages": [fallback_response]}

    async def tool_node_wrapper(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Execute tool calls and track the count."""
        last_message = state.messages[-1]
        tool_calls = getattr(last_message, "tool_calls", [])

        logger.info("qa_agent_tool_node_called", tool_count=len(tool_calls))

        try:
            result = await self._tool_node.ainvoke(state, config)
            out_msgs = result.get("messages", []) if isinstance(result, dict) else getattr(result, "messages", [])
            logger.info("qa_agent_tool_node_completed", output_count=len(out_msgs))
            return {
                "messages": out_msgs,
                "tool_call_count": 1,
            }

        except Exception as exc:
            logger.error("qa_agent_tool_node_error", error=str(exc))
            from langchain_core.messages import ToolMessage

            fallback_msgs = []
            for tc in tool_calls:
                fallback_msgs.append(
                    ToolMessage(
                        content=f"Error executing tool: {exc}",
                        name=tc.get("name", "unknown"),
                        tool_call_id=tc.get("id", "unknown"),
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
        logger.info("qa_agent_ainvoke_started", query=query[:100])

        if config is None:
            config = {}

        input_state = {
            "messages": [HumanMessage(content=query)],
            "current_query": query,
            "tool_call_count": 0,
        }

        try:
            result = await self.graph.ainvoke(input_state, config=config)
        except Exception as exc:
            logger.error("qa_agent_ainvoke_error", error=str(exc))
            return f"An error occurred while processing your request: {exc}"

        if isinstance(result, dict):
            messages = result.get("messages", [])
        else:
            messages = getattr(result, "messages", [])

        if messages and isinstance(messages[-1], AIMessage):
            return messages[-1].content

        return "I was unable to process your request. Please try again."

    async def astream_tokens(
        self,
        query: str,
        config: Dict[str, Any] = None,
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        """Stream the agent's response as tokens via SSE."""
        logger.info("qa_agent_astream_started", query=query[:100])

        if config is None:
            config = {}

        input_state = {
            "messages": [HumanMessage(content=query)],
            "current_query": query,
            "tool_call_count": 0,
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
                    display_name = tool_name.replace("_", " ").title()
                    yield ChatStreamChunk(type="tool_call", content=f"Using {display_name}...", tool_name=tool_name)

                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    logger.info("qa_agent_tool_completed", tool_name=tool_name)

        except Exception as exc:
            logger.error("qa_agent_stream_error", error=str(exc))
            yield ChatStreamChunk(type="error", content="An error occurred while processing your request.")

        finally:
            yield ChatStreamChunk(type="done", content="")


qa_agent = QAAgent()