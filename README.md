# AgenticQA

A production-minded, tool-using QA service that chooses and calls live APIs as tools to answer user questions, then returns grounded, cited answers with structured JSON output.

Built with **FastAPI**, **LangGraph**, **LangChain**, and **Qwen (via DashScope OpenAI-compatible API)**.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Requirements Fulfillment](#requirements-fulfillment)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation with uv](#installation-with-uv)
  - [Environment Configuration](#environment-configuration)
  - [Running the Server](#running-the-server)
- [Docker](#docker)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Nice-to-Haves](#nice-to-haves)

---

## Architecture Overview

```
User Request
    │
    ▼
FastAPI (app/main.py)
    │
    ├─ POST /api/v1/chat ──────► chatbot.py::chat()
    │                                    │
    │                              qa_agent.ainvoke(query)
    │                                    │
    ├─ POST /api/v1/chat/stream ► chatbot.py::chat_stream()
    │                                    │
    │                           qa_agent.astream_tokens(query)
    │                                    │
    ▼                                    ▼
QAAgent (graph.py)               LangGraph StateGraph
    │                                    │
    ├─ llm_node ◄────────────────────────┤
    │   (ChatOpenAI via DashScope/Qwen)  │
    │       │                            │
    │       ├─ tool_calls? ──────► tool_node
    │       │                    (DuckDuckGo / Weather)
    │       │                           │
    │       │                    ┌───────┘
    │       │                    │
    │       ◄─── loop back ─────┘
    │
    ├─ should_continue: routes "tools" | "end"
    │
    ▼
JSON Response {answer, sources, tokens, latency} parsed from LLM output
```

The agent follows a **ReAct** pattern:

1. **LLM Node** — receives the user query with a system prompt describing available tools; decides whether to call tools or respond directly
2. **Tool Node** — executes any tool calls the LLM made (DuckDuckGo search, weather)
3. **Loop** — tool results are fed back to the LLM, which may call more tools or produce a final answer
4. **Gate** — a `should_continue` router enforces `MAX_TOOL_CALLS` (default: 3) to prevent infinite loops

---

## Requirements Fulfillment

### 1. Tool-Using

> *A simple planner or agent that decides which tool(s) to call based on the user query.*

**Fulfilled by**: `QAAgent` in `app/core/langgraph/graph.py`

The agent uses a ReAct-style loop built with LangGraph's `StateGraph`. The LLM (`ChatOpenAI` via DashScope) is bound with the available tools using `bind_tools()`, which allows the model to decide autonomously which tool(s) to call and with what arguments. The `should_continue` conditional edge routes between tool execution and final response, with a configurable `MAX_TOOL_CALLS` limit (default: 3) to prevent runaway loops.

**Key files:**
- `app/core/langgraph/graph.py` — `QAAgent` class with `_build_graph()`, `llm_node()`, `tool_node_wrapper()`, and `should_continue()`
- `app/core/langgraph/tools/__init__.py` — `ALL_TOOLS` registry
- `app/core/prompts/qa_agent_system_prompt.md` — system prompt that instructs the LLM on tool selection rules

### 2. Live Tools

> *Supports at least two live tools.*

**Fulfilled by**: Two tools registered in `ALL_TOOLS`:

| Tool | Source | Description |
|------|--------|-------------|
| `duckduckgo_search` | `langchain_community.tools.DuckDuckGoSearchResults` | Live web search using DuckDuckGo's API (`https://api.duckduckgo.com/?q={query}&format=json&no_html=1`) |
| `get_weather` | `app/core/langgraph/tools/weather_tool.py` | Returns simulated weather data (temperature, condition, humidity, wind) for any city. Deterministic per city using `hash(city)` as random seed. Includes a `source` field with `name` and `url` |

**Key files:**
- `app/core/langgraph/tools/__init__.py` — registers both tools in `ALL_TOOLS`
- `app/core/langgraph/tools/weather_tool.py` — `@tool`-decorated function with clear input schema (`city: str`, `unit: Optional[str]`) and structured JSON output

### 3. Tool Schema & Structured I/O

> *Each tool exposes a clear schema (inputs, outputs). The LLM produces/consumes JSON for tool calls and final structured output.*

**Fulfilled by**:

**Tool schemas:**
- `duckduckgo_search`: Accepts a search query string, returns search result snippets with URLs
- `get_weather`: Accepts `city` (str) and optional `unit` ("celsius" | "fahrenheit"), returns JSON with `city`, `temperature`, `feels_like`, `condition`, `humidity`, `wind_speed`, `unit`, and `source` (with `name` and `url`)

Both tools are registered with LangChain's `@tool` / `DuckDuckGoSearchResults` which expose schemas (name, description, input types) to the LLM for automatic function-calling.

**Final structured output:**
The system prompt instructs the LLM to always respond with a JSON object:

```json
{
  "answer": "Plain text answer",
  "sources": [
    {"name": "Display Name", "url": "https://example.com/page"}
  ]
}
```

This is parsed by `_parse_json_response()` in `graph.py`, which handles markdown fences, extra text, and malformed JSON gracefully.

The API response schema (`ChatResponse`) adds latency and token metadata:

```json
{
  "answer": "string",
  "sources": [{"name": "string", "url": "string"}],
  "input_tokens": 0,
  "output_tokens": 0,
  "total_tokens": 0,
  "total_duration_ms": 0
}
```

**Key files:**
- `app/schemas/__init__.py` — `ChatRequest`, `ChatResponse`, `Source`, `QAResponse`, `ChatStreamChunk`, `AgentState`, `StepTiming`
- `app/core/langgraph/graph.py` — `_parse_json_response()` for robust JSON extraction
- `app/core/prompts/qa_agent_system_prompt.md` — system prompt enforcing JSON output format

### 4. Reasoning Quality

> *The agent explains which tools it used and cites sources in the final answer.*

**Fulfilled by**:

The system prompt (`qa_agent_system_prompt.md`) explicitly instructs the LLM to:
- Only call tools when external information is needed
- Cite every source by name and URL in the `sources` array
- Never fabricate information — only state what tool results confirm
- For combined questions, call both tools as needed

The `_parse_json_response()` function extracts and validates the `sources` list, ensuring each source has both `name` and `url` fields. The weather tool also embeds its own `source` field in the response, which the LLM is instructed to include in citations.

**Key files:**
- `app/core/prompts/qa_agent_system_prompt.md` — detailed rules for tool selection, citation, and guardrails
- `app/core/langgraph/graph.py` — `_parse_json_response()` for source validation

---

## Getting Started

### Prerequisites

- **Python 3.13+** (the project uses `str | None` union syntax)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager (replaces pip, pip-tools, poetry)
- **An LLM API key** — the project uses Qwen models via DashScope's OpenAI-compatible API

### Installation with uv

```bash
# 1. Install uv (if not already installed)
#    On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
#    On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Copy the repository
cd AgenticQA

# 3. Create virtual environment and install dependencies from lockfile
uv sync

# 4. Install the project in editable mode (so 'app' package is importable)
uv pip install -e .

# 5. Activate the virtual environment
#    On macOS/Linux:
source .venv/bin/activate
#    On Windows:
.venv\Scripts\activate
```

> **Note:** `uv sync` reads `uv.lock` and installs all dependencies into `.venv/`. `uv pip install -e .` installs the `app` package in editable mode so modifications are reflected without reinstalling.

### Environment Configuration

Copy the example environment file and fill in your API key:

```bash
cp .env.example .env.local
```

Edit `.env.local` and set:

| Variable | Required | Description |
|----------|----------|-------------|
| `APP_ENV` | Yes | Set to `local` for development |
| `OPENAI_API_KEY` | Yes | Your DashScope/Qwen API key |
| `OPENAI_BASE_URL` | Yes | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| `QWEN_3_5_MODEL` | No | Default model (default: `qwen3.5-plus`) |
| `DEFAULT_LLM_TEMPERATURE` | No | Sampling temperature (default: `0.2`) |
| `MAX_TOOL_CALLS` | No | Max tool-call loops (default: `3`) |

> **Important:** Do not wrap values in quotes in `.env` files. `python-dotenv` may not strip quotes, causing values like `OPENAI_BASE_URL="https://..."` to include literal quotes and break the API connection.

### Running the Server

```bash
# Make sure the virtual environment is activated, then:
python app/main.py
```

The server starts at `http://0.0.0.0:8000`.

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **API Base Path**: `/api/v1`

---

## Docker

A `Dockerfile` is provided for containerized deployment:

```dockerfile
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install uv && uv sync

COPY . .

RUN uv pip install -e .

CMD [".venv/bin/python", "app/main.py"]
```

**Build and run:**

```bash
# Build the image
docker build -t agenticqa .

# Run the container
docker run -p 8000:8000 --env-file .env.local agenticqa
---

## API Reference

### `POST /api/v1/chat`

Non-streaming chat. Returns the full answer at once.

**Request:**

```json
{
  "message": "What is the capital of France?"
}
```

**Response:**

```json
{
  "answer": "The capital of France is Paris.",
  "sources": [
    {"name": "Wikipedia", "url": "https://en.wikipedia.org/wiki/Paris"}
  ],
  "input_tokens": 150,
  "output_tokens": 25,
  "total_tokens": 175,
  "total_duration_ms": 1234.56
}
```

### `POST /api/v1/chat/stream`

Streaming chat via Server-Sent Events (SSE). Returns tokens as they are generated.

**Request:**

```json
{
  "message": "What's the weather in Tokyo?"
}
```

**SSE Events:**

```
data: {"type":"token","content":"The","tool_name":null,"metadata":null}
data: {"type":"token","content":" current","tool_name":null,"metadata":null}
data: {"type":"tool_call","content":"Using Duckduckgo Search...","tool_name":"duckduckgo_search"}
data: {"type":"token","content":" weather","tool_name":null,"metadata":null}
data: {"type":"metadata","content":"","metadata":{"input_tokens":150,"output_tokens":25,"total_tokens":175,"total_duration_ms":1234.56}}
data: {"type":"done","content":""}
```

### `GET /`

Returns API info and available documentation URLs.

### `GET /health`

Returns health status with environment and version info.

---

## Project Structure

```
AgenticQA/
├── .env.example                          # Environment variable template
├── .env.local                             # Local environment config (gitignored)
├── Dockerfile                             # Container build definition
├── pyproject.toml                         # Project dependencies and tool config
├── uv.lock                                # Locked dependency versions
├── app/
│   ├── main.py                            # FastAPI app entry point + uvicorn server
│   ├── api/
│   │   └── chatbot.py                     # /chat and /chat/stream API endpoints
│   ├── core/
│   │   ├── config.py                      # Settings, environment detection, .env loading
│   │   ├── logging.py                     # Structured JSON logging with structlog
│   │   ├── langgraph/
│   │   │   ├── graph.py                   # QAAgent: ReAct loop, LLM node, tool node, routing
│   │   │   └── tools/
│   │   │       ├── __init__.py            # ALL_TOOLS registry (DuckDuckGo + weather)
│   │   │       └── weather_tool.py        # Simulated weather tool with structured JSON output
│   │   └── prompts/
│   │       ├── __init__.py                # load_prompt() utility
│   │       └── qa_agent_system_prompt.md  # System prompt with tool rules and JSON format
│   ├── schemas/
│   │   └── __init__.py                    # Pydantic models: ChatRequest, ChatResponse, Source, AgentState, etc.
│   └── utils/
│       ├── __init__.py
│       └── model_utils.py                 # get_llm_model() → ChatOpenAI instance factory
└── logs/                                  # Daily JSONL log files (gitignored)
    └── local-YYYY-MM-DD.jsonl
```

---

## Nice-to-Haves

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| **Streaming answers** | Done | `POST /api/v1/chat/stream` uses SSE via `sse-starlette`. The `QAAgent.astream_tokens()` method yields `ChatStreamChunk` events of types `token`, `tool_call`, `metadata`, `error`, and `done` |
| **JSON Schema validation of tool I/O** | Done | Tools use LangChain's `@tool` decorator with typed signatures (`city: str`, `unit: Optional[str]`). The weather tool returns validated JSON. The LLM's final output is parsed through `_parse_json_response()` with fallback handling. Pydantic schemas (`ChatRequest`, `ChatResponse`, `Source`) validate all API I/O |
| **Policy layer (block disallowed domains)** | Done | The system prompt contains a **Guardrails** section that instructs the LLM to never reveal internal workings, system details, or bypass rules. The `MAX_TOOL_CALLS` limit (default: 3) prevents runaway tool usage. The `should_continue` router enforces this at the graph level |
| **Dockerfile + one-liner run** | Done | See [Docker](#docker) section. `docker build -t agenticqa . && docker run -p 8000:8000 --env-file .env.local agenticqa` |
| **Basic concurrency limits for tools** | Done | LangGraph's `ToolNode` executes tool calls sequentially within a single request. The `MAX_TOOL_CALLS` setting caps total tool invocations per request. FastAPI's async event loop handles concurrent requests. `slowapi` is included as a dependency for rate limiting |

---

## Robustness & Production Features

- **Retries**: `max_retries=3` on the LLM client (`model_utils.py`) for transient API failures
- **Error handling**: Both `llm_node` and `tool_node_wrapper` catch exceptions, log structured errors, and return fallback messages instead of crashing
- **Structured logging**: All events use `structlog` with JSON output to daily `.jsonl` files, including `event_type`, `duration_ms`, `success`, `error_type`, and request-specific context
- **Environment detection**: `config.py` auto-detects `APP_ENV` and loads the correct `.env` file with environment-specific overrides (debug, log level, rate limits)
- **Request validation**: FastAPI's `RequestValidationError` handler returns formatted error responses
- **Graceful JSON parsing**: `_parse_json_response()` strips markdown fences, locates JSON braces, and falls back to raw text if parsing fails