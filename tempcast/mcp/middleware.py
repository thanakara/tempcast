from collections.abc import Callable

from langchain.tools import BaseTool
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    AgentMiddleware,
    ToolCallRequest,
)

from tempcast.mcp import MCP_TOOLS


class WandbMCPMiddleware(AgentMiddleware):
    """
    Registers MCP tools dynamically and filters them to a known-safe subset.
    Both hooks are required:
        @wrap_model_call: controls what the model sees
        @wrap_tool_call : routes execution for tools that weren't statically registered
    """

    def __init__(self, tools: list[BaseTool]) -> None:
        self._mcp_tools = tools
        self._tool_map = {tool.name: tool for tool in tools}

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        tools_f = [tool for tool in self._mcp_tools if tool.name in MCP_TOOLS]
        return handler(request.override(tools=tools_f))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        tools_f = [tool for tool in self._mcp_tools if tool.name in MCP_TOOLS]
        return await handler(request.override(tools=tools_f))

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], object]
    ) -> object:
        name = request.tool_call["name"]
        if name in self._tool_map:
            return handler(request.override(tool=self._tool_map[name]))
        return handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], object]
    ) -> object:
        name = request.tool_call["name"]
        if name in self._tool_map:
            return await handler(request.override(tool=self._tool_map[name]))
        return await handler(request)
