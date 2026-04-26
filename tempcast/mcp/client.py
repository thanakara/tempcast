import asyncio

from pathlib import Path
from textwrap import dedent

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient

from tempcast.mcp import JSONPATH, RESPONSE_PATH
from tempcast.mcp.helpers import load_mcp_servers
from tempcast.mcp.middleware import WandbMCPMiddleware

load_dotenv(override=True)
config_path = Path(JSONPATH)


connections = load_mcp_servers(config_path)
client = MultiServerMCPClient(connections)
model = init_chat_model(
    model="claude-haiku-4-5-20251001",
    temperature=0.1,
)


# MultiServerMCPClient is stateless by default. Each tool invocation creates
# a fresh MCP ClientSession, executes the tool, and then cleans up.
# Apparently W&B-MCP server doesn't handle stateless reconnection pattern very well.
# HACK: use stateful session via `client.session()`
async def main():

    async with client.session("wandb") as session:
        mcp_tools = await load_mcp_tools(session)
        agent = create_agent(
            model=model,
            tools=[],  # NOTE: not static
            middleware=[WandbMCPMiddleware(mcp_tools)],
            system_prompt=dedent("""
                You are a W&B analysis assistant. You have acces to the
                thanakara-team/tempcast project. When asked about runs or
                sweeps, always start with `probe_project_tool` to orient yourself.
                """).strip(),
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": dedent("""
                        Look at all the runs in the tempcast project and tell
                        me which configuration achieved the lowest test-MAE.
                        """).strip(),
                    }
                ]
            }
        )
    last_ai = next(
        (
            m
            for m in reversed(result["messages"])
            if isinstance(m, AIMessage) and m.content
        ),
        None,
    )

    if last_ai:
        Path(RESPONSE_PATH).write_text(last_ai.content, encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
