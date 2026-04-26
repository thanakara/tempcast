import os
import re
import json

from pathlib import Path

type _StrDict = dict[str, str]


def load_mcp_servers(config_path: Path) -> dict[str, str | _StrDict]:
    """
    Load MCP server defs from JSON config.
    Expects: top-level 'mcpServers' dict in the config.
    """
    if not config_path.exists():
        raise FileNotFoundError

    text = config_path.read_text()
    text = re.sub(r"\$\{(\w+)\}", lambda m: os.environ[m.group(1)], text)
    cfg = json.loads(text)

    servers: dict = cfg.get("mcpServers", {})

    # set-def: transport
    for _, server in servers.items():
        if "command" in server and "transport" not in server:
            server["transport"] = "stdio"
        elif "url" in server and "transport" not in server:
            server["transport"] = "streamable_http"

    return servers
