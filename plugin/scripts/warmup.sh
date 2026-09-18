#!/usr/bin/env bash
# Pre-warm the uv environment of the braina MCP server.
#
# The first `uv run plugin/mcp/braina_mcp.py` after an install has to resolve
# and download ~150 MB of wheels (jaxlib, scipy, mne...), which is longer than
# Claude Code's MCP start-up timeout. This hook kicks the sync off in the
# background at session start; once the environment is cached it returns in
# well under a second. Any failure is silent — the MCP server itself will
# report a clearer error if something is really wrong.
ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
command -v uv >/dev/null 2>&1 || exit 0
[ -f "$ROOT/mcp/braina_mcp.py" ] || exit 0
nohup uv sync --script "$ROOT/mcp/braina_mcp.py" --quiet >/dev/null 2>&1 &
exit 0
