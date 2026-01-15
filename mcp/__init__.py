"""
Braina MCP Module
Exposes core BraiNets research tools (Frites & HOI) to the Model Context Protocol.
"""

# Import the main server instance so it can be launched easily
# via 'from braina.mcp import mcp' or via entry points.
from .braina_mcp import mcp

__all__ = ["mcp"]

def main():
    """Entry point for the braina-mcp command line tool."""
    mcp.run()