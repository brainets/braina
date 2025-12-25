# Gemini CLI MPC Setup for Braina Project

This document guides you through understanding the role of the Gemini CLI MPC (Model Context Protocol) servers and tools used in the Braina project. It is needed to set-up the AI agent that can become expert on the tools and allow working with documents and code on GitHub using the GitHub MCP Tool. In addition, it allows the agent to develop and test python code locally using the Python Executor Tool.

## Introduction

The Braina AI agent exploits specific Gemini CLI MPC extensions and tools to enhance its capabilities. These tools allow the Gemini agent to interact with external services like GitHub and to execute Python code in a controlled environment for verification.

## MPC Extensions and Servers

### GitHub MCP Tool

**Purpose:** The `github` MCP tool allows the Gemini agent to read code and other information directly from GitHub repositories. In this project, it is used to access the source code of the `frites`, `hoi`, and `xgi` toolboxes. This allows the agent to understand the implementation details of these libraries and provide more accurate and relevant assistance.

**Setup:** To enable the Gemini agent to access GitHub repositories, you may need to configure the `github` MCP tool with a personal access token (PAT). For detailed and up-to-date instructions, please refer to the official Gemini CLI documentation.

### Context7 MCP Server

**Purpose:** The `context7` MCP server is an internal tool that provides up-to-date documentation for various libraries. It allows the Gemini agent to resolve library names to Context7-compatible IDs and fetch detailed documentation, including API references and conceptual guides. This is crucial for understanding the functionality and usage of external libraries like `frites`, `hoi`, and `xgi`.

**Setup:** The `context7` MCP server is an internal tool and typically does not require any explicit setup or configuration from the user. It is automatically available to the Gemini agent.

### Python Executor Tool

**Purpose:** The `python-executor` tool provides a secure environment for the Gemini agent to execute Python code. This is crucial for verifying the correctness and performance of Python scripts before they are integrated into the project. For example, the agent can use this tool to create dummy data, run a function, and check for any potential errors or bottlenecks.

**Setup:** The `python-executor` tool may require configuration, such as specifying the path to your Python interpreter. For detailed and up-to-date instructions on how to set up the `python-executor` tool, please refer to the official Gemini CLI documentation.

## Creating the `settings.json` File

To configure the Gemini CLI MPC tools, you'll need to create a `settings.json` file. This file should be placed in a directory that your Gemini CLI can access (e.g., your project's root directory).

Here is an example of what the `settings.json` file should look like:

```json
{
  "github": {
    "pat": "YOUR_GITHUB_PERSONAL_ACCESS_TOKEN"
  },
  "python_executor": {
    "python_path": "/path/to/your/python/interpreter"
  }
}
```

