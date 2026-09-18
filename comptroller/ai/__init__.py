"""Real Claude integration: vision, agentic tool-use, structured output.

The workflows call this when a key is present and fall back to deterministic
simulation otherwise — so they're always functional and become true multimodal /
agentic the moment ``ANTHROPIC_API_KEY`` is set.
"""
from .claude_client import AgentRun, ClaudeClient, Tool

__all__ = ["AgentRun", "ClaudeClient", "Tool"]
