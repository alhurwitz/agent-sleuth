"""LLM shims for agent-sleuth.

Lazy-imported adapters for Anthropic and OpenAI SDKs.  The SDK packages are
optional extras — import errors are deferred to first instantiation.

Usage::

    from sleuth.llm.anthropic import Anthropic
    from sleuth.llm.openai import OpenAI
    from sleuth.llm.stub import StubLLM     # always available, no extra required
"""
