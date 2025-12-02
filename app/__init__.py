# app/__init__.py
"""
TB DBT bot package.

This package contains:
- config: env + paths
- state: conversation state schema
- llm: Bedrock client + chat helper
- prompts: loading/composing text prompts & KBs
- keywords: loading keyword lists
- nodes: LangGraph node implementations
- graph: function to build the compiled graph
"""

