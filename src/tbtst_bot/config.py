# config.py
from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

# Optional: fine-grained tool streaming on newer Claude models (only if you need it)
ANTHROPIC_BETA_FINE_GRAINED = os.getenv("ANTHROPIC_BETA_FINE_GRAINED", "").strip()


def get_llm(
    *,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> ChatBedrockConverse:
    additional_fields = None
    if ANTHROPIC_BETA_FINE_GRAINED:
        additional_fields = {"anthropic_beta": [ANTHROPIC_BETA_FINE_GRAINED]}

    return ChatBedrockConverse(
        model_id=MODEL_ID,
        region_name=AWS_REGION,
        temperature=temperature,
        max_tokens=max_tokens,
        additional_model_request_fields=additional_fields,
    )


# Convenience default (still safe to override per-node via get_llm(...))
bedrock_llm = get_llm()

