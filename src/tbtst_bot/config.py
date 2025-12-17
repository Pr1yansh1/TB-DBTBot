from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence

import boto3
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
AWS_PROFILE = os.getenv("AWS_PROFILE")  # optional named profile


def _bedrock_client():
    if AWS_PROFILE:
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        return session.client("bedrock-runtime", region_name=AWS_REGION)
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


client = _bedrock_client()


def bedrock_chat(
    messages: Sequence[BaseMessage],
    max_tokens: int = 350,
    temperature: float = 0.2,
) -> str:
    """
    Bedrock Anthropic Messages API call using LangChain BaseMessage objects.

    IMPORTANT:
    - system prompt must be top-level "system"
    - messages list must include ONLY roles "user" and "assistant"
    """
    system_parts: List[str] = []
    convo: List[Dict[str, str]] = []

    for m in messages:
        if isinstance(m, SystemMessage):
            system_parts.append(m.content or "")
        elif isinstance(m, HumanMessage):
            convo.append({"role": "user", "content": m.content or ""})
        elif isinstance(m, AIMessage):
            convo.append({"role": "assistant", "content": m.content or ""})
        else:
            # fallback: treat unknown as assistant text
            convo.append({"role": "assistant", "content": getattr(m, "content", str(m))})

    system_text = "\n".join([s for s in system_parts if s.strip()]).strip()

    body: Dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": convo,
    }
    if system_text:
        body["system"] = system_text

    resp = client.invoke_model(
        modelId=MODEL_ID,
        accept="application/json",
        contentType="application/json",
        body=json.dumps(body),
    )
    data = json.loads(resp["body"].read())

    out = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            out.append(block.get("text", ""))
    return "".join(out).strip()

