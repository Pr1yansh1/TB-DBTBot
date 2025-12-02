# app/llm.py
import json
from typing import List, Dict

import boto3

from .config import AWS_REGION, MODEL_ID, AWS_PROFILE


# Create a Bedrock runtime client
if AWS_PROFILE:
    _session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    bedrock = _session.client("bedrock-runtime", region_name=AWS_REGION)
else:
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def bedrock_chat(
    messages: List[Dict[str, str]],
    max_tokens: int = 220,
    temperature: float = 0.2,
) -> str:
    """
    Minimal wrapper around Anthropic chat models on Bedrock.

    Expects messages in Anthropic format:
      [{"role": "user"|"assistant", "content": "..."}, ...]
    plus an optional single system message (we'll pass it via "system").
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    convo = [m for m in messages if m["role"] in ("user", "assistant")]

    body: Dict[str, object] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": convo,
    }
    if system_parts:
        body["system"] = "\n\n".join(system_parts)

    resp = bedrock.invoke_model(
        modelId=MODEL_ID,
        accept="application/json",
        contentType="application/json",
        body=json.dumps(body),
    )
    data = json.loads(resp["body"].read())
    text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")
    return text.strip()

