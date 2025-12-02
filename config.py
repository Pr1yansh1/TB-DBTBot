# config.py
import os
import json
from typing import List, Dict, Any

import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
)
AWS_PROFILE = os.getenv("AWS_PROFILE")  # optional named profile


if AWS_PROFILE:
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    bedrock = session.client("bedrock-runtime", region_name=AWS_REGION)
else:
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def bedrock_chat(
    messages: List[Dict[str, str]],
    max_tokens: int = 220,
    temperature: float = 0.2,
) -> str:
    """
    Minimal Bedrock (Anthropic) chat call.

    messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
    """
    system = "\n".join(
        [m["content"] for m in messages if m["role"] == "system"]
    )
    convo = [m for m in messages if m["role"] in ("user", "assistant")]

    body: Dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": convo,
    }
    if system:
        body["system"] = system

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

