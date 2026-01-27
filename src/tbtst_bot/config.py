# config.py
from __future__ import annotations

import os
import socket
import urllib.request
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2"
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

# Optional: fine-grained tool streaming on newer Claude models (only if you need it)
ANTHROPIC_BETA_FINE_GRAINED = os.getenv("ANTHROPIC_BETA_FINE_GRAINED", "").strip()

# If you're on EC2 with an instance role, you usually want to IGNORE AWS_PROFILE (even if .env set it).
# Set TB_FORCE_AWS_PROFILE=1 if you really want to force using AWS_PROFILE on EC2.
TB_FORCE_AWS_PROFILE = os.getenv("TB_FORCE_AWS_PROFILE", "").strip().lower() in {"1", "true", "yes"}


def _imds_v2_token(timeout_sec: float = 0.25) -> str | None:
    """Fetch IMDSv2 token. Returns None if unavailable."""
    try:
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            token = resp.read().decode("utf-8", errors="ignore").strip()
            return token or None
    except Exception:
        return None


def _imds_get(path: str, timeout_sec: float = 0.25) -> str | None:
    """
    GET a metadata path from IMDS using IMDSv2 token when required.
    Returns body string if successful, else None.
    """
    url = f"http://169.254.169.254/latest/{path.lstrip('/')}"
    token = _imds_v2_token(timeout_sec=timeout_sec)

    headers = {}
    if token:
        headers["X-aws-ec2-metadata-token"] = token

    try:
        req = urllib.request.Request(url, method="GET", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return resp.read().decode("utf-8", errors="ignore").strip() or None
    except Exception:
        return None


def _ec2_instance_role_present(timeout_sec: float = 0.25) -> bool:
    """
    Detect whether we're on EC2 *and* an IAM role is attached by querying IMDS.
    Works with IMDSv2-required instances.
    """
    socket.setdefaulttimeout(timeout_sec)
    body = _imds_get("meta-data/iam/security-credentials/", timeout_sec=timeout_sec)
    return bool(body)


def _maybe_disable_profile_on_ec2() -> None:
    """
    If we're on EC2 with an instance role, remove AWS_PROFILE so the default
    credential chain uses the role (and doesn't fail because a local profile name
    from .env isn't present on the instance).
    """
    if TB_FORCE_AWS_PROFILE:
        return

    if _ec2_instance_role_present():
        os.environ.pop("AWS_PROFILE", None)
        os.environ.pop("AWS_DEFAULT_PROFILE", None)


def get_llm(
    *,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> ChatBedrockConverse:
    _maybe_disable_profile_on_ec2()

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

