# app/config.py
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

# -------- Paths --------
BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
KEYWORDS_DIR = BASE_DIR / "keywords"

# -------- AWS / model config --------
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
)
AWS_PROFILE = os.getenv("AWS_PROFILE")  # e.g., "tb-aws" (optional)

# -------- Thread / memory config --------
DEFAULT_THREAD_ID = os.getenv("THREAD_ID", f"cli-{uuid4().hex[:8]}")

