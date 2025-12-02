# main.py
import os
import json
from pathlib import Path
from typing import TypedDict, List, Dict, Literal, Optional
from uuid import uuid4

import boto3
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# ----------------- Config & AWS client -----------------

load_dotenv()  # loads .env if present

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
AWS_PROFILE = os.getenv("AWS_PROFILE")  # e.g., "tb-aws" (optional)

# Bedrock client
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
    """Minimal Bedrock (Anthropic) chat call."""
    system = "\n".join([m["content"] for m in messages if m["role"] == "system"])
    convo = [m for m in messages if m["role"] in ("user", "assistant")]

    body = {
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


# ----------------- Prompt loading -----------------

BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt text file from prompts/ by stem name."""
    path = PROMPTS_DIR / f"{name}.txt"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Fail loudly but in a way that's easy to debug
        raise RuntimeError(f"Missing prompt file: {path}")


# System prompts (files you will create in prompts/)
SAFETY_SYSTEM_PROMPT = load_prompt("safety_classifier")
CRISIS_SYSTEM_PROMPT = load_prompt("crisis_response")
OK_SYSTEM_PROMPT = load_prompt("ok_response")

# ----------------- State definition -----------------

RiskLevel = Literal["high", "ok"]


class State(TypedDict, total=False):
    # Full conversation so far
    messages: List[Dict[str, str]]
    # Safety routing
    risk_level: RiskLevel          # "high" or "ok"
    # For future routing (FAQ vs DBT etc.)
    path: Optional[str]            # "crisis" | "normal" | later: "faq", "dbt", ...


# ----------------- Safety: rules + LLM -----------------

CRISIS_KEYWORDS = [
    # English
    "kill myself",
    "killing myself",
    "end my life",
    "suicide",
    "suicidal",
    "take my life",
    "don't want to live",
    "dont want to live",
    "hurt myself",
    "self-harm",
    "self harm",
    # Spanish
    "no quiero vivir",
    "matarme",
    "hacerme daño",
]


def rule_based_high_risk(text: str) -> bool:
    """Hard-rule check for obvious crisis phrases."""
    t = text.lower()
    return any(kw in t for kw in CRISIS_KEYWORDS)


def safety_node(state: State) -> State:
    """
    Hybrid safety gate:
      1) If clear crisis keywords → high risk.
      2) Else, LLM safety classifier → 'high' or 'ok'.
    """
    if not state.get("messages"):
        state["risk_level"] = "ok"
        return state

    last_msg = state["messages"][-1]["content"]

    # 1) Rule-based layer
    if rule_based_high_risk(last_msg):
        state["risk_level"] = "high"
        return state

    # 2) LLM classifier for subtler cases
    msgs = [
        {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
        {"role": "user", "content": last_msg},
    ]
    verdict = bedrock_chat(msgs, max_tokens=3, temperature=0.0).strip().lower()

    if "high" in verdict:
        state["risk_level"] = "high"
    else:
        state["risk_level"] = "ok"

    return state


def safety_branch(state: State) -> str:
    """
    Conditional router after safety_node.
    Returns the name of the next node: 'crisis' or 'normal'.
    """
    level: RiskLevel = state.get("risk_level", "ok")
    return "crisis" if level == "high" else "normal"


# ----------------- Ingress & response nodes -----------------

def ingress_node(state: State) -> State:
    """
    Ingress node.
    For now this is a no-op that could later:
      - enforce global response rules,
      - normalize messages, language detection, etc.
    """
    # You could add things like: trim history, tag turn_id, etc.
    return state


def crisis_node(state: State) -> State:
    """
    Crisis responder.
    Uses CRISIS_SYSTEM_PROMPT and the last user message.
    """
    last_msg = state["messages"][-1]["content"]
    msgs = [
        {"role": "system", "content": CRISIS_SYSTEM_PROMPT},
        {"role": "user", "content": last_msg},
    ]
    reply = bedrock_chat(msgs, max_tokens=260, temperature=0.2)
    state["messages"].append({"role": "assistant", "content": reply})
    state["path"] = "crisis"
    return state


def normal_node(state: State) -> State:
    """
    Non-crisis responder for this MVP.

    Currently: uses OK_SYSTEM_PROMPT as a simple supportive assistant.
    Later: this node can become your 'router' to FAQ vs DBT skills paths.
    """
    last_msg = state["messages"][-1]["content"]
    msgs = [
        {"role": "system", "content": OK_SYSTEM_PROMPT},
        {"role": "user", "content": last_msg},
    ]
    reply = bedrock_chat(msgs, max_tokens=260, temperature=0.2)
    state["messages"].append({"role": "assistant", "content": reply})
    state["path"] = "normal"
    return state


# ----------------- Build LangGraph graph -----------------

def build_graph():
    g = StateGraph(State)

    # Nodes
    g.add_node("ingress", ingress_node)
    g.add_node("safety", safety_node)
    g.add_node("crisis", crisis_node)
    g.add_node("normal", normal_node)

    # Edges
    g.add_edge(START, "ingress")
    g.add_edge("ingress", "safety")

    # Conditional: crisis vs normal
    g.add_conditional_edges(
        "safety",
        safety_branch,
        {
            "crisis": "crisis",
            "normal": "normal",
        },
    )

    # Both end the flow
    g.add_edge("crisis", END)
    g.add_edge("normal", END)

    # MemorySaver so we can use persistent threads later if we want
    return g.compile(checkpointer=MemorySaver())


graph = build_graph()

# Stable thread id for the CLI session (for MemorySaver)
THREAD_ID = os.getenv("THREAD_ID", f"cli-{uuid4().hex[:8]}")


# ----------------- Simple REPL -----------------

BANNER = """DBT / TB Helper Bot (LangGraph + AWS Bedrock)
Flow: START → ingress → safety → (crisis | normal) → END
Type 'quit' to exit.
"""


def main():
    print(f"[config] region={AWS_REGION}, model={MODEL_ID}, profile={AWS_PROFILE or 'default'}, thread_id={THREAD_ID}")
    print(BANNER)

    state: State = {"messages": []}

    while True:
        try:
            user = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return

        if user.lower() in {"q", "quit", "exit"}:
            print("bye.")
            return
        if not user:
            continue

        state["messages"].append({"role": "user", "content": user})

        final = graph.invoke(
            state,
            config={"configurable": {"thread_id": THREAD_ID}},
        )
        reply = final["messages"][-1]["content"]
        print("\nbot:", reply, "\n")

        # Update local state with the returned one, in case nodes mutated it
        state = final


if __name__ == "__main__":
    main()

