# main.py
import os
import json
from typing import TypedDict, List, Dict, Literal
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

# Create a Bedrock client (uses named profile if provided)
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


# ----------------- DBT micro-coach prompts -----------------
PROMPTS: Dict[str, str] = {
    "mindfulness": (
        "You are a brief DBT Mindfulness coach. Keep replies ~6–8 sentences. "
        "Offer one in-the-moment grounding skill and a 60-second practice prompt."
    ),
    "distress": (
        "You are a brief DBT Distress Tolerance coach. Give 1–2 immediate skills "
        "(TIP, STOP, self-soothe, ACCEPTS). Focus on safety and short, actionable steps."
    ),
    "emotion": (
        "You are a brief DBT Emotion Regulation coach. Help name the emotion, check facts, "
        "propose opposite action, and one small experiment for the next hour."
    ),
    "interpersonal": (
        "You are a brief DBT Interpersonal Effectiveness coach. Help craft a short "
        "DEAR MAN or GIVE script. Provide one example sentence and a boundary variant."
    ),
}

AgentKey = Literal["mindfulness", "distress", "emotion", "interpersonal"]


# ----------------- LangGraph state & nodes -----------------
class State(TypedDict, total=False):
    messages: List[Dict[str, str]]  # [{"role": "...", "content": "..."}]
    route: AgentKey                 # set by router


def router(state: State) -> AgentKey:
    """Naive keyword router that picks a DBT micro-coach."""
    text = state["messages"][-1]["content"].lower()

    if any(k in text for k in ["panic", "crisis", "overwhelmed", "urge", "unsafe", "shut down", "self-harm", "self harm"]):
        state["route"] = "distress"
        return "distress"

    if any(k in text for k in ["argument", "ask", "boundary", "conflict", "conversation", "negotiate", "script"]):
        state["route"] = "interpersonal"
        return "interpersonal"

    if any(k in text for k in ["sad", "angry", "anger", "fear", "anxious", "guilt", "ashamed", "emotion", "feel"]):
        state["route"] = "emotion"
        return "emotion"

    state["route"] = "mindfulness"
    return "mindfulness"


def make_agent_node(domain: AgentKey):
    """Creates a node that calls Bedrock with the domain-specific system prompt."""
    def node(state: State) -> State:
        sys_prompt = PROMPTS[domain]
        msgs = [{"role": "system", "content": sys_prompt}] + state["messages"]
        reply = bedrock_chat(msgs, max_tokens=220, temperature=0.2)
        state["messages"].append({"role": "assistant", "content": reply})
        return state
    return node


def build_graph():
    g = StateGraph(State)

    # Add four agent nodes
    g.add_node("mindfulness", make_agent_node("mindfulness"))
    g.add_node("distress", make_agent_node("distress"))
    g.add_node("emotion", make_agent_node("emotion"))
    g.add_node("interpersonal", make_agent_node("interpersonal"))

    # Route from START → one of the agent nodes using the router function
    g.add_conditional_edges(
        START,
        router,
        {
            "mindfulness": "mindfulness",
            "distress": "distress",
            "emotion": "emotion",
            "interpersonal": "interpersonal",
        },
    )

    # Each agent ends the flow
    for n in ("mindfulness", "distress", "emotion", "interpersonal"):
        g.add_edge(n, END)

    # Use MemorySaver (requires a thread_id when invoking)
    return g.compile(checkpointer=MemorySaver())


graph = build_graph()

# A stable thread id for the CLI session (so MemorySaver is happy)
THREAD_ID = os.getenv("THREAD_ID", f"cli-{uuid4().hex[:8]}")


# ----------------- Simple REPL -----------------
BANNER = """DBT Helper Bot (LangGraph + AWS Bedrock)
Routes: mindfulness | distress | emotion | interpersonal
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
        # Pass a thread_id so the MemorySaver checkpointer works
        final = graph.invoke(state, config={"configurable": {"thread_id": THREAD_ID}})
        print("\nbot:", final["messages"][-1]["content"], "\n")


if __name__ == "__main__":
    main()

