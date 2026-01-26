# src/tbtst_bot/app.py
from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import argparse
import os

from langchain_core.messages import HumanMessage

try:
    from .graph import GRAPH_DBT_MINI, GRAPH_DBT_FULL
except Exception:  # pragma: no cover
    from tbtst_bot.graph import GRAPH_DBT_MINI, GRAPH_DBT_FULL


BANNER = """TB-TST Helper (LangGraph + AWS Bedrock)

Flow:
  classify (safety + route) → (crisis | RAG FAQ | DBT) → reply

Variants:
  mini  (baseline DBT node)
  full  (DBT subgraph: brain router → DT/MIND/ER/IE)

Commands:
  :graph               Save graph as PNG (default filename with timestamp)
  :graph <prefix>      Save graph as <prefix>.png (and <prefix>.mmd)
  q / quit / exit      Quit
"""


def select_graph(variant: str):
    v = (variant or "mini").strip().lower()
    if v == "full":
        return GRAPH_DBT_FULL, "full"
    if v == "mini":
        return GRAPH_DBT_MINI, "mini"
    raise ValueError(f"Unknown variant: {variant!r}. Use 'mini' or 'full'.")


def save_graph_files(graph, prefix: Optional[str] = None) -> tuple[Path, Path]:
    g = graph.get_graph()

    if prefix is None or not prefix.strip():
        prefix = f"tbtst_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    prefix_path = Path(prefix)
    mmd_path = prefix_path.with_suffix(".mmd")
    png_path = prefix_path.with_suffix(".png")

    mmd_path.write_text(g.draw_mermaid(), encoding="utf-8")
    png_path.write_bytes(g.draw_mermaid_png())

    return mmd_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["mini", "full"],
        default="mini",
        help="Which DBT implementation to use.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Optional thread id. If omitted, a new one is generated.",
    )
    args = parser.parse_args()

    graph, variant = select_graph(args.variant)

    # Separate memory per variant: suffix the thread id with :{variant}
    base_thread = args.thread_id or os.getenv("THREAD_ID") or f"cli-{uuid4().hex[:8]}"
    thread_id = f"{base_thread}:{variant}"

    print(BANNER)
    print(f"(variant: {variant}, thread_id: {thread_id})\n")

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

        if user.startswith(":graph"):
            parts = user.split(maxsplit=1)
            prefix = parts[1].strip() if len(parts) == 2 else None
            try:
                mmd_path, png_path = save_graph_files(graph, prefix)
                print(f"\n✅ Saved Mermaid: {mmd_path}")
                print(f"✅ Saved PNG:    {png_path}\n")
            except Exception as e:
                print(f"\n❌ Failed to save graph: {type(e).__name__}: {e}\n")
            continue

        try:
            final: Dict[str, Any] = graph.invoke(
                {"messages": [HumanMessage(content=user)]},
                config={"configurable": {"thread_id": thread_id}},
            )
            reply = final["messages"][-1].content
            print("\nbot:", reply, "\n")
        except Exception as e:
            print(f"\n❌ Error: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()

