from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import os

from langchain_core.messages import HumanMessage

# Prefer package import (works with `python -m tbtst_bot.app`)
try:
    from .graph import GRAPH
except Exception:  # pragma: no cover
    # Fallback for direct execution in some setups
    from tbtst_bot.graph import GRAPH


BANNER = """TB-TST Helper (LangGraph + AWS Bedrock)

Flow:
  classify (safety + route) → (crisis | RAG FAQ | DBT) → reply

Commands:
  :graph               Save graph as PNG (default filename with timestamp)
  :graph <prefix>      Save graph as <prefix>.png (and <prefix>.mmd)
  q / quit / exit      Quit
"""


def save_graph_files(prefix: Optional[str] = None) -> tuple[Path, Path]:
    """
    Saves:
      <prefix>.mmd  Mermaid diagram text
      <prefix>.png  Rendered PNG (via draw_mermaid_png; uses Mermaid.ink by default)

    Returns (mmd_path, png_path)
    """
    g = GRAPH.get_graph()

    if prefix is None or not prefix.strip():
        prefix = f"tbtst_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    prefix_path = Path(prefix)
    mmd_path = prefix_path.with_suffix(".mmd")
    png_path = prefix_path.with_suffix(".png")

    mmd_path.write_text(g.draw_mermaid(), encoding="utf-8")

    png_bytes = g.draw_mermaid_png()
    png_path.write_bytes(png_bytes)

    return mmd_path, png_path


def main() -> None:
    # Stable thread id for this CLI session
    thread_id = os.getenv("THREAD_ID", f"cli-{uuid4().hex[:8]}")

    print(BANNER)
    print(f"(thread_id: {thread_id})\n")

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

        # --- graph export command ---
        if user.startswith(":graph"):
            parts = user.split(maxsplit=1)
            prefix = parts[1].strip() if len(parts) == 2 else None
            try:
                mmd_path, png_path = save_graph_files(prefix)
                print(f"\n✅ Saved Mermaid: {mmd_path}")
                print(f"✅ Saved PNG:    {png_path}\n")
            except Exception as e:
                print(f"\n❌ Failed to save graph: {type(e).__name__}: {e}\n")
            continue

        # --- normal chat ---
        try:
            final: Dict[str, Any] = GRAPH.invoke(
                {"messages": [HumanMessage(content=user)]},
                config={"configurable": {"thread_id": thread_id}},
            )
            reply = final["messages"][-1].content
            print("\nbot:", reply, "\n")
        except Exception as e:
            print(f"\n❌ Error: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()

