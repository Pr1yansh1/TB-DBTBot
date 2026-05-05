import sqlite3
import os

conn = sqlite3.connect("data/chainlit.db")
conn.row_factory = sqlite3.Row

threads = conn.execute("SELECT * FROM threads ORDER BY createdAt").fetchall()

os.makedirs("transcripts-sqlite", exist_ok=True)

for thread in threads:
    tid = thread["id"]
    created = thread["createdAt"]
    user = thread["userIdentifier"] or "unknown"

    steps = conn.execute("""
        SELECT type, name, input, output, createdAt
        FROM steps
        WHERE threadId = ? AND type IN ('user_message', 'assistant_message')
        ORDER BY createdAt
    """, (tid,)).fetchall()

    filename = f"transcripts-sqlite/{created[:10]}_{tid[:8]}.txt"
    with open(filename, "w") as f:
        f.write(f"Thread: {tid}\n")
        f.write(f"Date: {created}\n")
        f.write(f"User: {user}\n")
        f.write("=" * 60 + "\n\n")

        for step in steps:
            if step["type"] == "user_message" and step["output"]:
                f.write(f"[USER]\n{step['output']}\n\n")
            elif step["type"] == "assistant_message" and step["output"]:
                f.write(f"[ASSISTANT]\n{step['output']}\n\n")

    print(f"Wrote {filename} ({len(steps)} messages)")

conn.close()
