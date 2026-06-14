import json
import sqlite3
from typing import Any


class ThreadStore:
    def __init__(self, db_path: str = "memories.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = sqlite3.Row     # access columns by name
        self._init_schema() 

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,   -- nullable: tool-only assistant messages
                tool_calls TEXT,    -- nullable: JSON string
                tool_call_id TEXT,  -- nullable: only set on role='tool'
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (thread_id) REFERENCES threads(id)
            )
        """)

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id)"
        )
        self.conn.commit()

    def create_thread(self, user_id, title) -> int:
        """
        Create a new thread and return its id.
        """
        cursor = self.conn.execute(
            "INSERT INTO threads (user_id, title) VALUES (?, ?)",
            (user_id, title)
        )
        self.conn.commit()
        return cursor.lastrowid

    def list_threads(self, user_id) -> list[dict]:
        """
        Return all threads for a user, newest updated first
        """
        cursor = self.conn.execute(
            "SELECT * FROM threads WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def append_message(self, thread_id, message: dict) -> None:
        """
        Normalize and store a message, updating the thread's updated_at timestamp.
        """
        norm = self.normalize_message(message)

        self.conn.execute(
            """
            INSERT INTO messages
                (thread_id, role, content, tool_calls, tool_call_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                norm.get("role"),
                norm.get("content"),
                norm.get("tool_calls"),
                norm.get("tool_call_id")
            )
        )

        # keep thread timestamp fresh
        self.conn.execute(
            "UPDATE threads SET updated_at = datetime('now') WHERE id = ?",
            (thread_id,)
        )
        self.conn.commit()

    def load_messages(self, thread_id, limit=20) -> list[dict]:
        """
        Return the most recent messages for a thread in chronological order.
        """
        cursor = self.conn.execute(
            """
            SELECT role, content, tool_calls, tool_call_id
            FROM messages
            WHERE thread_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (thread_id, limit)
        )
        rows = cursor.fetchall()

        messages = []
        for row in reversed(rows):
            message = {
                "role": row["role"],
                "content": row["content"],
            }

            if row["tool_calls"] is not None:
                message["tool_calls"] = json.loads(row["tool_calls"])
            if row["tool_call_id"] is not None:
                message["tool_call_id"] = row["tool_call_id"]
            
            # optional: skip empty content for tool calls
            if message["content"] is None and "tool_calls" in message:
                del message["content"]
            
            messages.append(message)
        return messages

    def normalize_message(self, message) -> dict:
        """
        Convert dicts, Pydantic models, or OpenAI SDK objects into a plain dict
        with keys matching the DB schema.
        """
        if hasattr(message, "model_dump"):      # Pydantic v2
            msg = message.model_dump()
        elif hasattr(message, "dict"):          # Pydantic v1
            msg = message.dict()
        elif isinstance(message, dict):
            msg = dict(message)                  # Shallow copy
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")

        norm: dict[str, Any] = {
            "role": msg.get("role"),
            "content": msg.get("content"),
            "tool_calls": None,
            "tool_call_id": msg.get("tool_call_id")
        }

        # Serialize tool_calls to JSON string for storage
        tc = msg.get("tool_calls")
        if tc is not None:
            norm["tool_calls"] = tc if isinstance(tc, str) else json.dumps(tc)

        return norm

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    with ThreadStore("demo.db") as store:
        thread_id = store.create_thread("user-123", "My first chat")
        print("Created thread: ", thread_id)

        store.append_message(thread_id, {
            "role": "user",
            "content": "Hello!"
        })
        store.append_message(thread_id, {
            "role": "assistant",
            "content": "Hi there."
        })

        store.append_message(thread_id, {
            "role": "user",
            "content": "What is the weather in Paris?"
        })

        store.append_message(
            thread_id,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}'
                        }
                    }
                ]
            }
        )

        store.append_message(
            thread_id,
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "It is 22°C and sunny in Paris."
            }
        )

        store.append_message(
            thread_id,
            {
                "role": "assistant",
                "content": "It is 22°C and sunny in Paris right now."
            }
        )

        threads = store.list_threads("user-123")
        print("Threads: ", json.dumps(threads, indent=2, default=str))

        messages = store.load_messages(thread_id)
        print("Messages: ", json.dumps(messages, indent=2, default=str))