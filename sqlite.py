import sqlite3


class MemoryStore:
    def __init__(self, db_path: str = "memories.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # access columns by name
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,                          -- NULL until Phase 3
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)"
        )
        self.conn.commit()

    @staticmethod
    def _escape_like(s: str) -> str:
        # %, _ are wildcards in LIKE — escape them so they match literally
        return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


    def store(self, user_id, content) -> int:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
            
        cur = self.conn.execute(
            "INSERT INTO memories (user_id, content) VALUES (?, ?)",
            (user_id, content), 
        )
        self.conn.commit()
        return cur.lastrowid

    def list_memories(self, user_id) -> list:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def search_by_keyword(self, user_id, keyword) -> list:
        rows = self.conn.execute(
            """
            SELECT * FROM memories
            WHERE user_id = ? AND content LIKE ? ESCAPE '\\'
            ORDER BY created_at DESC
            """,
            (user_id, f"%{self._escape_like(keyword)}%"),
        ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, memory_id) -> bool:
        cur = self.conn.execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def close(self):
        self.conn.close()

    # context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()