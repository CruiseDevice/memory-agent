import logging
import sqlite3


class MemoryStore:
    def __init__(self, db_path: str = "memories.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = sqlite3.Row  # access columns by name
        self._load_sqlite_vec_extension()
        self._init_schema()

    def _load_sqlite_vec_extension(self) -> None:
        """
        Load the sqlite-vec extension on the store's connection.
        If the extension cannot be loaded we swallow the error and let
        the caller fall back to keyword-only search - this keeps the
        memory system usable even on platforms where vec is not available.
        """
        try:
            self.conn.enable_load_extension(True)
            import sqlite_vec
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            self.conn.execute("SELECT vec_version()").fetchone()
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to load sqlite-vec extension: {e}"
            )
            # Make sure we leave the connection in a sane state..
            try:
                self.conn.enable_load_extension(False)
            except Exception:
                pass


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

    def _vec_available(self) -> bool:
        """Return True if sqlite-vec is loaded and functional on this connection."""
        try:
            self.conn.execute("SELECT vec_version()").fetchone()
            return True
        except Exception:
            return False

    def store(self, user_id, content, embedding: bytes | None = None) -> int:
        """
        Insert a new memory record.
        `embedding` is optional - pass None if you still want to store a row without a vector
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")

        cur = self.conn.execute(
            "INSERT INTO memories (user_id, content, embedding) VALUES (?, ?, ?)",
            (user_id, content, embedding), 
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

    def find_similar(self, user_id: str, embedding_blob: bytes, threshold: float=0.85, top_k: int=1) -> list[dict]:
        """
        Find memories whose embedding is semantically similar to `embedding_blob`.

        `threshold` is the minimum cosine similarity in [0, 1].  Higher means
        stricter.  `top_k` limits the number of returned rows.

        Returns an empty list if the sqlite-vec extension is unavailable, so
        callers can fall back to `search_by_keyword()`.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(embedding_blob, bytes) or not embedding_blob:
            raise ValueError("embedding_blob must be non-empty bytes")
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        if not self._vec_available():
            return []

        # sqlite-vec returns cosine *distance* = 1 - cosine_similarity
        max_distance = 1.0 - float(threshold)

        try:
            rows = self.conn.execute(
                """
                SELECT * FROM (
                    SELECT
                        m.*,
                        vec_distance_cosine(m.embedding, ?) AS distance
                    FROM memories m
                    WHERE m.user_id = ?
                      AND m.embedding IS NOT NULL
                      AND length(m.embedding) == length(?)
                )
                WHERE distance <= ?
                ORDER BY distance ASC
                LIMIT ?
                """,
                (embedding_blob, user_id, embedding_blob, max_distance, top_k),
            ).fetchall()
        except sqlite3.Error as e:
            logging.getLogger(__name__).warning(f"Vector search failed: {e}")
            return []

        results = []
        for row in rows:
            result = dict(row)
            result["distance"] = float(result["distance"])
            result["similarity"] = 1.0 - result["distance"]
            results.append(result)
        return results

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
