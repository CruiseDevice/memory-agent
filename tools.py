import json
import logging
import sys

from embeddings import _embed_text_to_floats, serialize_f32
from store import MemoryStore

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search stored memories by keyword. Call BEFORE "
                "answering any question about the user's preferences, history, "
                "or personal details, and before giving personalized "
                "recommendations. Input: 1-3 content keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "1-3 content keywords to search. e.g. 'hiking outdoor'"
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_list",
            "description": "Return all stored memories about the user. Call for "
                "broad questions ('what do you remember about me?'), open-ended "
                "personalization spanning many topics, or as a fallback when "
                "memory_search returns nothing but the fact likely exists under "
                "different wording. Takes no input.",
            "parameters": {
                "type": "object",
            },
        },
    }      
]

def make_tools(store: MemoryStore, user_id: str):
    # each tool closes over `store` and `user_id`

    def memory_search(keyword: str, *, top_k: int = 5, use_vector: bool = True) -> str:
        """
        Search memories for given keyword.

        - if `use_vector` is True and the sqlite-vec extension is available,
        we perform a nearest-neighbor vector search on the `embedding` column.
        - Regardless of the vector result we also run a simple LIKE-search on the raw
        `content` column - this guarantees you still get the hits when the vector
        index is missing or the query is very short.
        - The two result sets are de-duplicated (by `id`) and limited to `top_k`
        rows, preserving the vector-score order first, then the keyword matches. 
        """
        # try the vector path (if requested and extension is loadable)
        vec_rows = []
        if use_vector:
            try:
                # Load the sqlite-vec extension on the store's connection
                conn = store.conn

                # embed the keyword
                keyword_embedding = _embed_text_to_floats(keyword)
                keyword_blob = serialize_f32(keyword_embedding)

                # perform brute-force cosine-distance search.
                # For L2-normalized vectors, cosine distance = 1 - dot_product.
                # sqlite-vec provides `vec_distance_cosine(blob1, blob2)`.
                vec_sql = """
                    SELECT id, content, vec_distance_cosine(embedding, ?) AS distance
                    FROM memories
                    WHERE user_id = ?
                    ORDER BY distance
                    LIMIT ?
                """
                vec_rows = conn.execute(
                    vec_sql,
                    (keyword_blob, user_id, top_k),
                ).fetchall()
            except Exception as e:
                # if anything goes wrong (extension missing, error, etc) we just
                # fall back to the keyword search below
                logger.error(f"Vector search failed: {e}")
                vec_rows = []

        # keyword search (using the existing method in the store)
        kw_rows = store.search_by_keyword(user_id=user_id, keyword=keyword)
        # Note: store.search_by_keyword returns a list of dicts (because of the dict(row))

        seen_ids = set()
        merged_contents = []
        
        # process vector results first
        for row in vec_rows:
            # row is a sqlite3.Row (because we set row_factory to sqlite3.Row in the store's __init__)
            mem_id = row['id']
            if mem_id not in seen_ids:
                seen_ids.add(mem_id)
                merged_contents.append(row['content'])
                if len(merged_contents) >= top_k:
                    break
        
        # If we still need more results, add from keyword results
        if len(merged_contents) < top_k:
            for row in kw_rows:
                mem_id = row['id']
                if mem_id not in seen_ids:
                    seen_ids.add(mem_id)
                    merged_contents.append(row['content'])
                    if len(merged_contents) >= top_k:
                        break

        return "\n".join(merged_contents) if merged_contents else "No matching memories"


    def memory_list() -> str:
        rows = store.list_memories(user_id=user_id)
        hits = [row['content'] for row in rows]
        return "\n".join(hits) if hits else "No memories stored yet."


    registry = {
        "memory_search": memory_search,
        "memory_list": memory_list,
    }
    return registry


def dispatch(name: str, arguments: str, registry: dict) -> str:
    """
    Look up the tool, parse its JSON args, run it, return a string result
    """
    logger.info(f"Tool calls: {name}({arguments})")
    fn = registry.get(name)
    if fn is None:
        return f"Error: Unknown tool '{name}'"
    
    try:
        args = json.loads(arguments)
        return fn(**args)
    except (json.JSONDecodeError, TypeError) as e:
        return f"Error calling {name}: {e}"