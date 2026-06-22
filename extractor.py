import json
import sys

from client import MODEL, client
from embeddings import _embed_text_to_floats, serialize_f32
from prompts import EXTRACTION_PROMPT
from store import MemoryStore

DEDUP_THRESHOLD = 0.85


def extract_facts(
    user_input: str,
    model: str = MODEL,
) -> list[str]:
    """
    Ask the model to pull durable personal facts out of this turn.
    Returns a list of short, atomic statements.
    """
    prompt = (
        EXTRACTION_PROMPT
        + f"\n\nUser: {user_input}\n\nReturn JSON:"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""
        data = json.loads(content)
        if not isinstance(data, dict):
            return []
        facts = data.get("facts", [])
        if not isinstance(facts, list):
            return []
        return [f.strip() for f in facts if isinstance(f, str) and f.strip()]
    except Exception as e:
        # Surface extraction problems instead of silently dropping them.
        print(f"[extract_facts] error: {e}", file=sys.stderr)
        return []


def background_store_fact(fact: str, store: MemoryStore, user_id: str) -> None:
    """
    Store one extracted fact directly in the memory store.

    Deduplicates against existing memories before storing.
    """
    # 1. embed
    blob = serialize_f32(_embed_text_to_floats(fact))

    # 2. check for near-duplicate
    matches = store.find_similar(user_id, blob, threshold=DEDUP_THRESHOLD, top_k=1)

    if matches:
        # log and skip
        best = matches[0]
        print(f"[dedup] SKIP '{fact}' (similarity {best['similarity']:.3f} to '{best['content']}')", file=sys.stderr)
        return

    # 3. store directly
    store.store(user_id=user_id, content=fact, embedding=blob)
    print(f"[store] STORED '{fact}'", file=sys.stderr)


