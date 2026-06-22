import json
import sys

from client import MODEL, client
from prompts import EXTRACTION_PROMPT
from tools import dispatch


def _is_error(result: object) -> bool:
    return isinstance(result, str) and "error" in result.lower()


def extract_facts(
    user_input: str,
    assistant_reply: str,
    model: str = MODEL,
) -> list[str]:
    """
    Ask the model to pull durable personal facts out of this turn.
    Returns a list of short, atomic statements.
    """
    prompt = (
        EXTRACTION_PROMPT
        + f"\n\nUser: {user_input}\nAssistant: {assistant_reply}\n\nReturn JSON:"
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


def background_store_fact(fact: str, registry: dict) -> None:
    """
    Store one extracted fact using the same memory_store tool the agent uses.
    Make sure the argument name matches your tools.py schema.
    """
    # --- best-effort duplicate guard ---------------------------------------
    search_args = json.dumps({"keyword": fact, "top_k": 5})
    search_result = dispatch("memory_search", search_args, registry)

    if _is_error(search_result):
        print(
            f"[background] memory_search failed, "
            f"proceeding to store anyway: {search_result}",
            file=sys.stderr,
        )
    else:
        # Naive exact-match guard. For production, do embedding-based
        # deduplication inside MemoryStore instead.
        if fact.lower() in str(search_result).lower():
            return

    # --- store -------------------------------------------------------------
    # NOTE: change "content" below if your memory_store schema uses a
    # different parameter name (e.g. "memory", "text", "note").
    store_args = json.dumps({"content": fact})
    store_result = dispatch("memory_store", store_args, registry)

    if _is_error(store_result):
        print(f"[background] memory_store failed: {store_result}", file=sys.stderr)

