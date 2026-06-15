import json
import os
import sys

from client import MODEL, client
from store import MemoryStore
from thread import ThreadStore
from tools import TOOL_SCHEMAS, dispatch, make_tools

USER_ID = "agent_user"


# -------------------------------------------------------------------------
# Tool filtering
# -------------------------------------------------------------------------

def _tool_name(tool_schema: dict) -> str | None:
    return tool_schema.get("function", {}).get("name")


# The agent may READ memories but must never WRITE them.
# Writing is handled by the background extractor after the assistant reply.
AGENT_TOOLS = [t for t in TOOL_SCHEMAS if _tool_name(t) != "memory_store"]


# -------------------------------------------------------------------------
# Agent system prompt (recall only)
# -------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """You are a helpful assistant with persistent memory.
You have two tools available: memory_search and memory_list.
You do NOT have memory_store. Do not try to store memories yourself;
they are extracted automatically in the background.

## When to use memory_search
Call memory_search when the question is about a SPECIFIC topic with
clear keywords:
- "What outdoor stuff do I enjoy?" → search "hiking outdoor"
- "Where do I live?" → search "live location city"
- "Recommend a restaurant" → search "food restaurant diet"

Use 1-3 content keywords. If results seem incomplete or the keywords
might not match how the fact was stored, fall back to memory_list.

## When to use memory_list
Call memory_list when you need the full picture rather than a
keyword match:
- Broad recall: "What do you remember about me?", "What do you know
  about me?", "Summarize my profile"
- Open-ended personalization: "Plan my ideal weekend" — touches
  many possible memories, no single keyword set covers it
- After a memory_search that returned nothing but you suspect the
  fact exists under different wording

## When to use NO tool
General knowledge, coding, math, casual chat with no personal
component. Most turns need no memory call.

## Response style
If the user asks you to remember something, confirm naturally. You do
not need to call any tool to store it; that happens automatically.

If no relevant memories exist, say you don't have that information
yet — never invent one."""


# -------------------------------------------------------------------------
# Background extraction
# -------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a memory-extraction assistant.
Analyze the last user-assistant turn and identify durable personal facts
about the user that are worth remembering long-term.

Extract things like:
- Personal preferences (likes, dislikes, hobbies)
- Personal facts (family, location, job, age, constraints)
- Goals, routines, values, relationships

DO NOT extract:
- Transient requests ("Recommend a book", "What time is it?")
- General knowledge or assistant behavior
- Anything not clearly about the user

Return a JSON object with one key, "facts", containing short, atomic,
third-person statements about the user. Each fact must be self-contained.

Example:
{"facts": ["User has a 6-year-old daughter", "User's daughter is turning 7 next week"]}

If there are no durable personal facts, return {"facts": []}."""


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


def _is_error(result: object) -> bool:
    return isinstance(result, str) and "error" in result.lower()


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


# -------------------------------------------------------------------------
# Agent loop
# -------------------------------------------------------------------------

def agent_run(
    messages: list[dict],
    registry: dict,
    threads: ThreadStore,
    thread_id: int,
    max_iterations: int = 5,
) -> str:
    """
    Agent loop: keep calling the model until it stops requesting tools.
    """
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=AGENT_TOOLS,  # <-- agent cannot see memory_store
        )
        message = response.choices[0].message

        if not message.tool_calls:
            final_message = {"role": "assistant", "content": message.content}
            messages.append(final_message)
            threads.append_message(thread_id, final_message)
            return message.content

        messages.append(message)
        threads.append_message(thread_id, message)

        for call in message.tool_calls:
            result = dispatch(call.function.name, call.function.arguments, registry)
            tool_message = {
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            }
            messages.append(tool_message)
            threads.append_message(thread_id, tool_message)

    return "Reached max iterations without a final answer."


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    with MemoryStore("memories.db") as store, ThreadStore("memories.db") as threads:
        registry = make_tools(store, USER_ID)

        def system_only():
            return [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

        existing = threads.list_threads(USER_ID)
        if existing:
            thread_id = existing[0]["id"]
            messages = system_only()
            messages.extend(threads.load_messages(thread_id))
            print(
                f"Resuming thread {thread_id}: {existing[0]['title']} "
                f"({len(messages) - 1} prior messages)"
            )
        else:
            thread_id = threads.create_thread(USER_ID, "Agent chat")
            messages = system_only()
            print(f"Started new thread {thread_id}")

        while True:
            user_input = input("you> ").strip()
            if not user_input:
                continue

            if user_input == "/exit" or user_input in {"exit", "quit"}:
                break

            if user_input.startswith("/new"):
                title = user_input[len("/new"):].strip() or "Agent chat"
                thread_id = threads.create_thread(USER_ID, title)
                messages = system_only()
                print(f"Started new thread {thread_id}: {title}")
                continue

            if user_input.startswith("/resume"):
                thread_list = threads.list_threads(USER_ID)
                if not thread_list:
                    print("No saved threads to resume.")
                    continue

                print("\nSaved threads:")
                for t in thread_list:
                    print(f"  {t['id']}: {t['title']} "
                          f"(updated {t['updated_at']})")

                requested_id = user_input[len("/resume"):].strip()
                if requested_id.isdigit():
                    chosen_id = int(requested_id)
                else:
                    choice = input("Enter thread id to resume> ").strip()
                    if not choice.isdigit():
                        print("Invalid id. Staying on current thread.")
                        continue
                    chosen_id = int(choice)

                if not any(t["id"] == chosen_id for t in thread_list):
                    print(f"Thread {chosen_id} not found. Staying on current thread.")
                    continue

                thread_id = chosen_id
                messages = system_only()
                prior = threads.load_messages(thread_id)
                messages.extend(prior)
                print(f"Resumed thread {chosen_id} ({len(prior)} prior messages)")
                continue

            # Normal turn
            user_message = {"role": "user", "content": user_input}
            messages.append(user_message)
            threads.append_message(thread_id, user_message)

            reply = agent_run(messages, registry, threads, thread_id)
            print("agent>", reply)

            # --- background extraction: single source of truth for writes ---
            facts = extract_facts(user_input, reply)
            for fact in facts:
                background_store_fact(fact, registry)


if __name__ == "__main__":
    main()
