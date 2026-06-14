from client import MODEL, client
from store import MemoryStore
from thread import ThreadStore
from tools import TOOL_SCHEMAS, dispatch, make_tools

USER_ID = "agent_user"  # Fixed user ID for this session

SYSTEM_PROMPT = """You are a helpful assistant with persistent memory.
You have three tools: memory_store, memory_search, and memory_list.

## When to use memory_store
Call memory_store when the user shares durable personal information,
whether or not they explicitly say "remember":
- Explicit: "Remember that...", "Don't forget...", "Note that..."
- Implicit: stated preferences ("I like hiking"), personal facts
  ("I live in San Jose"), goals, constraints ("I'm vegetarian")

Before storing, check for existing or conflicting memories first
(memory_search for a specific topic, memory_list if unsure):
- Nothing related exists → store the new fact
- A memory contradicts it → store the updated fact
- Already stored → don't duplicate; just acknowledge

Store facts as short, atomic, third-person statements:
- Good: "User likes hiking"
- Bad: "The user mentioned they enjoy going hiking sometimes"

Do NOT store: transient requests, general knowledge questions, or
sensitive data the user didn't ask you to keep.

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
- Before storing, when you're unsure what's already saved
- After a memory_search that returned nothing but you suspect the
  fact exists under different wording

## When to use NO tool
General knowledge, coding, math, casual chat with no personal
component. Most turns need no memory call.

## Response style
After storing: confirm briefly and naturally. Never mention tool
names or the mechanics of your memory system. If no relevant
memories exist, say you don't have that information yet — never
invent one."""


def agent_run(
    messages: list[dict],
    registry: dict,
    threads: ThreadStore,
    thread_id: int,
    max_iterations: int = 5
) -> str:
    """
    Agent loop: keep calling the model until it stops requesting tools
    """
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        message = response.choices[0].message

        # no tool calls -> final answer, we're done
        if not message.tool_calls:
            final_message = {
                "role": "assistant",
                "content": message.content
            }
            messages.append(final_message)
            threads.append_message(thread_id, final_message)
            return message.content

        # append the assistant turn that requested the tools
        messages.append(message)
        threads.append_message(thread_id, message)


        # execute every tool call and append each result
        for call in message.tool_calls:
            result = dispatch(call.function.name, call.function.arguments, registry)
            tool_message = {
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            }
            messages.append(tool_message)
            threads.append_message(thread_id, tool_message)
        # loop continues: model now sees the tool results
    
    return "Reached max iterations without a final answer."


def main() -> None:
    with MemoryStore("memories.db") as store, ThreadStore("memories.db") as threads:
        registry = make_tools(store, USER_ID)

        def system_only():
            return [{
                "role": "system",
                "content": SYSTEM_PROMPT
            }]

        # Start with the latest existing thread, or create a fresh one
        existing = threads.list_threads(USER_ID)
        if existing:
            thread_id = existing[0]["id"]
            messages = system_only()
            messages.extend(threads.load_messages(thread_id))
            print(f"Resuming thread {thread_id}: {existing[0]['title']} "
                  f"({len(messages) - 1} prior messages)")
        else:
            thread_id = threads.create_thread(USER_ID, "Agent chat")
            messages = system_only()
            print(f"Started new thread {thread_id}")

        while True:
            user_input = input("you> ").strip()
            if not user_input:
                continue

            # --- slash commands ------------------------------------------------
            if user_input == "/exit" or user_input in {"exit", "quit"}:
                break

            if user_input.startswith("/new"):
                # Optional title after the command, e.g. /new Trip planning
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

                # Try to parse an id from the command, e.g. /resume 3
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
                print(f"Resumed thread {chosen_id} "
                      f"({len(prior)} prior messages)")
                continue
            # -------------------------------------------------------------------

            # Normal turn: save user message and run agent
            user_message = {
                "role": "user",
                "content": user_input
            }
            messages.append(user_message)
            threads.append_message(thread_id, user_message)

            reply = agent_run(messages, registry, threads, thread_id)
            print("agent>", reply)


if __name__ == "__main__":
    main()