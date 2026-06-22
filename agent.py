from client import MODEL, client
from extractor import background_store_fact, extract_facts
from prompts import AGENT_SYSTEM_PROMPT
from store import MemoryStore
from thread import ThreadStore
from tools import TOOL_SCHEMAS, dispatch, make_tools

USER_ID = "agent_user"


# The agent may READ memories but must never WRITE them.
# Writing is handled by the background extractor after the assistant reply.
AGENT_TOOLS = list(TOOL_SCHEMAS)

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
            tools=AGENT_TOOLS,  # <-- agent only reads memories; writes are handled by the extractor.
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
            facts = extract_facts(user_input)
            for fact in facts:
                background_store_fact(fact, store, USER_ID)


if __name__ == "__main__":
    main()
