from client import MODEL, client
from store import MemoryStore
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


def agent_run(messages: list[dict], registry: dict, max_iterations: int = 5) -> str:
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
            messages.append({
                "role": "assistant",
                "content": message.content
            })
            return message.content

        # append the assistant turn that requested the tools
        messages.append(message)

        # execute every tool call and append each result
        for call in message.tool_calls:
            result = dispatch(call.function.name, call.function.arguments, registry)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })
        # loop continues: model now sees the tool results
    
    return "Reached max iterations without a final answer."


def main() -> None:
    with MemoryStore("memories.db") as store:
        registry = make_tools(store, USER_ID)

        messages = [{
            "role": "system",
            "content": SYSTEM_PROMPT
        }]

        while True:
            user_input = input("you> ").strip()
            if user_input in {"exit", "quit"}:
                break
            messages.append({
                "role": "user",
                "content": user_input
            })
            print("agent> ", agent_run(messages, registry))


if __name__ == "__main__":
    main()