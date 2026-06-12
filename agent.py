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


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "memory_store",
            "description": "Store a durable fact about the user (preference, "
                "personal detail, goal, constraint). Call when the user shares "
                "lasting personal info, explicitly or implicitly. Search first "
                "to avoid duplicates. Input: a short third-person fact string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "A short, atomic, third-person fact. e.g. 'User likes hiking'"
                    }
                },
                "required": ["content"],
            },
        },
    },
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


def agent_run():
    pass


def main() -> None:
    print(agent_run())


if __name__ == "__main__":
    main()