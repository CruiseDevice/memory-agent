# Memory Agent

A memory-augmented agent that **remembers things about a user across sessions** — built from scratch with raw Python, the OpenAI API, SQLite, and `sqlite-vec`. No LangChain, no Strands, no mem0. Every line is owned.

```
> I enjoy hiking and outdoor photography
Got it. Stored.

> What nature stuff am I into?
You enjoy hiking and outdoor photography!
```

Close the terminal, come back tomorrow — it still remembers, because memories are embedded at store time and retrieved by **semantic similarity**, not keyword matching.

---

## How It Works

1. The LLM decides when to store, search, or list memories via three tools.
2. On store, the memory text is embedded (`text-embedding-3-small`, 1536-dim) and saved as a BLOB in SQLite.
3. On search, the query is embedded and compared against stored memories using cosine distance (`vec_distance_cosine` from `sqlite-vec`).
4. The LLM uses the retrieved memories to answer as if it simply knows.
5. Every conversation turn is saved to a `messages` table linked to `threads` table (via `thread_id`), each thread owned by a user_id.
6. On startup, the most recently updated thread is resumed automatically.

This is the same machinery mem0 runs under the hood — except here you can see every line of it.

## Setup

Requires an OpenAI API key and the `sqlite-vec` extension.

### Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | OpenAI API access |
| `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` | API base URL (for compatible providers) |
| `AGENT_MODEL` | no | `gpt-4o-mini` | Chat completion model |

### Run

```bash
export OPENAI_API_KEY="sk-..."
python agent.py
```

The agent resumes your latest thread automatically. Type `/exit` to leave. Memories and conversation history persist in `memories.db` between sessions.

## Slash commands

While chatting, you can use these commands:

| Command | Description|
|---|---|
| /new [title] | Start a brand-new conversation thread. Any title after the command is saved. |
| /resume [id] | List all saved threads, then switch to the one you pick. |
| /exit | Save and exit. |


## Example

```
you> I enjoy hiking and outdoor photography
agent> Got it. Stored.

you> /new Trip planning
Started new thread 2: Trip planning

you> Suggest a weekend trip
agent> ...

you> /resume
Saved threads:
  1: Agent chat (updated 2024-01-15 10:30:00)
  2: Trip planning (updated 2024-01-15 11:00:00)
Enter thread id to resume> 1
Resumed thread 1 (3 prior messages)

you> What do I enjoy outdoors?
agent> You enjoy hiking and outdoor photography.
```

Both memory facts and full conversation threads live in the same `memories.db` SQLite database, but in separate tables.