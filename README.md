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

Type `quit` or `exit` to leave. Memories persist in `memories.db` between sessions.
