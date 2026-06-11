# Build Plan: Memory Agent (No Framework)

## Project Goal
Build a memory-augmented agent that **remembers things about a user across sessions** using raw Python, an LLM API, and a vector store you build yourself. No Strands, no mem0, no LangChain, no orchestration frameworks. You own every line.

### The User Experience
```
> Remember that I prefer window seats on flights
Got it. Stored.

> I'm planning a trip to Japan next spring
Noted.

> What do you know about my travel preferences?
You prefer window seats on flights, you're planning a trip to Japan next spring,
and you like staying in Airbnbs over hotels.
```

Close the terminal. Come back tomorrow:
```
> Suggest a weekend activity for me
Based on what I know — you enjoy hiking and outdoor photography — how about
a sunrise hike at Mount Tamalpais? Great photo opportunities.
```

---

## Phase 0: Foundation & Mental Model

**Objective:** Understand the problem space and sketch your architecture before touching code.

### The Core Problem
Your weather agent was **stateless** — each query was independent. This agent is **stateful** — it accumulates context over time and uses it to personalize responses. That changes everything about how you think about storage, retrieval, and the agent loop.

### The Two Hard Questions
1. **Storage:** Where do memories live? How are they structured?
2. **Retrieval:** When the user asks "what do I like?", how do you find "I enjoy hiking" in a pile of stored memories? Keyword matching? Semantic similarity?

**Key insight:** Keyword matching is where you start. Semantic search (embeddings + vector similarity) is where you end up. The journey between them is the lesson.

**Decision to make:**
- Single user for MVP, or multi-user from day one?
- Should memories ever expire? If I say "I live in Seattle" in January and "I moved to Austin" in June, what happens?
- Do you want to build your own vector store (numpy arrays), or use a lightweight library like `sqlite-vss` or `chromadb`?

**Deliverable for this phase:**
- A hand-drawn or ASCII diagram showing the data flow: `User Input` → `Intent Classification` → `Memory Operation (store/retrieve/list)` → `Response Generation`.
- A list of 3–5 explicit assumptions (e.g., "single user for MVP," "OpenAI API for both chat and embeddings," "SQLite for persistence").

---

## Phase 1: Storage Layer (SQLite)

**Objective:** Build the persistence foundation. No LLM, no agent loop. Just raw database operations.

### Component: The Database Schema
Design a SQLite table that stores memories. Minimum columns to think about:

- `id` — unique identifier
- `user_id` — who does this memory belong to?
- `content` — the text of the memory
- `embedding` — a vector representation (placeholder for now, filled in Phase 3)
- `created_at` — when was it stored?

**Design questions:**
- What SQLite type stores a vector? (Hint: SQLite doesn't have a native vector type. How will you serialize it? BLOB? JSON?)
- Should you index anything? On what column(s)?
- What's the schema migration story? If you need to add a column later, how does that work?

### Component: The Data Access Layer
Build a `MemoryStore` class (or module) with these operations:

```
store(user_id, content) -> id
list_memories(user_id) -> list of (id, content, created_at)
search_by_keyword(user_id, keyword) -> list of (id, content, created_at)
delete(memory_id) -> bool
```

**Design questions:**
- Should `store` return the inserted ID? Why might that matter?
- Should `search_by_keyword` do exact match, substring match, or full-text search? SQLite has FTS5 — worth exploring.
- What happens if `user_id` doesn't exist yet? Auto-create?

**Acceptance Criteria:**
- You can write a script that stores 5 memories, lists them all, searches by keyword, and deletes one.
- You understand every column in your schema and why it exists.
- The database file persists on disk between script runs.

---

## Phase 2: Agent Loop + Memory Tools (Keyword Retrieval)

**Objective:** Connect your agent loop from the weather agent to the memory store. Use keyword matching for retrieval. Experience its limitations.

### Component: Tool Registry (Extended)
You're reusing the tool registry pattern from your weather agent. Three new tools:

| Tool | Purpose | Maps to |
|---|---|---|
| `memory_store` | Store a piece of information | `MemoryStore.store()` |
| `memory_search` | Find relevant memories | `MemoryStore.search_by_keyword()` |
| `memory_list` | List all memories for a user | `MemoryStore.list_memories()` |

Each tool needs a JSON Schema for the LLM's `functions` parameter. Think about:
- What parameters does each need?
- What does each return? (Remember: the LLM reads the output)
- How do you communicate errors back to the LLM?

### Component: The Agent Loop
Same ReAct pattern from your weather agent:

```
agent_run(user_query, user_id, tools, system_prompt) -> str
```

New concern: `user_id` is threaded through every operation. The agent needs to know *who* it's talking to.

### Component: The System Prompt
Draft a prompt that teaches the LLM:
- Its identity (a memory-augmented assistant).
- That it has three memory tools.
- When to store vs. retrieve vs. list.
- How to use retrieved memories to personalize responses.

**Design questions:**
- Should the LLM decide *when* to store, or should you use deterministic routing (like the Strands example did with `startswith("remember ")`)? Pros/cons of each?
- Should the system prompt include examples of when to store vs. retrieve?
- How do you tell the LLM "use these memories to answer the question but don't make things up"?

**Acceptance Criteria:**
- User says "remember I like hiking" → agent calls `memory_store` → memory is in SQLite.
- User says "what do I like?" → agent calls `memory_search("like")` → returns "I like hiking" → agent paraphrases it naturally.
- **Now try:** User says "what outdoor stuff do I enjoy?" → agent calls `memory_search("outdoor stuff enjoy")` → keyword search returns nothing → agent says "I don't have that information."
- **This is the wall.** This is where keyword matching fails. Sit with it. Understand why.

---

## Phase 3: Embeddings & Semantic Retrieval

**Objective:** Replace keyword matching with vector similarity. This is the core lesson of the entire project.

### Why You're Here
"I enjoy hiking and outdoor photography" should match "what outdoor stuff do I like?" — even though they share almost no keywords. Embeddings solve this by representing *meaning*, not just words.

### Component: The Embedding Function
Build a function that takes a string and returns a vector (list of floats).

```
embed(text: str) -> list[float]
```

**Design questions:**
- Which embedding model? OpenAI's `text-embedding-3-small` is cheap and good. But you could also use a local model via `sentence-transformers`. Tradeoffs?
- What's the dimension of your embedding vector? Does it matter for storage?
- What happens if the API is down? Do you cache embeddings?

### Component: Vector Similarity Search
Build a function that compares a query vector against all stored memory vectors and returns the most similar ones.

```
search_by_similarity(user_id, query_text, top_k=5) -> list of (id, content, score)
```

**The math is simple — cosine similarity:**
- Normalize both vectors to unit length.
- Dot product = cosine similarity (ranges -1 to 1, higher = more similar).
- Return top K results above a minimum threshold.

**Where do the vectors live?**
| Option | Complexity | When to use |
|---|---|---|
| Numpy array in memory | Low | MVP. Load all vectors on startup. |
| Serialize as BLOB in SQLite | Low-Medium | Persistent, but slow for large datasets (linear scan). |
| `sqlite-vss` extension | Medium | SQLite with vector indexing. Proper solution. |
| `chromadb` | Medium | Purpose-built vector DB. More dependencies. |

**Start with numpy.** Feel the linear scan. Understand why it doesn't scale. Then upgrade.

**Design questions:**
- What minimum similarity threshold do you use? 0.3? 0.5? How do you tune this?
- Should you embed the memory when it's stored, or lazily when it's searched?
- What happens if you have 10,000 memories? 100,000? At what point does numpy get slow?

**Acceptance Criteria:**
- Store "I enjoy hiking and outdoor photography."
- Search for "outdoor activities I like" → returns that memory with a high similarity score.
- Search for "my dog's name" → does NOT return the hiking memory (low similarity).
- **You can now explain what mem0 does under the hood and why it costs money.**

---

## Phase 4: Conversation Memory (Message Persistence)

**Objective:** Let the agent remember the *conversation* too, not just explicit memories.

### The Problem So Far
Phases 1–3 handle explicit memory ("remember X"). But what about:
- The user mentions their dog in passing: "Max was barking all night"
- Later asks: "What's my dog's name?"
- This was never explicitly stored. It's in the conversation history.

### Two Approaches

**Approach A: Persist the full message history**
Store the `messages` list (your agent loop's state) in SQLite. Load it on the next session. The LLM sees the full conversation as context.

- Pros: Simple. The LLM has everything.
- Cons: Token costs grow linearly. A 50-message conversation is expensive to re-send every time.

**Approach B: Auto-extract memories from conversation**
After each exchange, run a background pass that asks the LLM: "Is there anything in this conversation worth remembering long-term?" If yes, store it as a memory.

- Pros: Token-efficient. Only important facts are persisted.
- Cons: Two LLM calls per exchange. The extraction might miss things.

**You need to decide which approach fits your mental model.** Or combine both — persist recent messages (last N) for short-term context, and extract memories for long-term recall.

**Acceptance Criteria:**
- Close the terminal. Reopen. Ask "what were we talking about earlier?" → agent remembers.
- Token usage doesn't grow unboundedly across sessions.

---

## Phase 5: Resilience, Observability, Refinement

**Objective:** Same as your weather agent's Phase 5. Make it solid.

### Error Handling
- What happens if the embedding API fails mid-store? Is the memory saved without an embedding? Is that a problem?
- What if SQLite is locked (concurrent access)?
- What if cosine similarity returns all memories below threshold — does the agent say "I don't know" or hallucinate?

### Caching
- Embed the same text twice → should you cache the embedding? Where?
- Does the agent loop itself need caching? (Probably not — each query is different.)

### Logging & Tracing
Every agent run should log:
- Timestamp, user ID, query.
- Which memory operations were triggered (store, search, list).
- Embedding API calls (input tokens, latency).
- LLM calls (prompt tokens, completion tokens, model).
- Final response.

### Memory Management
- Should old memories be pruned? After how long?
- What about contradictory memories? ("I live in Seattle" vs. "I moved to Austin")
- Should there be a `memory_delete` tool? Who triggers it — the user or the agent?

**Acceptance Criteria:**
- You can look at a log and trace exactly why the agent retrieved a specific memory.
- The agent handles "I don't know" gracefully instead of hallucinating memories.
- Embedding failures don't corrupt the database.

---

## Phase 6: Extensions (Pick Your Own Adventure)

### A. Multi-User Support
Thread `user_id` through everything. Test with two users — their memories should be isolated.

### B. Memory Categories / Tags
Let users categorize memories ("this is about food", "this is about travel"). Store tags. Search by tag + semantic similarity.

### C. Memory Importance Scoring
Not all memories are equal. "My name is Alex" is more important than "I had a sandwich for lunch." Let the LLM assign an importance score on store. Weight retrieval by importance.

### D. Proactive Memory
The agent notices things in conversation and stores them without being asked. "You mentioned your dog Max — would you like me to remember that?"

### E. Web Interface
Same as your weather agent's Phase 6C. Wrap in FastAPI. Add a `thread_id` for conversation state.

---

## Architecture Appendix

### Directory Structure (Suggested)

```
memory-agent/
├── AGENTS.md               # How the AI coach works with you
├── PLAN.md                 # This file
├── requirements.txt        # When you're ready
├── src/
│   ├── __init__.py
│   ├── config.py           # API keys, DB path, embedding model config
│   ├── llm_client.py       # Wrapper around OpenAI SDK
│   ├── store.py            # SQLite schema + data access layer
│   ├── embeddings.py       # embed() function + caching
│   ├── similarity.py       # Vector similarity search
│   ├── tools/
│   │   ├── __init__.py     # Tool registry (schemas + dispatch)
│   │   ├── memory_store.py
│   │   ├── memory_search.py
│   │   └── memory_list.py
│   ├── agent.py            # The main loop
│   └── formatter.py        # Optional: response formatting pass
├── tests/
│   └── ...                 # Unit test store, embeddings, similarity independently
├── scripts/
│   └── demo.py             # Interactive CLI for manual testing
└── data/
    └── memories.db         # SQLite database (gitignored)
```

### Key Interfaces (Contracts)

Define these before you code:

1. **`MemoryStore`**: `store(user_id, content) -> id`, `search(user_id, query) -> list[Memory]`, `list(user_id) -> list[Memory]`
2. **`Embedder`**: `embed(text: str) -> list[float]`, `embed_batch(texts: list[str]) -> list[list[float]]`
3. **`SimilaritySearch`**: `search(query_vector, candidates, top_k, min_score) -> list[(id, score)]`
4. **`ToolFunction`**: Same pattern as your weather agent — `(params: dict) -> str`

### Dependency Philosophy
Keep it minimal. You probably need:
- An LLM SDK (`openai` — for both chat completions and embeddings)
- `httpx` (if calling embedding APIs directly instead of via SDK)
- `sqlite3` (stdlib — no install needed)
- `numpy` (for vector math — cosine similarity)

Everything else is a distraction until you need it.

---

## Exit Criteria

You know you're done with the MVP when:
1. A new developer can clone the repo, set one env var for the LLM key, run a CLI script, and have a conversation where the agent remembers things.
2. You can type "remember I like X", close the terminal, reopen it, ask "what do I like?", and get the right answer.
3. You can explain, step-by-step, how a memory is stored, embedded, retrieved by similarity, and injected into the LLM's context.
4. You can articulate what mem0/Strands would have abstracted away, and why you chose to own that complexity.
5. You've experienced keyword search failing and can explain *why* embeddings are necessary — not because a blog post told you, but because you hit the wall yourself.

---

**Next step:** Read Phase 0. Decide on single-user vs. multi-user. Decide on embedding provider. Sketch your ASCII diagram. Come back when you have answers.
