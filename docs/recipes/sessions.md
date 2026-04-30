# Sessions & multi-turn

`Session` maintains conversation context across multiple `ask` calls. Each turn is stored in a ring buffer and prepended to the LLM's message history on the next call.

---

## Construct and use

```python
from sleuth import Sleuth, Session
from sleuth.backends import WebBackend
from sleuth.llm.anthropic import Anthropic

agent = Sleuth(
    llm=Anthropic(model="claude-sonnet-4-6"),
    backends=[WebBackend(provider="tavily", api_key="...")],
)

# Create a session (default: last 20 turns)
session = Session(max_turns=20)

r1 = agent.ask("Who owns the auth middleware?", session=session)
print(r1.text)

# The LLM receives r1's Q+A as context
r2 = agent.ask("What tests cover it?", session=session)
print(r2.text)
```

The session is passed via the `session=` parameter. Turns are appended automatically after each successful call.

---

## Instance-level vs per-call session

Set a session at the `Sleuth` constructor level to apply it to every call:

```python
session = Session()
agent = Sleuth(llm=..., backends=[...], session=session)

agent.ask("First question")   # uses session
agent.ask("Follow-up")        # uses same session
```

Override per-call to use a different session for a specific call:

```python
other_session = Session()
agent.ask("Unrelated question", session=other_session)  # uses other_session
```

---

## `as_messages()`

`Session.as_messages()` returns the buffer as alternating `user` / `assistant` `Message` objects — the format expected by `LLMClient.stream()`. You can inspect the current conversation history:

```python
for msg in session.as_messages():
    print(f"[{msg.role}] {msg.content[:80]}")
```

---

## Ring buffer size

The default buffer size is 20 turns. Oldest turns are evicted when full:

```python
session = Session(max_turns=5)    # keep only the last 5 turns

print(session.max_turns)          # 5
print(len(session.turns))         # 0 (empty at start)
```

`session.turns` returns a snapshot as a list of `Turn` dataclasses (oldest first).

---

## Persistence

### Save

```python
session.save("./my_session.json")
```

Writes atomically. The file contains `max_turns` and the turn buffer (query + result text + citations).

### Load

```python
session = Session.load("./my_session.json")
```

`Session.load` is a classmethod. Raises `FileNotFoundError` if the path does not exist.

### Background save with `flush`

`_schedule_background_save` queues an async write task. Call `await session.flush()` to ensure it completes before shutdown or the next turn:

```python
session._schedule_background_save("./my_session.json")
# ... do other work ...
await session.flush()   # wait for the background write to finish
```

For most use cases, `session.save(path)` (synchronous) is simpler and safe.

---

## Complete example

```python
import asyncio
from sleuth import Sleuth, Session
from sleuth.backends.localfiles import LocalFiles
from sleuth.llm.anthropic import Anthropic

SAVE_PATH = "./auth_session.json"

async def main():
    agent = Sleuth(
        llm=Anthropic(model="claude-sonnet-4-6"),
        backends=[LocalFiles(path="./docs")],
    )

    # Load existing session if available
    try:
        session = Session.load(SAVE_PATH)
        print(f"Resuming session ({len(session.turns)} prior turns)")
    except FileNotFoundError:
        session = Session(max_turns=20)

    # Conversation loop
    while True:
        query = input("You: ").strip()
        if not query:
            break
        result = agent.ask(query, session=session)
        print(f"Sleuth: {result.text}\n")

    # Persist for next time
    session.save(SAVE_PATH)
    print(f"Session saved to {SAVE_PATH}")

asyncio.run(main())
```
