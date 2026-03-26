# Revan

A stateless decision engine that runs a 32B language model on consumer hardware.

## What this is

Running a 32B model on an RTX 2060 Super gives you about 3 tokens per second. That's way too slow for generating text — a single paragraph takes 17+ seconds. But it turns out 3 tok/s is plenty fast if you never ask it to write.

Revan treats the model as a decision-maker instead of a writer. You give it a short, focused prompt (~150-250 tokens) and it gives back a structured answer in 1-15 tokens. A yes/no answer comes back in 0.4 seconds. Tool routing in 0.5 seconds. Scoring 5 items in about 3 seconds. The model understands the question just as well — it just doesn't waste time explaining itself.

```
Input:  "Which tool should handle this? web_search / file_read / calculator / none"
Output: "calculator"  (1 token, 0.4 seconds)
```

The whole thing is stateless by design. The engine doesn't track history or accumulate context. Whatever calls it (an agent, a script, a pipeline) holds the state and feeds Revan a small question each time. That keeps every call constant-time — the 10th decision in a workflow is just as fast as the 1st.

## Architecture

Three layers:

```
  Your code / agents (hold all state, build prompts)
        |
        |  small prompt in (~150-250 tokens)
        v
  Engine (C++, ~1350 lines)
  - loads model once, links llama.cpp directly
  - Named Pipe IPC, binary protocol
  - 3-slot LRU cache for system prompts
  - hot model swap without restart
        |
        |  tiny answer out (1-15 tokens)
        v
  Hub (Rust, ~2100 lines)
  - async orchestrator (Tokio)
  - spawns/manages agent processes
  - routes agent "think" requests to the engine
  - VRAM monitoring via NVML
  - Job Objects kill all agents when hub exits
        |
        v
  Agents (any language, Python PoC included)
  - JSON Lines on stdin/stdout
  - send "think" to ask Revan a question
  - receive "thought" with the answer
  - send "result" when done
```

About 3,500 lines total across C++, Rust, and Python.

## Performance

All numbers from an RTX 2060 Super (8 GB), OLMo 3.1 32B Instruct Q4_K_M, 23/64 layers on GPU.

**Decision speed:**

- Yes/No → 1 token, ~0.4s, 100% accuracy (8/8 test cases)
- Tool routing → 1-2 tokens, ~0.5s, 100% accuracy (8/8)
- Sentiment → 1 token, ~0.4s
- Scoring 5 items → 5-9 tokens, ~3.0s
- File classification → 1 token, ~0.5s, 97% (29/30)

**Throughput:**

- Prompt processing: 82-128 tok/s cold, up to 185 tok/s with cache hits
- Generation: 3.23-3.31 tok/s

**KV cache savings** (same system prompt, repeat calls):

- Yes/No prompts: 4,746ms → ~1,893ms (60% faster)
- Routing prompts: 3,198ms → ~1,700ms (48% faster)
- Scoring prompts: 2,356ms → 1,203ms (49% faster)

**Multi-step workflows** stay fast because prompts don't grow:

- 1-2 decisions: 1-3 seconds
- 3 decisions: ~5 seconds
- 5 decisions: ~11 seconds
- 10 decisions: ~18 seconds (would be ~60s if context accumulated)

## Comparisons

**vs. cloud APIs** — Revan is comparable latency for simple decisions (0.4-3s vs 0.5-5s + network), costs nothing to run, and data never leaves your machine. Cloud wins on quality for anything open-ended or requiring 100B+ scale.

**vs. smaller local models (7B, 3B)** — A 32B model constrained to 1-15 tokens is more accurate than a 7B model generating 50+ tokens, and wall-clock time is similar because the output is so short. The 7B is better if you actually need text generation.

**vs. llama-server directly** — Revan links llama.cpp as a library instead of running it as a server. Named Pipe IPC is ~0.1ms overhead vs 1-5ms for HTTP. Built-in KV cache is tuned for repeated system prompts. Hot model swap without restart. The tradeoff is it's Windows-only and less portable.

## What it's good at

- Tool routing ("which tool handles this?") → 1 token, sub-second
- Yes/No decisions → 1 token, ~0.4s
- Classification (sentiment, intent, file type, error category) → 1 token
- Scoring and ranking → 5-10 tokens, 2-4s
- Confidence gating ("0-100, how sure?") → 1-3 tokens
- Short structured JSON → 10-15 tokens, 4-6s
- Anything where you need local, private, fast decisions at scale

## What it's not for

- Writing text. 3 tok/s makes paragraphs unusable.
- Code generation. Same reason.
- Summarization. Output length is unbounded.
- Conversation. It's a brain, not a chatbot.
- Anything needing more than ~15 tokens of output.

## Project structure

```
src/revan.cpp                  C++ engine
hub/
  revan-core/src/
    protocol.rs                Binary codec (20-byte req, 24-byte resp headers)
    client.rs                  Win32 Named Pipe client
    agent.rs                   Hub <-> Agent JSON protocol
  revan-hub/src/
    main.rs                    Entry point, CLI, config
    brain.rs                   FIFO request queue (mpsc + oneshot)
    agents.rs                  Agent lifecycle, Job Objects, routing
    monitor.rs                 VRAM monitoring (NVML)
agents/python/file_scanner.py  PoC agent (file classification)
tools/
  bench_revan.py               Benchmarks
  quant_ab_test.py             A/B quant comparison
  test_kv_cache.py             Cache validation
```

## Setup

See [BUILD.md](BUILD.md) for the full walkthrough. Short version:

1. Build llama.cpp b8067 as static libs with CUDA
2. Copy .lib and .h files into `bin_lib/`
3. `cmake --build . --config Release` for the engine
4. `cargo build --release` for the hub
5. Download a GGUF model, edit `revan.toml`
6. Start engine, start hub, run benchmarks

## Protocol

The engine uses a binary protocol over Windows Named Pipes. 20-byte request header, 24-byte response header, everything little-endian. No HTTP, no JSON on the wire.

Request types:

- `0x0001` — inference (V1)
- `0x0002` — health check
- `0x0003` — shutdown
- `0x0004` — inference V2 (GBNF grammar + logprobs)
- `0x0005` — hot model swap

Agents talk to the hub via JSON Lines on stdin/stdout:

```
Hub  → Agent:  {"type":"task", "id":"...", "action":"classify", "payload":{...}}
Agent → Hub:   {"type":"think", "id":"...", "system":"...", "user":"...", "max_tokens":5}
Hub  → Agent:  {"type":"thought", "id":"...", "output":"config", "gen_ms":362}
Agent → Hub:   {"type":"result", "id":"...", "status":"ok", "data":{...}}
```

Agents can be written in any language.

## Writing an agent

Minimal example — a yes/no classifier:

```python
import json, sys

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

send({"type": "progress", "id": "init", "message": "ready"})

for line in sys.stdin:
    msg = json.loads(line.strip())

    if msg["type"] == "task":
        send({
            "type": "think", "id": msg["id"],
            "system": "Respond with only yes or no.",
            "user": msg["payload"]["question"],
            "max_tokens": 3,
        })
    elif msg["type"] == "thought":
        send({
            "type": "result", "id": msg["id"],
            "status": "ok",
            "data": {"answer": msg["output"].strip()},
        })
    elif msg["type"] == "shutdown":
        break
```

Register in `hub.toml`, then `spawn` and `dispatch` from the hub CLI.

## Design notes

**Stateless** — If the engine tracked history, prompts would grow from 200 to 1000+ tokens over a multi-step workflow. At 82-128 tok/s that turns a 0.75s read into 5+ seconds by step 10. Keeping prompts flat means constant-time calls.

**Named Pipes** — Atomic message delivery, ~0.1ms overhead. HTTP adds 1-5ms per call for parsing and TCP. Small per-call, but it adds up.

**Library linking** — llama.cpp runs in-process, not as a server. Model loads once, inference is a function call. Hot swap, KV cache, GBNF grammar are all controlled at the library level.

**Rust hub** — Agent management is concurrency-heavy. Tokio handles it cleanly. Job Objects guarantee cleanup on exit or crash.

**KV cache** — 3-slot LRU using `llama_state_seq_get_data` / `llama_state_seq_set_data`. Each slot is ~1.6-1.8 MB. Saves 48-60% on repeat system prompts.

**Hot model swap** — Swaps models via request type `0x0005` without killing the process. Saves ~26 seconds of cold-start.

## Hardware

Built and tested on:

- AMD Ryzen 5 5600X, 32 GB DDR4-3200
- RTX 2060 Super (8 GB VRAM)
- NVMe SSD, Windows 11, CUDA 12.4
- llama.cpp b8067

VRAM breakdown: ~4.2 GB model (23 layers), ~0.4 GB KV cache, ~5.4 MB system prompt cache, ~1.0 GB CUDA overhead. Leaves about 2.4 GB headroom.

Works with any GGUF model. Tested with OLMo 3.1 32B, Llama 3.2 3B, and Mistral 7B. Bigger models at Q4_K_M give better decision accuracy than smaller models at higher precision.

## Status

The engine, hub, and protocol work. The file_scanner agent demonstrates the full think → thought → result chain end to end. 22 unit tests cover the protocol codec.

What's left to build: real-world agents for actual tasks, persistent KV cache that survives restarts, configurable cache slots, agent-to-agent routing.

## License

MIT. See [LICENSE](LICENSE).
