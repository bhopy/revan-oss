"""
file_scanner.py — Proof-of-concept agent for Revan Hub

Reads JSON Lines from stdin, classifies files using Revan (via the hub),
writes results as JSON Lines to stdout.

Protocol:
  Hub -> Agent (stdin):
    {"type": "task", "id": "...", "action": "classify", "payload": {"path": "..."}}
    {"type": "thought", "id": "...", "output": "config", "gen_ms": 362}
    {"type": "shutdown"}

  Agent -> Hub (stdout):
    {"type": "think", "id": "...", "system": "...", "user": "...", "max_tokens": 5}
    {"type": "result", "id": "...", "status": "ok", "data": {"category": "..."}}
    {"type": "progress", "id": "...", "message": "..."}
    {"type": "error", "id": "...", "message": "..."}
"""

import json
import sys
import os

# Pending think requests — waiting for hub to send back "thought" responses
pending_thinks = {}


def send(msg):
    """Send a JSON message to the hub via stdout."""
    line = json.dumps(msg, separators=(",", ":"))
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def handle_task(msg):
    """Handle a task dispatched by the hub."""
    task_id = msg["id"]
    action = msg["action"]
    payload = msg.get("payload", {})

    if action == "classify":
        classify_file(task_id, payload)
    else:
        send({
            "type": "error",
            "id": task_id,
            "message": f"unknown action: {action}",
        })


def classify_file(task_id, payload):
    """Classify a file by reading its first 500 bytes and asking Revan."""
    path = payload.get("path", "")

    if not path or not os.path.isfile(path):
        send({
            "type": "error",
            "id": task_id,
            "message": f"file not found: {path}",
        })
        return

    # Read first 500 bytes
    send({"type": "progress", "id": task_id, "message": f"reading {path}"})

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(500)
    except Exception as e:
        send({"type": "error", "id": task_id, "message": str(e)})
        return

    # Ask Revan to classify via the hub
    system_prompt = (
        "You are a file classifier. Given file content, respond with ONLY the category.\n"
        "Categories: config, code, data, docs, log, binary, unknown\n"
        "Respond with one category only."
    )

    send({
        "type": "think",
        "id": task_id,
        "system": system_prompt,
        "user": f"File: {os.path.basename(path)}\nContent:\n{content}",
        "max_tokens": 5,
    })

    # Mark as pending — we'll complete this task when "thought" comes back
    pending_thinks[task_id] = {"path": path}


def handle_thought(msg):
    """Handle Revan's response to our think request."""
    task_id = msg["id"]
    output = msg.get("output", "").strip().lower()
    gen_ms = msg.get("gen_ms", 0)

    if task_id not in pending_thinks:
        return

    context = pending_thinks.pop(task_id)

    send({
        "type": "result",
        "id": task_id,
        "status": "ok",
        "data": {
            "path": context["path"],
            "category": output or "unknown",
            "gen_ms": gen_ms,
        },
    })


def main():
    """Main loop: read JSON lines from stdin, process messages."""
    # Signal ready
    send({"type": "progress", "id": "init", "message": "file_scanner agent ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"[file_scanner] invalid JSON: {e}\n")
            continue

        msg_type = msg.get("type", "")

        if msg_type == "task":
            handle_task(msg)
        elif msg_type == "thought":
            handle_thought(msg)
        elif msg_type == "shutdown":
            send({"type": "progress", "id": "shutdown", "message": "shutting down"})
            break
        else:
            sys.stderr.write(f"[file_scanner] unknown message type: {msg_type}\n")


if __name__ == "__main__":
    main()
