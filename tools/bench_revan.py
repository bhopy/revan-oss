"""
bench_revan.py — Benchmark Revan inference speeds
Runs multiple sequential requests to find steady-state performance.
"""

import struct
import time
import sys
import ctypes
from ctypes import wintypes

PIPE_NAME = r'\\.\pipe\revan'
PROTO_VERSION = 1
REQ_INFERENCE = 0x0001

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.CreateFileW.restype = wintypes.HANDLE

GENERIC_READ  = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
PIPE_READMODE_MESSAGE = 0x00000002


def build_request(max_tokens, temperature_x100, system_prompt, user_prompt):
    sys_bytes = system_prompt.encode('utf-8')
    usr_bytes = user_prompt.encode('utf-8')
    sys_len = len(sys_bytes)
    payload = sys_bytes + usr_bytes
    msg_len = 20 + len(payload)
    header = struct.pack('<IHHHHII', msg_len, PROTO_VERSION, REQ_INFERENCE,
                         max_tokens, temperature_x100, 0, sys_len)
    return header + payload


def parse_response(data):
    if len(data) < 24:
        return None
    msg_len, version, status, tok_gen, tok_prompt, pp_ms, gen_ms, _ = \
        struct.unpack('<IHHHHIII', data[:24])
    output = data[24:].decode('utf-8', errors='replace') if len(data) > 24 else ""
    return {
        "status": status, "output": output,
        "tok_gen": tok_gen, "tok_prompt": tok_prompt,
        "pp_ms": pp_ms, "gen_ms": gen_ms,
    }


def send_request(request_bytes):
    handle = kernel32.CreateFileW(
        PIPE_NAME, GENERIC_READ | GENERIC_WRITE, 0, None, OPEN_EXISTING, 0, None)
    if handle == wintypes.HANDLE(-1).value:
        raise OSError(f"Cannot connect (error {ctypes.get_last_error()})")
    try:
        mode = wintypes.DWORD(PIPE_READMODE_MESSAGE)
        kernel32.SetNamedPipeHandleState(handle, ctypes.byref(mode), None, None)
        written = wintypes.DWORD(0)
        kernel32.WriteFile(handle, request_bytes, len(request_bytes),
                          ctypes.byref(written), None)
        buf = ctypes.create_string_buffer(65536)
        read_bytes = wintypes.DWORD(0)
        ok = kernel32.ReadFile(handle, buf, 65536, ctypes.byref(read_bytes), None)
        if not ok:
            raise OSError(f"ReadFile failed (error {ctypes.get_last_error()})")
        return parse_response(buf.raw[:read_bytes.value])
    finally:
        kernel32.CloseHandle(handle)


# ==================== BENCHMARKS ====================

def bench_yesno(n=8):
    """Repeated yes/no decisions — 1 token output."""
    print(f"\n{'='*60}")
    print(f"BENCH: Yes/No x{n} (expect ~1 tok output each)")
    print(f"{'='*60}")

    sys_prompt = "You are a yes/no classifier. Respond with only 'yes' or 'no'. Nothing else."
    questions = [
        "Is the sky blue?",
        "Is 2+2 equal to 5?",
        "Does water boil at 100C?",
        "Is Python a compiled language?",
        "Is the Earth flat?",
        "Does Rust have a garbage collector?",
        "Is 17 a prime number?",
        "Is JavaScript strongly typed?",
    ]

    results = []
    for i in range(min(n, len(questions))):
        req = build_request(3, 0, sys_prompt, questions[i])
        start = time.time()
        resp = send_request(req)
        wall = time.time() - start

        pp_s = resp['tok_prompt'] / (resp['pp_ms'] / 1000.0) if resp['pp_ms'] > 0 else 0
        gen_s = resp['tok_gen'] / (resp['gen_ms'] / 1000.0) if resp['gen_ms'] > 0 else 0

        print(f"  [{i+1}] \"{resp['output']:10s}\" prompt={resp['tok_prompt']:3d}tok "
              f"pp={resp['pp_ms']:5d}ms ({pp_s:5.1f}t/s) "
              f"gen={resp['gen_ms']:5d}ms ({gen_s:4.2f}t/s) wall={wall:.2f}s")
        results.append({"pp_s": pp_s, "gen_s": gen_s, "pp_ms": resp['pp_ms'], "gen_ms": resp['gen_ms']})

    # Summary (skip first 2 as warmup)
    steady = results[2:] if len(results) > 2 else results
    avg_pp = sum(r['pp_s'] for r in steady) / len(steady)
    avg_gen = sum(r['gen_s'] for r in steady) / len(steady)
    avg_pp_ms = sum(r['pp_ms'] for r in steady) / len(steady)
    avg_gen_ms = sum(r['gen_ms'] for r in steady) / len(steady)
    print(f"\n  Steady-state (skip first 2): pp={avg_pp:.1f} tok/s, gen={avg_gen:.2f} tok/s")
    print(f"  Average timing: pp={avg_pp_ms:.0f}ms, gen={avg_gen_ms:.0f}ms")


def bench_routing(n=8):
    """Repeated tool routing — 1-2 token output."""
    print(f"\n{'='*60}")
    print(f"BENCH: Tool Routing x{n}")
    print(f"{'='*60}")

    sys_prompt = ("You are a tool router. Given a user query, respond with ONLY the tool name.\n"
                  "Available tools: web_search, file_read, calculator, calendar, email, none\n"
                  "Respond with one tool name only.")
    questions = [
        "What is 42 * 17?",
        "What's the weather in London?",
        "Read the contents of config.toml",
        "When is my next meeting?",
        "Send a message to John",
        "What is the square root of 144?",
        "Search for Rust tutorials",
        "Tell me a joke",
    ]

    results = []
    for i in range(min(n, len(questions))):
        req = build_request(5, 10, sys_prompt, questions[i])
        start = time.time()
        resp = send_request(req)
        wall = time.time() - start

        pp_s = resp['tok_prompt'] / (resp['pp_ms'] / 1000.0) if resp['pp_ms'] > 0 else 0
        gen_s = resp['tok_gen'] / (resp['gen_ms'] / 1000.0) if resp['gen_ms'] > 0 else 0

        print(f"  [{i+1}] \"{resp['output']:15s}\" prompt={resp['tok_prompt']:3d}tok "
              f"pp={resp['pp_ms']:5d}ms ({pp_s:5.1f}t/s) "
              f"gen={resp['gen_ms']:5d}ms ({gen_s:4.2f}t/s) wall={wall:.2f}s")
        results.append({"pp_s": pp_s, "gen_s": gen_s, "pp_ms": resp['pp_ms'], "gen_ms": resp['gen_ms']})

    steady = results[2:] if len(results) > 2 else results
    avg_pp = sum(r['pp_s'] for r in steady) / len(steady)
    avg_gen = sum(r['gen_s'] for r in steady) / len(steady)
    print(f"\n  Steady-state (skip first 2): pp={avg_pp:.1f} tok/s, gen={avg_gen:.2f} tok/s")


def bench_scoring(n=5):
    """Repeated scoring — ~9 token output."""
    print(f"\n{'='*60}")
    print(f"BENCH: Scoring x{n}")
    print(f"{'='*60}")

    sys_prompt = ("You are a relevance scorer. Given a query and results, output a score 1-10 for each.\n"
                  "Format: just the numbers separated by commas. Nothing else.")
    user = ("Query: \"rust async patterns\"\n"
            "1. Blog post about Rust ownership\n"
            "2. Tokio runtime tutorial\n"
            "3. Python asyncio guide\n"
            "4. Rust async/await deep dive\n"
            "5. JavaScript promises explained")

    results = []
    for i in range(n):
        req = build_request(10, 10, sys_prompt, user)
        start = time.time()
        resp = send_request(req)
        wall = time.time() - start

        pp_s = resp['tok_prompt'] / (resp['pp_ms'] / 1000.0) if resp['pp_ms'] > 0 else 0
        gen_s = resp['tok_gen'] / (resp['gen_ms'] / 1000.0) if resp['gen_ms'] > 0 else 0

        print(f"  [{i+1}] \"{resp['output']:15s}\" prompt={resp['tok_prompt']:3d}tok gen={resp['tok_gen']}tok "
              f"pp={resp['pp_ms']:5d}ms ({pp_s:5.1f}t/s) "
              f"gen={resp['gen_ms']:5d}ms ({gen_s:4.2f}t/s) wall={wall:.2f}s")
        results.append({"pp_s": pp_s, "gen_s": gen_s})

    steady = results[2:] if len(results) > 2 else results
    avg_pp = sum(r['pp_s'] for r in steady) / len(steady)
    avg_gen = sum(r['gen_s'] for r in steady) / len(steady)
    print(f"\n  Steady-state (skip first 2): pp={avg_pp:.1f} tok/s, gen={avg_gen:.2f} tok/s")


if __name__ == "__main__":
    print("Revan Benchmark")
    print(f"Pipe: {PIPE_NAME}")

    bench_yesno()
    bench_routing()
    bench_scoring()

    print(f"\n{'='*60}")
    print("EXPECTED BASELINES (llama-bench b8067, ngl=23, q4_0 KV, fa=1, t=6):")
    print("  pp32=28 t/s, pp64=53 t/s, pp128=88 t/s, pp512=184 t/s")
    print("  tg128=2.74 t/s (verified baseline)")
    print(f"{'='*60}")
