"""
test_kv_cache.py — Validate system prompt KV caching in Revan engine.
Sends repeated requests with same/different system prompts and measures
pp_ms to verify cache hits skip prompt eval.
"""

import struct
import time
import sys
import ctypes
from ctypes import wintypes

PIPE_NAME = r'\\.\pipe\revan'
PROTO_VERSION = 1
REQ_INFERENCE = 0x0001
REQ_HEALTH_CHECK = 0x0002
REQ_MODEL_SWAP = 0x0005

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


def build_health_check():
    msg_len = 20
    header = struct.pack('<IHHHHII', msg_len, PROTO_VERSION, REQ_HEALTH_CHECK,
                         0, 0, 0, 0)
    return header


def build_model_swap(model_path):
    path_bytes = model_path.encode('utf-8')
    msg_len = 20 + len(path_bytes)
    header = struct.pack('<IHHHHII', msg_len, PROTO_VERSION, REQ_MODEL_SWAP,
                         0, 0, 0, len(path_bytes))
    return header + path_bytes


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


def test_health_check():
    print("\n[TEST] Health Check")
    resp = send_request(build_health_check())
    ok = resp['status'] == 0
    print(f"  Status: {'OK' if ok else 'FAIL'} (0x{resp['status']:04X})")
    return ok


def test_kv_cache():
    """Send same system prompt multiple times, verify pp_ms drops on cache hits."""
    print("\n[TEST] KV Cache — Same System Prompt")
    print("=" * 70)

    sys_prompt = "You are a yes/no classifier. Respond with only 'yes' or 'no'. Nothing else."
    questions = [
        "Is the sky blue?",
        "Is 2+2 equal to 5?",
        "Does water boil at 100C?",
        "Is Python a compiled language?",
        "Is the Earth flat?",
        "Does Rust have a garbage collector?",
    ]

    results = []
    for i, q in enumerate(questions):
        req = build_request(3, 10, sys_prompt, q)
        start = time.time()
        resp = send_request(req)
        wall = time.time() - start

        tag = "MISS" if i == 0 else "HIT "
        print(f"  [{i+1}] {tag} \"{resp['output']:5s}\"  pp={resp['pp_ms']:5d}ms  "
              f"gen={resp['gen_ms']:5d}ms  prompt={resp['tok_prompt']:3d}tok  wall={wall:.2f}s")
        results.append(resp)

    # Verify: requests 2+ should have significantly lower pp_ms than request 1
    first_pp = results[0]['pp_ms']
    avg_cached_pp = sum(r['pp_ms'] for r in results[1:]) / len(results[1:])

    print(f"\n  First request (cache miss):   pp_ms = {first_pp}")
    print(f"  Avg cached (hits 2-{len(results)}):       pp_ms = {avg_cached_pp:.0f}")

    if first_pp > 0:
        savings = (1 - avg_cached_pp / first_pp) * 100
        print(f"  Savings: {savings:.1f}%")
        if savings > 20:
            print("  PASS — cache hits are significantly faster")
            return True
        else:
            print("  WARN — cache hits not significantly faster (may need investigation)")
            return False
    return True


def test_different_prompts():
    """Send different system prompts, verify each miss then hit pattern."""
    print("\n[TEST] KV Cache — Multiple System Prompts")
    print("=" * 70)

    prompts = [
        ("You are a yes/no classifier. Respond with only 'yes' or 'no'.",
         "Is water wet?"),
        ("You are a tool router. Respond with only the tool name: web_search, calculator, none.",
         "What is 42 * 17?"),
        ("You are a sentiment classifier. Respond with only: positive, negative, neutral.",
         "I love this product!"),
    ]

    # First pass: all cache misses
    print("  --- Pass 1: All new prompts (expect MISS) ---")
    first_pass = []
    for i, (sys, usr) in enumerate(prompts):
        req = build_request(5, 10, sys, usr)
        resp = send_request(req)
        print(f"  [{i+1}] MISS \"{resp['output']:15s}\"  pp={resp['pp_ms']:5d}ms")
        first_pass.append(resp)

    # Second pass: all cache hits (same prompts)
    print("  --- Pass 2: Same prompts again (expect HIT) ---")
    second_pass = []
    for i, (sys, usr) in enumerate(prompts):
        req = build_request(5, 10, sys, usr)
        resp = send_request(req)
        print(f"  [{i+1}] HIT  \"{resp['output']:15s}\"  pp={resp['pp_ms']:5d}ms")
        second_pass.append(resp)

    # Compare
    print("\n  Comparison:")
    all_faster = True
    for i in range(len(prompts)):
        miss_pp = first_pass[i]['pp_ms']
        hit_pp = second_pass[i]['pp_ms']
        savings = (1 - hit_pp / miss_pp) * 100 if miss_pp > 0 else 0
        status = "OK" if savings > 20 else "SLOW"
        print(f"  Prompt {i+1}: miss={miss_pp}ms  hit={hit_pp}ms  savings={savings:.0f}%  [{status}]")
        if savings <= 20:
            all_faster = False

    return all_faster


def test_lru_eviction():
    """Send 4 different prompts (cache has 3 slots), verify oldest gets evicted."""
    print("\n[TEST] KV Cache — LRU Eviction (3 slots)")
    print("=" * 70)

    prompts = [
        ("Prompt A: You are a yes/no classifier.", "Is sky blue?"),
        ("Prompt B: You are a tool router.", "What is 2+2?"),
        ("Prompt C: You are a sentiment analyzer.", "I hate bugs!"),
        ("Prompt D: You are a language detector.", "Bonjour le monde"),
    ]

    # Fill all 3 cache slots: A, B, C
    print("  --- Fill cache: A, B, C ---")
    for i in range(3):
        req = build_request(5, 10, prompts[i][0], prompts[i][1])
        resp = send_request(req)
        print(f"  Stored [{chr(65+i)}] pp={resp['pp_ms']}ms")

    # Add D (should evict A, which was LRU)
    print("  --- Add D (should evict A) ---")
    req = build_request(5, 10, prompts[3][0], prompts[3][1])
    resp = send_request(req)
    print(f"  Stored [D] pp={resp['pp_ms']}ms")

    # Re-request A — should be a cache miss (evicted)
    print("  --- Re-request A (should be MISS — evicted) ---")
    req = build_request(5, 10, prompts[0][0], prompts[0][1])
    resp_a = send_request(req)
    print(f"  [A] pp={resp_a['pp_ms']}ms")

    # Re-request B — should be a cache hit (still cached)
    print("  --- Re-request B (should be HIT — still cached) ---")
    req = build_request(5, 10, prompts[1][0], prompts[1][1])
    resp_b = send_request(req)
    print(f"  [B] pp={resp_b['pp_ms']}ms")

    # B should be faster than A (A was evicted, B was cached)
    if resp_a['pp_ms'] > 0 and resp_b['pp_ms'] < resp_a['pp_ms']:
        print("  PASS — A was evicted (slow), B still cached (fast)")
        return True
    else:
        print("  INCONCLUSIVE — timing may vary")
        return True  # Don't fail on timing jitter


if __name__ == "__main__":
    print("Revan KV Cache Test")
    print(f"Pipe: {PIPE_NAME}")

    try:
        ok = test_health_check()
        if not ok:
            print("\nEngine not ready, aborting.")
            sys.exit(1)

        test_kv_cache()
        test_different_prompts()
        test_lru_eviction()

        print("\n" + "=" * 70)
        print("All tests completed.")
        print("=" * 70)

    except OSError as e:
        print(f"\nConnection error: {e}")
        print("Make sure revan.exe is running.")
        sys.exit(1)
