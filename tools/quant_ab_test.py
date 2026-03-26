"""
quant_ab_test.py — A/B comparison of quantization variants for Revan decision engine.

Runs identical prompts against quants via Revan's Named Pipe,
records accuracy + speed, and prints a comparison table.

Usage:
  python quant_ab_test.py                # Run test suite, save results to JSON
  python quant_ab_test.py --tag q4km     # Run and save as q4km_results.json
  python quant_ab_test.py --tag other    # Run and save as other_results.json
  python quant_ab_test.py --compare      # Compare all result files
"""

import struct
import time
import sys
import os
import json
import glob
import ctypes
from ctypes import wintypes

# ==================== PIPE PROTOCOL ====================

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
        raise OSError(f"Cannot connect to Revan pipe (error {ctypes.get_last_error()})")
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


# ==================== TEST CASES ====================

YESNO_SYSTEM = "You are a yes/no classifier. Respond with only 'yes' or 'no'. Nothing else."
YESNO_CASES = [
    ("Is the sky blue?", "yes"),
    ("Is 2+2 equal to 5?", "no"),
    ("Does water boil at 100C?", "yes"),
    ("Is Python a compiled language?", "no"),
    ("Is the Earth flat?", "no"),
    ("Does Rust have a garbage collector?", "no"),
    ("Is 17 a prime number?", "yes"),
    ("Is JavaScript strongly typed?", "no"),
]

ROUTING_SYSTEM = (
    "You are a tool router. Given a user query, respond with ONLY the tool name.\n"
    "Available tools: web_search, file_read, calculator, calendar, email, none\n"
    "Respond with one tool name only."
)
ROUTING_CASES = [
    ("What is 42 * 17?", "calculator"),
    ("What's the weather in London?", "web_search"),
    ("Read the contents of config.toml", "file_read"),
    ("When is my next meeting?", "calendar"),
    ("Send a message to John", "email"),
    ("What is the square root of 144?", "calculator"),
    ("Search for Rust tutorials", "web_search"),
    ("Tell me a joke", "none"),
]

FILE_CLASS_SYSTEM = (
    "You are a file classifier. Given file content, respond with ONLY the category.\n"
    "Categories: config, code, data, docs, log, binary, unknown\n"
    "Respond with one category only."
)
FILE_CLASS_CASES = [
    ("revan.toml", "config"),
    ("src/revan.cpp", "code"),
    ("hub.toml", "config"),
    ("tools/bench_revan.py", "code"),
    ("README.md", "docs"),
    ("CMakeLists.txt", "config"),
    ("hub/revan-hub/Cargo.toml", "config"),
    ("hub/revan-hub/src/main.rs", "code"),
    ("hub/revan-hub/src/agents.rs", "code"),
    ("hub/revan-hub/src/brain.rs", "code"),
    ("hub/revan-hub/src/monitor.rs", "code"),
    ("hub/revan-core/src/agent.rs", "code"),
    ("hub/revan-core/src/client.rs", "code"),
    ("agents/python/file_scanner.py", "code"),
]


def read_file_head(path, max_bytes=500):
    """Read first 500 bytes of a file for classification."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_bytes)
    except Exception as e:
        return f"[error reading {path}: {e}]"


# ==================== TEST RUNNER ====================

def run_suite():
    """Run all test categories and return structured results."""
    results = {"yesno": [], "routing": [], "file_class": [], "meta": {}}

    # -- Yes/No --
    print(f"\n{'='*60}")
    print("Category 1: Yes/No Classification (8 prompts)")
    print(f"{'='*60}")
    for i, (question, expected) in enumerate(YESNO_CASES):
        req = build_request(3, 0, YESNO_SYSTEM, question)
        start = time.time()
        resp = send_request(req)
        wall = time.time() - start
        answer = resp["output"].strip().lower().rstrip(".")
        correct = answer == expected
        pp_s = resp['tok_prompt'] / (resp['pp_ms'] / 1000.0) if resp['pp_ms'] > 0 else 0
        gen_s = resp['tok_gen'] / (resp['gen_ms'] / 1000.0) if resp['gen_ms'] > 0 else 0
        mark = "OK" if correct else "MISS"
        print(f"  [{i+1}] {mark:4s} got={answer:5s} exp={expected:5s}  "
              f"pp={pp_s:5.1f}t/s gen={gen_s:4.2f}t/s wall={wall:.2f}s")
        results["yesno"].append({
            "question": question, "expected": expected, "got": answer,
            "correct": correct, "tok_prompt": resp["tok_prompt"],
            "tok_gen": resp["tok_gen"], "pp_ms": resp["pp_ms"],
            "gen_ms": resp["gen_ms"], "wall_ms": round(wall * 1000),
        })

    # -- Routing --
    print(f"\n{'='*60}")
    print("Category 2: Tool Routing (8 prompts)")
    print(f"{'='*60}")
    for i, (question, expected) in enumerate(ROUTING_CASES):
        req = build_request(5, 10, ROUTING_SYSTEM, question)
        start = time.time()
        resp = send_request(req)
        wall = time.time() - start
        answer = resp["output"].strip().lower().rstrip(".")
        correct = answer == expected
        pp_s = resp['tok_prompt'] / (resp['pp_ms'] / 1000.0) if resp['pp_ms'] > 0 else 0
        gen_s = resp['tok_gen'] / (resp['gen_ms'] / 1000.0) if resp['gen_ms'] > 0 else 0
        mark = "OK" if correct else "MISS"
        print(f"  [{i+1}] {mark:4s} got={answer:15s} exp={expected:15s}  "
              f"pp={pp_s:5.1f}t/s gen={gen_s:4.2f}t/s wall={wall:.2f}s")
        results["routing"].append({
            "question": question, "expected": expected, "got": answer,
            "correct": correct, "tok_prompt": resp["tok_prompt"],
            "tok_gen": resp["tok_gen"], "pp_ms": resp["pp_ms"],
            "gen_ms": resp["gen_ms"], "wall_ms": round(wall * 1000),
        })

    # -- File Classification --
    print(f"\n{'='*60}")
    print("Category 3: File Classification (14 prompts)")
    print(f"{'='*60}")
    for i, (filepath, expected) in enumerate(FILE_CLASS_CASES):
        content = read_file_head(filepath)
        user_prompt = f"File: {os.path.basename(filepath)}\nContent:\n{content}"
        req = build_request(5, 10, FILE_CLASS_SYSTEM, user_prompt)
        start = time.time()
        resp = send_request(req)
        wall = time.time() - start
        answer = resp["output"].strip().lower().rstrip(".")
        correct = answer == expected
        pp_s = resp['tok_prompt'] / (resp['pp_ms'] / 1000.0) if resp['pp_ms'] > 0 else 0
        gen_s = resp['tok_gen'] / (resp['gen_ms'] / 1000.0) if resp['gen_ms'] > 0 else 0
        mark = "OK" if correct else "MISS"
        print(f"  [{i+1}] {mark:4s} got={answer:10s} exp={expected:10s}  "
              f"{os.path.basename(filepath):20s}  "
              f"pp={pp_s:5.1f}t/s gen={gen_s:4.2f}t/s wall={wall:.2f}s")
        results["file_class"].append({
            "file": filepath, "expected": expected, "got": answer,
            "correct": correct, "tok_prompt": resp["tok_prompt"],
            "tok_gen": resp["tok_gen"], "pp_ms": resp["pp_ms"],
            "gen_ms": resp["gen_ms"], "wall_ms": round(wall * 1000),
        })

    # -- Summary --
    for cat in ["yesno", "routing", "file_class"]:
        total = len(results[cat])
        correct = sum(1 for r in results[cat] if r["correct"])
        print(f"\n  {cat}: {correct}/{total}")

    return results


def compute_stats(results):
    """Compute accuracy and speed stats from results."""
    stats = {}
    for cat in ["yesno", "routing", "file_class"]:
        items = results[cat]
        total = len(items)
        correct = sum(1 for r in items if r["correct"])
        # Speed: skip first 2 as warmup
        steady = items[2:] if len(items) > 2 else items
        avg_pp_s = 0
        avg_gen_s = 0
        avg_wall = 0
        if steady:
            pp_speeds = []
            gen_speeds = []
            for r in steady:
                if r["pp_ms"] > 0:
                    pp_speeds.append(r["tok_prompt"] / (r["pp_ms"] / 1000.0))
                if r["gen_ms"] > 0:
                    gen_speeds.append(r["tok_gen"] / (r["gen_ms"] / 1000.0))
            avg_pp_s = sum(pp_speeds) / len(pp_speeds) if pp_speeds else 0
            avg_gen_s = sum(gen_speeds) / len(gen_speeds) if gen_speeds else 0
            avg_wall = sum(r["wall_ms"] for r in steady) / len(steady)
        stats[cat] = {
            "correct": correct, "total": total,
            "avg_pp_s": round(avg_pp_s, 1),
            "avg_gen_s": round(avg_gen_s, 2),
            "avg_wall_ms": round(avg_wall),
        }

    # Totals
    total_correct = sum(stats[c]["correct"] for c in stats)
    total_all = sum(stats[c]["total"] for c in stats)
    all_items = results["yesno"] + results["routing"] + results["file_class"]
    steady_all = all_items[2:] if len(all_items) > 2 else all_items
    pp_speeds = [r["tok_prompt"] / (r["pp_ms"] / 1000.0) for r in steady_all if r["pp_ms"] > 0]
    gen_speeds = [r["tok_gen"] / (r["gen_ms"] / 1000.0) for r in steady_all if r["gen_ms"] > 0]
    stats["total"] = {
        "correct": total_correct, "total": total_all,
        "avg_pp_s": round(sum(pp_speeds) / len(pp_speeds), 1) if pp_speeds else 0,
        "avg_gen_s": round(sum(gen_speeds) / len(gen_speeds), 2) if gen_speeds else 0,
        "avg_wall_ms": round(sum(r["wall_ms"] for r in steady_all) / len(steady_all)) if steady_all else 0,
    }
    return stats


def compare_results():
    """Load all result JSON files and print comparison table."""
    result_files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*_results.json")))
    if not result_files:
        print("No result files found. Run tests first with --tag <name>.")
        return

    datasets = {}
    for f in result_files:
        tag = os.path.basename(f).replace("_results.json", "")
        with open(f, "r") as fh:
            data = json.load(fh)
        datasets[tag] = compute_stats(data)

    tags = list(datasets.keys())

    # === ACCURACY TABLE ===
    print(f"\n{'='*70}")
    print("=== Quantization A/B Comparison ===")
    print(f"{'='*70}")

    print("\nACCURACY:")
    header = f"  {'Category':<15s}"
    for t in tags:
        header += f"  {t:<14s}"
    if len(tags) >= 2:
        header += "  Match"
    print(header)
    print(f"  {'-'*15}" + f"  {'-'*14}" * len(tags) + ("  -----" if len(tags) >= 2 else ""))

    categories = [("yesno", "Yes/No"), ("routing", "Routing"), ("file_class", "File Class"), ("total", "TOTAL")]
    for cat_key, cat_label in categories:
        row = f"  {cat_label:<15s}"
        vals = []
        for t in tags:
            s = datasets[t][cat_key]
            cell = f"{s['correct']}/{s['total']}"
            row += f"  {cell:<14s}"
            vals.append((s['correct'], s['total']))
        if len(vals) >= 2:
            # Match = % of prompts where first two agree
            row += f"  --"  # We'll compute agreement below
        print(row)

    # === AGREEMENT (prompt-by-prompt) ===
    if len(tags) >= 2:
        # Load raw data for first two tags
        tag_a, tag_b = tags[0], tags[1]
        with open(result_files[0]) as f:
            data_a = json.load(f)
        with open(result_files[1]) as f:
            data_b = json.load(f)
        print(f"\nAGREEMENT ({tag_a} vs {tag_b}):")
        for cat in ["yesno", "routing", "file_class"]:
            items_a = data_a[cat]
            items_b = data_b[cat]
            agree = sum(1 for a, b in zip(items_a, items_b) if a["got"] == b["got"])
            total = min(len(items_a), len(items_b))
            pct = (agree / total * 100) if total > 0 else 0
            cat_label = {"yesno": "Yes/No", "routing": "Routing", "file_class": "File Class"}[cat]
            print(f"  {cat_label:<15s} {agree}/{total} ({pct:.0f}%)")
            if agree < total:
                for i, (a, b) in enumerate(zip(items_a, items_b)):
                    if a["got"] != b["got"]:
                        q = a.get("question", a.get("file", "?"))
                        print(f"    ^ [{i+1}] {tag_a}={a['got']!r} vs {tag_b}={b['got']!r}  ({q})")
        # Total agreement
        all_a = data_a["yesno"] + data_a["routing"] + data_a["file_class"]
        all_b = data_b["yesno"] + data_b["routing"] + data_b["file_class"]
        agree = sum(1 for a, b in zip(all_a, all_b) if a["got"] == b["got"])
        total = min(len(all_a), len(all_b))
        pct = (agree / total * 100) if total > 0 else 0
        print(f"  {'TOTAL':<15s} {agree}/{total} ({pct:.0f}%)")

    # === SPEED TABLE ===
    print(f"\nSPEED (steady-state, skip first 2 per category):")
    header = f"  {'Metric':<15s}"
    for t in tags:
        header += f"  {t:<14s}"
    print(header)
    print(f"  {'-'*15}" + f"  {'-'*14}" * len(tags))

    for metric, key, unit in [
        ("Prompt eval", "avg_pp_s", "t/s"),
        ("Generation", "avg_gen_s", "t/s"),
        ("Avg wall time", "avg_wall_ms", "ms"),
    ]:
        row = f"  {metric:<15s}"
        for t in tags:
            s = datasets[t]["total"]
            val = s[key]
            cell = f"{val} {unit}"
            row += f"  {cell:<14s}"
        print(row)

    # === VERDICT ===
    if len(tags) >= 2:
        s_a = datasets[tags[0]]["total"]
        s_b = datasets[tags[1]]["total"]
        acc_diff = abs(s_a["correct"] - s_b["correct"])
        gen_diff = s_b["avg_gen_s"] - s_a["avg_gen_s"]
        print(f"\nVERDICT:")
        if acc_diff == 0:
            print(f"  Accuracy: IDENTICAL ({s_a['correct']}/{s_a['total']})")
        else:
            better = tags[0] if s_a["correct"] > s_b["correct"] else tags[1]
            print(f"  Accuracy: {better} wins by {acc_diff} prompt(s)")
        if abs(gen_diff) < 0.1:
            print(f"  Speed: NEGLIGIBLE difference ({s_a['avg_gen_s']} vs {s_b['avg_gen_s']} t/s)")
        else:
            faster = tags[1] if gen_diff > 0 else tags[0]
            print(f"  Speed: {faster} is faster ({s_a['avg_gen_s']} vs {s_b['avg_gen_s']} t/s)")
        # Final recommendation
        if acc_diff <= 1 and gen_diff >= -0.2:
            print(f"  >> RECOMMENDATION: Use smaller quant (saves VRAM, negligible quality loss)")
        elif acc_diff >= 3:
            print(f"  >> RECOMMENDATION: Stay with Q4_K_M (accuracy gap too large)")
        else:
            print(f"  >> RECOMMENDATION: Review results manually — tradeoff is close")

    print(f"\n{'='*70}")


# ==================== MAIN ====================

if __name__ == "__main__":
    if "--compare" in sys.argv:
        compare_results()
        sys.exit(0)

    # Determine output tag
    tag = "results"
    if "--tag" in sys.argv:
        idx = sys.argv.index("--tag")
        if idx + 1 < len(sys.argv):
            tag = sys.argv[idx + 1]

    print(f"Revan A/B Test — tag: {tag}")
    print(f"Pipe: {PIPE_NAME}")

    results = run_suite()

    # Save to JSON
    out_file = os.path.join(os.path.dirname(__file__), f"{tag}_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Print quick stats
    stats = compute_stats(results)
    s = stats["total"]
    print(f"\nTOTAL: {s['correct']}/{s['total']} correct, "
          f"pp={s['avg_pp_s']} t/s, gen={s['avg_gen_s']} t/s, "
          f"avg wall={s['avg_wall_ms']}ms")
