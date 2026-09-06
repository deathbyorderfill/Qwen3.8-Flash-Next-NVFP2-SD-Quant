#!/usr/bin/env python3
"""N concurrent streams decode probe (for the MiaAI comparison): short prose prompt and ~4k
context, 256 new tokens each, 3 reps; reports per-stream and aggregate tok/s and TTFT.
Usage: probe_streams.py <port> <tag> [streams=1,2,4]"""
import json, sys, time, threading, random, statistics, urllib.request
PORT, TAG = sys.argv[1], sys.argv[2]
NS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2,4").split(",")]
BASE = f"http://127.0.0.1:{PORT}/v1/chat/completions"
random.seed(11); words = open("/usr/share/dict/words").read().split()
PROSE = "Write a detailed technical explanation of how a write-ahead log guarantees durability, including fsync semantics, checkpointing and crash recovery. Be thorough."
CTX4K = " ".join(random.choice(words) for _ in range(2900)) + "\n\nContinue this text in the same style, at length."


def one(prompt, out, i):
    out[i] = (0, 0.0, 1e-6)          # default so a stalled stream is counted as failed, not a crash
    body = {"messages": [{"role": "user", "content": prompt + f" (stream {i})"}], "max_tokens": 256, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); tf = None; n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            try:
                d = json.loads(line[5:])
            except Exception:
                continue
            if d.get("usage"):
                n = d["usage"].get("completion_tokens", 0)
            ch = d.get("choices") or []
            if ch and tf is None and ((ch[0].get("delta") or {}).get("content") or (ch[0].get("delta") or {}).get("reasoning_content")):
                tf = time.time()
    te = time.time()
    out[i] = (n, (tf or t0) - t0, te - (tf or t0))


for label, prompt in (("short prose", PROSE), ("~4k context", CTX4K)):
    for ns in NS:
        per, agg, ttft = [], [], []
        for rep in range(3):  # NOTE: on a stall, capture docker logs before the server is recycled
            out = [None] * ns
            ts = [threading.Thread(target=one, args=(prompt, out, i)) for i in range(ns)]
            t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]; wall = time.time() - t0
            if any(o[0] == 0 for o in out):
                import subprocess
                print(f"  STALL/FAIL in {label} x{ns} rep {rep}: {[o[0] for o in out]} tokens; server log tail:", flush=True)
                print(subprocess.run(["docker", "logs", "--tail", "12", "nvfp2k_serve"], capture_output=True, text=True).stderr[-1500:], flush=True)
                continue
            toks = sum(o[0] for o in out)
            per.append(statistics.mean((o[0] - 1) / max(o[2], 1e-6) for o in out)); agg.append(toks / wall); ttft.append(statistics.mean(o[1] for o in out))
        if per:
            print(f"[{TAG}] {label:12} {ns} stream(s): per-stream {statistics.mean(per):6.1f} tok/s [{min(per):.1f}-{max(per):.1f}]  aggregate {statistics.mean(agg):6.1f}  TTFT {statistics.mean(ttft):.2f}s  ({len(per)}/3 reps ok)", flush=True)
        else:
            print(f"[{TAG}] {label:12} {ns} stream(s): ALL REPS FAILED", flush=True)
print("STREAMS_DONE", flush=True)
