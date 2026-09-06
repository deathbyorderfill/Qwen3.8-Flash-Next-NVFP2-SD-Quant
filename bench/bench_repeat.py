#!/usr/bin/env python3
"""Repeat-measure prefill/decode so single-run noise is visible.

Both servers get identical treatment: fresh boot, 2 warmups, then 3 timed
repeats per config, reporting mean and min-max spread. The 2-stream configs
are the ones that disagreed across formats, so they carry the most repeats.
"""
import json, sys, time, threading, urllib.request, random, statistics as st

PORT = sys.argv[1] if len(sys.argv) > 1 else "8931"
TAG = sys.argv[2] if len(sys.argv) > 2 else "?"
BASE = f"http://127.0.0.1:{PORT}/v1/chat/completions"
random.seed(5)
W = "system design latency cache token stream buffer kernel memory tensor".split()
filler = lambda n: " ".join(random.choice(W) for _ in range(n))


def one(prompt, max_tok, out, idx):
    req = urllib.request.Request(
        BASE, data=json.dumps({"messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": max_tok, "stream": True,
        "stream_options": {"include_usage": True}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time(); tf = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data:"): continue
            p = line[5:].strip()
            if p == b"[DONE]": continue
            try: d = json.loads(p)
            except Exception: continue
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            if ch:
                de = ch[0].get("delta") or {}
                if de.get("content") or de.get("reasoning_content"):
                    if tf is None: tf = time.time()
    te = time.time(); u = usage or {}
    pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    out[idx] = {"prefill": pt/(tf-t0) if tf and pt else 0,
                "decode": (ct-1)/(te-tf) if tf and ct > 1 else 0}


def once(nstream, words, max_tok):
    prompt = filler(words) + "\n\nExplain how a write-ahead log guarantees durability."
    out = [None]*nstream
    ths = [threading.Thread(target=one, args=(prompt, max_tok, out, i)) for i in range(nstream)]
    for t in ths: t.start()
    for t in ths: t.join()
    ok = [o for o in out if o]
    return (sum(o["decode"] for o in ok)/len(ok), sum(o["decode"] for o in ok),
            sum(o["prefill"] for o in ok))


def run(nstream, words, label, reps=3):
    ds, ags, pfs = [], [], []
    for _ in range(reps):
        d, a, p = once(nstream, words, 300)
        ds.append(d); ags.append(a); pfs.append(p)
    print(f"{TAG:6} {label:20} decode/stream {st.mean(ds):6.2f} "
          f"[{min(ds):5.2f}-{max(ds):5.2f}]  agg {st.mean(ags):7.2f}  "
          f"prefill {st.mean(pfs):8.1f} [{min(pfs):7.1f}-{max(pfs):7.1f}]", flush=True)


for _ in range(2):
    once(1, 200, 100)
run(1, 4000,  "1 stream / 4k")
run(2, 200,   "2 streams / short")
run(2, 4000,  "2 streams / 4k")
print("REPEAT_DONE")
