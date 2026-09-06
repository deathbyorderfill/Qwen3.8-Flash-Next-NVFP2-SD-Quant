#!/usr/bin/env python3
"""Tool-call eval for a served build, through /v1/chat/completions with `tools` (the path an agent
framework uses). 20 tasks: single calls, parallel calls, typed / enum / integer / nested-list
arguments, tool-result follow-up turns, and no-tool controls. A task passes only if every check
passes: call-vs-no-call decision, function name, arguments parse as JSON with the expected keys,
values and JSON types, no parser leftovers (`<tool_call>` text in content), and follow-up answers
quote the tool result. Greedy once, sampled (temp 0.6 / top-p 0.95) with several seeds, because
--speculative-accept-threshold-acc only changes sampled traffic.
Usage: tool_call_eval.py <port> <tag> [seeds=3]   -> results/toolcall_<tag>.{log,json}"""
import json, os, re, sys, time, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8934"
TAG = sys.argv[2] if len(sys.argv) > 2 else "run"
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=10).read())["data"][0]["id"]
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LOG = open(os.path.join(R, f"toolcall_{TAG}.log"), "a")


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}


T = {
    "get_weather": fn("get_weather", "Get the current weather for a city.",
                      {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    "convert_currency": fn("convert_currency", "Convert an amount between currencies.",
                           {"amount": {"type": "number"}, "from_currency": {"type": "string", "description": "ISO 4217 code"},
                            "to_currency": {"type": "string", "description": "ISO 4217 code"}}, ["amount", "from_currency", "to_currency"]),
    "send_email": fn("send_email", "Send an email.", {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, ["to", "subject", "body"]),
    "search_flights": fn("search_flights", "Search flights.", {"origin": {"type": "string", "description": "IATA airport code"},
                         "destination": {"type": "string", "description": "IATA airport code"}, "date": {"type": "string", "description": "YYYY-MM-DD"}}, ["origin", "destination", "date"]),
    "calculator": fn("calculator", "Evaluate an arithmetic expression exactly.", {"expression": {"type": "string"}}, ["expression"]),
    "create_calendar_event": fn("create_calendar_event", "Create a calendar event.",
                                {"title": {"type": "string"}, "start": {"type": "string", "description": "ISO 8601"}, "end": {"type": "string", "description": "ISO 8601"},
                                 "attendees": {"type": "array", "items": {"type": "string"}, "description": "email addresses"}}, ["title", "start", "end"]),
    "set_thermostat": fn("set_thermostat", "Set the thermostat.", {"temperature": {"type": "number"}, "mode": {"type": "string", "enum": ["heat", "cool", "auto"]}}, ["temperature", "mode"]),
    "get_stock_price": fn("get_stock_price", "Get a stock quote.", {"ticker": {"type": "string", "description": "ticker symbol"}, "exchange": {"type": "string"}}, ["ticker"]),
    "schedule_reminder": fn("schedule_reminder", "Schedule a reminder.", {"minutes_from_now": {"type": "integer"}, "message": {"type": "string"}}, ["minutes_from_now", "message"]),
    "run_sql": fn("run_sql", "Run a read-only SQL query against the analytics database.", {"query": {"type": "string"}}, ["query"]),
}
ALL = list(T.values())


def U(s):
    return [{"role": "user", "content": s}]


def tc(name, args, cid="call_1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


# expect: list of (name, {arg: matcher}) for calls; None for "no call expected"; "clarify" = no call, must ask.
# matcher: value | callable(v)->bool. "answer": regex the final text must match (follow-up tasks).
lower_eq = lambda x: (lambda v: isinstance(v, str) and v.strip().lower() == x)
contains = lambda *xs: (lambda v: isinstance(v, str) and all(x in v.lower() for x in xs))
num_eq = lambda x: (lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and abs(float(v) - x) < 1e-6)
int_eq = lambda x: (lambda v: isinstance(v, int) and not isinstance(v, bool) and v == x)
TASKS = [
    ("single_basic", U("What's the weather in Paris right now, in celsius?"), [T["get_weather"]], [("get_weather", {"city": contains("paris"), "unit": lower_eq("celsius")})]),
    ("single_optional", U("How is the weather in Berlin today?"), [T["get_weather"]], [("get_weather", {"city": contains("berlin")})]),
    ("single_number", U("Convert 250 US dollars to Japanese yen."), ALL, [("convert_currency", {"amount": num_eq(250), "from_currency": lower_eq("usd"), "to_currency": lower_eq("jpy")})]),
    ("single_strings", U("Email bob@example.com with the subject 'Lunch' telling him we will meet at noon on Friday."), ALL,
     [("send_email", {"to": lower_eq("bob@example.com"), "subject": contains("lunch"), "body": contains("noon")})]),
    ("single_codes", U("Find me flights from San Francisco to New York JFK on 12 October 2026."), ALL,
     [("search_flights", {"origin": lower_eq("sfo"), "destination": lower_eq("jfk"), "date": lower_eq("2026-10-12")})]),
    ("single_tooluse", U("Use the calculator to compute 1234 * 5678."), ALL, [("calculator", {"expression": contains("1234", "5678")})]),
    ("single_enum", U("Set the thermostat to 68 degrees in heat mode."), ALL, [("set_thermostat", {"temperature": num_eq(68), "mode": lower_eq("heat")})]),
    ("single_infer", U("What is Apple's stock trading at on NASDAQ?"), ALL, [("get_stock_price", {"ticker": lower_eq("aapl")})]),
    ("single_integer", U("Remind me in 45 minutes to call mom."), ALL, [("schedule_reminder", {"minutes_from_now": int_eq(45), "message": contains("mom")})]),
    ("single_sql", U("Using the analytics database, how many rows in the `users` table have a signup_date in 2025? Columns: id, signup_date."), ALL,
     [("run_sql", {"query": contains("count", "users", "2025")})]),
    ("nested_list", U("Create a calendar event 'Design review' on 2026-09-10 from 14:00 to 15:00 with ana@example.com and raj@example.com."), ALL,
     [("create_calendar_event", {"title": contains("design review"), "start": contains("2026-09-10", "14:00"), "end": contains("15:00"),
                                 "attendees": lambda v: isinstance(v, list) and sorted(x.lower() for x in v) == ["ana@example.com", "raj@example.com"]})]),
    ("parallel_same", U("What's the weather in Tokyo and in London? Use celsius."), ALL,
     [("get_weather", {"city": contains("tokyo")}), ("get_weather", {"city": contains("london")})]),
    ("parallel_currency", U("Convert 100 EUR to USD and also 100 GBP to USD."), ALL,
     [("convert_currency", {"amount": num_eq(100), "from_currency": lower_eq("eur"), "to_currency": lower_eq("usd")}),
      ("convert_currency", {"amount": num_eq(100), "from_currency": lower_eq("gbp"), "to_currency": lower_eq("usd")})]),
    ("chain_first_step", U("Look up the weather in Rome and then email a one-line summary of it to anna@example.com."), ALL,
     [("get_weather", {"city": contains("rome")})], {"forbid": ["send_email"]}),
    ("followup_weather", U("What's the weather in Oslo in celsius?") + [{"role": "assistant", "content": "", "tool_calls": [tc("get_weather", {"city": "Oslo", "unit": "celsius"})]},
     {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": json.dumps({"temperature": 17, "condition": "cloudy", "unit": "celsius"})}],
     ALL, None, {"answer": r"17\b.*cloud|cloud.*17\b"}),
    ("followup_currency", U("Convert 250 USD to JPY.") + [{"role": "assistant", "content": "", "tool_calls": [tc("convert_currency", {"amount": 250, "from_currency": "USD", "to_currency": "JPY"})]},
     {"role": "tool", "tool_call_id": "call_1", "name": "convert_currency", "content": json.dumps({"result": 37450.5, "rate": 149.802})}],
     ALL, None, {"answer": r"37,?450"}),
    ("no_tool_fact", U("What is the capital of France? Answer in one short sentence."), ALL, None, {"answer": r"paris"}),
    ("no_tool_creative", U("Write a haiku about rain."), ALL, None, {}),
    ("no_tool_explain", U("Explain what a token-bucket rate limiter is, in two sentences."), ALL, None, {}),
    ("clarify_missing", U("Book me a flight please."), ALL, "clarify", {}),
]


def call(messages, tools, sp):
    body = {"model": MODEL, "messages": messages, "tools": tools, "max_tokens": 3000, **sp}
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800).read())
    dt = time.time() - t0
    ch = r["choices"][0]; m = ch["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "", "tool_calls": m.get("tool_calls") or [],
            "finish": ch.get("finish_reason"), "ctok": r.get("usage", {}).get("completion_tokens", 0), "wall": dt}


def score(task, out):
    name, msgs, tools, expect = task[0], task[1], task[2], task[3]
    extra = task[4] if len(task) > 4 else {}
    fails = []
    calls = []
    for c in out["tool_calls"]:
        try:
            args = json.loads(c["function"]["arguments"]) if isinstance(c["function"]["arguments"], str) else c["function"]["arguments"]
        except Exception:
            fails.append(f"args not JSON: {c['function']['arguments'][:80]!r}"); args = None
        calls.append((c["function"]["name"], args))
    if re.search(r"<tool_call>|</tool_call>|<function=", out["content"]):
        fails.append("parser leftover in content")
    if out["finish"] == "length":
        fails.append("hit max_tokens")
    if expect is None or expect == "clarify":
        if calls:
            fails.append(f"unexpected call {[c[0] for c in calls]}")
        if not out["content"].strip():
            fails.append("empty answer")
        if expect == "clarify" and not re.search(r"\?|where|when|which|destination|date|from", out["content"], re.I):
            fails.append("did not ask for the missing details")
    else:
        for want_name, want_args in expect:
            hit = None
            for i, (n, a) in enumerate(calls):
                if n == want_name and a is not None and all(k in a and (m(a[k]) if callable(m) else a[k] == m) for k, m in want_args.items()):
                    hit = i; break
            if hit is None:
                fails.append(f"missing call {want_name}({list(want_args)}); got {[(n, a) for n, a in calls]}")
            else:
                calls.pop(hit)
        for forbid in extra.get("forbid", []):
            if any(n == forbid for n, _ in calls):
                fails.append(f"premature call {forbid}")
        # calls to unknown functions
        known = {t["function"]["name"] for t in tools}
        for n, _ in calls:
            if n not in known:
                fails.append(f"unknown function {n}")
    if "answer" in extra and not re.search(extra["answer"], out["content"], re.I | re.S):
        fails.append(f"answer mismatch: {out['content'][:100]!r}")
    return fails


configs = [("greedy", {"temperature": 0})] + [(f"temp0.6_seed{s}", {"temperature": 0.6, "top_p": 0.95, "seed": s}) for s in range(SEEDS)]
allres = {"model": MODEL, "tag": TAG, "configs": {}}
P(f"\n##### tool-call eval  tag={TAG}  model={MODEL}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
for cname, sp in configs:
    P(f"\n=== {cname} ===")
    rows = []; npass = 0; ctok = 0; wall = 0.0; rtok = 0
    for task in TASKS:
        try:
            out = call(task[1], task[2], sp)
        except Exception as e:
            out = {"content": "", "reasoning": "", "tool_calls": [], "finish": f"error {e}", "ctok": 0, "wall": 0}
        fails = score(task, out)
        ok = not fails; npass += ok; ctok += out["ctok"]; wall += out["wall"]
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in out["tool_calls"]]
        P(f"  {'PASS' if ok else 'FAIL'} {task[0]:18} ctok={out['ctok']:4d} {out['wall']:5.1f}s calls={calls if calls else '-'}" + ("" if ok else f"  <- {'; '.join(fails)}"))
        rows.append({"task": task[0], "ok": ok, "fails": fails, "calls": calls, "content": out["content"][:600], "reasoning_len": len(out["reasoning"]), "finish": out["finish"], "ctok": out["ctok"], "wall": out["wall"]})
    P(f"  --- {cname}: {npass}/{len(TASKS)} pass, {ctok} completion tokens in {wall:.0f} s ({ctok / max(wall, 1e-9):.1f} tok/s incl. TTFT)")
    allres["configs"][cname] = {"pass": npass, "n": len(TASKS), "rows": rows, "ctok": ctok, "wall": wall}
json.dump(allres, open(os.path.join(R, f"toolcall_{TAG}.json"), "w"), indent=1)
P("TOOLCALL_DONE")
