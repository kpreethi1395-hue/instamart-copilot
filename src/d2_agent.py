import json, os, sys, time, uuid
from datetime import datetime, timezone
import sqlite3
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))      # so `from d1_rag import retrieve` works
import mock_llm
from d1_rag import retrieve

DB = os.path.join(ROOT, "data", "instamart.db")
OUT = os.path.join(ROOT, "outputs")
STATE = os.path.join(ROOT, "state")
LEDGER = os.path.join(OUT, "refund_ledger.jsonl")
RUNS = os.path.join(OUT, "agent_runs.jsonl")
os.makedirs(OUT, exist_ok=True); os.makedirs(STATE, exist_ok=True)

# NOTE: the word AGENT must appear; CLASSIFY and POLICY ASSISTANT must NOT (the mock routes on them)
SYSTEM = ("You are a refund AGENT for Instamart support. Use the tools to look up the order, check the "
          "refund rules, calculate the refund, then issue it or escalate it. Finish with one line "
          "starting REFUND, ESCALATE, DENIED or NOT_FOUND.")

class SimulatedCrash(Exception):
    pass

# ---------- tools ----------
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        r = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders "
                        "WHERE order_id = ?", (order_id,)).fetchone()
    if r is None:
        return {"error": "order not found"}
    return {"order_id": r[0], "status": r[1], "order_amount": r[2], "delivery_minutes": r[3]}

def search_policy(query):
    return {"results": retrieve(query)}

def calculate_refund(order_amount, delivery_minutes):
    late = delivery_minutes is not None and delivery_minutes > 45
    return {"refund_inr": round(float(order_amount), 2) if late else 0}

def _ledger(tool, order_id, refund_inr):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)

def _spec(name, desc, props, req):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}

TOOLS_SPEC = [
    _spec("get_order", "Look up an order's status, amount and delivery time in the database.",
          {"order_id": {"type": "string"}}, ["order_id"]),
    _spec("search_policy", "Search the policy documents and return the most relevant passages.",
          {"query": {"type": "string"}}, ["query"]),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}},
          ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

# ---------- the loop, with stop limits and checkpoints ----------
def _checkpoint(s):
    keep = ("run_id", "scenario_id", "messages", "tokens_used", "llm_calls", "steps", "tools_executed",
            "max_iterations", "token_limit", "time_limit_seconds", "log")
    data = {k: s[k] for k in keep}
    data["step"] = len(s["steps"])
    with open(os.path.join(STATE, f"{s['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, default=str)

def _finish(s, status, answer):
    refund = 0.0
    for st in s["steps"]:
        if st["tool"] in ("issue_refund", "escalate_to_human"):
            refund = st["observation"].get("refund_inr", 0.0)
    res = {"run_id": s["run_id"], "scenario_id": s["scenario_id"], "status": status, "final_answer": answer,
           "steps": s["steps"], "llm_calls": s["llm_calls"], "tokens_used": s["tokens_used"], "refund_inr": refund}
    if s["log"]:
        with open(RUNS, "a", encoding="utf-8") as f:
            f.write(json.dumps({k: res[k] for k in ("scenario_id", "status", "final_answer", "llm_calls",
                                                    "tokens_used", "steps")}, default=str) + "\n")
    return res

def _handoff(s, reason):
    done = [st["tool"] for st in s["steps"]]
    return (f"Stopped automatically ({reason}). Handing over to a human agent. "
            f"Request: {s['messages'][1]['content']} Steps completed: {done or 'none'}.")

def _loop(s, crash_after_tools=None):
    start = time.monotonic()
    while s["llm_calls"] < s["max_iterations"]:
        if time.monotonic() - start >= s["time_limit_seconds"]:                    # timeout
            return _finish(s, "FALLBACK_TIMEOUT", _handoff(s, "time limit"))
        resp = mock_llm.chat(s["messages"], TOOLS_SPEC)
        s["llm_calls"] += 1
        s["tokens_used"] += resp.usage.total_tokens
        if s["tokens_used"] > s["token_limit"]:                                    # before executing the tool
            return _finish(s, "FALLBACK_TOKEN_LIMIT", _handoff(s, "token limit"))
        msg = resp.choices[0].message
        if not msg.tool_calls:                                                     # final answer
            text = msg.content or ""
            return _finish(s, text.split(":")[0].strip(), text)
        tc = msg.tool_calls[0]
        s["messages"].append({"role": "assistant", "content": None,
                              "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        fn = TOOL_MAP.get(tc.name)
        obs = fn(**tc.arguments) if fn else {"error": f"unknown tool {tc.name}"}
        s["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                              "content": json.dumps(obs, default=str)})
        s["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        s["tools_executed"].append(tc.name)
        _checkpoint(s)                                                             # after every observation
        if crash_after_tools and len(s["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools}")
    return _finish(s, "FALLBACK_MAX_ITERATIONS", _handoff(s, "max iterations"))

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              crash_after_tools=None, log=False, run_id=None):
    s = {"run_id": run_id or f"{scenario_id}_{uuid.uuid4().hex[:8]}", "scenario_id": scenario_id,
         "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message}],
         "tokens_used": 0, "llm_calls": 0, "steps": [], "tools_executed": [],
         "max_iterations": max_iterations, "token_limit": token_limit,
         "time_limit_seconds": time_limit_seconds, "log": log}
    return _loop(s, crash_after_tools)

def resume_agent(run_id):
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        s = json.load(f)
    return _loop(s)

# ================= main =================
if __name__ == "__main__":
    for p in (LEDGER, RUNS):
        if os.path.exists(p):
            os.remove(p)
    for f in os.listdir(STATE):
        if f.endswith(".json"):
            os.remove(os.path.join(STATE, f))

    sc = pd.read_csv(os.path.join(ROOT, "data", "raw", "agent_scenarios.csv"), keep_default_na=False)
    results = {}
    print("scenario | status | refund_inr | llm_calls | tokens_used")
    for r in sc.itertuples():
        res = run_agent(r.message, r.scenario_id, log=True)
        results[r.scenario_id] = res
        print(f"{r.scenario_id} | {res['status']} | {res['refund_inr']:.2f} | {res['llm_calls']} | {res['tokens_used']}")

    s1 = sc[sc.scenario_id == "S1"].iloc[0].message
    tok = run_agent(s1, "S1-token-demo", token_limit=200)
    tmo = run_agent(s1, "S1-timeout-demo", time_limit_seconds=0)
    print(f"\nS1 with token_limit=200      -> {tok['status']} (tokens {tok['tokens_used']}, calls {tok['llm_calls']})")
    print(f"S1 with time_limit_seconds=0 -> {tmo['status']} (calls {tmo['llm_calls']})")

    try:
        run_agent(s1, "S1-crash-demo", crash_after_tools=2, run_id="S1-crash-demo")
    except SimulatedCrash as e:
        print(f"\ncrashed: {e}")
    resumed = resume_agent("S1-crash-demo")
    with open(os.path.join(STATE, "S1-crash-demo.json"), encoding="utf-8") as f:
        executed = json.load(f)["tools_executed"]
    print(f"resumed -> {resumed['status']} {resumed['refund_inr']:.2f}; tools executed: {executed}")

    checks = {}
    for r in sc.itertuples():
        res = results[r.scenario_id]
        ok_status = res["status"] == r.expected_outcome or (r.expected_outcome == "FALLBACK" and res["status"].startswith("FALLBACK"))
        checks[f"{r.scenario_id} {r.expected_outcome}"] = ok_status and abs(res["refund_inr"] - float(r.expected_refund_inr)) <= 0.01
    checks.update({
        "S1 takes 5 model calls": results["S1"]["llm_calls"] == 5,
        "S7 FALLBACK_MAX_ITERATIONS (6 calls, default limit)": results["S7"]["status"] == "FALLBACK_MAX_ITERATIONS" and results["S7"]["llm_calls"] == 6,
        "token-limit demo": tok["status"] == "FALLBACK_TOKEN_LIMIT",
        "timeout demo": tmo["status"] == "FALLBACK_TIMEOUT" and tmo["llm_calls"] == 0,
        "crash+resume: 4 tools, none repeated": executed == ["get_order", "search_policy", "calculate_refund", "issue_refund"],
        "crash+resume: REFUND 128.25": resumed["status"] == "REFUND" and abs(resumed["refund_inr"] - 128.25) <= 0.01,
        "agent_runs.jsonl has 7 lines": sum(1 for _ in open(RUNS, encoding="utf-8")) == 7})
    print()
    for n, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + n)