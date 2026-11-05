import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, random, re, time
import pandas as pd
import mock_llm
from mock_llm import est_tokens, TransientLLMError

random.seed(42)                      # makes the jitter reproducible
MAX_INPUT_TOKENS = 400
MAX_RETRIES, BASE_DELAY = 3, 0.05
CATS = ["billing", "delivery", "technical", "fraud"]

# ---------- Task 2: redaction (this exact order) ----------
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

def redact(text, counts=None):
    for name, pat, ph in PATTERNS:
        text, n = pat.subn(ph, text)
        if counts is not None:
            counts[name] += n
    return text

# ---------- Task 3: prompts ----------
ZERO = ('You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. '
        'Reply with JSON only, no other text, in the form {"category": "...", "reason": "..."}.')
FEW = ZERO + '''
EXAMPLES:
Ticket: I was charged twice for order [ORDER_ID]. Please refund the extra payment. -> {"category": "billing", "reason": "double charge"}
Ticket: Someone hacked my account and used my wallet balance. This is fraud. -> {"category": "fraud", "reason": "unauthorized account use"}
Ticket: The app crashes every time I open the cart. -> {"category": "technical", "reason": "app crash"}
Ticket: My order [ORDER_ID] has not arrived and the rider is not picking up calls. -> {"category": "delivery", "reason": "order not received"}'''
assert "CLASSIFY" in ZERO and "EXAMPLES:" not in ZERO
assert "CLASSIFY" in FEW and "EXAMPLES:" in FEW

# ---------- Task 4: retry with exponential backoff and jitter ----------
def call_with_retry(ticket_id, messages, stats):
    failed = False
    for n in range(MAX_RETRIES + 1):
        try:
            resp = mock_llm.chat(messages)
            if failed:
                stats["recovered"] += 1
            return resp
        except TransientLLMError:
            failed = True
            stats["transient"] += 1
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            print(f"  [retry] ticket={ticket_id} retry_number={n} wait={wait:.4f}s")
            time.sleep(wait)

# ---------- Task 5: validate every answer ----------
def validate(text):
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        found = set(re.findall(r"[a-z]+", (text or "").lower())) & set(CATS)
        return ("classified_keyword", found.pop()) if len(found) == 1 else ("flagged_invalid_output", "")
    if isinstance(obj, dict) and obj.get("category") in CATS:
        return "classified_json", obj["category"]
    return "flagged_invalid_output", ""              # valid JSON, bad category: never guess

# ---------- pipeline ----------
def run_pipeline(df, system_prompt, token_budget=float("inf"), label=""):
    print(f"\n--- run: {label} ---")
    mock_llm.reset()                                  # before every full run
    stats = {"transient": 0, "recovered": 0, "calls": 0}
    sent, rows, total = [], [], 0
    for t in df.itertuples():
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "true_category": t.true_category,
               "status": "", "category": "", "tokens_used": 0}
        if not t.text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(t.text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif total >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            msgs = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": redact(t.text)}]      # user content = redacted text only
            sent.append((t.ticket_id, msgs))
            try:
                resp = call_with_retry(t.ticket_id, msgs, stats)
            except TransientLLMError:
                row["status"] = "flagged_invalid_output"              # permanent failure -> human review
                rows.append(row); continue
            stats["calls"] += 1
            used = resp.usage.total_tokens
            total += used
            row["tokens_used"] = used
            row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)

    # guardrail assertions: no PII in any prompt, no empty/long ticket was sent
    skipped_ids = {r["ticket_id"] for r in rows if r["status"] in ("skipped_empty", "flagged_long")}
    for tid, msgs in sent:
        assert tid not in skipped_ids
        for m in msgs:
            for _, pat, _ in PATTERNS:
                assert not pat.search(m["content"]), f"PII leaked in prompt for {tid}"
    res = pd.DataFrame(rows)
    stats["total_tokens"] = total
    print(f"status counts: {res['status'].value_counts().to_dict()}")
    print(f"transient errors: {stats['transient']}, recovered: {stats['recovered']}")
    return res, stats

def summarise(res, stats):
    ok = res[res["status"].isin(["classified_json", "classified_keyword"])]
    return {"classified": len(ok),
            "accuracy_%": round((ok["category"] == ok["true_category"]).mean() * 100, 1),
            "total_tokens": stats["total_tokens"],
            "tokens_per_classified": round(stats["total_tokens"] / len(ok), 1)}

# ---------- Task 7 routing ----------
seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")

def to_output(res):
    out = res.copy()
    out["segment_name"] = out["customer_id"].map(seg["segment_name"]).fillna("")
    prem = out["customer_id"].map(seg["is_premium"]).fillna(False).astype(bool)
    human = out["status"].isin(["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"])
    fraud = ~human & (out["category"] == "fraud")
    premium = ~human & ~fraud & prem
    out["route"] = "support_queue"; out["priority"] = "NORMAL"
    out.loc[human, "route"] = "human_review"
    out.loc[fraud, ["route", "priority"]] = ["security_team", "HIGH"]
    out.loc[premium, ["route", "priority"]] = ["premium_desk", "HIGH"]
    return out[["ticket_id", "customer_id", "status", "category", "segment_name",
                "priority", "route", "tokens_used"]]

# ================= main =================
tick = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

# Task 1 and 2 audit figures
empty = (tick["text"].str.strip() == "").sum()
long_mask = tick["text"].map(est_tokens) > MAX_INPUT_TOKENS
long_ = long_mask.sum()
print(f"empty tickets: {empty} | long tickets: {long_} | long est tokens: "
      f"{sorted(set(tick.loc[long_mask, 'text'].map(est_tokens)))}")

counts = {"email": 0, "phone": 0, "order_id": 0}
redacted_all = [redact(t, counts) for t in tick["text"]]
print("redactions:", counts, "total", sum(counts.values()))
left = sum(len(p.findall(t)) for t in redacted_all for _, p, _ in PATTERNS)
print(f"PII found after re-applying the patterns: {left}")

# Task 5-6: zero-shot vs few-shot (no budget)
zero, zs = run_pipeline(tick, ZERO, label="zero-shot, no budget")
few, fs = run_pipeline(tick, FEW, label="few-shot, no budget")
cmp_ = pd.DataFrame({"zero-shot": summarise(zero, zs), "few-shot": summarise(few, fs)})
print("\n", cmp_.to_string())
zok = zero[zero["status"].isin(["classified_json", "classified_keyword"])]
print("\nZero-shot confusion (rows = true, cols = predicted):\n",
      pd.crosstab(zok["true_category"], zok["category"]).to_string())

# Task 7a: budget demo
BUDGET_DEMO = 6000
demo, ds = run_pipeline(tick, ZERO, token_budget=BUDGET_DEMO, label=f"zero-shot, budget {BUDGET_DEMO}")
n_ex = int((demo["status"] == "budget_exhausted").sum())
print(f"budget {BUDGET_DEMO}: {n_ex} tickets budget_exhausted")
to_output(demo).to_csv("outputs/triage_budget_demo.csv", index=False)

# Task 7b: final routed output
final, fns = run_pipeline(tick, FEW, token_budget=100000, label="few-shot, budget 100000 (final)")
out = to_output(final)
out.to_csv("outputs/triage_results.csv", index=False)
print("\nroute counts:", out["route"].value_counts().to_dict())
print("priority counts:", out["priority"].value_counts().to_dict())

# ---------- checkpoints ----------
zc = zero["status"].value_counts().to_dict()
checks = {
    "4 empty, 4 long": (empty, long_) == (4, 4),
    "redactions 55/91/132 = 278": (counts["email"], counts["phone"], counts["order_id"]) == (55, 91, 132),
    "0 PII left": left == 0,
    "19 transient errors, all recovered": zs["transient"] == 19 and zs["recovered"] == 19,
    "status 135 / 12 / 9": (zc.get("classified_json"), zc.get("classified_keyword"), zc.get("flagged_invalid_output")) == (135, 12, 9),
    "zero-shot accuracy 85.7 (+-0.5)": abs(summarise(zero, zs)["accuracy_%"] - 85.7) <= 0.5,
    "few-shot accuracy 100 (+-0.5)": abs(summarise(few, fs)["accuracy_%"] - 100) <= 0.5,
    "budget demo has exhausted tickets": n_ex >= 1,
    "results has 164 rows": len(out) == 164,
}
print()
for n, ok in checks.items():
    print(("PASS  " if ok else "FAIL  ") + n)