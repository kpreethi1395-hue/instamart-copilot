import re, sqlite3
import pandas as pd

text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"^-- (Q\d)\b.*$", text, flags=re.M)
queries = dict(zip(parts[1::2], (b.strip() for b in parts[2::2])))

con = sqlite3.connect("data/instamart.db")
r = {k: pd.read_sql_query(q, con) for k, q in queries.items()}
con.close()

for k, df in r.items():
    print(f"\n=== {k}: {len(df)} rows ===")
    print(df.head(8).to_string(index=False))

q2, q5, q6, q7 = r["Q2"], r["Q5"], r["Q6"], r["Q7"]
top = q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]]
n_tie = int((q6["rnk"] != q6["row_num"]).sum())
best = q2.iloc[0]

checks = {
    "Q1 30 customers": len(r["Q1"]) == 30,
    "Q2 30 rows": len(q2) == 30,
    "Q2 total 2,998,668.80": round(q2["revenue"].sum(), 2) == 2998668.80,
    "Q2 max Bengaluru/Electronics 233,909.45": (best["city"], best["category"], best["revenue"]) == ("Bengaluru", "Electronics", 233909.45),
    "Q3 11 products": len(r["Q3"]) == 11,
    "Q4 25 customers": len(r["Q4"]) == 25,
    "Q5 550 customers": len(q5) == 550,
    "Q5 average 2,552.06": q5["avg_total"].iloc[0] == 2552.06,
    "Q6 8 rows rank != row_number": n_tie == 8,
    "Q7 183 days": len(q7) == 183,
    "Q7 final cumulative 2,998,668.80": q7["cumulative_revenue"].iloc[-1] == 2998668.80,
}
print("\nTop customer per city (Q6):\n", top.to_string(index=False))
print(f"Q6 rows with RANK != ROW_NUMBER: {n_tie}\n")
for name, ok in checks.items():
    print(("PASS  " if ok else "FAIL  ") + name)