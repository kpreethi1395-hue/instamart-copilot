import sqlite3
import pandas as pd

with sqlite3.connect("data/instamart.db") as con:
    c = pd.read_sql_query("SELECT * FROM customers", con)
    o = pd.read_sql_query("SELECT * FROM orders", con)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}

comp = o[o["status"] == "completed"]
canc = o[o["status"] == "cancelled"]
cities = c["city"].value_counts().to_dict()
genders = c["gender"].value_counts().to_dict()

checks = {
    "3 tables in db": tables >= {"customers", "products", "orders"},
    "12,309 order rows": len(o) == 12309,
    "unique order_id": o["order_id"].is_unique,
    "11,110 completed": len(comp) == 11110,
    "1,199 cancelled": len(canc) == 1199,
    "no pending": (o["status"] == "pending").sum() == 0,
    "0 missing amount": o["order_amount"].isna().sum() == 0,
    "0 missing delivery (completed)": comp["delivery_minutes"].isna().sum() == 0,
    "1,199 missing delivery (cancelled)": canc["delivery_minutes"].isna().sum() == 1199,
    "completed revenue 2,998,668.80": round(comp["order_amount"].sum(), 2) == 2998668.80,
    "six cities": len(cities) == 6,
    "city counts": cities == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                              "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "gender counts": genders == {"female": 610, "male": 560, "unknown": 30},
    "0 missing age, all 18-60": c["age"].notna().all() and c["age"].between(18, 60).all(),
}
for name, ok in checks.items():
    print(("PASS  " if ok else "FAIL  ") + name)
assert all(checks.values()), "some checkpoints failed"
print("ALL CHECKPOINTS PASSED")