import os, sqlite3
import pandas as pd



RAW, CLEAN = "data/raw/", "data/clean/"
os.makedirs(CLEAN, exist_ok=True)

# ---------- customers ----------
c = pd.read_csv(RAW + "customers.csv")
c["city"] = c["city"].str.strip().str.title()

g = c["gender"].fillna("").str.strip().str.lower()
c["gender"] = g.map(lambda x: "male" if x.startswith("m") else "female" if x.startswith("f") else "unknown")

bad_age = c["age"].notna() & ((c["age"] < 10) | (c["age"] > 100))
print(f"ages set to missing (outside 10-100): {bad_age.sum()}")
c.loc[bad_age, "age"] = pd.NA
c["age"] = c["age"].astype(float)
n_fill = c["age"].isna().sum()
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
print(f"ages filled with city median: {n_fill}")
c["signup_date"] = pd.to_datetime(c["signup_date"])

# ---------- orders ----------
p = pd.read_csv(RAW + "products.csv")
o = pd.read_csv(RAW + "orders.csv")
print(f"raw rows: {len(o)}")

o["order_ts"] = pd.to_datetime(o["order_ts"])                       # rule 1

n0 = len(o); o = o.drop_duplicates()                                 # rule 2
print(f"exact duplicates removed: {n0 - len(o)}")

n0 = len(o)                                                          # rule 3
o = o.sort_values("order_ts").drop_duplicates("order_id", keep="last")
print(f"superseded (earlier) records removed: {n0 - len(o)}")
o = o.sort_values("order_id").reset_index(drop=True)

o = o.merge(p[["product_id", "category", "price_inr"]], on="product_id", how="left")

o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")   # rule 4
miss = o["order_amount"].isna()
o.loc[miss, "order_amount"] = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)[miss]
print(f"amounts filled from product price: {miss.sum()}")

o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")   # rule 5
neg = o["delivery_minutes"] < 0
o.loc[neg, "delivery_minutes"] = pd.NA
o["delivery_minutes"] = o["delivery_minutes"].astype(float)
print(f"negative delivery_minutes set to missing: {neg.sum()}")
comp = o["status"] == "completed"
med = o.loc[comp].groupby("category")["delivery_minutes"].transform("median")
n_fill = o.loc[comp, "delivery_minutes"].isna().sum()
o.loc[comp, "delivery_minutes"] = o.loc[comp, "delivery_minutes"].fillna(med)
print(f"completed delivery_minutes filled with category median: {n_fill}")

o = o.drop(columns=["category", "price_inr"])

# ---------- store ----------
c_out, o_out = c.copy(), o.copy()
c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

c_out.to_csv(CLEAN + "customers.csv", index=False)
p.to_csv(CLEAN + "products.csv", index=False)
o_out.to_csv(CLEAN + "orders.csv", index=False)

with sqlite3.connect("data/instamart.db") as con:
    c_out.to_sql("customers", con, if_exists="replace", index=False)
    p.to_sql("products", con, if_exists="replace", index=False)
    o_out.to_sql("orders", con, if_exists="replace", index=False)
print("clean CSVs and data/instamart.db written")