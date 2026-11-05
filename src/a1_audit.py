import pandas as pd

RAW = "data/raw/"
cust = pd.read_csv(RAW + "customers.csv")
prod = pd.read_csv(RAW + "products.csv")
orders = pd.read_csv(RAW + "orders.csv")
tick = pd.read_csv(RAW + "tickets.csv", keep_default_na=False)  # empty text stays ""

for name, df in [("customers", cust), ("products", prod), ("orders", orders), ("tickets", tick)]:
    print(f"\n=== {name}: shape {df.shape} ===")
    print(df.dtypes.to_string())
    miss = df.isna().sum()
    print(pd.DataFrame({"missing": miss, "pct": (miss / len(df) * 100).round(2)}).to_string())

amt_bad = pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum()
dm = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
age = pd.to_numeric(cust["age"], errors="coerce")

audit = {
    "raw order rows": len(orders),                                   # 13084
    "exact duplicate rows": int(orders.duplicated().sum()),          # 119
    "pending rows": int((orders["status"] == "pending").sum()),      # 656
    "distinct order_id": orders["order_id"].nunique(),               # 12309
    "non-numeric order_amount": int(amt_bad),                        # 120
    "negative delivery_minutes": int((dm < 0).sum()),                # 106
    "blank delivery_minutes": int(dm.isna().sum()),                  # 2541
    "blank age": int(age.isna().sum()),                              # 61
    "age outside 10-100": int(((age < 10) | (age > 100)).sum()),     # 6
    "distinct raw city spellings": cust["city"].nunique(),           # 24
    "distinct raw gender spellings": cust["gender"].dropna().nunique(),  # 10
    "blank gender": int(cust["gender"].isna().sum()),                # 30
}
print("\n=== AUDIT FIGURES ===")
print(pd.Series(audit).to_string())