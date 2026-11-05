import numpy as np
import pandas as pd

o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
d = (o[o["status"] == "completed"]
     .merge(c[["customer_id", "city"]], on="customer_id")
     .merge(p[["product_id", "category"]], on="product_id"))

# (a) pivot cross-check against SQL Q2
pv = pd.pivot_table(d, index="city", columns="category", values="order_amount",
                    aggfunc="sum", margins=True, margins_name="Total")
print(pv.round(2).to_string())
grand = round(pv.loc["Total", "Total"], 2)
print(f"\npivot grand total: {grand:,.2f}  (SQL: 2,998,668.80) -> match: {grand == 2998668.80}")

city = d.groupby("city").agg(total=("order_amount", "sum"),
                             mean=("order_amount", "mean"),
                             count=("order_amount", "count")).round(2)
print("\nPer-city sum, mean, count:\n", city.to_string())

# (b) distribution statistics
x = d["order_amount"]
q1, q3 = x.quantile(.25), x.quantile(.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean {x.mean():.2f} | median {x.median():.2f} | skew {x.skew():.2f}")
print(f"Q1 {q1:.2f} | Q2 {x.median():.2f} | Q3 {q3:.2f} | IQR {iqr:.2f}")
print(f"fences: lower {lo:.2f}, upper {hi:.2f}")
print(f"outliers below lower fence: {(x < lo).sum()} | above upper fence: {(x > hi).sum()}")

cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = d[cols].corr()
print("\nCorrelation matrix:\n", corr.round(3).to_string())
pairs = corr.abs().where(~np.eye(len(corr), dtype=bool)).stack()
a, b = pairs.idxmax()
print(f"\nlargest |correlation|: {a} vs {b} = {corr.loc[a, b]:.3f}")