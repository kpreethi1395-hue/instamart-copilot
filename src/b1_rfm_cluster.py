import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

os.makedirs("outputs", exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
COLS = ["recency", "frequency", "monetary"]
FINAL_K = 4          # chosen from the elbow and silhouette table

# ---------- Task 1: RFM ----------
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
d = o[o["status"] == "completed"]
rfm = d.groupby("customer_id").agg(last_ts=("order_ts", "max"),
                                   frequency=("order_id", "nunique"),
                                   monetary=("order_amount", "sum")).reset_index()
rfm["recency"] = (SNAPSHOT - rfm["last_ts"]).dt.days
rfm["monetary"] = rfm["monetary"].round(2)
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

print(f"customers: {len(rfm)}")
print(f"recency median {rfm.recency.median():.0f}, min {rfm.recency.min()}, max {rfm.recency.max()}")
print(f"frequency mean {rfm.frequency.mean():.2f}, max {rfm.frequency.max()}")
print(f"monetary mean {rfm.monetary.mean():.2f}, max {rfm.monetary.max():.2f}")

# ---------- Task 2: K-means K = 2..8 ----------
X = StandardScaler().fit_transform(rfm[COLS])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1),
                 "silhouette": round(silhouette_score(X, km.labels_), 3)})
ktab = pd.DataFrame(rows)
print("\n", ktab.to_string(index=False))

REF = {2: (1820.8, .465), 3: (1268.2, .445), 4: (743.1, .526), 5: (617.4, .481),
       6: (503.2, .482), 7: (408.5, .416), 8: (366.5, .412)}
for _, r in ktab.iterrows():
    ri, rs = REF[int(r.K)]
    ok = abs(r.inertia - ri) / ri <= .05 and abs(r.silhouette - rs) <= .05
    print(f"K={int(r.K)} within tolerance: {ok}")

# ---------- Task 3: final model, profile, name ----------
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"),
                                     recency=("recency", "mean"),
                                     frequency=("frequency", "mean"),
                                     monetary=("monetary", "mean")).round(2)
print("\n", prof.to_string())

# name segments from their profile (cluster numbers are only labels, with no order)
premium = prof["monetary"].idxmax()
rest = prof.drop(index=premium)
loyal = rest["frequency"].idxmax()
rest = rest.drop(index=loyal)
at_risk = rest["recency"].idxmax()
names = {premium: "Premium Big Spenders", loyal: "Loyal Frequent Buyers", at_risk: "At-Risk / Lapsing"}
for s in prof.index:
    names.setdefault(s, "Occasional Buyers")
actions = {"Premium Big Spenders": "VIP perks, early access to electronics deals, priority support",
           "Loyal Frequent Buyers": "Loyalty rewards and subscriptions, protect delivery quality",
           "At-Risk / Lapsing": "Win-back coupon and reminder before they churn",
           "Occasional Buyers": "Bundle offers and free-delivery nudges to raise order frequency"}
prof["segment_name"] = prof.index.map(names)
prof["action"] = prof["segment_name"].map(actions)
print("\n", prof[["size", "segment_name", "action"]].to_string())

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium

pca = PCA(n_components=2).fit(X)
evr = pca.explained_variance_ratio_
print(f"\nPCA explained variance: {evr[0]:.3f}, {evr[1]:.3f}, total {evr.sum():.3f}")

rfm[["customer_id", "recency", "frequency", "monetary",
     "segment_id", "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)
print("wrote outputs/customer_segments.csv")

# ---------- checkpoints ----------
checks = {
    "1,175 customers": len(rfm) == 1175,
    "recency median 19 / min 0 / max 180": (rfm.recency.median(), rfm.recency.min(), rfm.recency.max()) == (19, 0, 180),
    "frequency mean 9.46 / max 33": round(rfm.frequency.mean(), 2) == 9.46 and rfm.frequency.max() == 33,
    "monetary mean 2552.06 / max 14766.55": round(rfm.monetary.mean(), 2) == 2552.06 and rfm.monetary.max() == 14766.55,
    "PCA total ~0.852 (+-0.02)": abs(evr.sum() - 0.852) <= .02,
    "exactly one premium segment": rfm.loc[rfm.is_premium, "segment_id"].nunique() == 1,
}
print()
for n, ok in checks.items():
    print(("PASS  " if ok else "FAIL  ") + n)