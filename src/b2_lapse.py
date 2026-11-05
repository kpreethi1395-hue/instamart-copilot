import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")

# ---------- Task 4: cohort, label, features (orders BEFORE cutoff only) ----------
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
c = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"]).set_index("customer_id")
p = pd.read_csv("data/clean/products.csv")[["product_id", "category"]]
o = o.merge(p, on="product_id", how="left")

before = o[o["order_ts"] < CUTOFF]                      # nothing on/after 2025-09-01 below this line
comp = before[before["status"] == "completed"]
g = comp.groupby("customer_id")

feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount": g["discount_pct"].mean(),
    "avg_delivery": g["delivery_minutes"].mean(),
})
final = before[before["status"].isin(["completed", "cancelled"])]
feat["cancel_rate"] = (final["status"] == "cancelled").groupby(final["customer_id"]).mean().reindex(feat.index)

cnt = (comp.groupby(["customer_id", "category"]).size().reset_index(name="n")
       .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
       .drop_duplicates("customer_id").set_index("customer_id"))
feat["fav_category"] = cnt["category"]

feat = feat.join(c[["city", "age", "gender"]])
feat["tenure_days"] = (CUTOFF - c.loc[feat.index, "signup_date"]).dt.days

active = o[(o["status"] == "completed") & (o["order_ts"] >= CUTOFF) & (o["order_ts"] < END)]["customer_id"].unique()
y = pd.Series((~feat.index.isin(active)).astype(int), index=feat.index, name="lapsed")

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X = feat[NUM + CAT]
assert X.isna().sum().sum() == 0, "feature columns contain missing values"
print("no missing values in any feature column: OK")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"cohort {len(X)} | lapsed {y.sum()} ({y.mean():.2%}) | train {len(X_train)} | test {len(X_test)}")

# ---------- Task 5: preprocessing, fitted on training rows only (inside a Pipeline) ----------
def make_pre():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])

# ---------- Task 6: baseline, tuning, test evaluation ----------
def metrics(yt, yp):
    return {"precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0)}

base = np.zeros(len(y_test), dtype=int)
print(f"\nBASELINE (always 'not lapsed'): accuracy {accuracy_score(y_test, base):.3f}, "
      f"lapsed F1 {f1_score(y_test, base, zero_division=0):.3f}")

lr_grid = GridSearchCV(Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000))]),
                       {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1", n_jobs=-1).fit(X_train, y_train)
rf_grid = GridSearchCV(Pipeline([("pre", make_pre()),
                                 ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
                       {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                       cv=5, scoring="f1", n_jobs=-1).fit(X_train, y_train)

print(f"\nLogReg best {lr_grid.best_params_}  CV F1 {lr_grid.best_score_:.3f}")
print(f"RandForest best {rf_grid.best_params_}  CV F1 {rf_grid.best_score_:.3f}")

# scaler learned on train only -> its mean differs from the full-cohort mean
sc = lr_grid.best_estimator_.named_steps["pre"].named_transformers_["num"]
print(f"\nfrequency mean: scaler (train only) {sc.mean_[NUM.index('frequency')]:.4f} "
      f"vs full cohort {X['frequency'].mean():.4f}")

results = {}
for name, grid in [("LogisticRegression", lr_grid), ("RandomForest", rf_grid)]:
    pred = grid.predict(X_test)
    results[name] = metrics(y_test, pred)
    print(f"\n{name} TEST  confusion [[TN FP],[FN TP]]:\n{confusion_matrix(y_test, pred)}")
    print({k: round(v, 3) for k, v in results[name].items()})

# select on CROSS-VALIDATION F1 (not on the test set)
sel_name, sel_grid = (("RandomForest", rf_grid) if rf_grid.best_score_ >= lr_grid.best_score_
                      else ("LogisticRegression", lr_grid))
selected = sel_grid.best_estimator_
print(f"\nSELECTED: {sel_name} (test F1 {results[sel_name]['f1']:.3f})")
assert results[sel_name]["f1"] >= 0.55, "selected model must reach test F1 >= 0.55"

# ---------- Task 7: oversampling the TRAIN set only ----------
n_add = int((y_train == 0).sum() - (y_train == 1).sum())
extra = X_train[y_train == 1].sample(n=n_add, replace=True, random_state=42)
X_bal = pd.concat([X_train, extra])
y_bal = pd.concat([y_train, y_train.loc[extra.index]])
print(f"\nbalanced train: {y_bal.value_counts().to_dict()}")

lr_bal = clone(lr_grid.best_estimator_).fit(X_bal, y_bal)      # preprocessor refitted on train rows only
m_orig, m_bal = results["LogisticRegression"], metrics(y_test, lr_bal.predict(X_test))
cmp_ = pd.DataFrame({"original LR": m_orig, "balanced LR": m_bal}).round(3)
cmp_["change"] = (cmp_["balanced LR"] - cmp_["original LR"]).round(3)
print(cmp_.to_string())

# ---------- Task 8: interpretation, persistence, retention list ----------
rf_best = rf_grid.best_estimator_
names = rf_best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop 5 random-forest features:\n", imp.head(5).round(3).to_string())

lr_names = lr_grid.best_estimator_.named_steps["pre"].get_feature_names_out()
coef = pd.Series(lr_grid.best_estimator_.named_steps["clf"].coef_[0], index=lr_names)
print("\nLogReg coefficients (numeric features):\n", coef[[f"num__{n}" for n in NUM]].round(3).to_string())

joblib.dump(selected, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(selected.predict(X_test), reloaded.predict(X_test))
print(f"\nreloaded model reproduces test predictions exactly: {same}")
assert same

seg = pd.read_csv("outputs/customer_segments.csv")
premium_ids = set(seg.loc[seg["is_premium"], "customer_id"])
proba = selected.predict_proba(X_test)[:, 1]
rl = pd.DataFrame({"customer_id": X_test.index, "lapse_probability": proba.round(4),
                   "actually_lapsed": y_test.values})
rl = rl[rl["customer_id"].isin(premium_ids)].sort_values("lapse_probability", ascending=False).head(10)
rl.to_csv("outputs/retention_list.csv", index=False)
print(f"\nretention list: {len(rl)} premium test customers, {int(rl['actually_lapsed'].sum())} truly lapsed")
print("NOTE: the Premium flag uses whole-window RFM (includes September), so this list is illustrative, "
      "not a leakage-free evaluation.")

checks = {"cohort 1,156": len(X) == 1156, "443 lapsed": int(y.sum()) == 443,
          "train 924 / test 232": (len(X_train), len(X_test)) == (924, 232),
          "baseline accuracy 0.616": round(accuracy_score(y_test, base), 3) == 0.616,
          "balanced 570 each": set(y_bal.value_counts()) == {570}}
print()
for n, ok in checks.items():
    print(("PASS  " if ok else "FAIL  ") + n)