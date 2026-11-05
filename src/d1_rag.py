import glob, os, sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                       # so `import mock_llm` works from src/
import mock_llm

POLICY_DIR = os.path.join(ROOT, "data", "policies")
CHUNK_WORDS = 40

# ---------- Task 1: chunking ----------
def chunk_text(text, size=CHUNK_WORDS, overlap=10):
    words, step, out, start = text.split(), size - overlap, [], 0
    while True:
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):          # this chunk reaches the last word
            break
        start += step
    return out

def build_chunks(overlap=10):
    chunks = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            for t in chunk_text(f.read(), overlap=overlap):
                chunks.append({"source": name, "text": t})
    return chunks

# ---------- Task 2: TF-IDF retrieval ----------
class Retriever:
    def __init__(self, overlap=10):
        self.chunks = build_chunks(overlap)
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]),
                 "text": self.chunks[i]["text"]} for i in top]

_default = Retriever(overlap=10)

def retrieve(query, k=3):
    return _default.retrieve(query, k)

# ---------- Task 3: augment and answer ----------
SYSTEM = ("You are a POLICY ASSISTANT for Instamart support. Answer ONLY from the supplied CONTEXT. "
          "If the answer is not in the context, say: I could not find that information in the policy documents.")

def answer(question, retriever=None):
    hits = (retriever or _default).retrieve(question, 3)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]

# ---------- Task 4: evaluation ----------
def evaluate(overlap):
    r = Retriever(overlap)
    qa = pd.read_csv(os.path.join(ROOT, "data", "raw", "policy_eval.csv"), keep_default_na=False)
    doc_ok = fact_ok = fallback_ok = n_ans = n_unans = 0
    failures = []
    for q in qa.itertuples():
        text, sources = answer(q.question, r)
        if q.expected_doc == "NONE":
            n_unans += 1
            fallback_ok += text.startswith("I could not find")
        else:
            n_ans += 1
            d, f = q.expected_doc in sources, q.expected_fact in text
            doc_ok += d; fact_ok += f
            if not (d and f):
                failures.append((q.question, d, f, text))
    return {"chunks": len(r.chunks), "answerable": n_ans, "doc_in_top3": doc_ok,
            "fact_in_answer": fact_ok, "unanswerable": n_unans, "fallback": fallback_ok}, failures

if __name__ == "__main__":
    for ov in (10, 0):
        r = Retriever(ov)
        per_doc = pd.Series([c["source"] for c in r.chunks]).value_counts().to_dict()
        print(f"\n=== overlap {ov}: {len(r.chunks)} chunks {per_doc} ===")
        res, fails = evaluate(ov)
        print(pd.Series(res).to_string())
        for q, d, f, t in fails:
            print(f"FAILS: {q!r} | expected_doc in top3: {d} | fact in answer: {f}\n   answer: {t}")

    print("\nquery 'refund policy late delivery' top 3 sources:",
          [h["source"] for h in retrieve("refund policy late delivery")])
    res10, _ = evaluate(10)
    res0, f0 = evaluate(0)
    n_chunks = len(_default.chunks)
    checks = {"15 chunks": n_chunks == 15,
              "top 3 all refund_policy.txt": all(h["source"] == "refund_policy.txt" for h in retrieve("refund policy late delivery")),
              "overlap 10: 12/12, 12/12, 2/2": (res10["doc_in_top3"], res10["fact_in_answer"], res10["fallback"]) == (12, 12, 2),
              "overlap 0: at least one answerable fails": len(f0) >= 1}
    print()
    for n, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + n)