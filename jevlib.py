"""Plumbing for the Jev vs GPT-6 Luna notebook: API calls, budget cap, response cache, batch runner, metrics.

Nothing in here is specific to decision models conceptually; it exists so the notebook can stay focused on them.
"""
import os, json, time, math, hashlib, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import f1_score

load_dotenv()
API_KEY = os.environ.get("OPENROUTER_API_KEY")
assert API_KEY, "Set OPENROUTER_API_KEY in your environment or in a .env file next to this notebook"

BASE = "https://openrouter.ai/api"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "X-Title": "Jev CLINC150 demo"}
RUNS_DIR = Path("runs"); RUNS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- budget + cache
class BudgetExceeded(RuntimeError):
    pass

class Budget:
    """Sums usage.cost from every live response; refuses a request once the cap would be exceeded."""
    def __init__(self, cap_usd):
        self.cap, self.spent, self.calls, self._lock = cap_usd, 0.0, 0, threading.Lock()
    def charge(self, cost):
        with self._lock:
            self.spent += cost; self.calls += 1
    def check(self, est=0.0):
        if self.spent + est > self.cap:
            raise BudgetExceeded(f"spent ${self.spent:.4f} + est ${est:.4f} would exceed cap ${self.cap:.2f}")
    def __repr__(self):
        return f"Budget(spent=${self.spent:.4f} of ${self.cap:.2f}, live calls={self.calls})"

budget = Budget(float(os.environ.get("BUDGET_CAP_USD", 20)))

CACHE_PATH = RUNS_DIR / "cache.jsonl"
_cache, _cache_lock = {}, threading.Lock()
if CACHE_PATH.exists():
    for line in CACHE_PATH.open():
        rec = json.loads(line); _cache[rec["key"]] = rec

def _post(url, body, est_cost=0.0, retries=4, cache=True):
    """POST with disk cache keyed on the request body. Returns (response_json, latency_s, was_cached).
    cache=False always makes a live call and does not store it (used for repeat probes)."""
    key = hashlib.sha256((url + json.dumps(body, sort_keys=True)).encode()).hexdigest()
    if cache and key in _cache:
        return _cache[key]["response"], _cache[key]["latency_s"], True
    budget.check(est_cost)
    for attempt in range(retries):
        t0 = time.perf_counter()
        resp = requests.post(url, headers=HEADERS, json=body, timeout=120)
        lat = time.perf_counter() - t0
        if resp.status_code == 502 and "refused" in resp.text:      # OpenAI's `refusal` answer type; do not retry
            raise RuntimeError(f"refusal: {resp.text[:200]}")
        if resp.status_code in (429, 500, 502, 503, 504, 520) and attempt < retries - 1:
            time.sleep(1.5 * 2 ** attempt); continue
        if resp.status_code != 200:
            raise RuntimeError(f"{resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        budget.charge(float((data.get("usage") or {}).get("cost") or 0.0))
        if not cache:
            return data, lat, False
        rec = {"key": key, "url": url, "request": body, "response": data, "latency_s": lat, "ts": time.time()}
        with _cache_lock:
            _cache[key] = rec
            with CACHE_PATH.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return data, lat, False

# ----------------------------------------------------------------------------- prices
def fetch_prices(model_ids):
    """Live $/M-token prices and context length for the given model IDs."""
    models = {m["id"]: m for m in requests.get(f"{BASE}/v1/models", timeout=30).json()["data"]}
    models.update({m["id"]: m for m in requests.get(f"{BASE}/v1/models", params={"output_modalities": "decisions"}, timeout=30).json()["data"]})
    out = {}
    for mid in model_ids:
        p = models[mid]["pricing"]
        out[mid] = {"in": float(p["prompt"]) * 1e6, "out": float(p["completion"]) * 1e6,
                    "cached": float(p.get("input_cache_read") or 0) * 1e6, "ctx": models[mid].get("context_length")}
    return out

PRICE = {}

def est_tokens(obj):
    return math.ceil(len(json.dumps(obj)) / 4)      # ~4 chars/token on English JSON; pre-flight estimates only

# ----------------------------------------------------------------------------- the two request types
def decide(model, state, questions, cache=True):
    """OpenRouter Decisions API. Works for Jev and for openai/gpt-6-luna-decisions."""
    body = {"model": model, "state": state, "questions": questions}
    est = est_tokens(body) / 1e6 * PRICE.get(model, {"in": 0.1})["in"]
    return _post(f"{BASE}/alpha/decisions", body, est_cost=est, cache=cache)

def chat_classify(model, system, user, labels, cache=True):
    """Chat completions with a strict JSON schema whose only field is an enum of the labels."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "reasoning": {"enabled": False},
        "response_format": {"type": "json_schema", "json_schema": {"name": "intent", "strict": True, "schema": {
            "type": "object", "properties": {"label": {"type": "string", "enum": labels}},
            "required": ["label"], "additionalProperties": False}}},
        "usage": {"include": True},
    }
    est = est_tokens(body) / 1e6 * PRICE.get(model, {"in": 0.1})["in"]
    return _post(f"{BASE}/v1/chat/completions", body, est_cost=est, cache=cache)

# ----------------------------------------------------------------------------- classification runners
INSTR = "Which intent does the user's message express?"

def classify_decisions(model, text, criteria, cache=True):
    resp, lat, cached = decide(model, {"message": text}, {"intent": {"type": "choice", "instructions": INSTR, "criteria": criteria}}, cache=cache)
    a = resp["answers"]["intent"]; u = resp["usage"]
    return {"pred": a["choice"], "p_max": max(a["probabilities"].values()), "conf": a["confidence"], "probs": a["probabilities"],
            "latency_s": lat, "cost": float(u.get("cost") or 0), "in_tok": u.get("input_tokens"), "cached": cached}

def classify_chat(model, text, criteria, cache=True):
    system = INSTR + " Reply with JSON only.\n\nIntents:\n" + "\n".join(f"- {k}: {v}" for k, v in criteria.items())
    resp, lat, cached = chat_classify(model, system, text, list(criteria), cache=cache)
    try:
        label = json.loads(resp["choices"][0]["message"]["content"])["label"]
    except Exception:
        label = None
    u = resp["usage"]
    return {"pred": label, "p_max": np.nan, "conf": np.nan, "probs": None,
            "latency_s": lat, "cost": float(u.get("cost") or 0), "in_tok": u.get("prompt_tokens"), "cached": cached}

def run_batch(model, items, criteria, tag, workers=8, chat=False):
    """Classify every row of `items` (needs a `text` column) and return items joined with the results."""
    fn = classify_chat if chat else classify_decisions
    out = [None] * len(items)
    def work(i, text):
        try:
            return i, fn(model, text, criteria)
        except BudgetExceeded:
            raise
        except Exception as e:
            return i, {"pred": None, "error": str(e)[:200], "latency_s": np.nan, "cost": 0.0, "p_max": np.nan, "conf": np.nan, "probs": None}
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(work, i, t) for i, t in enumerate(items.text)]
        for k, f in enumerate(as_completed(futs)):
            i, r = f.result(); out[i] = r
            if k % 500 == 0:
                print(f"  {tag} {model}: {k}/{len(items)}  {budget}")
    res = items.reset_index(drop=True).join(pd.DataFrame(out))
    res["model"], res["condition"] = model, tag
    res.to_pickle(RUNS_DIR / f"{tag}_{model.replace('/', '_')}.pkl")
    return res

# ----------------------------------------------------------------------------- data
def load_clinc150(path="data/clinc150_data_full.json"):
    raw = json.load(open(path))
    test = pd.DataFrame(raw["test"], columns=["text", "label"])
    oos = pd.DataFrame(raw["oos_test"], columns=["text", "label"])
    return test, oos

def make_sample(test, oos, n_per_intent, n_oos, seed=7):
    ins = pd.concat([g.sample(n_per_intent, random_state=seed) for _, g in test.groupby("label")])
    s = pd.concat([ins, oos.sample(n_oos, random_state=seed)], ignore_index=True)
    s["is_oos"] = s.label.eq("oos")
    return s

# ----------------------------------------------------------------------------- metrics
def bootstrap_ci(correct, n=2000, seed=7):
    rng = np.random.default_rng(seed); c = np.asarray(correct, float)
    bs = [rng.choice(c, len(c)).mean() for _ in range(n)]
    return c.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def ece(p, correct, bins=10):
    p, c = np.asarray(p, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - c[m].mean())
    return e

def summarize(res):
    """One row of headline metrics for a classification run (in-scope accuracy, F1, latency, cost)."""
    ins = res[~res.is_oos]
    ok = ins.pred.eq(ins.label)
    acc, lo, hi = bootstrap_ci(ok)
    row = {"in-scope acc": f"{acc:.3f} [{lo:.3f}, {hi:.3f}]",
           "macro-F1": round(f1_score(ins.label.tolist(), ins.pred.fillna("__none__").tolist(), average="macro"), 3),
           "invalid / refused": int(res.pred.isna().sum()),
           "p50 ms": round(res.latency_s.median() * 1000), "p95 ms": round(res.latency_s.quantile(.95) * 1000),
           "$ / 1k req": round(res.cost.sum() / len(res) * 1000, 4),
           "mean input tok": round(res.in_tok.mean())}
    if ins.p_max.notna().any():
        row["ECE (p_max)"] = round(ece(ins.p_max, ok), 3)
    return row


# ----------------------------------------------------------------------------- repeat probe (consistency)
def repeat_probe(models_chat, texts, criteria, n=10):
    """Send each text n times to each model with the cache bypassed. models_chat: {model_id: use_chat_completions}.
    Returns one row per call: text, model, label, p_max, runner_up."""
    rows = []
    for text in texts:
        for model, chat in models_chat.items():
            fn = classify_chat if chat else classify_decisions
            for k in range(n):
                try:
                    r = fn(model, text, criteria, cache=False)
                except Exception as e:
                    rows.append({"text": text, "model": model, "rep": k, "label": "REFUSED" if "refus" in str(e) else "ERROR", "p_max": np.nan, "runner_up": None}); continue
                ru = None
                if r["probs"]:
                    top2 = sorted(r["probs"].items(), key=lambda x: -x[1])[1]
                    ru = f"{top2[0]} {top2[1]:.2f}"
                rows.append({"text": text, "model": model, "rep": k, "label": r["pred"], "p_max": r["p_max"], "runner_up": ru})
    return pd.DataFrame(rows)

def repeat_summary(df):
    import collections
    def agg(g):
        c = collections.Counter(g.label)
        return pd.Series({"labels seen (count)": ", ".join(f"{k} ({v})" for k, v in c.most_common()),
                          "p_max mean": round(g.p_max.mean(), 3), "sd": round(g.p_max.std(), 3),
                          "min": g.p_max.min(), "max": g.p_max.max()})
    return df.groupby(["text", "model"]).apply(agg, include_groups=False)
