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


# ----------------------------------------------------------------------------- chart for section 2.3
def plot_decomposition(dec, signal_names, one_col="one question: P(escalate)", band=(0.2, 0.8)):
    """Two panels: where each framing's answers land on the 0-1 scale, and how the compound answer varies with the
    true signal count. `dec` is the table built in section 2.3."""
    import matplotlib.pyplot as plt
    SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    C_ONE, C_MANY = "#2a78d6", "#eb6834"          # validated pair (colourblind-safe, >= 3:1 on the surface)
    one = dec[one_col].to_numpy(float)
    many = dec[list(signal_names)].to_numpy(float).ravel()
    gold = dec["should escalate"].to_numpy(bool)
    in_band = lambda v: int(((v > band[0]) & (v < band[1])).sum())
    ok_one = int(((one >= 0.5) == gold).sum()); ok_many = int((dec["decision in code"] == dec["should escalate"]).sum())

    def stack(vals, step=0.105, width=0.025):      # dot histogram: equal values pile upward from the row's baseline
        ys, seen = [], {}
        for v in vals:
            k = round(v / width); n = seen.get(k, 0); seen[k] = n + 1
            ys.append(n * step)
        return np.array(ys)

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.25, 1]}, facecolor=SURF)
    for ax in (a, b):
        ax.set_facecolor(SURF)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(AXIS); ax.tick_params(colors=MUTED, length=0, labelsize=9)

    # panel A: one row per framing, x = the probability returned
    a.axvspan(*band, color=GRID, alpha=0.55, lw=0)
    a.text(sum(band) / 2, 4.12, "unsure band, 0.2 to 0.8", ha="center", va="center", fontsize=9, color=INK2)
    rows = [(2.55, one, C_ONE, "One compound question", "each dot: should this ticket escalate?",
             f"{len(one)} answers, {in_band(one)} in the band, {ok_one} of {len(one)} decisions correct"),
            (0.0, many, C_MANY, "Five atomic questions", "each dot: is this one signal present in this ticket?",
             f"{len(many)} answers, {in_band(many)} in the band, {ok_many} of {len(dec)} decisions correct after counting in code")]
    for y0, vals, col, name, sub, note in rows:
        a.scatter(vals, y0 + stack(vals), s=46, color=col, edgecolor=SURF, linewidth=1.2, zorder=3)
        a.axhline(y0 - 0.09, color=AXIS, lw=0.8, zorder=1)
        a.text(0.5, y0 + 1.04, name, fontsize=10.5, color=INK, fontweight="bold", ha="center", va="bottom")
        a.text(0.5, y0 + 0.77, sub, fontsize=9, color=INK, style="italic", ha="center", va="bottom")
        a.text(0.5, y0 + 0.50, note, fontsize=9, color=INK2, ha="center", va="bottom")
    a.set_xlim(-0.03, 1.03); a.set_ylim(-0.12, 4.3); a.set_yticks([])
    a.set_xticks([0, 0.2, 0.5, 0.8, 1.0]); a.set_xlabel("probability returned by the model", color=INK2, fontsize=9.5)
    a.set_title("Where each framing's answers land", loc="left", fontsize=12, color=INK, pad=12)

    # panel B: the compound answer against the true signal count
    n_true = dec["signals present"].to_numpy(float)
    offs, seen = [], {}
    for v in n_true:
        n = seen.get(v, 0); seen[v] = n + 1; offs.append(n)
    counts = {v: c for v, c in seen.items()}
    xs = np.array([v + (o - (counts[v] - 1) / 2) * 0.16 for v, o in zip(n_true, offs)])
    wrong = (one >= 0.5) != gold
    b.axhspan(*band, color=GRID, alpha=0.55, lw=0)
    b.axvline(2.5, color=AXIS, lw=1, ls=(0, (4, 3)))
    b.text(2.58, 1.09, "rule: escalate at 3 or more", fontsize=9, color=INK2, ha="left", va="center")
    b.scatter(xs[~wrong], one[~wrong], s=70, color=C_ONE, edgecolor=SURF, linewidth=1.4, zorder=3)
    b.scatter(xs[wrong], one[wrong], s=70, facecolor=SURF, edgecolor=C_ONE, linewidth=2, zorder=3)
    for x, y in zip(xs[wrong], one[wrong]):
        b.annotate("should escalate,\nanswered below 0.5", (x, y), xytext=(12, -24), textcoords="offset points", fontsize=9, color=INK2)
    b.axhline(0.5, color=AXIS, lw=0.8)
    b.set_xlim(-0.5, 5.5); b.set_ylim(-0.05, 1.13); b.set_xticks(range(6)); b.set_yticks([0, 0.2, 0.5, 0.8, 1.0])
    b.set_xlabel("signals actually present in the ticket", color=INK2, fontsize=9.5)
    b.set_ylabel("P(escalate), one compound question", color=INK2, fontsize=9.5)
    b.set_title("The compound question hedges at the threshold", loc="left", fontsize=12, color=INK, pad=12)
    fig.tight_layout(w_pad=3)
    return fig


# ----------------------------------------------------------------------------- chart for sections 3 and 6
MODEL_NAMES = {"typesafe/jev-1.13": "Jev 1.13", "openai/gpt-6-luna-decisions": "Luna decisions", "openai/gpt-6-luna": "Luna chat"}

def plot_comparison(results, models):
    """One small panel per measure, the same model rows in each: accuracy (with 95% interval), latency (median to p95),
    cost per 1,000 requests, and, once the `with_other` runs exist in `results`, out-of-scope queries caught."""
    import matplotlib.pyplot as plt
    SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]      # validated trio; every mark is also labelled directly
    models = list(models); ys = {m: len(models) - 1 - i for i, m in enumerate(models)}
    base = {m: results[("no_other", m)] for m in models}
    has_other = all(("with_other", m) in results for m in models)
    n_pan = 4 if has_other else 3
    fig, axes = plt.subplots(1, n_pan, figsize=(3.3 * n_pan + 1.2, 3.3), facecolor=SURF, sharey=True)

    def frame(ax, title, note, xlim, ticks, fmt, xlabel):
        ax.set_facecolor(SURF)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(AXIS); ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
        ax.set_xlim(*xlim); ax.set_xticks(ticks); ax.set_xticklabels([fmt(t) for t in ticks])
        ax.set_ylim(-0.6, len(models) - 0.35); ax.set_yticks([])
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=18)
        ax.text(0, 1.035, note, transform=ax.transAxes, fontsize=8.5, color=INK2, ha="left", va="bottom")
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK2, labelpad=6)

    def dot(ax, x, m, label, dx=7, ha="left"):
        ax.scatter([x], [ys[m]], s=70, color=COLORS[models.index(m)], edgecolor=SURF, linewidth=1.4, zorder=3)
        ax.annotate(label, (x, ys[m]), xytext=(dx, 9), textcoords="offset points", fontsize=9, color=INK, ha=ha, va="center")

    # 1 accuracy with bootstrap interval (zoomed axis)
    ax = axes[0]; frame(ax, "In-scope accuracy", "higher is better", (0.90, 1.0), [0.90, 0.95, 1.0], lambda t: f"{t:.0%}", "line = 95% interval · axis starts at 90%")
    for m in models:
        r = base[m]; ins = r[~r.is_oos]; acc, lo, hi = bootstrap_ci(ins.pred.eq(ins.label))
        ax.plot([lo, hi], [ys[m]] * 2, color=COLORS[models.index(m)], lw=2, solid_capstyle="round", zorder=2)
        dot(ax, acc, m, f"{acc:.1%}", dx=0, ha="center")
        refused, refused_in = int(r.pred.isna().sum()), int(ins.pred.isna().sum())
        if refused:
            ax.annotate(f"{refused} refusals, {refused_in} of them in-scope", (lo, ys[m]), xytext=(0, -13), textcoords="offset points", fontsize=8, color=INK2, ha="left", va="center")
    for m in models:   # row labels, once, left of the first panel
        ax.text(-0.04, ys[m], MODEL_NAMES.get(m, m), transform=ax.get_yaxis_transform(), fontsize=10, color=INK, ha="right", va="center", fontweight="bold")

    # 2 latency: median to 95th percentile
    ax = axes[1]; top = max(base[m].latency_s.quantile(.95) for m in models) * 1000
    frame(ax, "Latency", "lower is better", (0, top * 1.12), [0, 500, 1000, 1500], lambda t: f"{t:,.0f}", "milliseconds · dot = median, line to 95th percentile")
    for m in models:
        p50, p95 = base[m].latency_s.median() * 1000, base[m].latency_s.quantile(.95) * 1000
        ax.plot([p50, p95], [ys[m]] * 2, color=COLORS[models.index(m)], lw=2, solid_capstyle="round", zorder=2)
        dot(ax, p50, m, f"{p50:,.0f} ms", dx=0, ha="left")

    # 3 cost per 1,000 requests
    ax = axes[2]; costs = {m: base[m].cost.sum() / len(base[m]) * 1000 for m in models}
    frame(ax, "Cost per 1,000 requests", "lower is better", (0, max(costs.values()) * 1.3), [0, 0.1, 0.2], lambda t: f"${t:.2f}", "billed cost, US dollars")
    for m in models: dot(ax, costs[m], m, f"${costs[m]:.2f}", dx=0, ha="center")

    # 4 out-of-scope caught once `other` is offered
    if has_other:
        ax = axes[3]; frame(ax, "Out-of-scope caught", "higher is better", (0.80, 1.0), [0.80, 0.90, 1.0], lambda t: f"{t:.0%}", "with an `other` option (0% without) · axis starts at 80%")
        for m in models:
            r = results[("with_other", m)]; o = r[r.is_oos]; v = o.pred.eq("other").mean()
            dot(ax, v, m, f"{v:.1%}", dx=0, ha="center")
    fig.tight_layout(); fig.subplots_adjust(left=1.5 / fig.get_figwidth(), wspace=0.16)
    return fig
