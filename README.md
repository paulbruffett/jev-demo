# Jev vs GPT-6 Luna on CLINC150

A notebook that shows what a decision model is using TypeSafe's Jev on OpenRouter, compares it with GPT-6 Luna
(Decisions API head and plain chat completions), and measures what happens on out-of-scope input with and
without an `other` option.

- `jev_clinc150.ipynb` - the notebook (five sections: what a decision model is, prompting, head-to-head, unknown input, take-aways)
- `jevlib.py` - plumbing: API calls, budget cap, response cache, batch runner, metrics
- `data/clinc150_data_full.json` - CLINC150 (Larson et al. 2019), from github.com/clinc/oos-eval
- `data/clinc150_criteria.json` - one-line description per intent, the "prompt"
- `build_notebook.py` - regenerates the notebook from source (clears outputs; re-execute to refill, cached calls are free)
- `OUTLINE.md` - research notes, sources, and the original plan
- `runs/` - response cache and per-run pickles (created on first run, gitignored)

## Run

```bash
cp .env.example .env   # then paste your OpenRouter key
uv sync
uv run jupyter lab jev_clinc150.ipynb
```

Spend is capped at $20 by default; set `BUDGET_CAP_USD` in `.env` to change it. Every response is cached in
`runs/cache.jsonl`, so re-running cells does not re-bill. The default sample is 1,500 in-scope + 500 OOS items
(about $1.30 for all six passes); set `N_PER_INTENT, N_OOS = 30, 1000` in section 3 for the full test set.
