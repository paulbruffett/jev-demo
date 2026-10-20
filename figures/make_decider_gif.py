"""Animated walk-through of a Decider-style readout: one forward pass, slot row, letter rows, scores, temperature, softmax.
Writes figures/decider_readout.gif. Run: uv run python figures/make_decider_gif.py"""
import io, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
AMBER, AMBER_FILL = "#c98500", "#fbedcc"        # computed from this prompt
VIOLET, VIOLET_FILL = "#4a3aa7", "#e6e2f7"      # fixed trained weights
BLUE = "#2a78d6"
W, H, FPS = 16, 9, 12

LETTERS = ["A", "B", "C", "D", "E"]
NAMES = ["billing", "technical", "sales", "fraud", "account"]
SCORES = np.array([3.1, 1.2, -0.4, -1.0, -2.2])
T = 1.164
SCALED = SCORES / T
PROBS = np.exp(SCALED) / np.exp(SCALED).sum()

# stage boundaries in frames: (name, length)
STAGES = [("prompt", 22), ("pass", 30), ("slot", 14), ("to_slotbox", 14), ("vocab", 22), ("to_dot", 14),
          ("scores", 20), ("hold1", 10), ("temp", 20), ("hold2", 8), ("softmax", 24), ("choice", 36)]
STARTS, acc = {}, 0
for name, n in STAGES:
    STARTS[name] = (acc, n); acc += n
N_FRAMES = acc

def prog(f, name):
    s, n = STARTS[name]
    x = (f - s) / n
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def ease(x): return x * x * (3 - 2 * x)

CAPTIONS = [("prompt", "1  The prompt: context, the question, lettered options, then an answer slot"),
            ("pass", "2  One forward pass: every token gets a hidden-state vector, building a 1,000 × d matrix"),
            ("slot", "3  Take the single row at the answer slot: a 1 × d vector shaped by everything before it"),
            ("vocab", "4  From the vocabulary head, select the fixed rows for the option letters A to E"),
            ("scores", "5  Dot the slot vector with each letter row: one score per option"),
            ("temp", "6  Divide the scores by the fitted temperature to calibrate them"),
            ("softmax", "7  Softmax turns the scores into probabilities that sum to 100%"),
            ("choice", "8  The highest-probability option is returned as the choice")]

def caption_at(f):
    cur = CAPTIONS[0][1]
    for name, text in CAPTIONS:
        if f >= STARTS[name][0]: cur = text
    return cur

def box(ax, x, y, w, h, title, sub, edge, fill, alpha=1.0, bold_title=True):
    if alpha <= 0: return
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fill, ec=edge, lw=1.2, alpha=alpha))
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=11.5, color=INK, alpha=alpha, fontweight="bold" if bold_title else "normal")
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=9.5, color=INK2, alpha=alpha)

def draw_line(ax, pts, p, color, lw=1.8):
    """Draw a polyline progressively; p in [0,1]."""
    if p <= 0: return
    seg = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total, todo = sum(seg), sum(seg) * p
    xs, ys = [pts[0][0]], [pts[0][1]]
    for i, L in enumerate(seg):
        if todo >= L:
            xs.append(pts[i + 1][0]); ys.append(pts[i + 1][1]); todo -= L
        else:
            r = todo / L
            xs.append(pts[i][0] + (pts[i + 1][0] - pts[i][0]) * r); ys.append(pts[i][1] + (pts[i + 1][1] - pts[i][1]) * r); break
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=4)
    if p >= 1:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        ax.annotate("", xy=(x1, y1), xytext=(x1 - (x1 - x0) * 0.01 - 1e-6, y1 - (y1 - y0) * 0.01), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), zorder=4)

def frame(f):
    fig = plt.figure(figsize=(W * 0.6, H * 0.6), dpi=110, facecolor=SURF)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.set_facecolor(SURF)

    # ---- prompt (top left), typed line by line
    lines = ["Context: My card was declined at the store.", "Which team should handle this?",
             "(A) billing (B) technical (C) sales (D) fraud (E) account", "Answer 1: ("]
    p = prog(f, "prompt")
    ax.text(0.45, 8.55, "Prompt", fontsize=11.5, color=INK, fontweight="bold", va="center")
    for k, line in enumerate(lines):
        frac = min(max(p * len(lines) - k, 0), 1)
        if frac <= 0: continue
        txt = line[: max(1, int(len(line) * frac))]
        is_slot = k == len(lines) - 1
        ax.text(0.45, 8.1 - k * 0.38, txt, fontsize=9.5, family="monospace", color=AMBER if is_slot else INK2,
                fontweight="bold" if is_slot else "normal", va="center")

    # ---- hidden-state matrix (left), filled top to bottom
    mx, my, mw, mh = 0.45, 1.0, 3.0, 4.95
    pp = ease(prog(f, "pass"))
    if f >= STARTS["pass"][0]:
        ax.text(mx, my + mh + 0.3, "Hidden states, 1,000 × d", ha="left", fontsize=11.5, color=INK, fontweight="bold")
        ax.add_patch(Rectangle((mx, my), mw, mh, fc=SURF, ec=AXIS, lw=1.2))
        n_rows = 40; filled = int(n_rows * pp)
        for r in range(filled):
            y = my + mh - (r + 1) * mh / n_rows
            shade = 0.80 + 0.12 * ((r * 7) % 5) / 5
            ax.add_patch(Rectangle((mx + 0.05, y + 0.02), mw - 0.1, mh / n_rows - 0.04, fc=str(shade), ec="none"))
        if 0 < pp < 1:
            yb = my + mh - filled * mh / n_rows
            ax.plot([mx, mx + mw], [yb, yb], color=BLUE, lw=2)
        ax.text(mx + mw / 2, my + mh - 0.25, "token 1", ha="center", fontsize=9, color=INK2)
        ax.text(mx + mw / 2, my + 0.18, "token 1,000", ha="center", fontsize=9, color=INK2)

    # ---- slot row highlight
    slot_y = my + 0.62
    ps = prog(f, "slot")
    if ps > 0:
        a = min(1, ps * 2)
        ax.add_patch(Rectangle((mx, slot_y - 0.09), mw, 0.18, fc=AMBER, ec="none", alpha=a, zorder=3))
        ax.text(mx + mw + 0.12, slot_y - 0.34, "answer slot", fontsize=9.5, color=AMBER, va="center", alpha=a, fontweight="bold")

    # ---- slot box
    sbx, sby, sbw, sbh = 5.0, 2.55, 2.6, 1.05
    draw_line(ax, [(mx + mw, slot_y), (4.6, slot_y), (4.6, sby + sbh / 2), (sbx, sby + sbh / 2)], ease(prog(f, "to_slotbox")), AMBER)
    if prog(f, "to_slotbox") >= 1:
        box(ax, sbx, sby, sbw, sbh, "Slot row", "1 × d vector", AMBER, AMBER_FILL)

    # ---- vocabulary head and letter rows
    pv = prog(f, "vocab")
    vx, vy, vw, vh = 5.0, 4.6, 2.6, 1.05
    lx, ly, lw_, lh = 8.2, 4.6, 2.6, 1.05
    if pv > 0:
        box(ax, vx, vy, vw, vh, "Vocabulary head", "≈150,000 × d", VIOLET, VIOLET_FILL, alpha=min(1, pv * 3))
        draw_line(ax, [(vx + vw, vy + vh / 2), (lx, ly + vh / 2)], ease(min(1, max(0, pv * 2 - 0.5))), VIOLET)
        if pv > 0.75:
            box(ax, lx, ly, lw_, lh, "Rows A to E", "5 × d", VIOLET, VIOLET_FILL, alpha=min(1, (pv - 0.75) * 4))

    # ---- dot product box, fed by both
    dx, dy, dw, dh = 8.2, 2.55, 2.6, 1.05
    pd_ = ease(prog(f, "to_dot"))
    draw_line(ax, [(sbx + sbw, sby + sbh / 2), (dx, dy + dh / 2)], pd_, AMBER)
    draw_line(ax, [(lx + lw_ / 2, ly), (dx + dw / 2, dy + dh)], pd_, VIOLET)
    if pd_ >= 1:
        box(ax, dx, dy, dw, dh, "Dot products", "slot · letter rows", INK2, "#f1efe8")

    # ---- bar panel (right)
    px0, base, pw = 11.15, 2.9, 4.2
    sc, tp, sm = ease(prog(f, "scores")), ease(prog(f, "temp")), ease(prog(f, "softmax"))
    if f >= STARTS["scores"][0]:
        draw_line(ax, [(dx + dw, dy + dh / 2), (px0 - 0.15, dy + dh / 2)], min(1, sc * 3), INK2)
        kS, kP = 0.55, 3.6        # screen units per score / per probability
        vals = SCORES * kS * sc
        if tp > 0: vals = (SCORES + (SCALED - SCORES) * tp) * kS
        if sm > 0: vals = SCALED * kS + (PROBS * kP - SCALED * kS) * sm
        label = "probability" if sm >= 1 else "score ÷ T" if tp >= 1 else "score"
        head = {"score": "Scores", "score ÷ T": "Scores ÷ T (T = 1.164)", "probability": "Probabilities, sum = 100%"}[label]
        ax.text(px0 + pw / 2, 6.6, head, ha="center", fontsize=11.5, color=INK, fontweight="bold")
        ax.plot([px0 - 0.1, px0 + pw], [base, base], color=AXIS, lw=1.2)
        bw = pw / 5 * 0.62
        win = sm >= 1 and f >= STARTS["choice"][0]
        for i in range(5):
            cx = px0 + (i + 0.5) * pw / 5
            col = BLUE
            if win: col = BLUE if i == 0 else "#b7d3f6"
            h = vals[i]
            ax.add_patch(Rectangle((cx - bw / 2, base if h >= 0 else base + h), bw, abs(h), fc=col, ec="none"))
            shown = PROBS[i] if sm >= 1 else (SCALED[i] if tp >= 1 else SCORES[i])
            if sc >= 1 and (sm >= 1 or tp >= 1 or tp == 0):
                txt = f"{shown:.0%}" if sm >= 1 else f"{shown:+.2f}" if tp >= 1 else f"{shown:+.1f}"
                if 0 < tp < 1 or 0 < sm < 1: txt = ""
                ax.text(cx, (base + h + 0.18) if h >= 0 else (base + h - 0.22), txt, ha="center", fontsize=9.5, color=INK)
            ax.text(cx, base - 1.5, LETTERS[i], ha="center", fontsize=11, color=INK, fontweight="bold")
        ax.text(px0 + pw / 2, base - 1.95, "A billing · B technical · C sales · D fraud · E account", ha="center", fontsize=8, color=INK2)
        if win:
            a = min(1, (f - STARTS["choice"][0]) / 8)
            ax.text(px0 + pw / 2, 7.25, f"choice: A, billing ({PROBS[0]:.0%})", ha="center", fontsize=13, color=BLUE, fontweight="bold", alpha=a)

    # ---- caption
    ax.add_patch(Rectangle((0, 0), W, 0.62, fc="#f1efe8", ec="none"))
    ax.text(0.45, 0.31, caption_at(f), fontsize=11.5, color=INK, va="center")

    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=SURF); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB")

if __name__ == "__main__":
    frames = [frame(f) for f in range(N_FRAMES)]
    pal = frames[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in frames]
    durations = [int(1000 / FPS)] * len(q); durations[-1] = 3500
    q[0].save("figures/decider_readout.gif", save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True)
    frames[-1].save("figures/decider_readout_final.png")
    for name in ("pass", "vocab", "temp"):
        s, n = STARTS[name]; frames[s + n // 2].save(f"figures/_check_{name}.png")
    print(N_FRAMES, "frames")
