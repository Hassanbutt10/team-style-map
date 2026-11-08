"""
Team Style Map - Cycle 11, Day 3 project
Can a computer group the 16 Copa America 2024 teams by how they play, and do
the groups survive when the matches change? Each team becomes one point in an
8-feature style space, is grouped with k-means, drawn on a 2D map, and then
the whole thing is repeated 1,000 times on re-sampled matches to see how
stable the groups are.

Method:
  1. Periods 1 and 2 only (90 minutes plus stoppage).
  2. Eight style features per team, from event data:
       pass_share   share of both teams' passes (a possession proxy)
       long_pct     % of passes that travel 30 yards or more
       forward_pct  % of passes that gain 10 yards or more towards goal
       pass_x       average pass start, yards from own goal line
       press_x      average pressure position, yards from own goal line
       press_rate   pressures per 100 opponent passes
       cross_pct    % of passes that are crosses
       carry_rate   carries per 100 passes
  3. Features are z-scored across teams. PCA (NumPy SVD) gives the 2D map.
  4. k-means (NumPy, k-means++ starts, 100 restarts) with k = 2 to 5. The k
     with the best silhouette score is used.
  5. Stability: re-sample each team's matches with replacement 1,000 times,
     rebuild the features, re-cluster with the same k, and compare with the
     original groups using the adjusted Rand index (1 = identical, 0 = no
     better than chance). The share of resamples in which two teams share a
     group is drawn as a heat map.
  6. Outcomes (xG for and against per match) are shown next to the groups but
     are NOT used to build them.

Limitations: 16 teams and 3 to 6 matches each, so the groups can move a lot
with one match. Features describe how a team played against its own
opponents and scorelines, not how it would play in general. Group names are
not given: a group is a summary of the features chosen, and a different set
of features can give different groups. Resampling matches shows how stable
the groups are, not whether they are right.

Outputs: team_style_map.png, team_style_stability.png,
team_style_replay.mp4 (clustering and resampling replay plus the two charts
as end cards), team_style_features.csv

Reuses: copa_america_2024_matches.csv, copa_america_2024_events.csv
"""

import itertools
import os
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

N_BOOT = 1000
SEED = 3
K_RANGE = range(2, 6)
LONG_PASS = 30
FORWARD_GAIN = 10
N_VIDEO_BOOT = 10
END_CARDS = ["team_style_map.png", "team_style_stability.png"]
END_CARD_SECONDS = 4

BG = "#0d1b2a"
GRID = "#3a4a5a"
BLUE = "#4a90d9"
ORANGE = "#f4a300"
MUTED = "#a8b8c8"
GREEN = "#6fcf97"
RED = "#e76f51"
PURPLE = "#b39ddb"
GROUP_COLORS = [BLUE, ORANGE, GREEN, RED, PURPLE]

TEAM_CODES = {
    "Argentina": "ARG", "Brazil": "BRA", "Colombia": "COL", "Uruguay": "URU", "Canada": "CAN",
    "Venezuela": "VEN", "Ecuador": "ECU", "Panama": "PAN", "Mexico": "MEX", "United States": "USA",
    "Chile": "CHI", "Peru": "PER", "Paraguay": "PAR", "Bolivia": "BOL", "Jamaica": "JAM",
    "Costa Rica": "CRC"}
FEATS = ["pass_share", "long_pct", "forward_pct", "pass_x", "press_x", "press_rate", "cross_pct", "carry_rate"]
FEAT_NAME = {"pass_share": "possession share", "long_pct": "long passes", "forward_pct": "forward passes",
             "pass_x": "pass height", "press_x": "pressing height", "press_rate": "pressing rate",
             "cross_pct": "crosses", "carry_rate": "carries"}
FEAT_ARROW = {"pass_share": "possession", "long_pct": "long", "forward_pct": "forward", "pass_x": "pass height",
              "press_x": "press height", "press_rate": "press rate", "cross_pct": "crosses", "carry_rate": "carries"}

matches = pd.read_csv("copa_america_2024_matches.csv")
events = pd.read_csv("copa_america_2024_events.csv", low_memory=False)
events = events[events["period"].isin([1, 2])].copy()
for col in ["pass_length", "pass_cross", "location", "pass_end_location", "shot_type",
            "shot_statsbomb_xg", "shot_outcome"]:
    if col not in events.columns:
        events[col] = np.nan


def parse(col, a, b):
    xy = events[col].astype(str).str.extract(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)")
    events[a] = pd.to_numeric(xy[0], errors="coerce")
    events[b] = pd.to_numeric(xy[1], errors="coerce")


parse("location", "lx", "ly")
parse("pass_end_location", "ex", "ey")
events["is_cross"] = events["pass_cross"].astype(str).str.lower().eq("true")

# ---------- counts per team and match ----------
pairs = pd.concat([matches[["match_id", "home_team", "away_team"]].rename(
    columns={"home_team": "team", "away_team": "opp"}),
    matches[["match_id", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opp"})], ignore_index=True)
key = ["team", "match_id"]
p = events[events["type"] == "Pass"]
cnt = pd.DataFrame(index=pd.MultiIndex.from_frame(pairs[key]))
cnt["n_pass"] = p.groupby(key).size()
cnt["n_long"] = p[p["pass_length"] >= LONG_PASS].groupby(key).size()
cnt["n_fwd"] = p[(p["ex"] - p["lx"]) >= FORWARD_GAIN].groupby(key).size()
cnt["n_cross"] = p[p["is_cross"]].groupby(key).size()
cnt["sum_px"] = p[p["lx"].notna()].groupby(key)["lx"].sum()
cnt["n_px"] = p[p["lx"].notna()].groupby(key).size()
pr = events[events["type"] == "Pressure"]
cnt["n_press"] = pr.groupby(key).size()
cnt["sum_prx"] = pr[pr["lx"].notna()].groupby(key)["lx"].sum()
cnt["n_prx"] = pr[pr["lx"].notna()].groupby(key).size()
cnt["n_carry"] = events[events["type"] == "Carry"].groupby(key).size()
sh = events[(events["type"] == "Shot") & events["shot_statsbomb_xg"].notna()
            & (events["shot_type"].fillna("") != "Penalty")]
cnt["xg_for"] = sh.groupby(key)["shot_statsbomb_xg"].sum()
cnt = cnt.fillna(0.0)
opp_map = pairs.set_index(key)["opp"]
opp_idx = pd.MultiIndex.from_arrays([opp_map.to_numpy(), cnt.index.get_level_values("match_id")])
cnt["opp_pass"] = cnt["n_pass"].reindex(opp_idx).to_numpy()
cnt["xg_against"] = cnt["xg_for"].reindex(opp_idx).to_numpy()
cnt = cnt.fillna(0.0)
COLS = list(cnt.columns)
teams = sorted(cnt.index.get_level_values("team").unique())
nT = len(teams)
team_rows = {t: np.where(cnt.index.get_level_values("team") == t)[0] for t in teams}
CNT = cnt.to_numpy(dtype=float)
ci = {c: i for i, c in enumerate(COLS)}


def safe(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def features_from(S):
    """S: array (teams x counts) of summed counts -> (teams x 8) feature matrix."""
    g = lambda c: S[:, ci[c]]
    F = np.column_stack([
        safe(g("n_pass"), g("n_pass") + g("opp_pass")) * 100,
        safe(g("n_long"), g("n_pass")) * 100,
        safe(g("n_fwd"), g("n_pass")) * 100,
        safe(g("sum_px"), g("n_px")),
        safe(g("sum_prx"), g("n_prx")),
        safe(g("n_press"), g("opp_pass")) * 100,
        safe(g("n_cross"), g("n_pass")) * 100,
        safe(g("n_carry"), g("n_pass")) * 100])
    col_mean = np.nanmean(F, axis=0)
    return np.where(np.isnan(F), col_mean, F)


S0 = np.vstack([CNT[team_rows[t]].sum(axis=0) for t in teams])
X0 = features_from(S0)
n_matches = np.array([len(team_rows[t]) for t in teams])


# ---------- NumPy PCA, k-means, silhouette, ARI ----------
def zscore(X):
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1)


def kmeans(Z, k, rng, n_init=30, max_iter=100, record=False):
    n = len(Z)
    best = None
    for _ in range(n_init):
        idx = [int(rng.integers(n))]
        for _ in range(1, k):
            d2 = np.min(((Z[:, None, :] - Z[idx][None, :, :]) ** 2).sum(2), axis=1)
            pr_ = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
            idx.append(int(rng.choice(n, p=pr_)))
        C = Z[idx].copy()
        labels, hist = None, []
        for _ in range(max_iter):
            d = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(2)
            new = d.argmin(1)
            if record:
                hist.append((new.copy(), C.copy()))
            if labels is not None and (new == labels).all():
                break
            labels = new
            C = np.array([Z[labels == j].mean(0) if (labels == j).any() else C[j] for j in range(k)])
        inertia = float(((Z - C[labels]) ** 2).sum())
        if best is None or inertia < best[0] - 1e-12:
            best = (inertia, labels.copy(), C.copy(), hist)
    return best


def silhouette(Z, labels):
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(2))
    s = np.zeros(len(Z))
    for i in range(len(Z)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == c].mean() for c in set(labels.tolist()) if c != labels[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


def ari(a, b):
    n = len(a)
    ca, cb = sorted(set(a.tolist())), sorted(set(b.tolist()))
    M = np.array([[np.sum((a == x) & (b == y)) for y in cb] for x in ca], dtype=float)
    c2 = lambda v: v * (v - 1) / 2
    sum_ij = c2(M).sum()
    sa, sb = c2(M.sum(1)).sum(), c2(M.sum(0)).sum()
    tot = c2(n)
    exp = sa * sb / tot
    den = 0.5 * (sa + sb) - exp
    return 1.0 if den == 0 else float((sum_ij - exp) / den)


rng = np.random.default_rng(SEED)
Z0 = zscore(X0)
print(f"\n{nT} teams, {int(n_matches.sum())} team-matches ({n_matches.min()} to {n_matches.max()} matches per team)")

U, Sv, Vt = np.linalg.svd(Z0, full_matrices=False)
V2 = Vt[:2].T
for j in range(2):
    if V2[np.abs(V2[:, j]).argmax(), j] < 0:
        V2[:, j] *= -1
scores = Z0 @ V2
expl = Sv ** 2 / np.sum(Sv ** 2)
print(f"PCA: PC1 {expl[0] * 100:.0f}% and PC2 {expl[1] * 100:.0f}% of the variance in the 8 features")
print("Loadings (PC1, PC2):")
for f, l1, l2 in zip(FEATS, V2[:, 0], V2[:, 1]):
    print(f"  {f:<12}{l1:+.2f} {l2:+.2f}")

print("\nChoosing k by silhouette score:")
best_k, best_sil, fits = None, -9, {}
for k in K_RANGE:
    fit = kmeans(Z0, k, rng, n_init=100, record=True)
    sil = silhouette(Z0, fit[1])
    fits[k] = fit
    print(f"  k={k}: silhouette {sil:.3f}")
    if sil > best_sil:
        best_k, best_sil = k, sil
K = best_k
_, lab_raw, C_raw, hist_raw = fits[K]
sizes = np.bincount(lab_raw, minlength=K)
order_ids = sorted(range(K), key=lambda c: (-sizes[c], scores[lab_raw == c, 0].mean()))
mp = np.zeros(K, dtype=int)
for new, old in enumerate(order_ids):
    mp[old] = new
labels0 = mp[lab_raw]
inv = np.argsort(mp)
hist = [(mp[l], C[inv]) for (l, C) in hist_raw]
print(f"Using k={K} (silhouette {best_sil:.3f}; below about 0.25 means weak structure)")

# ---------- stability by re-sampling matches ----------
co = np.zeros((nT, nT))
aris, boot_labels = [], []
perms = list(itertools.permutations(range(K)))


def align(lbl, ref):
    best, bp = -1, None
    for pm in perms:
        m = np.array(pm)[lbl]
        agree = int((m == ref).sum())
        if agree > best:
            best, bp = agree, m
    return bp


for b in range(N_BOOT):
    Sb = np.vstack([CNT[rng.choice(team_rows[t], len(team_rows[t]), replace=True)].sum(axis=0) for t in teams])
    Zb = zscore(features_from(Sb))
    lb = kmeans(Zb, K, rng, n_init=10)[1]
    co += (lb[:, None] == lb[None, :])
    aris.append(ari(labels0, lb))
    if b < N_VIDEO_BOOT:
        boot_labels.append(align(lb, labels0))
co /= N_BOOT
aris = np.array(aris)
stay = np.full(nT, np.nan)
for i in range(nT):
    mates = [j for j in range(nT) if j != i and labels0[j] == labels0[i]]
    if mates:
        stay[i] = co[i, mates].mean()
print(f"\nStability over {N_BOOT} resamples of each team's matches:")
print(f"  adjusted Rand index vs the original groups: mean {aris.mean():.2f}, median {np.median(aris):.2f}, "
      f"5th to 95th {np.percentile(aris, 5):.2f} to {np.percentile(aris, 95):.2f}")
print(f"  resamples with ARI of at least 0.7: {(aris >= 0.7).mean() * 100:.0f}%; at least 0.5: {(aris >= 0.5).mean() * 100:.0f}%")

codes = [TEAM_CODES.get(t, t[:3].upper()) for t in teams]
xg_pm = cnt.groupby(level="team")[["xg_for", "xg_against"]].sum().reindex(teams)
xg_pm = xg_pm.div(n_matches, axis=0)
tab = pd.DataFrame(X0, columns=FEATS, index=teams)
tab.insert(0, "matches", n_matches)
tab["group"] = labels0 + 1
tab["pc1"], tab["pc2"] = scores[:, 0], scores[:, 1]
tab["stays_with_group"] = stay
tab["xg_for_pm"], tab["xg_against_pm"] = xg_pm["xg_for"], xg_pm["xg_against"]
tab.sort_values(["group", "pc1"]).to_csv("team_style_features.csv")

print("\nGroups (xG columns are outcomes, not used to build the groups):")
for c in range(K):
    mem = [i for i in range(nT) if labels0[i] == c]
    zc = Z0[mem].mean(0)
    hi = [FEAT_NAME[FEATS[j]] for j in np.argsort(-zc)[:2] if zc[j] > 0.3]
    lo = [FEAT_NAME[FEATS[j]] for j in np.argsort(zc)[:2] if zc[j] < -0.3]
    print(f"  Group {c + 1} ({len(mem)} teams): {', '.join(teams[i] for i in mem)}")
    print(f"     higher: {', '.join(hi) or 'none'}; lower: {', '.join(lo) or 'none'}; "
          f"xG for {xg_pm['xg_for'].iloc[mem].mean():.2f}, against {xg_pm['xg_against'].iloc[mem].mean():.2f} per match")
print("\nHow often each team stays with its group-mates (1.00 = always):")
for i in np.argsort(-np.nan_to_num(stay, nan=-1)):
    if not np.isnan(stay[i]):
        print(f"  {teams[i]:<14}group {labels0[i] + 1}  {stay[i]:.2f}  ({n_matches[i]} matches)")
single = [teams[i] for i in range(nT) if np.isnan(stay[i])]
if single:
    print("  Alone in their group:", ", ".join(single))


# ---------- label placement ----------
def place_labels(fig, ax, xs, ys, labels, sizes, bold=None):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    box = ax.get_window_extent()
    placed, result = [], {}
    crowd = [sum(np.hypot(*(pts[i] - pts[j])) < 70 for j in range(len(xs)) if j != i) for i in range(len(xs))]
    for i in sorted(range(len(xs)), key=lambda k: -crowd[k]):
        t = ax.text(0, 0, labels[i], fontsize=sizes[i], fontweight="bold" if (bold and bold[i]) else "normal")
        bb = t.get_window_extent(rend)
        w, h = bb.width, bb.height
        t.remove()
        cands = [(-w / 2, 9), (-w / 2, -9 - h), (9, -h / 2), (-9 - w, -h / 2),
                 (8, 7), (-8 - w, 7), (8, -7 - h), (-8 - w, -7 - h),
                 (-w / 2, 20), (-w / 2, -20 - h), (20, -h / 2), (-20 - w, -h / 2)]
        best = None
        for cj, (dx, dy) in enumerate(cands):
            x0, y0 = pts[i][0] + dx, pts[i][1] + dy
            pen = cj * 0.5
            if x0 < box.x0 + 2 or x0 + w > box.x1 - 2 or y0 < box.y0 + 2 or y0 + h > box.y1 - 2:
                pen += 200
            for (a0, b0, a1, b1) in placed:
                ox = max(0.0, min(x0 + w, a1) - max(x0, a0))
                oy = max(0.0, min(y0 + h, b1) - max(y0, b0))
                pen += 3 * ox * oy
            for j in range(len(xs)):
                px_, py_ = pts[j]
                ox = max(0.0, min(x0 + w, px_ + 7) - max(x0, px_ - 7))
                oy = max(0.0, min(y0 + h, py_ + 7) - max(y0, py_ - 7))
                pen += 2 * ox * oy
            if best is None or pen < best[0]:
                best = (pen, dx, dy, x0, y0, w, h)
        _, dx, dy, x0, y0, w, h = best
        placed.append((x0, y0, x0 + w, y0 + h))
        result[i] = (dx, dy)
    return result


xr = scores[:, 0].max() - scores[:, 0].min()
yr = scores[:, 1].max() - scores[:, 1].min()
XLIM = (scores[:, 0].min() - 0.18 * xr, scores[:, 0].max() + 0.18 * xr)
YLIM = (scores[:, 1].min() - 0.22 * yr, scores[:, 1].max() + 0.22 * yr)
arrow_scale = 0.62 * min(xr, yr) / max(np.linalg.norm(V2, axis=1).max(), 1e-9)
arrow_tips = V2 * arrow_scale

# ---------- chart 1: the map ----------
fig1 = plt.figure(figsize=(11, 9), dpi=150)
fig1.patch.set_facecolor(BG)
ax1 = fig1.add_axes([0.08, 0.31, 0.88, 0.57])
ax1.set_facecolor(BG)
ax1.set_xlim(*XLIM)
ax1.set_ylim(*YLIM)
for j, f in enumerate(FEATS):
    ax1.annotate("", xy=(arrow_tips[j, 0], arrow_tips[j, 1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color=GRID, linewidth=1.1), zorder=1)
for c in range(K):
    m = labels0 == c
    ax1.scatter(scores[m, 0], scores[m, 1], s=120, color=GROUP_COLORS[c], edgecolors="white", linewidths=1, zorder=4)
ax1.axhline(0, color=GRID, linewidth=0.6, zorder=0)
ax1.axvline(0, color=GRID, linewidth=0.6, zorder=0)
ax1.set_xlabel(f"PC1 ({expl[0] * 100:.0f}% of the variance in the 8 features)", color="white", fontsize=10.5)
ax1.set_ylabel(f"PC2 ({expl[1] * 100:.0f}%)", color="white", fontsize=10.5)
ax1.tick_params(colors="white", labelsize=9)
for sp in ax1.spines.values():
    sp.set_color(GRID)
all_x = list(scores[:, 0]) + list(arrow_tips[:, 0])
all_y = list(scores[:, 1]) + list(arrow_tips[:, 1])
all_lab = codes + [FEAT_ARROW[f] for f in FEATS]
all_sz = [11] * nT + [9] * len(FEATS)
all_bold = [True] * nT + [False] * len(FEATS)
offs1 = place_labels(fig1, ax1, all_x, all_y, all_lab, all_sz, all_bold)
for i, lab in enumerate(all_lab):
    is_team = i < nT
    ax1.annotate(lab, xy=(all_x[i], all_y[i]), xytext=offs1[i], textcoords="offset pixels", ha="left", va="bottom",
                 color="white" if is_team else MUTED, fontsize=all_sz[i], fontweight="bold" if is_team else "normal",
                 fontstyle="normal" if is_team else "italic", zorder=6)
fig1.text(0.5, 0.955, "Team Style Map: How Copa America 2024 Teams Group by Play", ha="center", color="white",
          fontsize=17, fontweight="bold")
fig1.text(0.5, 0.918, f"16 teams, 8 style features, k-means with k = {K} (silhouette {best_sil:.2f}). "
          f"Grey arrows show which features push a team that way", ha="center", color=MUTED, fontsize=10)
yy = 0.255
fig1.text(0.04, yy, "Groups (the colours carry no name)", color="white", fontsize=11, fontweight="bold")
yy -= 0.04
for c in range(K):
    mem = [i for i in range(nT) if labels0[i] == c]
    zc = Z0[mem].mean(0)
    hi = [FEAT_NAME[FEATS[j]] for j in np.argsort(-zc)[:2] if zc[j] > 0.3]
    lo = [FEAT_NAME[FEATS[j]] for j in np.argsort(zc)[:2] if zc[j] < -0.3]
    fig1.text(0.04, yy, f"Group {c + 1}  ({len(mem)} teams)", color=GROUP_COLORS[c], fontsize=10.5, fontweight="bold")
    fig1.text(0.21, yy, f"higher: {', '.join(hi) or 'none'};  lower: {', '.join(lo) or 'none'}", color=MUTED, fontsize=10)
    fig1.text(0.04, yy - 0.022, ", ".join(codes[i] for i in sorted(mem, key=lambda i: scores[i, 0])), color="white",
              fontsize=9.5)
    yy -= 0.052
fig1.text(0.5, 0.012, "A 2D view of 8 features, so distances on the map are approximate. Based on the first 90 minutes.",
          ha="center", color=MUTED, fontsize=9)
fig1.savefig("team_style_map.png", dpi=150, facecolor=BG)
plt.close(fig1)
print("\nSaved chart: team_style_map.png")

# ---------- chart 2: stability heat map ----------
ordr = sorted(range(nT), key=lambda i: (labels0[i], scores[i, 0]))
Mo = co[np.ix_(ordr, ordr)]
cmap = LinearSegmentedColormap.from_list("sty", ["#10253a", BLUE, "#ffffff"])
fig2 = plt.figure(figsize=(10, 9), dpi=150)
fig2.patch.set_facecolor(BG)
ax2 = fig2.add_axes([0.14, 0.14, 0.66, 0.70])
im = ax2.imshow(Mo, cmap=cmap, vmin=0, vmax=1, aspect="equal")
ax2.set_xticks(range(nT))
ax2.set_yticks(range(nT))
ax2.set_xticklabels([codes[i] for i in ordr], rotation=90, color="white", fontsize=9)
ax2.set_yticklabels([codes[i] for i in ordr], color="white", fontsize=9)
ax2.tick_params(length=0)
bounds = np.cumsum([int((labels0 == c).sum()) for c in range(K)])[:-1]
for bnd in bounds:
    ax2.axhline(bnd - 0.5, color=ORANGE, linewidth=1.4)
    ax2.axvline(bnd - 0.5, color=ORANGE, linewidth=1.4)
for sp in ax2.spines.values():
    sp.set_color(GRID)
cax = fig2.add_axes([0.84, 0.14, 0.025, 0.70])
cb = fig2.colorbar(im, cax=cax)
cb.ax.tick_params(colors="white", labelsize=9)
cb.outline.set_edgecolor(GRID)
cb.set_label("Share of resamples where both teams share a group", color="white", fontsize=9.5)
fig2.text(0.5, 0.945, "How Stable Are the Groups?", ha="center", color="white", fontsize=17, fontweight="bold")
fig2.text(0.5, 0.905, f"Each team's matches re-sampled {N_BOOT:,} times, features rebuilt, teams re-grouped (k = {K}). "
          f"Orange lines separate the original groups", ha="center", color=MUTED, fontsize=9.5)
fig2.text(0.5, 0.07, f"Agreement with the original groups (adjusted Rand index): mean {aris.mean():.2f}, "
          f"5th to 95th percentile {np.percentile(aris, 5):.2f} to {np.percentile(aris, 95):.2f}",
          ha="center", color="white", fontsize=10.5)
fig2.text(0.5, 0.04, f"ARI of 1 means identical groups, near 0 means no better than chance. At least 0.7 in "
          f"{(aris >= 0.7).mean() * 100:.0f}% of resamples.", ha="center", color=MUTED, fontsize=9.5)
fig2.savefig("team_style_stability.png", dpi=150, facecolor=BG)
plt.close(fig2)
print("Saved chart: team_style_stability.png")

# ---------- replay: k-means rounds, then re-sampling ----------
fig = plt.figure(figsize=(10, 10), dpi=108)  # 1080 x 1080 pixels
fig.patch.set_facecolor(BG)
axm = fig.add_axes([0.07, 0.30, 0.86, 0.52])
fig.text(0.5, 0.965, "How a Computer Groups 16 Teams by Style", ha="center", color="white",
         fontsize=23, fontweight="bold")
fig.text(0.5, 0.928, "Copa America 2024, 8 style features, first 90 minutes", ha="center", color=MUTED, fontsize=14)
step_t = fig.text(0.5, 0.885, "", ha="center", color=ORANGE, fontsize=17, fontweight="bold")
info1_t = fig.text(0.5, 0.235, "", ha="center", color="white", fontsize=19, fontweight="bold")
info2_t = fig.text(0.5, 0.19, "", ha="center", color=MUTED, fontsize=14)
fig.text(0.5, 0.012, "StatsBomb open data, Copa America 2024. Grouping uses all 8 features, the map shows 2 of them combined.",
         ha="center", color=MUTED, fontsize=9)
for c in range(K):
    fig.text(0.5 + (c - (K - 1) / 2) * 0.16, 0.135, f"Group {c + 1}", ha="center", color=GROUP_COLORS[c],
             fontsize=14, fontweight="bold")
axm.set_xlim(*XLIM)
axm.set_ylim(*YLIM)
offs_v = place_labels(fig, axm, list(scores[:, 0]), list(scores[:, 1]), codes, [11] * nT, [True] * nT)


def ease(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def proj(C):
    return C @ V2


def draw_state(st):
    axm.clear()
    axm.set_facecolor(BG)
    axm.set_xlim(*XLIM)
    axm.set_ylim(*YLIM)
    axm.axhline(0, color=GRID, linewidth=0.6)
    axm.axvline(0, color=GRID, linewidth=0.6)
    axm.set_xticks([])
    axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_color(GRID)
    lab = st.get("labels")
    cents = st.get("cents")
    if lab is not None and cents is not None and st.get("lines"):
        for i in range(nT):
            axm.plot([scores[i, 0], cents[lab[i], 0]], [scores[i, 1], cents[lab[i], 1]],
                     color=GROUP_COLORS[lab[i]], linewidth=1, alpha=0.35, zorder=2)
    cols = [MUTED] * nT if lab is None else [GROUP_COLORS[l] for l in lab]
    ec = ["white"] * nT
    lw = [1.0] * nT
    for i in st.get("changed", []):
        ec[i] = "white"
        lw[i] = 3.2
    axm.scatter(scores[:, 0], scores[:, 1], s=190, c=cols, edgecolors=ec, linewidths=lw, zorder=4)
    if cents is not None:
        for c in range(K):
            axm.scatter([cents[c, 0]], [cents[c, 1]], s=260, marker="X", color=GROUP_COLORS[c],
                        edgecolors=BG, linewidths=1.6, zorder=6)
    for i in range(nT):
        axm.annotate(codes[i], xy=(scores[i, 0], scores[i, 1]), xytext=offs_v[i], textcoords="offset pixels",
                     ha="left", va="bottom", color="white", fontsize=11, fontweight="bold", zorder=7)
    step_t.set_text(st["step"])
    info1_t.set_text(st.get("info1", ""))
    info2_t.set_text(st.get("info2", ""))


states = []
for _ in range(10):
    states.append(dict(labels=None, cents=None, step="Step 1: every team is one point in style space"))
n_it = min(len(hist), 6)
for t in range(n_it):
    lab_t, C_t = hist[t]
    P_t = proj(C_t)
    for _ in range(10):
        states.append(dict(labels=lab_t, cents=P_t, lines=True,
                           step=f"Step 2: round {t + 1}, each team joins the nearest centre",
                           info1="Centres (X) are the average of their group", info2="Then the centres move to the new average"))
    if t + 1 < len(hist):
        P_n = proj(hist[t + 1][1])
        for f in range(8):
            e = ease((f + 1) / 8)
            states.append(dict(labels=lab_t, cents=P_t + (P_n - P_t) * e, lines=True,
                               step=f"Step 2: round {t + 1}, the centres move",
                               info1="Centres (X) are the average of their group", info2="Then every team is checked again"))
for _ in range(12):
    states.append(dict(labels=labels0, cents=None,
                       step="The groups stop changing", info1=f"{K} groups found (silhouette {best_sil:.2f})",
                       info2="Now: would they survive different matches?"))
for b, bl in enumerate(boot_labels):
    ch = [i for i in range(nT) if bl[i] != labels0[i]]
    for _ in range(12):
        states.append(dict(labels=bl, cents=None, changed=ch,
                           step=f"Step 3: re-sample each team's matches, group again ({b + 1} of {len(boot_labels)})",
                           info1=f"{len(ch)} of {nT} teams changed group (white rings)",
                           info2=f"Agreement with the original groups: ARI {ari(labels0, bl):.2f}"))
for _ in range(45):
    states.append(dict(labels=labels0, cents=None,
                       step=f"After {N_BOOT:,} re-samples",
                       info1=f"Average agreement (ARI): {aris.mean():.2f}, ARI 0.7 or more in {(aris >= 0.7).mean() * 100:.0f}%",
                       info2="1 means identical groups, near 0 means no better than chance"))

FPS = 20
RAW_VIDEO = "team_style_replay_raw.mp4"
FINAL_VIDEO = "team_style_replay.mp4"
anim = FuncAnimation(fig, lambda i: draw_state(states[i]), frames=len(states), interval=1000 / FPS)
writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
print(f"Rendering {len(states)} frames ({len(states) / FPS:.0f} seconds), this takes a minute...")
anim.save(RAW_VIDEO, writer=writer, dpi=108, savefig_kwargs={"facecolor": BG})
print("Saved replay:", RAW_VIDEO)


# ---------- append the charts to the video as end cards ----------
def append_end_cards(video_in, video_out, cards, seconds, size=1080, fps=20, bg="0d1b2a"):
    cards = [c for c in cards if os.path.exists(c)]
    if not cards:
        print("No chart PNGs found, keeping the replay without end cards.")
        os.replace(video_in, video_out)
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-i", video_in]
    for c in cards:
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", str(seconds), "-i", c]
    norm = (f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
            f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=0x{bg},"
            f"setsar=1,fps={fps},format=yuv420p")
    n = len(cards) + 1
    parts_f = [f"[{i}:v]{norm}[v{i}]" for i in range(n)]
    labels_f = "".join(f"[v{i}]" for i in range(n))
    fc = ";".join(parts_f) + f";{labels_f}concat=n={n}:v=1:a=0[out]"
    cmd += ["-filter_complex", fc, "-map", "[out]", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", video_out]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1500:])
        print("End cards failed, keeping the replay without them.")
        os.replace(video_in, video_out)
        return
    os.remove(video_in)
    print("Added end cards:", ", ".join(cards))


append_end_cards(RAW_VIDEO, FINAL_VIDEO, END_CARDS, END_CARD_SECONDS, fps=FPS)
print("Saved video:", FINAL_VIDEO)