"""Generate the 5 publication-quality figures for the WR Route Kinetics submission.

Outputs (300 DPI, white background, colorblind-safe palette) into ``figures/``:
  fig1_hero_palmer_vs_dike.png      Combine trajectories + NFL in-game speed density
  fig2_scouting_blindspot.png       Added-variable (partial regression) plots on N=75
  fig3_combine_to_nfl_bridge.png    Combine sensor -> NFL on-field translation
  fig4_lodco_cv_performance.png     Leave-One-Draft-Class-Out out-of-sample proof
  fig5_draft_room_quadrant.png      Draft-room scouting board (40-yd vs kinetics)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import polars as pl
from scipy import stats
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "figures"

BLUE = "#0072B2"
ORANGE = "#D55E00"
SKY = "#56B4E9"
GREEN = "#009E73"
GRAY = "#8A8A8A"
DARK = "#222222"
MID = "#555555"
LIGHT_GRID = "#E6E6E6"

DPI = 300
MIN_SNAPS = 50


def setup_style() -> None:
    """Apply a clean, FiveThirtyEight/The Athletic-like white style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.9,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.labelcolor": DARK,
            "xtick.color": MID,
            "ytick.color": MID,
            "axes.grid": True,
            "grid.color": LIGHT_GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def _load_module(name: str, path: Path) -> ModuleType:
    """Import a script whose filename starts with digits."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def add_titles(fig, title: str, subtitle: str, footer: str, top: float = 0.80) -> None:
    """Place headline (conclusion), subtitle (source / N) and footer without overlaps."""
    fig.text(0.012, 0.985, title, fontsize=17, fontweight="bold", color=DARK, va="top", ha="left")
    fig.text(0.012, 0.985 - 0.058, subtitle, fontsize=10.5, color=MID, va="top", ha="left")
    fig.text(0.012, 0.012, footer, fontsize=8.5, color=GRAY, va="bottom", ha="left")
    fig.subplots_adjust(top=top)


def save(fig, name: str) -> None:
    """Save a figure at 300 DPI to figures/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


def stars(p: float) -> str:
    """Significance marker."""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return ""


def load_modeling_frame(mod08: ModuleType) -> pl.DataFrame:
    """Join advanced kinetics, SOE targets, NFL game kinetics and football production."""
    adv = pl.read_parquet(PROCESSED_DIR / "wr_advanced_kinetics.parquet")
    targets = pl.read_parquet(PROCESSED_DIR / "wr_targets.parquet")
    game_kin = pl.read_parquet(PROCESSED_DIR / "wr_game_kinetics.parquet")
    prod = mod08.load_football_production(RAW_DIR)
    t_cols = [c for c in targets.columns if c not in ("display_name", "draft_year")]
    return (
        adv.join(targets.select(t_cols), on="nfl_id", how="inner")
        .join(game_kin, on="nfl_id", how="inner")
        .join(prod, on="nfl_id", how="left")
        .sort("nfl_id")
    )


def regression_line_ci(x: np.ndarray, y: np.ndarray, n_grid: int = 100):
    """OLS line with 95% confidence band."""
    X = sm.add_constant(x)
    fit = sm.OLS(y, X).fit()
    grid = np.linspace(x.min(), x.max(), n_grid)
    pred = fit.get_prediction(sm.add_constant(grid)).summary_frame(alpha=0.05)
    return grid, pred["mean"].to_numpy(), pred["mean_ci_lower"].to_numpy(), pred["mean_ci_upper"].to_numpy()


def place_labels(fig, ax, pts, names, colors, all_pts, fontsize: float = 9.5) -> None:
    """Annotate points and iteratively repel labels so none overlap each other or dots."""
    anns = []
    for (x, y), name, col in zip(pts, names, colors):
        ann = ax.annotate(
            name,
            xy=(x, y),
            xytext=(16, 14),
            textcoords="offset points",
            fontsize=fontsize,
            fontweight="bold",
            color=col,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "none", "alpha": 0.9},
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.8, "shrinkA": 0, "shrinkB": 3},
            zorder=12,
        )
        anns.append(ann)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = [Text.get_window_extent(a, rend) for a in anns]
    c = np.array([[b.x0 + b.width / 2, b.y0 + b.height / 2] for b in boxes], dtype=float)
    c0 = c.copy()
    w = np.array([b.width for b in boxes]) + 6.0
    h = np.array([b.height for b in boxes]) + 4.0
    pd_ = ax.transData.transform(np.asarray(all_pts, dtype=float))
    axbb = ax.get_window_extent()
    for _ in range(500):
        force = np.zeros_like(c)
        for i in range(len(c)):
            force[i] += 0.01 * (c0[i] - c[i])
            for j in range(len(c)):
                if i == j:
                    continue
                dx, dy = c[i] - c[j]
                ox = (w[i] + w[j]) / 2 - abs(dx)
                oy = (h[i] + h[j]) / 2 - abs(dy)
                if ox > 0 and oy > 0:
                    if ox < oy:
                        force[i, 0] += np.sign(dx if dx != 0 else 1.0) * ox * 0.5
                    else:
                        force[i, 1] += np.sign(dy if dy != 0 else 1.0) * oy * 0.5
            for p in pd_:
                dx, dy = c[i] - p
                ox = w[i] / 2 + 5 - abs(dx)
                oy = h[i] / 2 + 5 - abs(dy)
                if ox > 0 and oy > 0:
                    if ox < oy:
                        force[i, 0] += np.sign(dx if dx != 0 else 1.0) * ox * 0.4
                    else:
                        force[i, 1] += np.sign(dy if dy != 0 else 1.0) * oy * 0.4
        c += force
        c[:, 0] = np.clip(c[:, 0], axbb.x0 + w / 2, axbb.x1 - w / 2)
        c[:, 1] = np.clip(c[:, 1], axbb.y0 + h / 2, axbb.y1 - h / 2)
    inv = ax.transData.inverted()
    data_pos = inv.transform(c)
    for ann, (x, y) in zip(anns, data_pos):
        ann.anncoords = "data"
        ann.xyann = (x, y)


# ------------------------------------------------------------------ FIGURE 1
def extract_player_frames(mod06: ModuleType, ids: list[int]) -> pl.DataFrame:
    """Stream game tracking for the given WRs: active route frames snap -> pass_forward."""
    plays = mod06.load_valid_wr_route_plays(RAW_DIR).filter(pl.col("nfl_id").is_in(ids))
    plays_lazy = plays.select(["game_id", "play_id", "nfl_id"]).lazy()
    frames = []
    for yr in mod06.TRACKING_SEASONS:
        f = (
            pl.scan_csv(
                RAW_DIR / f"game_tracking_{yr}.csv",
                null_values=["NA", ""],
                infer_schema_length=50_000,
            )
            .select(["game_id", "play_id", "nfl_id", "time", "s", "event"])
            .filter(pl.col("nfl_id").is_in(ids))
            .join(plays_lazy, on=["game_id", "play_id", "nfl_id"], how="inner")
            .collect(engine="streaming")
        )
        frames.append(f)
    gt = pl.concat(frames).sort(["nfl_id", "game_id", "play_id", "time"])
    key = ["game_id", "play_id", "nfl_id"]
    return (
        gt.with_columns(pl.int_range(0, pl.len()).over(key).alias("f_idx"))
        .with_columns(
            pl.col("f_idx").filter(pl.col("event") == "ball_snap").min().over(key).alias("snap_idx"),
            pl.col("f_idx")
            .filter(pl.col("event").is_in(["pass_forward", "pass_shovel"]))
            .min()
            .over(key)
            .alias("pass_idx"),
        )
        .filter(
            pl.col("snap_idx").is_not_null()
            & pl.col("pass_idx").is_not_null()
            & (pl.col("f_idx") >= pl.col("snap_idx"))
            & (pl.col("f_idx") <= pl.col("pass_idx"))
        )
    )


def best_rep(ct_wr: pl.DataFrame, nfl_id: int, drill: str) -> pl.DataFrame | None:
    """Return the fastest rep (by max s) of a drill for a player."""
    sub = ct_wr.filter(
        (pl.col("nfl_id") == nfl_id)
        & (pl.col("drill_type") == "SKILL_DRILLS_WR")
        & (pl.col("drill_name") == drill)
    )
    if sub.height == 0:
        return None
    best = (
        sub.group_by("event_id").agg(pl.col("s").max().alias("ms")).sort("ms", descending=True)["event_id"][0]
    )
    return sub.filter(pl.col("event_id") == best).sort("ts")


def draw_trajectory(ax, rep: pl.DataFrame, drill: str, mod04: ModuleType, color: str) -> dict[str, float]:
    """Draw one player's trajectory (line width encodes speed) with peak-speed & break-apex markers."""
    seg = mod04.segment_wr_rep(rep, drill)
    x = rep["x"].to_numpy()
    y = rep["y"].to_numpy()
    s = rep["s"].to_numpy()
    x = x - x[0]
    y = y - y[0]
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lw = 1.0 + 0.62 * s[:-1]
    ax.add_collection(LineCollection(segs, linewidths=lw, colors=color, alpha=0.88, capstyle="round", zorder=3))
    ip = int(np.argmax(s))
    ia = int(seg["apex_idx"])
    ax.scatter(x[ip], y[ip], marker="*", s=260, color=color, edgecolor="white", linewidth=1.2, zorder=6)
    ax.scatter(x[ia], y[ia], marker="o", s=95, color="white", edgecolor=color, linewidth=2.4, zorder=6)
    ax.scatter(x[0], y[0], marker="s", s=38, color=color, edgecolor="white", zorder=6)
    return {"x": x, "y": y, "peak": float(s[ip]), "apex_s": float(s[ia])}  # type: ignore[dict-item]


def fig1_hero(df: pl.DataFrame, mod04: ModuleType, mod06: ModuleType) -> None:
    """Hero visual: Palmer vs Dike trajectories + NFL speed density."""
    palmer = df.filter(pl.col("display_name") == "Trey Palmer").row(0, named=True)
    dike = df.filter(pl.col("display_name") == "Chimere Dike").row(0, named=True)
    ids = [int(palmer["nfl_id"]), int(dike["nfl_id"])]
    _, _, ct_wr = mod04.load_wr_tracking(RAW_DIR)

    drills = ["GO_ROUTE_RIGHT"]
    for cand in ("COMEBACK_ROUTE_RIGHT", "CURL_ROUTE_RIGHT"):
        if best_rep(ct_wr, ids[0], cand) is not None and best_rep(ct_wr, ids[1], cand) is not None:
            drills.append(cand)
            break
    print(f"    fig1 shared drills used: {drills}")

    fig = plt.figure(figsize=(15.5, 8.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0], hspace=0.62, wspace=0.16)
    axes_left = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    ax_r = fig.add_subplot(gs[:, 1])

    for ax, drill in zip(axes_left, drills):
        info = {}
        for who, row, col in (("palmer", palmer, BLUE), ("dike", dike, ORANGE)):
            rep = best_rep(ct_wr, int(row["nfl_id"]), drill)
            info[who] = draw_trajectory(ax, rep, drill, mod04, col)
        allx = np.concatenate([info["palmer"]["x"], info["dike"]["x"]])
        ally = np.concatenate([info["palmer"]["y"], info["dike"]["y"]])
        ax.set_xlim(allx.min() - 3, allx.max() + 4)
        ypad = max(2.5, 0.12 * (ally.max() - ally.min()))
        ax.set_ylim(ally.min() - ypad, ally.max() + ypad)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(
            f"{drill.replace('_', ' ').title()} — peak speed Palmer {info['palmer']['peak']:.2f} vs Dike {info['dike']['peak']:.2f} yd/s",
            loc="left", fontsize=11, color=DARK,
        )
        ax.set_xlabel("Yards downfield from start", fontsize=9.5)
        ax.set_ylabel("Lateral yards", fontsize=9.5)

    handles = [
        Line2D([0], [0], color=BLUE, lw=4, label="Trey Palmer (Rd 6 #191)"),
        Line2D([0], [0], color=ORANGE, lw=4, label="Chimere Dike (Rd 4 #103)"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=GRAY, markeredgecolor="white", markersize=15, label="Peak speed"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=GRAY, markeredgewidth=2, markersize=9, label="Break apex"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=GRAY, markersize=6, label="Start"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.865), ncol=5, fontsize=9.5, frameon=False)

    # Right: NFL in-game speed density
    frames = extract_player_frames(mod06, ids)
    p95 = {}
    ymax = 0.0
    xs = np.linspace(0, 11.5, 400)
    for who, nid, col, nm in (("palmer", ids[0], BLUE, "Trey Palmer"), ("dike", ids[1], ORANGE, "Chimere Dike")):
        sp = frames.filter(pl.col("nfl_id") == nid)["s"].to_numpy()
        kde = stats.gaussian_kde(sp, bw_method=0.12)
        dens = kde(xs)
        ymax = max(ymax, float(dens.max()))
        ax_r.fill_between(xs, dens, color=col, alpha=0.22, linewidth=0)
        ax_r.plot(xs, dens, color=col, lw=2.4, label=f"{nm} ({len(sp):,} frames)")
        p95[who] = float(np.quantile(sp, 0.95))
    expected = {"palmer": float(palmer["nfl_route_p95_s"]), "dike": float(dike["nfl_route_p95_s"])}
    print(f"    fig1 p95 recomputed {p95} vs stored {expected}")
    for who, col in (("palmer", BLUE), ("dike", ORANGE)):
        ax_r.plot([expected[who], expected[who]], [0.0, 0.285], color=col, lw=2, ls="--", zorder=3)
    ax_r.set_xlim(0, 11.5)
    ax_r.set_ylim(0, ymax * 2.0)
    ax_r.set_xlabel("In-game speed on route, snap → pass release (yd/s)")
    ax_r.set_ylabel("Density of tracking frames")
    ax_r.set_title("NFL regular-season route speed", loc="left", fontsize=11.5, color=DARK)
    ax_r.legend(loc="upper left", bbox_to_anchor=(0.0, 0.58), fontsize=9)
    ax_r.text(
        expected["palmer"] + 0.1, ymax * 1.14, f"p95 {expected['palmer']:.2f}", color=BLUE, fontsize=9.5, fontweight="bold", ha="left"
    )
    ax_r.text(
        expected["dike"] - 0.1, ymax * 1.14, f"p95 {expected['dike']:.2f}", color=ORANGE, fontsize=9.5, fontweight="bold", ha="right"
    )
    box = (
        "Same body, same stopwatch time\n"
        f"Palmer {int(palmer['combine_weight'])} lb · {palmer['forty']:.2f}s 40-yd\n"
        f"Dike    {int(dike['combine_weight'])} lb · {dike['forty']:.2f}s 40-yd\n\n"
        "Separation over expected (break routes)\n"
        f"Palmer {palmer['soe_break']:+.3f} yd\n"
        f"Dike    {dike['soe_break']:+.3f} yd"
    )
    ax_r.text(
        0.97, 0.97, box, transform=ax_r.transAxes, ha="right", va="top", fontsize=9.5, color=DARK, zorder=10,
        bbox={"boxstyle": "round,pad=0.55", "fc": "#F7F7F7", "ec": "#BBBBBB", "lw": 0.8, "alpha": 1.0}, linespacing=1.35,
    )
    add_titles(
        fig,
        "Equal 40-yard times, unequal route kinetics: Palmer wins where Dike does not",
        "Left: Combine tracking (fastest rep per drill, line width grows with speed). Right: NFL 2023–25 regular season tracking, "
        "snap to pass release. Both receivers ran all shared drills.",
        "Source: NFL Big Data Bowl 2027 combine_tracking.csv & game_tracking_2023–2025.csv; non-nullified REG plays with a route.",
        top=0.78,
    )
    save(fig, "fig1_hero_palmer_vs_dike.png")


# ------------------------------------------------------------------ FIGURE 2
def added_variable(y: np.ndarray, X: np.ndarray, j: int):
    """Residualise y and column j of X on the remaining columns (Frisch–Waugh–Lovell)."""
    others = np.delete(X, j, axis=1)
    Xo = sm.add_constant(others)
    ry = sm.OLS(y, Xo).fit().resid
    rx = sm.OLS(X[:, j], Xo).fit().resid
    return rx, ry


def fig2_blindspot(df: pl.DataFrame) -> None:
    """Three added-variable plots from the same Model 1 on the complete-40-yd cohort."""
    cols = ["forty", "ngs_athleticism_score", "combine_weight", "combine_height", "ten_yd_split", "pure_kinetics_z"]
    sub = df.filter(pl.col("forty").is_not_null() & pl.col("ten_yd_split").is_not_null())
    n = sub.height
    y = sub["soe_break"].to_numpy()
    X = sub.select(cols).to_numpy().astype(float)
    m1 = sm.OLS(y, sm.add_constant(X)).fit()
    base_idx = [i for i in range(len(cols)) if cols[i] != "pure_kinetics_z"]
    m0 = sm.OLS(y, sm.add_constant(X[:, base_idx])).fit()
    f_stat, f_p, _ = m1.compare_f_test(m0)
    d_adj = m1.rsquared_adj - m0.rsquared_adj
    t_val = float(m1.tvalues[-1])
    partial_r = float(np.sign(t_val) * np.sqrt(t_val**2 / (t_val**2 + m1.df_resid)))
    print(f"    fig2 N={n} partial_r={partial_r:+.3f} F_p={f_p:.4f} dAdjR2={d_adj:+.4f}")

    panels = [
        ("forty", "(a) 40-yard dash", "40-yd time, residual (s)", GRAY),
        ("ngs_athleticism_score", "(b) NGS Athleticism Score", "NGS score, residual", GRAY),
        ("pure_kinetics_z", "(c) Pure route-kinetics (sensor only)", "pure_kinetics_z, residual (z)", BLUE),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0), sharey=True)
    names = sub["display_name"].to_list()
    for ax, (col, title, xlabel, color) in zip(axes, panels):
        j = cols.index(col)
        rx, ry = added_variable(y, X, j)
        ax.scatter(rx, ry, s=40, color=color, alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3)
        grid, mean, lo, hi = regression_line_ci(rx, ry)
        lw = 3.0 if col == "pure_kinetics_z" else 2.0
        ax.fill_between(grid, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(grid, mean, color=DARK if col != "pure_kinetics_z" else BLUE, lw=lw, zorder=4)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.axvline(0, color="#999999", lw=0.8)
        beta = float(m1.params[1 + j])
        p = float(m1.pvalues[1 + j])
        ax.set_title(title, loc="left", fontsize=12, color=DARK)
        ax.set_xlabel(xlabel)
        txt = f"β = {beta:+.3f} yd per unit\np = {p:.2f}"
        if col == "pure_kinetics_z":
            txt = (
                f"partial r = {partial_r:+.3f}\nF-test p = {f_p:.3f}\nΔ adj. R² = {d_adj:+.3f}\nβ = {beta:+.3f}, p = {p:.3f}"
            )
        ax.text(
            0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left", fontsize=10,
            fontweight="bold" if col == "pure_kinetics_z" else "normal",
            color=BLUE if col == "pure_kinetics_z" else DARK,
            bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#CCCCCC", "lw": 0.8},
        )
    axes[0].set_ylabel("soe_break, residual (yd)")
    # label Palmer and Dike on panel (c)
    rx_c, ry_c = added_variable(y, X, cols.index("pure_kinetics_z"))
    for nm, dx, dy, col in (("Trey Palmer", -10, 16, BLUE), ("Chimere Dike", 8, -30, ORANGE)):
        if nm in names:
            i = names.index(nm)
            axes[2].annotate(
                nm, (rx_c[i], ry_c[i]), xytext=(dx, dy), textcoords="offset points", fontsize=9.5,
                fontweight="bold", color=col, ha="center",
                arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.8},
            )
    add_titles(
        fig,
        "Stopwatch and NGS score are blind to route separation; route kinetics are not",
        f"Added-variable plots from one OLS model (N = {n} receivers with a measured 40-yd & 10-yd split): soe_break ~ weight + height + "
        "40-yd + 10-yd split + NGS + pure_kinetics_z. Each panel controls for the other five variables; band = 95% CI.",
        "Source: NFL Big Data Bowl 2027 combine_results/combine_tracking & player_play (REG). soe_break = separation over expected on break routes "
        "(OUT, IN, HITCH, SLANT, CORNER), out-of-fold by season, empirical-Bayes shrunk.",
        top=0.78,
    )
    fig.subplots_adjust(bottom=0.13, wspace=0.08)
    save(fig, "fig2_scouting_blindspot.png")


# ------------------------------------------------------------------ FIGURE 3
def fig3_bridge(df50: pl.DataFrame) -> None:
    """Combine sensor -> NFL on-field translation."""
    n = df50.height
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={"width_ratios": [1.0, 1.1], "wspace": 0.22})
    x = df50["vertical_route_max_s_z"].to_numpy()
    y = df50["nfl_route_p95_s"].to_numpy()
    rho, p = stats.spearmanr(x, y)
    grid, mean, lo, hi = regression_line_ci(x, y)
    axl.scatter(x, y, s=48, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3)
    axl.fill_between(grid, lo, hi, color=BLUE, alpha=0.18, linewidth=0)
    axl.plot(grid, mean, color=BLUE, lw=2.8)
    axl.set_xlabel("Combine vertical-route speed (within-drill z-score, weighted)")
    axl.set_ylabel("NFL in-game route speed, 95th pct (yd/s)")
    axl.set_title("(a) Combine sensor speed carries into NFL games", loc="left", fontsize=12, color=DARK)
    p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    axl.text(
        0.04, 0.96, f"Spearman ρ = {rho:+.3f}{stars(p)}\n{p_txt} · N = {n}", transform=axl.transAxes, va="top",
        fontsize=10.5, fontweight="bold", color=BLUE,
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#CCCCCC", "lw": 0.8},
    )
    print(f"    fig3 left rho={rho:+.3f} p={p:.2e} N={n}")

    feats = [
        ("pure_kinetics_z", "Pure route-kinetics (sensor)", BLUE),
        ("vertical_route_max_s_z", "Vertical-route speed z (sensor)", SKY),
        ("ngs_athleticism_score", "NGS Athleticism Score", GRAY),
    ]
    targets = [
        ("soe_break", "Separation over\nexpected (break)"),
        ("mean_expected_yac", "Expected YAC\n(NFL tracking model)"),
        ("nfl_route_p95_s", "NFL route speed\n(95th pct)"),
    ]
    width = 0.26
    for fi, (fcol, flabel, col) in enumerate(feats):
        for ti, (tcol, _) in enumerate(targets):
            pair = df50.select([fcol, tcol]).drop_nulls()
            r, pv = stats.spearmanr(pair[fcol].to_numpy(), pair[tcol].to_numpy())
            xpos = ti + (fi - 1) * width
            axr.bar(xpos, r, width=width * 0.92, color=col, edgecolor="white", linewidth=0.8, label=flabel if ti == 0 else None, zorder=3)
            off = 0.012 if r >= 0 else -0.012
            axr.text(
                xpos, r + off, f"{r:+.2f}{stars(pv)}", ha="center", va="bottom" if r >= 0 else "top",
                fontsize=9.2, fontweight="bold", color=DARK,
            )
    axr.axhline(0, color="#444444", lw=0.9)
    axr.set_xticks(range(len(targets)))
    axr.set_xticklabels([t[1] for t in targets], fontsize=10)
    axr.set_ylabel("Spearman ρ with NFL outcome")
    axr.set_ylim(-0.17, 0.62)
    axr.set_title("(b) Sensor features predict on-field outcomes; NGS score does not", loc="left", fontsize=12, color=DARK)
    axr.legend(loc="upper right", fontsize=9.2)
    axr.grid(axis="x", visible=False)
    add_titles(
        fig,
        "What the Combine sensor measures shows up on NFL Sundays",
        f"Receivers with ≥ {MIN_SNAPS} valid regular-season route snaps (N = {n}). Stars: ** p<0.01, * p<0.05, . p<0.10 (Spearman).",
        "Source: NFL Big Data Bowl 2027 combine_tracking.csv, game_tracking_2023–2025.csv, player_play.csv (REG, non-nullified). "
        "NFL route speed = 95th percentile of frame speed from snap to pass release.",
        top=0.80,
    )
    fig.subplots_adjust(bottom=0.15)
    save(fig, "fig3_combine_to_nfl_bridge.png")


# ------------------------------------------------------------------ FIGURE 4
def fig4_lodco(df50: pl.DataFrame, df_all: pl.DataFrame, mod08: ModuleType) -> None:
    """Leave-One-Draft-Class-Out performance of Models A, B, C (RidgeCV, no weight-mixing)."""
    specs = [
        ("Model A\nSize + track baseline", mod08.MODEL_A_COLS, False, GRAY),
        ("Model B\nPure sensor only", mod08.MODEL_B_PURE_COLS, False, SKY),
        ("Model C\nCombined (clean)", mod08.MODEL_C_PURE_BASE_COLS, True, BLUE),
    ]
    results: dict[str, list[dict]] = {}
    for cname, sub in ((f"N = {df50.height} (≥ {MIN_SNAPS} snaps)", df50), (f"N = {df_all.height} (all receivers)", df_all)):
        results[cname] = []
        for label, cols, resid, _ in specs:
            _, met = mod08.fit_predict_clean_lodco(sub, cols, "soe_break", "RidgeCV", add_fold_weight_residual=resid)
            results[cname].append(met)
            print(f"    fig4 {cname} {label.splitlines()[0]}: R2={met['r2']:+.4f} rho={met['spearman_rho']:+.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), gridspec_kw={"hspace": 0.5, "wspace": 0.18})
    for ci, (cname, mets) in enumerate(results.items()):
        ax = axes[0, ci]
        r2s = [m["r2"] for m in mets]
        for i, (r2, (label, _, _, col)) in enumerate(zip(r2s, specs)):
            ax.bar(i, r2, color=col, width=0.62, edgecolor="white", zorder=3)
            ax.text(i, r2 + (0.006 if r2 >= 0 else -0.006), f"{r2:+.3f}", ha="center", va="bottom" if r2 >= 0 else "top", fontsize=11, fontweight="bold", color=DARK)
        ax.annotate(
            "", xy=(2, r2s[2] + 0.03), xytext=(0, r2s[0] + 0.03),
            arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 2.2, "connectionstyle": "arc3,rad=-0.25"},
        )
        ax.text(1.0, max(r2s) + 0.075, f"R² {r2s[0]:.3f} → {r2s[2]:.3f}", ha="center", fontsize=11.5, fontweight="bold", color=ORANGE)
        ax.axhline(0, color="#444444", lw=0.9)
        ax.set_xticks(range(3))
        ax.set_xticklabels([s[0] for s in specs], fontsize=9.5)
        ax.set_ylim(min(0, min(r2s)) - 0.05, max(r2s) + 0.13)
        ax.set_title(f"Out-of-sample R², {cname}", loc="left", fontsize=11.5, color=DARK)
        if ci == 0:
            ax.set_ylabel("Pooled OOF R²")
        ax.grid(axis="x", visible=False)

        ax2 = axes[1, ci]
        years = [f["draft_year"] for f in mets[0]["by_draft_year"]]
        labels = [f"{y}\n(held out)" for y in years] + ["All\n(pooled)"]
        w = 0.275
        for mi, (m, (label, _, _, col)) in enumerate(zip(mets, specs)):
            vals = [f["spearman_rho"] for f in m["by_draft_year"]] + [m["spearman_rho"]]
            xs = np.arange(len(vals)) + (mi - 1) * w
            ax2.bar(xs, vals, width=w * 0.86, color=col, edgecolor="white", label=label.splitlines()[0], zorder=3)
            for xv, v in zip(xs, vals):
                ax2.text(xv, v + (0.015 if v >= 0 else -0.015), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.6, color=DARK)
        ax2.axhline(0, color="#444444", lw=0.9)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, fontsize=9.5)
        ax2.set_ylim(-0.3, 0.72)
        ax2.set_title(f"Rank accuracy (Spearman ρ) by held-out draft class, {cname}", loc="left", fontsize=11.5, color=DARK)
        if ci == 0:
            ax2.set_ylabel("Spearman ρ  (OOF prediction vs soe_break)")
            ax2.legend(loc="upper left", fontsize=9, ncol=3)
        ax2.grid(axis="x", visible=False)
    add_titles(
        fig,
        "Route kinetics improve predictions for draft classes the model has never seen",
        "Leave-One-Draft-Class-Out CV (train on two classes, test blind on the third), RidgeCV, target = soe_break. Imputation, scaling, "
        "weight residualisation and feature screening are all fitted inside each training fold.",
        "Source: NFL Big Data Bowl 2027 combine & NFL tracking (REG). Model C uses fold-internal sensor-minus-weight terms only (no weight mixing). "
        "Small samples: bootstrap CIs for Δρ include 0; full bootstrap 95% CIs reported in Appendix.",
        top=0.82,
    )
    fig.subplots_adjust(bottom=0.10)
    save(fig, "fig4_lodco_cv_performance.png")


# ------------------------------------------------------------------ FIGURE 5
def fig5_quadrant(df50: pl.DataFrame) -> None:
    """Draft-room scouting board: 40-yd vs size-adjusted kinetics, colored by soe_break."""
    sub = df50.filter(pl.col("has_go_route") & pl.col("forty").is_not_null())
    n = sub.height
    x = sub["forty"].to_numpy()
    y = sub["size_adjusted_route_kinetics"].to_numpy()
    c = sub["soe_break"].to_numpy()
    names = sub["display_name"].to_list()

    cmap = LinearSegmentedColormap.from_list("cb", [ORANGE, "#F5F5F5", BLUE])
    vmax = float(np.max(np.abs(c)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(13.5, 8.6))
    sc = ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=95, edgecolor="#555555", linewidth=0.6, zorder=3)
    xmed = float(np.median(x))
    ax.axvline(xmed, color="#888888", lw=1.1, ls="--", zorder=1)
    ax.axhline(0, color="#888888", lw=1.1, ls="--", zorder=1)
    ax.invert_xaxis()
    ax.set_xlabel("40-yard dash (s) — faster to the right →")
    ax.set_ylabel("Size-adjusted route kinetics (sensor residual vs. weight, z units)")
    cb = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.035)
    cb.set_label("NFL soe_break (yd): separation over expected, break routes", fontsize=9.5)
    cb.outline.set_visible(False)

    xl, xr = ax.get_xlim()
    yl, yh = ax.get_ylim()
    ax.set_ylim(yl, yh + 0.45)
    yl, yh = ax.get_ylim()
    quad_kw = {"fontsize": 10.5, "fontweight": "bold", "color": "#9A9A9A", "zorder": 1, "transform": ax.transAxes}
    ax.text(0.012, 0.985, "SLOW 40 · HIGH KINETICS\nHidden gems the stopwatch misses", ha="left", va="top", **quad_kw)
    ax.text(0.988, 0.985, "FAST 40 · HIGH KINETICS\nComplete route burners", ha="right", va="top", **quad_kw)
    ax.text(0.012, 0.015, "SLOW 40 · LOW KINETICS", ha="left", va="bottom", **quad_kw)
    ax.text(0.988, 0.015, "FAST 40 · LOW KINETICS\nTrack-only profiles", ha="right", va="bottom", **quad_kw)

    gems = ["Tez Johnson", "Trey Palmer", "Jalin Hyatt", "Jimmy Horn", "Tre Tucker"]
    lows = ["Chimere Dike", "Jack Bech", "Jamari Thrash"]
    sel_pts, sel_names, sel_cols, missing = [], [], [], []
    for nm in gems + lows:
        if nm in names:
            i = names.index(nm)
            sel_pts.append((x[i], y[i]))
            sel_names.append(nm)
            sel_cols.append(BLUE if nm in gems else ORANGE)
            ax.scatter(x[i], y[i], s=210, facecolor="none", edgecolor=BLUE if nm in gems else ORANGE, linewidth=2.4, zorder=5)
        else:
            missing.append(nm)
    print(f"    fig5 N={n}; labelled={sel_names}; not plotted={missing}")
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.07, right=0.93)
    fig.canvas.draw()
    place_labels(fig, ax, sel_pts, sel_names, sel_cols, np.column_stack([x, y]))

    note_missing = ""
    if "Jack Bech" in missing:
        note_missing = " Jack Bech (Rd 2) skipped the 40-yd dash, so he cannot be placed on this axis."
    add_titles(
        fig,
        "The draft board that finds route talent the 40-yard dash cannot see",
        f"{n} NFL receivers (2023–25 draft classes, ≥ {MIN_SNAPS} regular-season route snaps). Color = realised NFL separation over expected; "
        "dashed lines = cohort median 40-yd and zero kinetics residual.",
        "Note: receivers who opted out of GO_ROUTE_RIGHT (e.g. Dont'e Thornton, KeAndre Lambert-Smith) are excluded so every dot ran the full vertical-route battery."
        + note_missing,
        top=0.84,
    )
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.07, right=0.93)
    save(fig, "fig5_draft_room_quadrant.png")


def main() -> None:
    """Build all 5 figures."""
    t0 = time.perf_counter()
    setup_style()
    mod04 = _load_module("wr_bio_mod", SRC_DIR / "04_wr_biomechanics.py")
    mod06 = _load_module("wr_bridge_mod", SRC_DIR / "06_wr_game_tracking_bridge.py")
    mod08 = _load_module("wr_audit_mod", SRC_DIR / "08_wr_robustness_audit.py")

    print("Loading modeling frame...")
    df_all = load_modeling_frame(mod08)
    df50 = df_all.filter(pl.col("n_all") >= MIN_SNAPS)
    print(f"    N_all={df_all.height}, N_50={df50.height}")

    print("Figure 1 — hero visual")
    fig1_hero(df_all, mod04, mod06)
    print("Figure 2 — scouting blind spot")
    fig2_blindspot(df_all)
    print("Figure 3 — combine-to-NFL bridge")
    fig3_bridge(df50)
    print("Figure 4 — LODCO performance")
    fig4_lodco(df50, df_all, mod08)
    print("Figure 5 — draft-room quadrant")
    fig5_quadrant(df50)
    print(f"Done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

