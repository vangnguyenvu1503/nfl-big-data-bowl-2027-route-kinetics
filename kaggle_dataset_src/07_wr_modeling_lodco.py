"""Train & evaluate regularized WR models via Leave-One-Draft-Class-Out CV.

Evaluates Model A (Combine/size baselines), Model B (Combine Route Kinetics),
and Model C (Combined) using RidgeCV, BayesianRidge, and ElasticNetCV strictly
inside Leave-One-Draft-Class-Out CV (2023, 2024, 2025) on `soe_break`, `soe_all`,
and `nfl_route_p95_s`, with paired Bootstrap 95% confidence intervals and real
player Hidden Gems vs Overvalued Track Stars tables written to
``wr_modeling_report.md``.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import time

import numpy as np
import polars as pl
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
_KAGGLE_RAW = Path("/kaggle/input/nfl-big-data-bowl-2027")
_KAGGLE_WORK = Path("/kaggle/working")


def _resolve_raw_dir() -> Path:
    env = os.environ.get("BDB_RAW_DIR")
    if env:
        return Path(env).resolve()
    if _KAGGLE_RAW.exists():
        return _KAGGLE_RAW
    return PROJECT_ROOT / "data" / "raw"


def _resolve_out_dir() -> Path:
    env = os.environ.get("BDB_OUT_DIR")
    if env:
        return Path(env).resolve()
    if _KAGGLE_WORK.exists():
        return _KAGGLE_WORK
    return PROJECT_ROOT


RAW_DIR = _resolve_raw_dir()
OUT_DIR = _resolve_out_dir()
PROCESSED_DIR = OUT_DIR / "data" / "processed"
TARGETS_PARQUET = PROCESSED_DIR / "wr_targets.parquet"
ADV_KINETICS_PARQUET = PROCESSED_DIR / "wr_advanced_kinetics.parquet"
GAME_KINETICS_PARQUET = PROCESSED_DIR / "wr_game_kinetics.parquet"
ADV_SUMMARY_JSON = PROCESSED_DIR / "wr_advanced_kinetics_summary.json"
TRANSLATION_JSON = PROCESSED_DIR / "wr_kinetic_translation.json"
OUTPUT_REPORT = OUT_DIR / "wr_modeling_report.md"

MIN_SNAPS_COHORT = 50
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42

MODEL_A_COLS = [
    "forty",
    "ten_yd_split",
    "ngs_athleticism_score",
    "combine_height",
    "combine_weight",
]
MODEL_B_COLS = [
    "route_max_s",
    "vertical_route_max_s_z",
    "pure_kinetics_z",
    "peak_centripetal_accel_z",
    "braking_power_index_z",
    "speed_cut_retention_z",
    "pre_break_jerk",
]
MODEL_C_COLS = MODEL_A_COLS + MODEL_B_COLS + [
    "route_kinetics_index",
    "functional_speed_ratio",
]

MODEL_SPECS = [
    ("Model A (Baseline)", MODEL_A_COLS),
    ("Model B (Route Kinetics)", MODEL_B_COLS),
    ("Model C (Combined)", MODEL_C_COLS),
]
ESTIMATOR_NAMES = ["RidgeCV", "BayesianRidge", "ElasticNetCV"]


def _make_estimator(name: str):
    """Instantiate a fresh regularized linear estimator."""
    if name == "RidgeCV":
        return RidgeCV(alphas=np.logspace(-1, 3.5, 40))
    if name == "BayesianRidge":
        return BayesianRidge()
    if name == "ElasticNetCV":
        return ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 0.99],
            alphas=np.logspace(-3, 1.5, 30),
            cv=3,
            random_state=42,
            max_iter=5000,
        )
    raise ValueError(f"Unknown estimator: {name}")


def _rank_features_by_abs_corr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rank feature indices on training data only by |Pearson correlation| with y."""
    scores = np.zeros(X.shape[1], dtype=np.float64)
    y_std = float(np.std(y))
    for j in range(X.shape[1]):
        col = X[:, j]
        if y_std == 0.0 or np.std(col) == 0.0:
            scores[j] = 0.0
        else:
            scores[j] = abs(float(np.corrcoef(col, y)[0, 1]))
    return np.argsort(-scores, kind="mergesort")


def _select_top_k_inner_cv(
    X_tr_scaled: np.ndarray,
    y_tr: np.ndarray,
    ranked_idx: np.ndarray,
    estimator_name: str,
) -> int:
    """Select top-k feature subset inside training fold via 3-fold inner CV."""
    n_feats = X_tr_scaled.shape[1]
    if n_feats <= 2:
        return n_feats

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    best_k = n_feats
    best_mse = float("inf")
    # Candidate subset sizes: 1, 2, 3, ..., n_feats
    for k in range(1, n_feats + 1):
        cols = ranked_idx[:k]
        preds = np.empty_like(y_tr)
        for in_tr, in_va in kf.split(X_tr_scaled):
            est = _make_estimator(estimator_name)
            est.fit(X_tr_scaled[in_tr][:, cols], y_tr[in_tr])
            preds[in_va] = est.predict(X_tr_scaled[in_va][:, cols])
        mse = float(np.mean((y_tr - preds) ** 2))
        if mse < best_mse - 1e-9:
            best_mse = mse
            best_k = k
    return best_k


def fit_predict_lodco(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    estimator_name: str = "ElasticNetCV",
) -> tuple[np.ndarray, dict[str, object]]:
    """Run Leave-One-Draft-Class-Out CV with fold-internal imputation, scaling & screening."""
    draft_years = df["draft_year"].to_numpy()
    X_raw = df.select(feature_cols).to_numpy().astype(np.float64)
    y_all = df[target_col].to_numpy().astype(np.float64)
    oof_pred = np.empty(df.height, dtype=np.float64)

    unique_years = sorted(int(yr) for yr in np.unique(draft_years))
    fold_metrics: list[dict[str, object]] = []
    coef_matrix = np.zeros((len(unique_years), len(feature_cols)), dtype=np.float64)
    selected_counts = np.zeros(len(feature_cols), dtype=np.int64)

    for fold_i, holdout_yr in enumerate(unique_years):
        tr_mask = draft_years != holdout_yr
        va_mask = draft_years == holdout_yr

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(imputer.fit_transform(X_raw[tr_mask]))
        X_va = scaler.transform(imputer.transform(X_raw[va_mask]))
        y_tr = y_all[tr_mask]
        y_va = y_all[va_mask]

        ranked_idx = _rank_features_by_abs_corr(X_tr, y_tr)
        top_k = _select_top_k_inner_cv(X_tr, y_tr, ranked_idx, estimator_name)
        sel_idx = ranked_idx[:top_k]
        selected_counts[sel_idx] += 1

        est = _make_estimator(estimator_name)
        est.fit(X_tr[:, sel_idx], y_tr)
        pred_va = est.predict(X_va[:, sel_idx])
        oof_pred[va_mask] = pred_va
        coef_matrix[fold_i, sel_idx] = est.coef_

        sr = stats.spearmanr(y_va, pred_va)
        fold_metrics.append(
            {
                "draft_year": holdout_yr,
                "n_train": int(tr_mask.sum()),
                "n_val": int(va_mask.sum()),
                "top_k": int(top_k),
                "selected_features": [feature_cols[j] for j in sel_idx],
                "r2": float(r2_score(y_va, pred_va)),
                "rmse": float(root_mean_squared_error(y_va, pred_va)),
                "spearman_rho": float(sr.statistic) if np.isfinite(sr.statistic) else 0.0,
                "spearman_p": float(sr.pvalue) if np.isfinite(sr.pvalue) else 1.0,
            }
        )

    sr_all = stats.spearmanr(y_all, oof_pred)
    pr_all = stats.pearsonr(y_all, oof_pred)
    metrics: dict[str, object] = {
        "n": int(df.height),
        "r2": float(r2_score(y_all, oof_pred)),
        "rmse": float(root_mean_squared_error(y_all, oof_pred)),
        "mae": float(mean_absolute_error(y_all, oof_pred)),
        "spearman_rho": float(sr_all.statistic) if np.isfinite(sr_all.statistic) else 0.0,
        "spearman_p": float(sr_all.pvalue) if np.isfinite(sr_all.pvalue) else 1.0,
        "pearson_r": float(pr_all.statistic) if np.isfinite(pr_all.statistic) else 0.0,
        "pearson_p": float(pr_all.pvalue) if np.isfinite(pr_all.pvalue) else 1.0,
        "by_draft_year": fold_metrics,
        "feature_roles": [
            {
                "feature": feature_cols[j],
                "selected_folds": int(selected_counts[j]),
                "mean_std_coef": float(np.mean(coef_matrix[:, j])),
                "mean_abs_coef": float(np.mean(np.abs(coef_matrix[:, j]))),
            }
            for j in range(len(feature_cols))
        ],
    }
    return oof_pred, metrics


def paired_bootstrap_comparison(
    y: np.ndarray,
    pred_cand: np.ndarray,
    pred_base: np.ndarray,
    n_boot: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, tuple[float, float]]:
    """Compute paired Bootstrap 95% CIs for Spearman rho, RMSE, and deltas (cand - base)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    rho_c_list, rho_b_list, drho_list = [], [], []
    rmse_c_list, rmse_b_list, drmse_list = [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        pc = pred_cand[idx]
        pb = pred_base[idx]
        if np.std(yb) == 0 or np.std(pc) == 0 or np.std(pb) == 0:
            continue
        rc = float(stats.spearmanr(yb, pc).statistic)
        rb = float(stats.spearmanr(yb, pb).statistic)
        if not (np.isfinite(rc) and np.isfinite(rb)):
            continue
        ec = float(root_mean_squared_error(yb, pc))
        eb = float(root_mean_squared_error(yb, pb))
        rho_c_list.append(rc)
        rho_b_list.append(rb)
        drho_list.append(rc - rb)
        rmse_c_list.append(ec)
        rmse_b_list.append(eb)
        drmse_list.append(ec - eb)

    def _ci(arr: list[float]) -> tuple[float, float]:
        a = np.asarray(arr, dtype=np.float64)
        return float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))

    return {
        "rho_cand_ci": _ci(rho_c_list),
        "rho_base_ci": _ci(rho_b_list),
        "delta_rho_ci": _ci(drho_list),
        "rmse_cand_ci": _ci(rmse_c_list),
        "rmse_base_ci": _ci(rmse_b_list),
        "delta_rmse_ci": _ci(drmse_list),
    }


def md_escape(value: object) -> str:
    """Format a scalar safely for a Markdown table cell."""
    if value is None:
        return "—"
    return str(value).replace("\r\n", " ").replace("\n", " ").replace("|", "\\|")


def format_markdown_table(
    headers: list[str],
    rows: list[list[object]],
    alignments: list[str] | None = None,
) -> list[str]:
    """Render a GitHub-flavored Markdown table."""
    align_map = {"left": ":---", "right": "---:", "center": ":---:"}
    if alignments is None:
        alignments = ["left"] * len(headers)
    sep = [align_map[a] for a in alignments]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_escape(c) for c in row) + " |")
    return lines


def sig_star(p_val: float | None) -> str:
    """Return significance marker for a p-value."""
    if p_val is None or np.isnan(p_val):
        return ""
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    if p_val < 0.10:
        return "."
    return ""


def load_modeling_dataset() -> pl.DataFrame:
    """Join targets, advanced Combine kinetics, and NFL game kinetics on `nfl_id`."""
    targets = pl.read_parquet(TARGETS_PARQUET)
    adv = pl.read_parquet(ADV_KINETICS_PARQUET)
    game_kin = pl.read_parquet(GAME_KINETICS_PARQUET)

    # Avoid duplicate metadata columns when joining
    t_cols = [c for c in targets.columns if c not in ("display_name", "draft_year")]
    df = (
        adv.join(targets.select(t_cols), on="nfl_id", how="inner")
        .join(game_kin, on="nfl_id", how="inner")
        .sort("nfl_id")
    )
    return df


def build_report(
    df_all: pl.DataFrame,
    adv_summary: dict[str, object],
    translation: dict[str, object],
) -> str:
    """Run all LODCO evaluations and generate wr_modeling_report.md."""
    df_50 = df_all.filter(pl.col("n_all") >= MIN_SNAPS_COHORT)
    lines: list[str] = [
        "# NFL Big Data Bowl 2027 — WR Advanced Route Kinetics & LODCO Modeling Report",
        "",
        f"- **Thời gian tạo**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "- **Đề tài**: *Wide Receiver — From Track Speed to Route Kinetics*",
        f"- **Cỡ mẫu đánh giá**: Nhóm chuẩn `n_all >= {MIN_SNAPS_COHORT}` snaps (**N = {df_50.height} WR**) và Toàn bộ WR có dữ liệu NFL REG (**N = {df_all.height} WR**).",
        "- **Ký hiệu ý nghĩa thống kê**: `**` (*p < 0.01*), `*` (*p < 0.05*), `.` (*p < 0.10*).",
        "",
        "## 1. Chuẩn hóa theo nhóm bài tập thuần nhất & Đặc trưng vật lý ghép (`05_wr_advanced_kinetics.py`)",
        "",
        f"- Tổng hợp từ **{int(adv_summary['n_wrs'])} WR** trên **{int(adv_summary['n_reps_grouped']):,} reps** thuộc 10 bài `SKILL_DRILLS_WR` chạy ổn định cả 3 khóa Combine (`2023`–`2025`), sau khi chuẩn hóa Z-score trong nội bộ từng `drill_name`.",
        "",
    ]

    dg_rows = [
        [
            r["drill_group"],
            f"`{r['drill_name']}`",
            f"{int(r['n_wrs']):,}",
            f"{float(r['med_max_s']):.2f}",
            f"{float(r['med_decel']):.2f}",
            f"{float(r['med_centripetal_a']):.2f}",
            f"{float(r['med_braking_power']):.2f}",
            f"{float(r['med_retention']):.3f}",
        ]
        for r in adv_summary["drill_groups"]  # type: ignore[attr-defined]
    ]
    lines.extend(
        format_markdown_table(
            [
                "Nhóm vật lý",
                "drill_name",
                "WRs",
                "Med max_s",
                "Med Decel",
                "Med a_lat (yd/s²)",
                "Med Braking Power",
                "Med Retention",
            ],
            dg_rows,
            ["left", "left", "right", "right", "right", "right", "right", "right"],
        )
    )
    lines.extend(["", "### 1.1. Tương quan đơn biến của bộ đặc trưng vật lý ghép trên `N = 69` WR", ""])

    adv_features = [
        ("route_max_s", "`route_max_s` (yd/s)"),
        ("vertical_route_max_s_z", "`vertical_route_max_s_z` (Weighted Z)"),
        ("speed_cut_retention_z", "`speed_cut_retention_z` (Z)"),
        ("peak_centripetal_accel", "`peak_centripetal_accel` (yd/s²)"),
        ("peak_centripetal_accel_z", "`peak_centripetal_accel_z` (Z)"),
        ("braking_power_index", "`braking_power_index` (Hard-Break)"),
        ("braking_power_index_z", "`braking_power_index_z` (Hard-Break Z)"),
        ("pure_kinetics_z", "`pure_kinetics_z` (Sensor-Only Composite)"),
        ("route_kinetics_index", "`route_kinetics_index` (Weight-Controlled)"),
        ("forty", "Baseline: `forty` (s)"),
        ("ten_yd_split", "Baseline: `ten_yd_split` (s)"),
        ("ngs_athleticism_score", "Baseline: `ngs_athleticism_score`"),
        ("combine_weight", "Baseline: `combine_weight` (lbs)"),
    ]
    adv_corr_rows: list[list[object]] = []
    for fcol, flabel in adv_features:
        cells: list[object] = [flabel]
        for tcol in ("soe_break", "soe_all", "soe_man"):
            pair = df_50.select([fcol, tcol]).drop_nulls()
            x = pair[fcol].cast(pl.Float64).to_numpy()
            y = pair[tcol].cast(pl.Float64).to_numpy()
            sr = stats.spearmanr(x, y)
            pr = stats.pearsonr(x, y)
            cells.append(f"{float(sr.statistic):+.3f}{sig_star(float(sr.pvalue))} ({float(sr.pvalue):.3f})")
            cells.append(f"{float(pr.statistic):+.3f}{sig_star(float(pr.pvalue))} ({float(pr.pvalue):.3f})")
        adv_corr_rows.append(cells)

    lines.extend(
        format_markdown_table(
            [
                "Đặc trưng",
                "soe_break ρ (p)",
                "soe_break r (p)",
                "soe_all ρ (p)",
                "soe_all r (p)",
                "soe_man ρ (p)",
                "soe_man r (p)",
            ],
            adv_corr_rows,
            ["left", "right", "right", "right", "right", "right", "right"],
        )
    )

    # Section 2: Combine-to-NFL Game Tracking Bridge
    t_counts = translation["counts"]  # type: ignore[index]
    t_c50 = translation["cohorts"]["cohort_50"]  # type: ignore[index]
    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Cầu nối Combine-to-NFL Kinetic Translation (`06_wr_game_tracking_bridge.py`)",
            "",
            f"- Quét lazy streaming `game_tracking_2023..2025.csv` trên **{int(t_counts['valid_route_plays']):,} route plays** hợp lệ (`REG`, `play_nullified_by_penalty == \"N\"`), trích xuất **{int(t_counts['active_route_frames']):,} active frames** (từ `ball_snap` đến `pass_forward`) cho **{int(t_counts['n_wrs'])} WR**.",
            f"- Bảng kiểm định Combine-to-NFL Kinetic Translation trên nhóm chuẩn **N = {int(t_c50['n_wrs'])} WR** (`n_all >= {MIN_SNAPS_COHORT}`):",
            "",
        ]
    )

    def _fmt_tr_cell(sd: dict[str, object]) -> str:
        if sd["rho"] is None:
            return "—"
        return f"{float(sd['rho']):+.3f}{sig_star(float(sd['sp']))} / {float(sd['r']):+.3f} (p={float(sd['sp']):.3f})"

    tr_rows = []
    for entry in t_c50["correlations"]:  # type: ignore[index]
        tr_rows.append(
            [
                entry["label"],
                _fmt_tr_cell(entry["nfl_route_p95_s"]),
                _fmt_tr_cell(entry["nfl_route_max_s"]),
                _fmt_tr_cell(entry["nfl_break_centripetal_p95"]),
                _fmt_tr_cell(entry["nfl_break_decel_p05"]),
            ]
        )
    lines.extend(
        format_markdown_table(
            [
                "Đặc trưng Combine (ρ / r, p-val Spearman)",
                "`nfl_route_p95_s`",
                "`nfl_route_max_s`",
                "`nfl_break_centripetal_p95`",
                "`nfl_break_decel_p05`",
            ],
            tr_rows,
            ["left", "right", "right", "right", "right"],
        )
    )

    # Section 3: LODCO Modeling Results
    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Kết quả mô hình hóa Leave-One-Draft-Class-Out CV (`2023`, `2024`, `2025`)",
            "",
            "- Quy trình chống rò rỉ tuyệt đối: trong từng fold (train trên 2 khóa Draft, kiểm thử mù trên khóa thứ 3), toàn bộ bước `SimpleImputer(median)`, `StandardScaler`, xếp hạng biến và chọn `top_k` qua inner 3-fold CV đều chạy **chỉ trên tập train**.",
            "",
            f"### 3.1. So sánh Model A vs Model B vs Model C trên nhóm chuẩn (`N = {df_50.height}` WR)",
            "",
        ]
    )

    # Run all combinations for N=69 and N=83
    lodco_preds_50: dict[tuple[str, str, str], np.ndarray] = {}
    lodco_metrics_50: dict[tuple[str, str, str], dict[str, object]] = {}
    comp_rows_50: list[list[object]] = []

    for tgt in ("soe_break", "soe_all", "nfl_route_p95_s"):
        y_vec = df_50[tgt].to_numpy()
        for est_name in ESTIMATOR_NAMES:
            for mname, mcols in MODEL_SPECS:
                pred, met = fit_predict_lodco(df_50, mcols, tgt, est_name)
                lodco_preds_50[(tgt, est_name, mname)] = pred
                lodco_metrics_50[(tgt, est_name, mname)] = met
            # Compute paired Bootstrap CI for Model C - Model A and Model B - Model A
            ci_ca = paired_bootstrap_comparison(
                y_vec,
                lodco_preds_50[(tgt, est_name, "Model C (Combined)")],
                lodco_preds_50[(tgt, est_name, "Model A (Baseline)")],
            )
            ci_ba = paired_bootstrap_comparison(
                y_vec,
                lodco_preds_50[(tgt, est_name, "Model B (Route Kinetics)")],
                lodco_preds_50[(tgt, est_name, "Model A (Baseline)")],
            )
            for mname, _ in MODEL_SPECS:
                met = lodco_metrics_50[(tgt, est_name, mname)]
                if mname == "Model A (Baseline)":
                    drho_str = "Ref"
                    drmse_str = "Ref"
                elif mname == "Model B (Route Kinetics)":
                    d_rho = float(met["spearman_rho"]) - float(lodco_metrics_50[(tgt, est_name, "Model A (Baseline)")]["spearman_rho"])
                    d_rmse = float(met["rmse"]) - float(lodco_metrics_50[(tgt, est_name, "Model A (Baseline)")]["rmse"])
                    c1, c2 = ci_ba["delta_rho_ci"]
                    r1, r2 = ci_ba["delta_rmse_ci"]
                    drho_str = f"{d_rho:+.3f} [{c1:+.3f}, {c2:+.3f}]"
                    drmse_str = f"{d_rmse:+.4f} [{r1:+.4f}, {r2:+.4f}]"
                else:
                    d_rho = float(met["spearman_rho"]) - float(lodco_metrics_50[(tgt, est_name, "Model A (Baseline)")]["spearman_rho"])
                    d_rmse = float(met["rmse"]) - float(lodco_metrics_50[(tgt, est_name, "Model A (Baseline)")]["rmse"])
                    c1, c2 = ci_ca["delta_rho_ci"]
                    r1, r2 = ci_ca["delta_rmse_ci"]
                    drho_str = f"{d_rho:+.3f} [{c1:+.3f}, {c2:+.3f}]"
                    drmse_str = f"{d_rmse:+.4f} [{r1:+.4f}, {r2:+.4f}]"

                comp_rows_50.append(
                    [
                        f"`{tgt}`",
                        est_name,
                        mname,
                        f"{float(met['r2']):+.4f}",
                        f"{float(met['rmse']):.4f}",
                        f"{float(met['spearman_rho']):+.3f}{sig_star(float(met['spearman_p']))} ({float(met['spearman_p']):.3f})",
                        f"{float(met['pearson_r']):+.3f}{sig_star(float(met['pearson_p']))}",
                        drho_str,
                        drmse_str,
                    ]
                )

    lines.extend(
        format_markdown_table(
            [
                "Target",
                "Estimator",
                "Model",
                "OOF $R^2$",
                "OOF RMSE",
                "Spearman ρ (p)",
                "Pearson r",
                "Δρ vs A [95% Boot CI]",
                "ΔRMSE vs A [95% Boot CI]",
            ],
            comp_rows_50,
            ["left", "left", "left", "right", "right", "right", "right", "right", "right"],
        )
    )

    # Full N=83 table (with ElasticNetCV & RidgeCV & BayesianRidge)
    lines.extend(["", f"### 3.2. Kiểm tra trên toàn bộ mẫu (`N = {df_all.height}` WR — EB-shrunk)", ""])
    comp_rows_all: list[list[object]] = []
    lodco_preds_all: dict[tuple[str, str, str], np.ndarray] = {}
    lodco_metrics_all: dict[tuple[str, str, str], dict[str, object]] = {}
    for tgt in ("soe_break", "soe_all", "nfl_route_p95_s"):
        y_vec = df_all[tgt].to_numpy()
        for est_name in ESTIMATOR_NAMES:
            for mname, mcols in MODEL_SPECS:
                pred, met = fit_predict_lodco(df_all, mcols, tgt, est_name)
                lodco_preds_all[(tgt, est_name, mname)] = pred
                lodco_metrics_all[(tgt, est_name, mname)] = met
            ci_ca = paired_bootstrap_comparison(
                y_vec,
                lodco_preds_all[(tgt, est_name, "Model C (Combined)")],
                lodco_preds_all[(tgt, est_name, "Model A (Baseline)")],
            )
            for mname, _ in MODEL_SPECS:
                met = lodco_metrics_all[(tgt, est_name, mname)]
                if mname == "Model C (Combined)":
                    d_rho = float(met["spearman_rho"]) - float(lodco_metrics_all[(tgt, est_name, "Model A (Baseline)")]["spearman_rho"])
                    c1, c2 = ci_ca["delta_rho_ci"]
                    drho_str = f"{d_rho:+.3f} [{c1:+.3f}, {c2:+.3f}]"
                else:
                    drho_str = "—"
                comp_rows_all.append(
                    [
                        f"`{tgt}`",
                        est_name,
                        mname,
                        f"{float(met['r2']):+.4f}",
                        f"{float(met['rmse']):.4f}",
                        f"{float(met['spearman_rho']):+.3f}{sig_star(float(met['spearman_p']))} ({float(met['spearman_p']):.3f})",
                        drho_str,
                    ]
                )
    lines.extend(
        format_markdown_table(
            ["Target", "Estimator", "Model", "OOF $R^2$", "OOF RMSE", "Spearman ρ (p)", "Δρ (C−A) [95% Boot CI]"],
            comp_rows_all,
            ["left", "left", "left", "right", "right", "right", "right"],
        )
    )

    # 3.3 Per-fold breakdown & feature roles for primary ElasticNetCV on soe_break (N=69)
    lines.extend(["", "### 3.3. Chi tiết theo từng khóa Draft & Vai trò từng biến (`soe_break`, `ElasticNetCV`, `N = 69`)", ""])
    fold_rows: list[list[object]] = []
    for mname, _ in MODEL_SPECS:
        met = lodco_metrics_50[("soe_break", "ElasticNetCV", mname)]
        for fm in met["by_draft_year"]:  # type: ignore[attr-defined]
            fold_rows.append(
                [
                    mname,
                    fm["draft_year"],
                    f"{int(fm['n_train'])} / {int(fm['n_val'])}",
                    fm["top_k"],
                    f"{float(fm['r2']):+.4f}",
                    f"{float(fm['rmse']):.4f}",
                    f"{float(fm['spearman_rho']):+.3f}",
                    ", ".join(f"`{f}`" for f in fm["selected_features"]),
                ]
            )
    lines.extend(
        format_markdown_table(
            ["Model", "Holdout Year", "Train / Val", "Top-K", "OOF $R^2$", "RMSE", "Spearman ρ", "Selected Features"],
            fold_rows,
            ["left", "center", "center", "right", "right", "right", "right", "left"],
        )
    )

    lines.extend(["", "#### Trọng số chuẩn hóa trung bình qua 3 fold (`Model C (Combined)`, `soe_break`, `ElasticNetCV`)", ""])
    role_rows = sorted(
        lodco_metrics_50[("soe_break", "ElasticNetCV", "Model C (Combined)")]["feature_roles"],  # type: ignore[arg-type]
        key=lambda r: float(r["mean_abs_coef"]),
        reverse=True,
    )
    lines.extend(
        format_markdown_table(
            ["Feature", "Selected Folds (/3)", "Mean Std Coef", "Mean |Std Coef|"],
            [
                [
                    f"`{r['feature']}`",
                    f"{int(r['selected_folds'])}/3",
                    f"{float(r['mean_std_coef']):+.4f}",
                    f"{float(r['mean_abs_coef']):.4f}",
                ]
                for r in role_rows
            ],
            ["left", "center", "right", "right"],
        )
    )

    # Section 4: Top 10 Hidden Gems vs Top 5 Overvalued Track Stars
    df_with_pred = df_50.with_columns(
        pl.Series("pred_c_soe_break", lodco_preds_50[("soe_break", "ElasticNetCV", "Model C (Combined)")]),
        pl.Series("pred_a_soe_break", lodco_preds_50[("soe_break", "ElasticNetCV", "Model A (Baseline)")]),
    ).with_columns(
        (
            ((pl.col("soe_break") - pl.col("soe_break").mean()) / pl.col("soe_break").std())
            + 0.75 * ((pl.col("route_kinetics_index") - pl.col("route_kinetics_index").mean()) / pl.col("route_kinetics_index").std())
        ).alias("gem_score")
    )

    hidden_gems = (
        df_with_pred.filter(
            (pl.col("draft_round").is_null() | (pl.col("draft_round") >= 3))
            & (pl.col("soe_break") > 0.0)
            & (pl.col("route_kinetics_index") > 0.0)
        )
        .sort("gem_score", descending=True)
        .head(10)
    )
    if hidden_gems.height < 10:
        hidden_gems = (
            df_with_pred.filter(pl.col("draft_round").is_null() | (pl.col("draft_round") >= 3))
            .sort("gem_score", descending=True)
            .head(10)
        )

    overvalued = (
        df_with_pred.filter(
            (pl.col("forty") <= 4.48) | ((pl.col("draft_round").is_not_null()) & (pl.col("draft_round") <= 2))
        )
        .with_columns(
            (
                -1.0 * ((pl.col("soe_break") - pl.col("soe_break").mean()) / pl.col("soe_break").std())
                - 0.75 * ((pl.col("route_kinetics_index") - pl.col("route_kinetics_index").mean()) / pl.col("route_kinetics_index").std())
            ).alias("overval_score")
        )
        .sort("overval_score", descending=True)
        .head(5)
    )

    def _format_player_rows(tbl: pl.DataFrame) -> list[list[object]]:
        rows_out: list[list[object]] = []
        for r in tbl.iter_rows(named=True):
            d_round = "UDFA" if r["draft_round"] is None else f"Rd {int(r['draft_round'])}"
            d_pick = "UDFA" if r["draft_overall_pick"] is None else f"#{int(r['draft_overall_pick'])}"
            f40 = "—" if r["forty"] is None else f"{float(r['forty']):.2f}s"
            rows_out.append(
                [
                    r["display_name"],
                    int(r["draft_year"]),
                    d_round,
                    d_pick,
                    f40,
                    f"{int(r['combine_weight'])}",
                    f"{float(r['route_max_s']):.2f}",
                    f"{float(r['route_kinetics_index']):+.2f}",
                    f"{float(r['nfl_route_p95_s']):.2f}",
                    f"{float(r['soe_break']):+.3f}",
                    f"{float(r['soe_all']):+.3f}",
                ]
            )
        return rows_out

    player_headers = [
        "Cầu thủ (`display_name`)",
        "Draft Year",
        "Round",
        "Overall Pick",
        "40-Yd",
        "Wt (lbs)",
        "Combine `route_max_s`",
        "`route_kinetics_index`",
        "NFL `route_p95_s`",
        "`soe_break`",
        "`soe_all`",
    ]
    player_align = ["left", "center", "center", "center", "right", "right", "right", "right", "right", "right", "right"]

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Ứng dụng tuyển trạch: Top 10 Hidden Gems vs Top 5 Overvalued Track Stars",
            "",
            "### 4.1. Top 10 Hidden Gems (Vòng muộn >= Rd 3 / UDFA nhưng Route Kinetics & SOE cao)",
            "",
        ]
    )
    lines.extend(format_markdown_table(player_headers, _format_player_rows(hidden_gems), player_align))
    lines.extend(
        [
            "",
            "### 4.2. Top 5 Overvalued Track Stars (40-yard <= 4.48s hoặc Draft <= Rd 2 nhưng Route Kinetics & SOE thấp)",
            "",
        ]
    )
    lines.extend(format_markdown_table(player_headers, _format_player_rows(overvalued), player_align))
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run LODCO modeling comparisons and write wr_modeling_report.md."""
    t0 = time.perf_counter()
    print("1/3 Loading WR targets, advanced Combine kinetics, and NFL game kinetics...")
    df_all = load_modeling_dataset()
    adv_summary = json.loads(ADV_SUMMARY_JSON.read_text(encoding="utf-8"))
    translation = json.loads(TRANSLATION_JSON.read_text(encoding="utf-8"))
    print(f"    Loaded merged WR dataset: {df_all.height} WRs ({df_all.filter(pl.col('n_all') >= MIN_SNAPS_COHORT).height} with >=50 snaps).")

    print("2/3 Running Leave-One-Draft-Class-Out CV & Bootstrap 95% CIs...")
    report = build_report(df_all, adv_summary, translation)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("3/3 Writing wr_modeling_report.md...")
    OUTPUT_REPORT.write_text(report, encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s. Report written to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
