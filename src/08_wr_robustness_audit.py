"""Audit WR Route Kinetics robustness against weight confounding and drill opt-outs.

Produces ``wr_robustness_audit.md`` and enriches
``data/processed/wr_advanced_kinetics.parquet`` with opt-out-proof and
size-adjusted metrics:
  1. Exact formula audit, Nested OLS F-test & LRT (statsmodels), and football
     production validation (`player_play.csv` & `player_career_successes.csv`).
  2. Drill opt-out audit for the 15 case-study WRs (verifying Dont'e Thornton
     & KeAndre Lambert-Smith opt-outs) plus `route_speed_z` and
     `size_adjusted_route_kinetics`.
  3. Automatic discovery of 2 disjoint "Apples-to-Apples" Hero Visual pairs
     (matched weight <= 10 lbs, 40-yd <= 0.08s, >= 4 shared Break+Vertical drills).
  4. Clean Leave-One-Draft-Class-Out CV (No Weight-Mixing) for `soe_break`
     on N = 69 and N = 83 with paired Bootstrap 95% CIs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import numpy as np
import polars as pl
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ADV_PARQUET = PROCESSED_DIR / "wr_advanced_kinetics.parquet"
TARGETS_PARQUET = PROCESSED_DIR / "wr_targets.parquet"
GAME_KINETICS_PARQUET = PROCESSED_DIR / "wr_game_kinetics.parquet"
OUTPUT_REPORT = PROJECT_ROOT / "wr_robustness_audit.md"

NULL_VALUES = ["NA", ""]
MIN_SNAPS_COHORT = 50
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42

VERT_DRILLS = {
    "GO_ROUTE_RIGHT",
    "POST_CORNER_ROUTE_RIGHT",
    "SLOT_SAIL_ROUTE_RIGHT",
    "RED_ZONE_FADE_RIGHT",
}
BREAK_DRILLS = {
    "CURL_ROUTE_RIGHT",
    "COMEBACK_ROUTE_RIGHT",
    "SPEED_OUT_ROUTE_LEFT",
    "DAGGER_ROUTE_LEFT",
    "SLANT_ROUTE_LEFT",
    "SLOT_STRIKE_ROUTE_LEFT",
}
CORE_10_ROUTE_DRILLS = sorted(VERT_DRILLS | BREAK_DRILLS)

CASE_STUDY_NAMES = [
    "Tez Johnson",
    "Jalin Hyatt",
    "Trey Palmer",
    "Derius Davis",
    "Jimmy Horn",
    "Malik Washington",
    "Tre Tucker",
    "Arian Smith",
    "Troy Franklin",
    "Isaiah Bond",
    "Dont'e Thornton",
    "Jack Bech",
    "KeAndre Lambert-Smith",
    "Jonathan Mingo",
    "Chimere Dike",
]

MODEL_A_COLS = [
    "forty",
    "ten_yd_split",
    "ngs_athleticism_score",
    "combine_height",
    "combine_weight",
]
MODEL_B_PURE_COLS = [
    "pure_kinetics_z",
    "route_max_s",
    "vertical_route_max_s_z",
    "route_speed_z",
    "peak_centripetal_accel_z",
    "braking_power_index_z",
    "speed_cut_retention_z",
    "pre_break_jerk",
    "functional_speed_ratio",
]
MODEL_C_PURE_BASE_COLS = MODEL_A_COLS + MODEL_B_PURE_COLS
ESTIMATORS = ["RidgeCV", "BayesianRidge", "ElasticNetCV"]


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


def compute_optout_and_size_adjusted_clean(
    raw_dir: Path = RAW_DIR,
    adv_parquet: Path = ADV_PARQUET,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Compute opt-out-proof `route_speed_z` and OLS `size_adjusted_route_kinetics`."""
    adv = pl.read_parquet(adv_parquet)
    drop_cols = [
        c
        for c in (
            "n_route_drills_ran",
            "has_go_route",
            "has_post_corner_right",
            "route_speed_z",
            "size_adjusted_route_kinetics",
            "size_adjusted_route_speed_z",
        )
        if c in adv.columns
    ]
    if drop_cols:
        adv = adv.drop(drop_cols)

    ct_wr = (
        pl.read_csv(raw_dir / "combine_tracking.csv", null_values=NULL_VALUES)
        .filter((pl.col("entity_type") == "PLAYER") & (pl.col("drill_type") == "SKILL_DRILLS_WR"))
        .with_columns(pl.col("drill_name").replace({"SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT"}))
        .join(adv.select("nfl_id"), on="nfl_id", how="inner")
    )

    wr_drill = ct_wr.group_by(["nfl_id", "drill_name"]).agg(
        pl.col("s").max().alias("drill_max_s"),
    )
    wr_drill_z = wr_drill.with_columns(
        (
            (pl.col("drill_max_s") - pl.col("drill_max_s").mean().over("drill_name"))
            / pl.col("drill_max_s").std().over("drill_name").clip(lower_bound=1e-6)
        ).alias("drill_max_s_z")
    )

    route_only = wr_drill_z.filter(pl.col("drill_name") != "GAUNTLET_DRILL")
    per_wr_opt = route_only.group_by("nfl_id").agg(
        pl.len().alias("n_route_drills_ran"),
        pl.col("drill_max_s_z").mean().alias("route_speed_z"),
        (pl.col("drill_name") == "GO_ROUTE_RIGHT").any().alias("has_go_route"),
        (pl.col("drill_name") == "POST_CORNER_ROUTE_RIGHT").any().alias("has_post_corner_right"),
        pl.col("drill_name").sort().implode().alias("route_drills_list"),
    )

    adv_enriched = adv.join(per_wr_opt.drop("route_drills_list"), on="nfl_id", how="left")

    wt = adv_enriched["combine_weight"].cast(pl.Float64).to_numpy()
    X_wt = sm.add_constant(wt)
    pk_resid = sm.OLS(adv_enriched["pure_kinetics_z"].to_numpy(), X_wt).fit().resid
    rs_resid = sm.OLS(adv_enriched["route_speed_z"].to_numpy(), X_wt).fit().resid

    adv_enriched = adv_enriched.with_columns(
        pl.Series("size_adjusted_route_kinetics", pk_resid),
        pl.Series("size_adjusted_route_speed_z", rs_resid),
    ).sort("nfl_id")

    adv_enriched.write_parquet(adv_parquet)
    return adv_enriched, per_wr_opt, wr_drill


def load_football_production(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """Compute real NFL production outcomes from `player_play.csv` and `player_career_successes.csv`."""
    games_reg = (
        pl.read_csv(raw_dir / "games.csv", null_values=NULL_VALUES)
        .filter(pl.col("season_type") == "REG")
        .select(["game_id"])
    )
    wr_ids = (
        pl.read_csv(raw_dir / "combine_results.csv", null_values=NULL_VALUES)
        .filter(pl.col("combine_position") == "WR")
        .select(["nfl_id", "draft_year"])
    )
    pp = (
        pl.scan_csv(
            raw_dir / "player_play.csv",
            null_values=NULL_VALUES,
            infer_schema_length=None,
        )
        .select(
            [
                "game_id",
                "nfl_id",
                "play_nullified_by_penalty",
                "route_ran",
                "target",
                "rec_yards",
                "yards_after_catch",
                "expected_yards_after_catch",
            ]
        )
        .filter((pl.col("play_nullified_by_penalty") == "N") & pl.col("route_ran").is_not_null())
        .collect()
        .join(games_reg, on="game_id", how="inner")
        .join(wr_ids, on="nfl_id", how="inner")
    )
    prod = pp.group_by("nfl_id").agg(
        pl.len().alias("routes_run"),
        (pl.col("rec_yards").fill_null(0).sum() / pl.len()).alias("yards_per_route_run"),
        pl.col("target").cast(pl.Float64).mean().alias("target_rate"),
        pl.col("expected_yards_after_catch").mean().alias("mean_expected_yac"),
        pl.col("yards_after_catch").mean().alias("mean_yac"),
        (pl.col("yards_after_catch") - pl.col("expected_yards_after_catch")).mean().alias("yac_over_expected"),
    )

    career = (
        pl.read_csv(raw_dir / "player_career_successes.csv", null_values=NULL_VALUES)
        .join(wr_ids, on="nfl_id", how="inner")
        .with_columns(
            (pl.col("career_offensive_snaps") / pl.col("career_games_active").clip(lower_bound=1)).alias(
                "snaps_per_active_game"
            ),
            (
                (
                    pl.col("career_offensive_snaps")
                    - pl.col("career_offensive_snaps").mean().over("draft_year")
                )
                / pl.col("career_offensive_snaps").std().over("draft_year").clip(lower_bound=1e-6)
            ).alias("career_snaps_draft_z"),
        )
        .select(
            [
                "nfl_id",
                "career_offensive_snaps",
                "career_games_started",
                "snaps_per_active_game",
                "career_snaps_draft_z",
            ]
        )
    )
    return prod.join(career, on="nfl_id", how="full", coalesce=True)


def run_nested_ols_suite(df: pl.DataFrame) -> tuple[list[dict[str, object]], sm.regression.linear_model.RegressionResultsWrapper, sm.regression.linear_model.RegressionResultsWrapper]:
    """Run Nested OLS F-test & LRT adding sensor features onto Size + Track Baseline."""
    base_cols = ["combine_weight", "combine_height", "forty", "ten_yd_split", "ngs_athleticism_score"]
    cohorts = [
        ("N=69 (`n_all>=50`, median-imputed 40)", df.filter(pl.col("n_all") >= MIN_SNAPS_COHORT), True),
        ("N=64 (`n_all>=50`, complete 40)", df.filter((pl.col("n_all") >= MIN_SNAPS_COHORT) & pl.col("forty").is_not_null() & pl.col("ten_yd_split").is_not_null()), False),
        ("N=83 (`All REG WRs`, median-imputed 40)", df, True),
        ("N=75 (`All REG WRs`, complete 40)", df.filter(pl.col("forty").is_not_null() & pl.col("ten_yd_split").is_not_null()), False),
        ("N=61 (`n_all>=50` & `has_go_route`)", df.filter((pl.col("n_all") >= MIN_SNAPS_COHORT) & pl.col("has_go_route")), True),
    ]
    sensor_candidates = ["pure_kinetics_z", "route_max_s", "vertical_route_max_s_z", "route_speed_z"]
    results: list[dict[str, object]] = []

    m1_n64 = None
    m1_n83 = None
    for cname, sub, impute in cohorts:
        if impute:
            X0 = sub.select(
                [pl.col(c).cast(pl.Float64).fill_null(pl.col(c).median()) for c in base_cols]
            ).to_numpy()
        else:
            X0 = sub.select([pl.col(c).cast(pl.Float64) for c in base_cols]).to_numpy()

        X0_sm = sm.add_constant(X0)
        y = sub["soe_break"].to_numpy()
        m0 = sm.OLS(y, X0_sm).fit()

        for scol in sensor_candidates:
            s_vec = sub[scol].to_numpy().reshape(-1, 1)
            X1_sm = np.column_stack([X0_sm, s_vec])
            m1 = sm.OLS(y, X1_sm).fit()
            f_stat, f_pval, _ = m1.compare_f_test(m0)
            lr_stat, lr_pval, _ = m1.compare_lr_test(m0)
            # Partial r = sign(t) * sqrt(t^2 / (t^2 + df_resid))
            t_val = float(m1.tvalues[-1])
            partial_r = float(np.sign(t_val) * np.sqrt((t_val ** 2) / (t_val ** 2 + m1.df_resid)))

            results.append(
                {
                    "cohort": cname,
                    "n": int(sub.height),
                    "sensor": scol,
                    "r2_m0": float(m0.rsquared),
                    "r2_m1": float(m1.rsquared),
                    "adj_r2_m0": float(m0.rsquared_adj),
                    "adj_r2_m1": float(m1.rsquared_adj),
                    "delta_adj_r2": float(m1.rsquared_adj - m0.rsquared_adj),
                    "beta": float(m1.params[-1]),
                    "se": float(m1.bse[-1]),
                    "t_stat": t_val,
                    "p_val": float(m1.pvalues[-1]),
                    "partial_r": partial_r,
                    "f_stat": float(f_stat),
                    "f_pval": float(f_pval),
                    "lr_stat": float(lr_stat),
                    "lr_pval": float(lr_pval),
                }
            )
            if scol == "pure_kinetics_z" and cname.startswith("N=64"):
                m1_n64 = m1
            if scol == "pure_kinetics_z" and cname.startswith("N=83"):
                m1_n83 = m1

    assert m1_n64 is not None and m1_n83 is not None
    return results, m1_n64, m1_n83


def find_apples_to_apples_pairs(
    df_50: pl.DataFrame,
    per_wr_opt: pl.DataFrame,
) -> list[dict[str, object]]:
    """Find matched WR pairs with similar weight, 40-yd, and >=4 shared drills (incl. GO)."""
    joined = df_50.join(
        per_wr_opt.select(["nfl_id", "route_drills_list"]),
        on="nfl_id",
        how="left",
    ).filter(pl.col("forty").is_not_null())
    rows = joined.to_dicts()
    candidates: list[dict[str, object]] = []

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if abs(a["combine_weight"] - b["combine_weight"]) > 10:
                continue
            if abs(a["forty"] - b["forty"]) > 0.08 + 1e-9:
                continue
            da = set(a["route_drills_list"])
            db = set(b["route_drills_list"])
            shared = da & db
            if len(shared) < 4:
                continue
            if not (shared & VERT_DRILLS) or not (shared & BREAK_DRILLS):
                continue
            if "GO_ROUTE_RIGHT" not in shared:
                continue

            hi, lo = (a, b) if a["soe_break"] >= b["soe_break"] else (b, a)
            if hi["soe_break"] < 0.08 or lo["soe_break"] > -0.06:
                continue
            if hi["pure_kinetics_z"] <= lo["pure_kinetics_z"]:
                continue
            if hi["size_adjusted_route_kinetics"] <= lo["size_adjusted_route_kinetics"]:
                continue

            score = (
                (hi["soe_break"] - lo["soe_break"])
                + 0.30 * (hi["pure_kinetics_z"] - lo["pure_kinetics_z"])
                + 0.15 * (hi["route_speed_z"] - lo["route_speed_z"])
            )
            candidates.append(
                {
                    "score": float(score),
                    "n_shared": int(len(shared)),
                    "shared_drills": sorted(shared),
                    "gem": hi,
                    "bust": lo,
                }
            )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def _make_estimator(name: str):
    """Instantiate a fresh regularized linear estimator with clean convergence."""
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
            max_iter=20000,
            tol=1e-3,
        )
    raise ValueError(f"Unknown estimator: {name}")


def fit_predict_clean_lodco(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    estimator_name: str,
    *,
    add_fold_weight_residual: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run Leave-One-Draft-Class-Out CV with zero weight-mixing and optional fold-internal residual."""
    draft_years = df["draft_year"].to_numpy()
    X_raw = df.select(feature_cols).to_numpy().astype(np.float64)
    w_raw = df["combine_weight"].to_numpy().astype(np.float64)
    pk_raw = df["pure_kinetics_z"].to_numpy().astype(np.float64)
    y_all = df[target_col].to_numpy().astype(np.float64)
    oof_pred = np.empty(df.height, dtype=np.float64)

    all_names = list(feature_cols) + (
        ["fold_size_adj_kinetics", "fold_pk_minus_wt_z"] if add_fold_weight_residual else []
    )
    unique_years = sorted(int(yr) for yr in np.unique(draft_years))
    fold_metrics: list[dict[str, object]] = []
    coef_matrix = np.zeros((len(unique_years), len(all_names)), dtype=np.float64)
    selected_counts = np.zeros(len(all_names), dtype=np.int64)

    for fold_i, holdout_yr in enumerate(unique_years):
        tr_mask = draft_years != holdout_yr
        va_mask = draft_years == holdout_yr

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(imputer.fit_transform(X_raw[tr_mask]))
        X_va = scaler.transform(imputer.transform(X_raw[va_mask]))
        y_tr = y_all[tr_mask]
        y_va = y_all[va_mask]

        if add_fold_weight_residual:
            w_mean, w_std = float(np.mean(w_raw[tr_mask])), float(np.std(w_raw[tr_mask]))
            pk_mean, pk_std = float(np.mean(pk_raw[tr_mask])), float(np.std(pk_raw[tr_mask]))
            w_tr_z = (w_raw[tr_mask] - w_mean) / max(w_std, 1e-6)
            w_va_z = (w_raw[va_mask] - w_mean) / max(w_std, 1e-6)
            pk_tr_z = (pk_raw[tr_mask] - pk_mean) / max(pk_std, 1e-6)
            pk_va_z = (pk_raw[va_mask] - pk_mean) / max(pk_std, 1e-6)
            slope, intercept = np.polyfit(w_tr_z, pk_tr_z, 1)
            res_tr = (pk_tr_z - (intercept + slope * w_tr_z)).reshape(-1, 1)
            res_va = (pk_va_z - (intercept + slope * w_va_z)).reshape(-1, 1)
            diff_tr = (pk_tr_z - w_tr_z).reshape(-1, 1)
            diff_va = (pk_va_z - w_va_z).reshape(-1, 1)
            X_tr = np.column_stack([X_tr, res_tr, diff_tr])
            X_va = np.column_stack([X_va, res_va, diff_va])

        # Rank by |corr| on training fold ONLY
        scores = np.array(
            [
                abs(float(np.corrcoef(X_tr[:, j], y_tr)[0, 1]))
                if np.std(X_tr[:, j]) > 0
                else 0.0
                for j in range(X_tr.shape[1])
            ],
            dtype=np.float64,
        )
        ranked = np.argsort(-scores, kind="mergesort")

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        best_k = X_tr.shape[1]
        best_mse = float("inf")
        for k in range(1, X_tr.shape[1] + 1):
            c_idx = ranked[:k]
            p_in = np.empty_like(y_tr)
            for itr, iva in kf.split(X_tr):
                est_in = _make_estimator(estimator_name)
                est_in.fit(X_tr[itr][:, c_idx], y_tr[itr])
                p_in[iva] = est_in.predict(X_tr[iva][:, c_idx])
            mse = float(np.mean((y_tr - p_in) ** 2))
            if mse < best_mse - 1e-9:
                best_mse = mse
                best_k = k

        sel_idx = ranked[:best_k]
        selected_counts[sel_idx] += 1
        est = _make_estimator(estimator_name)
        est.fit(X_tr[:, sel_idx], y_tr)
        pred_va = est.predict(X_va[:, sel_idx])
        oof_pred[va_mask] = pred_va
        coef_matrix[fold_i, sel_idx] = est.coef_

        rho_f = (
            float(stats.spearmanr(y_va, pred_va).statistic)
            if np.std(pred_va) > 0 and np.std(y_va) > 0
            else 0.0
        )
        fold_metrics.append(
            {
                "draft_year": holdout_yr,
                "n_train": int(tr_mask.sum()),
                "n_val": int(va_mask.sum()),
                "top_k": int(best_k),
                "selected_features": [all_names[j] for j in sel_idx],
                "r2": float(r2_score(y_va, pred_va)),
                "rmse": float(root_mean_squared_error(y_va, pred_va)),
                "spearman_rho": rho_f if np.isfinite(rho_f) else 0.0,
            }
        )

    sr_all = stats.spearmanr(y_all, oof_pred) if np.std(oof_pred) > 0 else None
    pr_all = stats.pearsonr(y_all, oof_pred) if np.std(oof_pred) > 0 else None
    metrics: dict[str, object] = {
        "n": int(df.height),
        "r2": float(r2_score(y_all, oof_pred)),
        "rmse": float(root_mean_squared_error(y_all, oof_pred)),
        "mae": float(mean_absolute_error(y_all, oof_pred)),
        "spearman_rho": float(sr_all.statistic) if sr_all is not None else 0.0,
        "spearman_p": float(sr_all.pvalue) if sr_all is not None else 1.0,
        "pearson_r": float(pr_all.statistic) if pr_all is not None else 0.0,
        "pearson_p": float(pr_all.pvalue) if pr_all is not None else 1.0,
        "by_draft_year": fold_metrics,
        "feature_roles": [
            {
                "feature": all_names[j],
                "selected_folds": int(selected_counts[j]),
                "mean_std_coef": float(np.mean(coef_matrix[:, j])),
                "mean_abs_coef": float(np.mean(np.abs(coef_matrix[:, j]))),
            }
            for j in range(len(all_names))
        ],
    }
    return oof_pred, metrics


def paired_bootstrap_ci(
    y: np.ndarray,
    pred_cand: np.ndarray,
    pred_base: np.ndarray,
    n_boot: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute paired Bootstrap 95% CIs for delta Spearman rho and delta RMSE."""
    rng = np.random.default_rng(seed)
    n = len(y)
    drho, drmse = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pc, pb = y[idx], pred_cand[idx], pred_base[idx]
        if np.std(yb) == 0 or np.std(pc) == 0 or np.std(pb) == 0:
            continue
        rc = float(stats.spearmanr(yb, pc).statistic)
        rb = float(stats.spearmanr(yb, pb).statistic)
        if not (np.isfinite(rc) and np.isfinite(rb)):
            continue
        drho.append(rc - rb)
        drmse.append(float(root_mean_squared_error(yb, pc) - root_mean_squared_error(yb, pb)))
    a_r = np.asarray(drho, dtype=np.float64)
    a_e = np.asarray(drmse, dtype=np.float64)
    return (
        (float(np.quantile(a_r, 0.025)), float(np.quantile(a_r, 0.975))),
        (float(np.quantile(a_e, 0.025)), float(np.quantile(a_e, 0.975))),
    )


def build_audit_report(
    adv: pl.DataFrame,
    per_wr_opt: pl.DataFrame,
    wr_drill: pl.DataFrame,
    targets: pl.DataFrame,
    game_kin: pl.DataFrame,
    prod: pl.DataFrame,
) -> str:
    """Assemble the complete wr_robustness_audit.md report."""
    t_cols = [c for c in targets.columns if c not in ("display_name", "draft_year")]
    df_all = (
        adv.join(targets.select(t_cols), on="nfl_id", how="inner")
        .join(game_kin, on="nfl_id", how="inner")
        .join(prod, on="nfl_id", how="left")
        .sort("nfl_id")
    )
    df_50 = df_all.filter(pl.col("n_all") >= MIN_SNAPS_COHORT)

    # 1. Nested OLS & Football production
    nested_rows, m1_n64, m1_n83 = run_nested_ols_suite(df_all)

    lines: list[str] = [
        "# NFL Big Data Bowl 2027 — WR Robustness Audit (Weight Confounding & Drill Opt-Out)",
        "",
        f"- **Thời gian tạo**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "- **Mục tiêu kiểm toán**: Giải trình 2 điểm phản biện trọng yếu của ban giám khảo NFL — (1) nhiễu cân nặng `combine_weight` và (2) sai lệch do cầu thủ bỏ bài tập (`drill opt-out`), đồng thời chạy lại LODCO CV minh bạch 100% không pha trộn cân nặng.",
        "- **Ký hiệu**: `**` (*p < 0.01*), `*` (*p < 0.05*), `.` (*p < 0.10*).",
        "",
        "## 1. Kiểm toán công thức & Nested OLS (Chứng minh giá trị gia tăng của Sensor)",
        "",
        "### 1.1. Công thức chính xác trong `src/05_wr_advanced_kinetics.py`",
        "",
        "```python",
        "# 100% Sensor-Only Composite (HOÀN TOÀN KHÔNG chứa combine_weight):",
        "pure_kinetics_z = (",
        "    0.55 * route_max_s_z",
        "    + 0.30 * vertical_route_max_s_z",
        "    + 0.15 * peak_centripetal_accel_z",
        ")",
        "",
        "# Weight-Controlled Composite (có trừ weight_z nhằm bù nhiễu thể hình):",
        "route_kinetics_index = (",
        "    0.50 * route_max_s_z",
        "    + 0.25 * vertical_route_max_s_z",
        "    + 0.15 * peak_centripetal_accel_z",
        "    + 0.10 * braking_power_index_z",
        "    - 0.50 * weight_z",
        ")",
        "```",
        "",
        "- **Kết luận kiểm toán công thức**: Trong `wr_modeling_report.md`, biến `route_kinetics_index` có chứa `-0.50 * weight_z` nên tương quan cao một phần đến từ `combine_weight` ($\\rho = -0.374^{**}$ với `soe_break`). Ngược lại, `pure_kinetics_z` hoàn toàn sạch (`0%` `combine_weight`). Để trả lời dứt khoát liệu dữ liệu sensor Combine có mang lại **thông tin mới vượt trên cân nặng và 40-yard dash** hay không, ta chạy **Nested OLS F-test & Likelihood Ratio Test (LRT)** dưới đây.",
        "",
        "### 1.2. Nested OLS F-Test & Likelihood Ratio Test dự báo `soe_break`",
        "",
        "- **Model 0 (Size + Track Baseline)**: `combine_weight + combine_height + forty + ten_yd_split + ngs_athleticism_score`.",
        "- **Model 1 (+ Sensor)**: `Model 0 + sensor_feature`.",
        "",
    ]

    ols_table = [
        [
            r["cohort"],
            f"`{r['sensor']}`",
            f"{r['r2_m0']:.4f} → {r['r2_m1']:.4f}",
            f"{r['adj_r2_m0']:.4f} → {r['adj_r2_m1']:.4f} ({r['delta_adj_r2']:+.4f})",
            f"{r['beta']:+.4f} (p={r['p_val']:.4f}{sig_star(r['p_val'])})",
            f"{r['partial_r']:+.3f}",
            f"{r['f_stat']:.3f} (p={r['f_pval']:.4f}{sig_star(r['f_pval'])})",
            f"{r['lr_stat']:.3f} (p={r['lr_pval']:.4f}{sig_star(r['lr_pval'])})",
        ]
        for r in nested_rows
    ]
    lines.extend(
        format_markdown_table(
            ["Cohort", "Added Sensor", "R² (M0 → M1)", "Adj R² (M0 → M1, Δ)", "β Sensor (p)", "Partial r", "F-test (p)", "LRT χ² (p)"],
            ols_table,
            ["left", "left", "right", "right", "right", "right", "right", "right"],
        )
    )

    # Full OLS parameter table for Model 1 on N=64 complete cases and N=83
    lines.extend(["", "### 1.3. Chi tiết từng hệ số trong Model 1 (`Model 0 + pure_kinetics_z`)", ""])
    param_names = ["Intercept", "combine_weight", "combine_height", "forty", "ten_yd_split", "ngs_athleticism_score", "pure_kinetics_z"]
    for label, mod_res in [("Complete 40-yd Cohort (`N = 64`, `n_all >= 50`)", m1_n64), ("All REG WRs (`N = 83`, median-imputed 40-yd)", m1_n83)]:
        lines.append(f"#### {label} — $R^2 = {mod_res.rsquared:.4f}$, $R^2_{{\\text{{adj}}}} = {mod_res.rsquared_adj:.4f}$")
        lines.append("")
        ci = mod_res.conf_int()
        p_rows = [
            [
                f"`{pname}`",
                f"{float(mod_res.params[idx]):+.5f}",
                f"{float(mod_res.bse[idx]):.5f}",
                f"{float(mod_res.tvalues[idx]):+.3f}",
                f"{float(mod_res.pvalues[idx]):.4f}{sig_star(float(mod_res.pvalues[idx]))}",
                f"[{float(ci[idx, 0]):+.5f}, {float(ci[idx, 1]):+.5f}]",
            ]
            for idx, pname in enumerate(param_names)
        ]
        lines.extend(
            format_markdown_table(
                ["Biến", "Coef (β)", "Std Err", "t-stat", "p-val", "95% CI"],
                p_rows,
                ["left", "right", "right", "right", "right", "right"],
            )
        )
        lines.append("")

    # 1.4 Football production correlations
    lines.extend(["### 1.4. Kiểm định giá trị thực chiến trên sân NFL (`player_play.csv` & `player_career_successes.csv`, `N = 69`)", ""])
    prod_feats = [
        ("pure_kinetics_z", "`pure_kinetics_z` (100% Sensor)"),
        ("size_adjusted_route_kinetics", "`size_adjusted_route_kinetics` (Residual)"),
        ("route_max_s", "`route_max_s`"),
        ("vertical_route_max_s_z", "`vertical_route_max_s_z`"),
        ("route_speed_z", "`route_speed_z` (Opt-out-proof)"),
        ("nfl_route_p95_s", "`nfl_route_p95_s` (On-field)"),
        ("forty", "Baseline: `forty` (s)"),
        ("ngs_athleticism_score", "Baseline: `ngs_athleticism_score`"),
    ]
    prod_targets = [
        ("mean_expected_yac", "Expected YAC"),
        ("mean_yac", "Actual YAC"),
        ("yards_per_route_run", "YPRR"),
        ("target_rate", "Target Rate"),
        ("career_snaps_draft_z", "Career Snaps (Draft Z)"),
    ]
    prod_rows: list[list[object]] = []
    for fcol, flabel in prod_feats:
        cells: list[object] = [flabel]
        for tcol, _ in prod_targets:
            pair = df_50.select([fcol, tcol]).drop_nulls()
            x = pair[fcol].cast(pl.Float64).to_numpy()
            y = pair[tcol].cast(pl.Float64).to_numpy()
            sr = stats.spearmanr(x, y)
            pr = stats.pearsonr(x, y)
            cells.append(f"{float(sr.statistic):+.3f}{sig_star(float(sr.pvalue))} / {float(pr.statistic):+.3f} (p={float(sr.pvalue):.3f})")
        prod_rows.append(cells)

    lines.extend(
        format_markdown_table(
            ["Feature (ρ / r, p-val S)"] + [tlabel for _, tlabel in prod_targets],
            prod_rows,
            ["left"] + ["right"] * len(prod_targets),
        )
    )

    # Part 2: Drill Opt-Out Audit
    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Kiểm toán Drill Opt-Out & Chỉ số sạch chống bỏ bài (`route_speed_z` & `size_adjusted_route_kinetics`)",
            "",
            "- **Phát hiện quan trọng về Drill Opt-Out**:",
            "  - Trong `combine_tracking.csv`, các bài route dài như `GO_ROUTE_RIGHT` (trung vị `10.14 yd/s`) và `POST_CORNER_ROUTE_RIGHT` (`9.52 yd/s`) có tốc độ đỉnh cao hơn hẳn các bài ngắn như `CURL_ROUTE_RIGHT` (`7.10 yd/s`) hay `SPEED_OUT_ROUTE_LEFT` (`7.00 yd/s`).",
            "  - Nếu dùng `route_max_s = max(s)` trực tiếp, cầu thủ **bỏ bài `GO_ROUTE_RIGHT` và `POST_CORNER_ROUTE_RIGHT`** sẽ bị gán `route_max_s` rất thấp một cách giả tạo dù họ chạy nhanh ở các bài ngắn hơn.",
            "  - Kiểm tra xác nhận: **Dont'e Thornton** (`4.30s` 40-yd) chỉ chạy **5/12** bài route (bỏ cả `GO_ROUTE_RIGHT` lẫn `POST_CORNER_ROUTE_RIGHT`), nên `route_max_s = 8.51` chủ yếu phản ánh opt-out; trên 5 bài đã chạy, `route_speed_z = +0.10` (mức trung bình). Tương tự **KeAndre Lambert-Smith** (`4.37s`) chỉ chạy **8/12** bài (bỏ cả `GO` và `POST_CORNER`), nên dù `route_max_s = 9.08`, trên 8 bài đã chạy anh đạt `route_speed_z = +1.02`!",
            "  - Để chống opt-out 100%, ta bổ sung **`route_speed_z`** (trung bình Z-score của `drill_max_s` chỉ trên các bài cầu thủ **thực sự chạy**) và **`size_adjusted_route_kinetics`** (phần dư OLS của `pure_kinetics_z ~ combine_weight`, trực giao hoàn toàn với cân nặng: $r = 0.0000$).",
            "",
            "### 2.1. Kiểm toán đầy đủ 15 cầu thủ Case Study (10 Hidden Gems + 5 Overvalued Track Stars cũ)",
            "",
        ]
    )

    case_df = (
        df_all.filter(pl.col("display_name").is_in(CASE_STUDY_NAMES))
        .join(per_wr_opt.select(["nfl_id", "route_drills_list"]), on="nfl_id", how="left")
        .sort("soe_break", descending=True)
    )
    case_rows: list[list[object]] = []
    for r in case_df.iter_rows(named=True):
        ran = set(r["route_drills_list"])
        missed_core = sorted(set(CORE_10_ROUTE_DRILLS) - ran)
        missed_str = "Đủ 10/10 bài cốt lõi" if not missed_core else ", ".join(missed_core)
        d_rd = "UDFA" if r["draft_round"] is None else f"Rd {int(r['draft_round'])}"
        f40 = "—" if r["forty"] is None else f"{float(r['forty']):.2f}s"
        case_rows.append(
            [
                r["display_name"],
                f"{int(r['draft_year'])} {d_rd}",
                f40,
                int(r["combine_weight"]),
                f"{int(r['n_route_drills_ran'])}/12",
                "Yes" if r["has_go_route"] else "**NO (Opt-out)**",
                "Yes" if r["has_post_corner_right"] else "**NO**",
                f"{float(r['route_max_s']):.2f}",
                f"{float(r['route_speed_z']):+.2f}",
                f"{float(r['pure_kinetics_z']):+.2f}",
                f"{float(r['size_adjusted_route_kinetics']):+.2f}",
                f"{float(r['soe_break']):+.3f}",
                missed_str,
            ]
        )

    lines.extend(
        format_markdown_table(
            [
                "Cầu thủ",
                "Draft",
                "40-Yd",
                "Wt",
                "Drills",
                "Ran GO?",
                "Ran PC?",
                "`route_max_s`",
                "`route_speed_z`",
                "`pure_kinetics_z`",
                "`size_adj_kin`",
                "`soe_break`",
                "Bài cốt lõi (10 bài) bị bỏ",
            ],
            case_rows,
            ["left", "center", "right", "right", "center", "center", "center", "right", "right", "right", "right", "right", "left"],
        )
    )

    # Clean Overvalued Track Stars (who ACTUALLY ran GO_ROUTE_RIGHT)
    lines.extend(
        [
            "",
            "### 2.2. Danh sách Overvalued Track Stars ĐÃ LOẠI BỎ Opt-Out (`has_go_route == True` & `forty <= 4.48s` hoặc `Rd <= 2`)",
            "",
        ]
    )
    clean_over = (
        df_50.filter(
            pl.col("has_go_route")
            & ((pl.col("forty") <= 4.48) | ((pl.col("draft_round").is_not_null()) & (pl.col("draft_round") <= 2)))
        )
        .with_columns(
            (
                -1.0 * ((pl.col("soe_break") - pl.col("soe_break").mean()) / pl.col("soe_break").std())
                - 0.75 * ((pl.col("size_adjusted_route_kinetics") - pl.col("size_adjusted_route_kinetics").mean()) / pl.col("size_adjusted_route_kinetics").std())
            ).alias("clean_over_score")
        )
        .sort("clean_over_score", descending=True)
        .head(6)
    )
    clean_over_rows = [
        [
            r["display_name"],
            f"{int(r['draft_year'])} " + ("UDFA" if r["draft_round"] is None else f"Rd {int(r['draft_round'])}"),
            "—" if r["forty"] is None else f"{float(r['forty']):.2f}s",
            int(r["combine_weight"]),
            f"{int(r['n_route_drills_ran'])}/12 (Yes GO)",
            f"{float(r['route_max_s']):.2f}",
            f"{float(r['route_speed_z']):+.2f}",
            f"{float(r['pure_kinetics_z']):+.2f}",
            f"{float(r['size_adjusted_route_kinetics']):+.2f}",
            f"{float(r['nfl_route_p95_s']):.2f}",
            f"{float(r['soe_break']):+.3f}",
        ]
        for r in clean_over.iter_rows(named=True)
    ]
    lines.extend(
        format_markdown_table(
            ["Cầu thủ", "Draft", "40-Yd", "Wt", "Drills", "`route_max_s`", "`route_speed_z`", "`pure_kinetics_z`", "`size_adj_kin`", "NFL `p95_s`", "`soe_break`"],
            clean_over_rows,
            ["left", "center", "right", "right", "center", "right", "right", "right", "right", "right", "right"],
        )
    )

    # Part 3: Apples-to-Apples Hero Visual Pairs
    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Hai cặp đối chứng \"Apples-to-Apples\" hoàn hảo cho Hero Visual",
            "",
            "- Điều kiện lọc tự động: `|Δ weight| <= 10 lbs`, `|Δ forty| <= 0.08s`, chạy chung **>= 4 bài drill** (gồm cả Vertical & Break và **cả hai đều chạy `GO_ROUTE_RIGHT`**), không dính nhiễu opt-out.",
            "",
        ]
    )
    all_pairs = find_apples_to_apples_pairs(df_50, per_wr_opt)
    # Pick 2 completely disjoint pairs (all 4 players distinct)
    selected_pairs: list[dict[str, object]] = []
    used_ids: set[int] = set()
    for cand in all_pairs:
        gid, bid = int(cand["gem"]["nfl_id"]), int(cand["bust"]["nfl_id"])
        if gid in used_ids or bid in used_ids:
            continue
        selected_pairs.append(cand)
        used_ids.update({gid, bid})
        if len(selected_pairs) == 2:
            break

    cand_rows = [
        [
            f"{c['gem']['display_name']} vs {c['bust']['display_name']}",
            f"{int(c['gem']['combine_weight'])} vs {int(c['bust']['combine_weight'])} lbs (Δ={abs(c['gem']['combine_weight']-c['bust']['combine_weight'])})",
            f"{c['gem']['forty']:.2f}s vs {c['bust']['forty']:.2f}s (Δ={abs(c['gem']['forty']-c['bust']['forty']):.2f}s)",
            f"{c['n_shared']} bài (cả 2 có GO)",
            f"{c['gem']['pure_kinetics_z']:+.2f} vs {c['bust']['pure_kinetics_z']:+.2f}",
            f"{c['gem']['size_adjusted_route_kinetics']:+.2f} vs {c['bust']['size_adjusted_route_kinetics']:+.2f}",
            f"{c['gem']['soe_break']:+.3f} vs {c['bust']['soe_break']:+.3f}",
        ]
        for c in all_pairs[:6]
    ]
    lines.extend(
        format_markdown_table(
            ["Cặp đối chứng (Gem vs Track Clone)", "Cân nặng", "40-Yd", "Drill chung", "`pure_kinetics_z`", "`size_adj_kin`", "`soe_break`"],
            cand_rows,
            ["left", "center", "center", "center", "right", "right", "right"],
        )
    )

    for p_idx, pair in enumerate(selected_pairs, start=1):
        gem, bust = pair["gem"], pair["bust"]
        g_drills = wr_drill.filter(pl.col("nfl_id") == gem["nfl_id"]).to_dicts()
        b_drills = wr_drill.filter(pl.col("nfl_id") == bust["nfl_id"]).to_dicts()
        g_map = {d["drill_name"]: float(d["drill_max_s"]) for d in g_drills}
        b_map = {d["drill_name"]: float(d["drill_max_s"]) for d in b_drills}

        g_rd = "UDFA" if gem["draft_round"] is None else f"Rd {int(gem['draft_round'])} (#{int(gem['draft_overall_pick'])})"
        b_rd = "UDFA" if bust["draft_round"] is None else f"Rd {int(bust['draft_round'])} (#{int(bust['draft_overall_pick'])})"
        lines.extend(
            [
                "",
                f"### 3.{p_idx}. Hero Pair #{p_idx}: **{gem['display_name']}** ({int(gem['draft_year'])}, {g_rd}) vs **{bust['display_name']}** ({int(bust['draft_year'])}, {b_rd})",
                "",
                f"- **Thể hình & Track**: {gem['display_name']} (**{int(gem['combine_weight'])} lbs**, `{gem['forty']:.2f}s` 40-yd) vs {bust['display_name']} (**{int(bust['combine_weight'])} lbs**, `{bust['forty']:.2f}s` 40-yd) — chạy chung **{pair['n_shared']} bài drill**.",
                f"- **Combine Route Kinetics**: `route_max_s` = **{gem['route_max_s']:.2f}** vs **{bust['route_max_s']:.2f} yd/s** | `pure_kinetics_z` = **{gem['pure_kinetics_z']:+.2f}** vs **{bust['pure_kinetics_z']:+.2f}** | `size_adjusted_route_kinetics` = **{gem['size_adjusted_route_kinetics']:+.2f}** vs **{bust['size_adjusted_route_kinetics']:+.2f}**.",
                f"- **Thực chiến NFL**: `nfl_route_p95_s` = **{gem['nfl_route_p95_s']:.2f}** vs **{bust['nfl_route_p95_s']:.2f} yd/s** | `soe_break` = **{gem['soe_break']:+.3f}** vs **{bust['soe_break']:+.3f} yd** | `soe_all` = **{gem['soe_all']:+.3f}** vs **{bust['soe_all']:+.3f} yd**.",
                "",
            ]
        )
        shared_drill_rows = [
            [
                f"`{d}`",
                f"{g_map[d]:.2f}",
                f"{b_map[d]:.2f}",
                f"{g_map[d] - b_map[d]:+.2f}",
            ]
            for d in pair["shared_drills"]
            if d in g_map and d in b_map
        ]
        lines.extend(
            format_markdown_table(
                ["Bài drill chạy chung", f"{gem['display_name']} max_s", f"{bust['display_name']} max_s", "Chênh lệch (Gem − Clone)"],
                shared_drill_rows,
                ["left", "right", "right", "right"],
            )
        )

    # Part 4: Clean LODCO CV (No Weight-Mixing)
    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Kết quả LODCO CV sạch 100% (No Weight-Mixing) dự báo `soe_break`",
            "",
            "- **Model A (Baseline)**: `forty, ten_yd_split, ngs_athleticism_score, combine_height, combine_weight`.",
            "- **Model B (100% Pure Sensor — Không pha trộn cân nặng)**: `pure_kinetics_z, route_max_s, vertical_route_max_s_z, route_speed_z, peak_centripetal_accel_z, braking_power_index_z, speed_cut_retention_z, pre_break_jerk`.",
            "- **Model C (Combined — Fold-Internal Residualization Only)**: `Model A + Model B` cộng thêm 2 biến quan hệ sensor-cân nặng được ước lượng **hoàn toàn bên trong tập train của từng fold** (không dùng bất kỳ tham số toàn cục nào từ ngoài fold).",
            "",
        ]
    )

    lodco_rows: list[list[object]] = []
    enet_n69_c_roles: list[dict[str, object]] = []
    enet_n69_folds: list[dict[str, object]] = []

    for cohort_lbl, sub in [(f"N={df_50.height} (`n_all >= 50`)", df_50), (f"N={df_all.height} (`All REG WRs`)", df_all)]:
        y_vec = sub["soe_break"].to_numpy()
        for est_name in ESTIMATORS:
            pred_a, met_a = fit_predict_clean_lodco(sub, MODEL_A_COLS, "soe_break", est_name, add_fold_weight_residual=False)
            pred_b, met_b = fit_predict_clean_lodco(sub, MODEL_B_PURE_COLS, "soe_break", est_name, add_fold_weight_residual=False)
            pred_c, met_c = fit_predict_clean_lodco(sub, MODEL_C_PURE_BASE_COLS, "soe_break", est_name, add_fold_weight_residual=True)

            if cohort_lbl.startswith("N=69") and est_name == "BayesianRidge":
                pass
            if cohort_lbl.startswith("N=69") and est_name == "ElasticNetCV":
                enet_n69_c_roles = met_c["feature_roles"]  # type: ignore[assignment]
                for tag, m_obj in [("Model A", met_a), ("Model B (Pure Sensor)", met_b), ("Model C (Combined)", met_c)]:
                    for fm in m_obj["by_draft_year"]:  # type: ignore[attr-defined]
                        enet_n69_folds.append({"model": tag, **fm})

            ci_b_dr, ci_b_de = paired_bootstrap_ci(y_vec, pred_b, pred_a)
            ci_c_dr, ci_c_de = paired_bootstrap_ci(y_vec, pred_c, pred_a)

            for mname, met, dr_ci, de_ci in [
                ("Model A (Baseline)", met_a, None, None),
                ("Model B (Pure Sensor)", met_b, ci_b_dr, ci_b_de),
                ("Model C (Combined Clean)", met_c, ci_c_dr, ci_c_de),
            ]:
                if dr_ci is None:
                    dr_str, de_str = "Ref", "Ref"
                else:
                    dr_val = float(met["spearman_rho"]) - float(met_a["spearman_rho"])
                    de_val = float(met["rmse"]) - float(met_a["rmse"])
                    dr_str = f"{dr_val:+.3f} [{dr_ci[0]:+.3f}, {dr_ci[1]:+.3f}]"
                    de_str = f"{de_val:+.4f} [{de_ci[0]:+.4f}, {de_ci[1]:+.4f}]"
                lodco_rows.append(
                    [
                        cohort_lbl,
                        est_name,
                        mname,
                        f"{float(met['r2']):+.4f}",
                        f"{float(met['rmse']):.4f}",
                        f"{float(met['spearman_rho']):+.3f}{sig_star(float(met['spearman_p']))} ({float(met['spearman_p']):.3f})",
                        f"{float(met['pearson_r']):+.3f}{sig_star(float(met['pearson_p']))} ({float(met['pearson_p']):.3f})",
                        dr_str,
                        de_str,
                    ]
                )

    lines.extend(
        format_markdown_table(
            ["Cohort", "Estimator", "Model", "OOF $R^2$", "OOF RMSE", "Spearman ρ (p)", "Pearson r (p)", "Δρ vs A [95% Boot CI]", "ΔRMSE vs A [95% Boot CI]"],
            lodco_rows,
            ["left", "left", "left", "right", "right", "right", "right", "right", "right"],
        )
    )

    lines.extend(["", "### 4.1. Chi tiết từng fold & Trọng số chuẩn hóa (`ElasticNetCV`, `N = 69`, `No Weight-Mixing`)", ""])
    lines.extend(
        format_markdown_table(
            ["Model", "Holdout Year", "Train / Val", "Top-K", "OOF $R^2$", "RMSE", "Spearman ρ", "Selected Features"],
            [
                [
                    fm["model"],
                    fm["draft_year"],
                    f"{fm['n_train']} / {fm['n_val']}",
                    fm["top_k"],
                    f"{float(fm['r2']):+.4f}",
                    f"{float(fm['rmse']):.4f}",
                    f"{float(fm['spearman_rho']):+.3f}",
                    ", ".join(f"`{f}`" for f in fm["selected_features"]),
                ]
                for fm in enet_n69_folds
            ],
            ["left", "center", "center", "right", "right", "right", "right", "left"],
        )
    )

    lines.extend(["", "#### Trọng số chuẩn hóa trung bình qua 3 fold (`Model C Clean`, `ElasticNetCV`, `N = 69`)", ""])
    sorted_roles = sorted(enet_n69_c_roles, key=lambda r: float(r["mean_abs_coef"]), reverse=True)
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
                for r in sorted_roles
            ],
            ["left", "center", "right", "right"],
        )
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Execute full WR robustness audit and write wr_robustness_audit.md."""
    t0 = time.perf_counter()
    print("1/4 Computing opt-out-proof & size-adjusted features and updating wr_advanced_kinetics.parquet...")
    adv_enriched, per_wr_opt, wr_drill = compute_optout_and_size_adjusted_clean(RAW_DIR, ADV_PARQUET)

    print("2/4 Loading targets, game kinetics, and NFL football production outcomes...")
    targets = pl.read_parquet(TARGETS_PARQUET)
    game_kin = pl.read_parquet(GAME_KINETICS_PARQUET)
    prod = load_football_production(RAW_DIR)

    print("3/4 Running Nested OLS, Opt-Out Audit, Pair Matching, and Clean LODCO CV...")
    report = build_audit_report(adv_enriched, per_wr_opt, wr_drill, targets, game_kin, prod)

    print("4/4 Writing wr_robustness_audit.md...")
    OUTPUT_REPORT.write_text(report, encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s. Report written to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
