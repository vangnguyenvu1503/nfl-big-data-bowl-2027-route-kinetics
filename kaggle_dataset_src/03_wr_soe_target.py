"""Build denoised NFL Wide Receiver Separation Over Expected (SOE) targets.

Filters NFL Regular Season WR snaps with valid route and separation tracking,
fits an Out-of-Fold (Leave-One-Season-Out) LightGBM expected_separation model
on play context features to avoid cross-season data leakage, computes play-level
SOE = separation_at_pass_forward - expected_separation, and aggregates
Empirical-Bayes-shrunk player targets (`soe_all`, `soe_man`, `soe_break`) into
``data/processed/wr_targets.parquet``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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
OUTPUT_PARQUET = PROCESSED_DIR / "wr_targets.parquet"
OUTPUT_METRICS = PROCESSED_DIR / "wr_oof_metrics.json"

NULL_VALUES = ["NA", ""]
BREAK_ROUTES = ["OUT", "IN", "HITCH", "SLANT", "CORNER"]
CAT_COLS = [
    "route_ran",
    "team_coverage_man_zone",
    "team_coverage_type",
    "lined_up_position",
]
FEATURE_NAMES = CAT_COLS + ["in_motion_at_ball_snap", "cushion", "pass_length"]

SHRINKAGE_PRIORS = {
    "all": 50.0,
    "man": 30.0,
    "break": 40.0,
}


def load_wr_separation_plays(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """Load and filter REG non-nullified WR plays with route and separation values."""
    games_reg = (
        pl.read_csv(raw_dir / "games.csv", null_values=NULL_VALUES)
        .filter(pl.col("season_type") == "REG")
        .with_columns(
            pl.col("home_team_abbr").replace({"LA": "LAR"}),
            pl.col("visitor_team_abbr").replace({"LA": "LAR"}),
        )
        .select(["game_id", "season"])
    )
    wr_ids = (
        pl.read_csv(raw_dir / "combine_results.csv", null_values=NULL_VALUES)
        .filter(pl.col("combine_position") == "WR")
        .select(["nfl_id", "combine_position"])
    )
    players_meta = pl.read_csv(raw_dir / "players.csv", null_values=NULL_VALUES).select(
        ["nfl_id", "display_name", "draft_year"]
    )

    needed_cols = [
        "game_id",
        "play_id",
        "nfl_id",
        "team_abbr",
        "possession_team",
        "yardline_side",
        "play_nullified_by_penalty",
        "separation_at_pass_forward",
        "route_ran",
        "team_coverage_man_zone",
        "team_coverage_type",
        "cushion",
        "lined_up_position",
        "in_motion_at_ball_snap",
        "pass_length",
    ]
    plays = (
        pl.scan_csv(
            raw_dir / "player_play.csv",
            null_values=NULL_VALUES,
            infer_schema_length=None,
        )
        .select(needed_cols)
        .filter(
            (pl.col("play_nullified_by_penalty") == "N")
            & pl.col("route_ran").is_not_null()
            & pl.col("separation_at_pass_forward").is_not_null()
        )
        .collect()
        .with_columns(
            pl.col("team_abbr").replace({"LA": "LAR"}),
            pl.col("possession_team").replace({"LA": "LAR"}),
            pl.col("yardline_side").replace({"LA": "LAR"}),
        )
        .join(games_reg, on="game_id", how="inner")
        .join(wr_ids, on="nfl_id", how="inner")
        .join(players_meta, on="nfl_id", how="left")
        .sort(["season", "game_id", "play_id", "nfl_id"])
    )
    return plays


def encode_fold_features(
    train_df: pl.DataFrame,
    eval_df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit categorical mappings strictly on train_df and transform both train_df and eval_df."""

    def _to_matrix(df: pl.DataFrame, vocabs: dict[str, dict[str, int]]) -> np.ndarray:
        cols: list[np.ndarray] = []
        for col in CAT_COLS:
            mapping = vocabs[col]
            vals = df[col].fill_null("MISSING").to_list()
            cols.append(np.array([mapping.get(v, -1) for v in vals], dtype=np.float64))
        cols.append(
            df["in_motion_at_ball_snap"].cast(pl.Float64).fill_null(-1.0).to_numpy()
        )
        cols.append(df["cushion"].cast(pl.Float64).to_numpy())
        cols.append(df["pass_length"].cast(pl.Float64).to_numpy())
        return np.column_stack(cols)

    vocabs: dict[str, dict[str, int]] = {}
    for col in CAT_COLS:
        uniq = sorted(train_df[col].fill_null("MISSING").unique().to_list())
        vocabs[col] = {val: idx for idx, val in enumerate(uniq)}

    return _to_matrix(train_df, vocabs), _to_matrix(eval_df, vocabs)


def fit_predict_oof_by_season(
    df: pl.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run Leave-One-Season-Out LightGBM expected_separation predictions."""
    seasons_arr = df["season"].to_numpy()
    y_all = df["separation_at_pass_forward"].cast(pl.Float64).to_numpy()
    oof_pred = np.empty(df.height, dtype=np.float64)

    unique_seasons = sorted(int(s) for s in np.unique(seasons_arr))
    fold_metrics: list[dict[str, object]] = []
    gain_importances = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    split_importances = np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    for holdout_season in unique_seasons:
        tr_mask = seasons_arr != holdout_season
        va_mask = seasons_arr == holdout_season
        train_df = df.filter(pl.col("season") != holdout_season)
        eval_df = df.filter(pl.col("season") == holdout_season)

        X_tr, X_va = encode_fold_features(train_df, eval_df)
        y_tr = y_all[tr_mask]
        y_va = y_all[va_mask]

        model = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_tr, y_tr, categorical_feature=[0, 1, 2, 3])
        pred_va = model.predict(X_va)
        oof_pred[va_mask] = pred_va

        gain_importances += model.booster_.feature_importance(importance_type="gain")
        split_importances += model.booster_.feature_importance(importance_type="split")

        fold_metrics.append(
            {
                "season": holdout_season,
                "n_train": int(tr_mask.sum()),
                "n_val": int(va_mask.sum()),
                "r2": float(r2_score(y_va, pred_va)),
                "rmse": float(root_mean_squared_error(y_va, pred_va)),
                "mae": float(mean_absolute_error(y_va, pred_va)),
            }
        )

    n_folds = max(len(unique_seasons), 1)
    gain_avg = (gain_importances / n_folds).tolist()
    split_avg = (split_importances / n_folds).tolist()
    total_gain = sum(gain_avg) or 1.0

    feature_importance = [
        {
            "feature": fname,
            "avg_gain": float(g),
            "gain_share": float(g / total_gain),
            "avg_split": float(sp),
        }
        for fname, g, sp in sorted(
            zip(FEATURE_NAMES, gain_avg, split_avg),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    metrics: dict[str, object] = {
        "n_plays": int(df.height),
        "n_players": int(df["nfl_id"].n_unique()),
        "overall": {
            "r2": float(r2_score(y_all, oof_pred)),
            "rmse": float(root_mean_squared_error(y_all, oof_pred)),
            "mae": float(mean_absolute_error(y_all, oof_pred)),
            "target_mean": float(np.mean(y_all)),
            "target_std": float(np.std(y_all)),
            "oof_residual_mean": float(np.mean(y_all - oof_pred)),
            "oof_residual_std": float(np.std(y_all - oof_pred)),
        },
        "by_season": fold_metrics,
        "feature_importance": feature_importance,
    }
    return oof_pred, metrics


def _aggregate_eb_subset(
    sub_df: pl.DataFrame,
    prefix: str,
    k_prior: float,
) -> tuple[pl.DataFrame, float]:
    """Compute raw mean and Empirical Bayes shrunk SOE for a play subset."""
    mu0 = float(sub_df["soe"].mean())
    agg = sub_df.group_by("nfl_id").agg(
        pl.len().alias(f"n_{prefix}"),
        pl.col("separation_at_pass_forward").mean().alias(f"mean_sep_{prefix}"),
        pl.col("expected_separation").mean().alias(f"mean_exp_sep_{prefix}"),
        pl.col("soe").mean().alias(f"raw_soe_{prefix}"),
    )
    agg = agg.with_columns(
        (
            (pl.col(f"n_{prefix}") * pl.col(f"raw_soe_{prefix}") + k_prior * mu0)
            / (pl.col(f"n_{prefix}") + k_prior)
        ).alias(f"soe_{prefix}")
    )
    return agg, mu0


def aggregate_wr_soe_targets(df_with_soe: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, float]]:
    """Aggregate play-level SOE into EB-shrunk WR metrics (`soe_all`, `soe_man`, `soe_break`)."""
    meta = (
        df_with_soe.group_by("nfl_id")
        .agg(
            pl.col("display_name").first().alias("display_name"),
            pl.col("draft_year").first().alias("draft_year"),
        )
        .sort("nfl_id")
    )

    all_agg, mu0_all = _aggregate_eb_subset(df_with_soe, "all", SHRINKAGE_PRIORS["all"])
    man_agg, mu0_man = _aggregate_eb_subset(
        df_with_soe.filter(pl.col("team_coverage_man_zone") == "MAN_COVERAGE"),
        "man",
        SHRINKAGE_PRIORS["man"],
    )
    brk_agg, mu0_brk = _aggregate_eb_subset(
        df_with_soe.filter(pl.col("route_ran").is_in(BREAK_ROUTES)),
        "break",
        SHRINKAGE_PRIORS["break"],
    )

    targets = (
        meta.join(all_agg, on="nfl_id", how="left")
        .join(man_agg, on="nfl_id", how="left")
        .join(brk_agg, on="nfl_id", how="left")
        .with_columns(
            pl.col("n_man").fill_null(0),
            pl.col("mean_sep_man").fill_null(pl.col("mean_sep_all")),
            pl.col("mean_exp_sep_man").fill_null(pl.col("mean_exp_sep_all")),
            pl.col("raw_soe_man").fill_null(mu0_man),
            pl.col("soe_man").fill_null(mu0_man),
            pl.col("n_break").fill_null(0),
            pl.col("mean_sep_break").fill_null(pl.col("mean_sep_all")),
            pl.col("mean_exp_sep_break").fill_null(pl.col("mean_exp_sep_all")),
            pl.col("raw_soe_break").fill_null(mu0_brk),
            pl.col("soe_break").fill_null(mu0_brk),
        )
        .sort("nfl_id")
    )
    priors = {
        "mu0_all": mu0_all,
        "mu0_man": mu0_man,
        "mu0_break": mu0_brk,
        **{f"k_{k}": v for k, v in SHRINKAGE_PRIORS.items()},
    }
    return targets, priors


def main() -> None:
    """Run WR SOE target construction and save Parquet + OOF metrics."""
    t0 = time.perf_counter()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("1/4 Loading REG WR plays with valid routes and separation...")
    plays = load_wr_separation_plays(RAW_DIR)
    print(f"    Loaded {plays.height:,} plays across {plays['nfl_id'].n_unique()} WRs.")

    print("2/4 Fitting Leave-One-Season-Out LightGBM expected_separation model...")
    oof_pred, metrics = fit_predict_oof_by_season(plays)
    overall = metrics["overall"]
    print(
        f"    Overall OOF: R2={overall['r2']:.4f}, RMSE={overall['rmse']:.4f}, MAE={overall['mae']:.4f}"  # type: ignore[index]
    )

    print("3/4 Computing play-level SOE and Empirical Bayes shrinkage per WR...")
    plays_with_soe = plays.with_columns(
        pl.Series("expected_separation", oof_pred),
        (pl.col("separation_at_pass_forward") - pl.Series(oof_pred)).alias("soe"),
    )
    targets, priors = aggregate_wr_soe_targets(plays_with_soe)
    metrics["shrinkage_priors"] = priors

    print("4/4 Writing wr_targets.parquet and wr_oof_metrics.json...")
    targets.write_parquet(OUTPUT_PARQUET)
    OUTPUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s. Saved {targets.height} WRs to {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
