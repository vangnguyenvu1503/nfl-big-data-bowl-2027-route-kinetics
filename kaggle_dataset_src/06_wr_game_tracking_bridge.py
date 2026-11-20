"""Bridge Combine WR tracking features to real NFL Regular Season game tracking.

Uses lazy streaming ``polars.scan_csv`` across ``game_tracking_2023..2025.csv``
to isolate active route frames (``ball_snap`` to ``pass_forward``) for the 83
NFL Regular Season WRs, extracts real-game route kinetics into
``data/processed/wr_game_kinetics.parquet``, and computes Combine-to-NFL
kinetic translation correlations into ``data/processed/wr_kinetic_translation.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import polars as pl
from scipy import stats


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
ADV_KINETICS_PARQUET = PROCESSED_DIR / "wr_advanced_kinetics.parquet"
TARGETS_PARQUET = PROCESSED_DIR / "wr_targets.parquet"
OUTPUT_PARQUET = PROCESSED_DIR / "wr_game_kinetics.parquet"
OUTPUT_TRANSLATION_JSON = PROCESSED_DIR / "wr_kinetic_translation.json"

NULL_VALUES = ["NA", ""]
BREAK_ROUTES = ["OUT", "IN", "HITCH", "SLANT", "CORNER"]
DT_SECONDS = 0.1
TRACKING_SEASONS = (2023, 2024, 2025)
MIN_SNAPS_COHORT = 50


def load_valid_wr_route_plays(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """Load non-nullified REG WR route plays across 2023-2025."""
    games_reg = (
        pl.read_csv(raw_dir / "games.csv", null_values=NULL_VALUES)
        .filter(pl.col("season_type") == "REG")
        .select(["game_id", "season"])
    )
    wr_ids = (
        pl.read_csv(raw_dir / "combine_results.csv", null_values=NULL_VALUES)
        .filter(pl.col("combine_position") == "WR")
        .select("nfl_id")
    )
    plays = (
        pl.scan_csv(
            raw_dir / "player_play.csv",
            null_values=NULL_VALUES,
            infer_schema_length=None,
        )
        .select(["game_id", "play_id", "nfl_id", "play_nullified_by_penalty", "route_ran"])
        .filter(
            (pl.col("play_nullified_by_penalty") == "N")
            & pl.col("route_ran").is_not_null()
        )
        .collect()
        .join(games_reg, on="game_id", how="inner")
        .join(wr_ids, on="nfl_id", how="inner")
        .with_columns(pl.col("route_ran").is_in(BREAK_ROUTES).alias("is_break"))
        .select(["game_id", "play_id", "nfl_id", "season", "route_ran", "is_break"])
    )
    return plays


def extract_nfl_game_kinetics(raw_dir: Path = RAW_DIR) -> tuple[pl.DataFrame, dict[str, int]]:
    """Lazily scan 2023-2025 game tracking and aggregate active WR route kinetics."""
    valid_plays = load_valid_wr_route_plays(raw_dir)
    valid_plays_lazy = valid_plays.select(
        ["game_id", "play_id", "nfl_id", "is_break"]
    ).lazy()

    season_frames: list[pl.DataFrame] = []
    for yr in TRACKING_SEASONS:
        path = raw_dir / f"game_tracking_{yr}.csv"
        gt_yr = (
            pl.scan_csv(path, null_values=NULL_VALUES, infer_schema_length=50_000)
            .select(["game_id", "play_id", "nfl_id", "time", "s", "a", "dir", "event"])
            .join(valid_plays_lazy, on=["game_id", "play_id", "nfl_id"], how="inner")
            .collect(engine="streaming")
        )
        season_frames.append(gt_yr)

    gt_all = pl.concat(season_frames).sort(["nfl_id", "game_id", "play_id", "time"])
    total_matched_rows = int(gt_all.height)

    gt_active = (
        gt_all.with_columns(
            pl.int_range(0, pl.len()).over(["game_id", "play_id", "nfl_id"]).alias("f_idx")
        )
        .with_columns(
            pl.col("f_idx")
            .filter(pl.col("event") == "ball_snap")
            .min()
            .over(["game_id", "play_id", "nfl_id"])
            .alias("snap_idx"),
            pl.col("f_idx")
            .filter(pl.col("event").is_in(["pass_forward", "pass_shovel"]))
            .min()
            .over(["game_id", "play_id", "nfl_id"])
            .alias("pass_idx"),
            ((pl.col("s") - pl.col("s").shift(1)) / DT_SECONDS)
            .over(["game_id", "play_id", "nfl_id"])
            .alias("signed_a"),
            (
                pl.min_horizontal(
                    (pl.col("dir") - pl.col("dir").shift(1)).abs(),
                    360.0 - (pl.col("dir") - pl.col("dir").shift(1)).abs(),
                )
                * (np.pi / 180.0)
                / DT_SECONDS
            )
            .over(["game_id", "play_id", "nfl_id"])
            .alias("omega_rad"),
        )
        .filter(
            pl.col("snap_idx").is_not_null()
            & pl.col("pass_idx").is_not_null()
            & (pl.col("f_idx") >= pl.col("snap_idx"))
            & (pl.col("f_idx") <= pl.col("pass_idx"))
        )
        .with_columns((pl.col("s") * pl.col("omega_rad")).alias("centripetal_a"))
    )
    active_route_frames = int(gt_active.height)

    all_agg = gt_active.group_by("nfl_id").agg(
        pl.len().alias("nfl_route_frames"),
        pl.struct(["game_id", "play_id"]).n_unique().alias("nfl_route_plays"),
        pl.col("s").quantile(0.95).alias("nfl_route_p95_s"),
        pl.col("s").max().alias("nfl_route_max_s"),
        pl.col("centripetal_a")
        .filter(pl.col("s") > 2.5)
        .quantile(0.95)
        .alias("nfl_all_centripetal_p95"),
        pl.col("signed_a")
        .filter(pl.col("s") > 2.0)
        .quantile(0.05)
        .alias("nfl_all_decel_p05"),
    )

    brk_agg = (
        gt_active.filter(pl.col("is_break"))
        .group_by("nfl_id")
        .agg(
            pl.len().alias("nfl_break_frames"),
            pl.struct(["game_id", "play_id"]).n_unique().alias("nfl_break_plays"),
            pl.col("s").quantile(0.95).alias("nfl_break_p95_s"),
            pl.col("s").max().alias("nfl_break_max_s"),
            pl.col("centripetal_a")
            .filter(pl.col("s") > 2.5)
            .quantile(0.95)
            .alias("nfl_break_centripetal_p95"),
            pl.col("signed_a")
            .filter(pl.col("s") > 2.0)
            .quantile(0.05)
            .alias("nfl_break_decel_p05"),
        )
    )

    game_kinetics = (
        all_agg.join(brk_agg, on="nfl_id", how="left")
        .with_columns(
            pl.col("nfl_break_frames").fill_null(0),
            pl.col("nfl_break_plays").fill_null(0),
            pl.col("nfl_break_p95_s").fill_null(pl.col("nfl_route_p95_s")),
            pl.col("nfl_break_max_s").fill_null(pl.col("nfl_route_max_s")),
            pl.col("nfl_break_centripetal_p95").fill_null(pl.col("nfl_all_centripetal_p95")),
            pl.col("nfl_break_decel_p05").fill_null(pl.col("nfl_all_decel_p05")),
        )
        .sort("nfl_id")
    )

    counts = {
        "valid_route_plays": int(valid_plays.height),
        "total_matched_rows": total_matched_rows,
        "active_route_frames": active_route_frames,
        "n_wrs": int(game_kinetics.height),
    }
    return game_kinetics, counts


def compute_translation_report(
    game_kinetics: pl.DataFrame,
    adv_kinetics: pl.DataFrame,
    targets: pl.DataFrame,
    counts: dict[str, int],
) -> dict[str, object]:
    """Measure Combine-to-NFL Kinetic Translation correlations on N=69 and N=83."""
    merged = (
        game_kinetics.join(adv_kinetics, on="nfl_id", how="inner")
        .join(targets.select(["nfl_id", "n_all"]), on="nfl_id", how="inner")
    )
    combine_features = [
        ("forty_track_max_s", "Combine 40-Yd Tracking `forty_track_max_s`"),
        ("route_max_s", "Combine Route `route_max_s`"),
        ("vertical_route_max_s_z", "Combine Vertical `vertical_route_max_s_z`"),
        ("apex_speed_retention", "Combine Break `apex_speed_retention`"),
        ("pre_break_decel", "Combine Break `pre_break_decel`"),
        ("peak_centripetal_accel", "Combine Break `peak_centripetal_accel`"),
        ("braking_power_index", "Combine Hard-Break `braking_power_index`"),
        ("forty", "Baseline `forty` (s)"),
        ("ten_yd_split", "Baseline `ten_yd_split` (s)"),
        ("ngs_athleticism_score", "Baseline `ngs_athleticism_score`"),
    ]
    nfl_targets = [
        "nfl_route_p95_s",
        "nfl_route_max_s",
        "nfl_break_centripetal_p95",
        "nfl_break_decel_p05",
    ]

    cohorts: dict[str, object] = {}
    for label, min_n in [("cohort_50", MIN_SNAPS_COHORT), ("cohort_all", 0)]:
        sub = merged.filter(pl.col("n_all") >= min_n)
        rows: list[dict[str, object]] = []
        for fcol, flabel in combine_features:
            entry: dict[str, object] = {"feature": fcol, "label": flabel}
            for tcol in nfl_targets:
                pair = sub.select([fcol, tcol]).drop_nulls()
                n = int(pair.height)
                if n < 3:
                    entry[tcol] = {"n": n, "rho": None, "sp": None, "r": None, "pp": None}
                    continue
                x = pair[fcol].cast(pl.Float64).to_numpy()
                y = pair[tcol].cast(pl.Float64).to_numpy()
                sr = stats.spearmanr(x, y)
                pr = stats.pearsonr(x, y)
                entry[tcol] = {
                    "n": n,
                    "rho": float(sr.statistic),
                    "sp": float(sr.pvalue),
                    "r": float(pr.statistic),
                    "pp": float(pr.pvalue),
                }
            rows.append(entry)
        cohorts[label] = {"n_wrs": int(sub.height), "correlations": rows}

    return {"counts": counts, "targets": nfl_targets, "cohorts": cohorts}


def main() -> None:
    """Extract NFL WR game kinetics and compute Combine-to-NFL translation metrics."""
    t0 = time.perf_counter()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("1/3 Streaming 2023-2025 game tracking for REG WR route plays...")
    game_kinetics, counts = extract_nfl_game_kinetics(RAW_DIR)
    print(
        f"    Extracted {counts['active_route_frames']:,} active route frames "
        f"({counts['n_wrs']} WRs) from {counts['total_matched_rows']:,} matched rows."
    )

    print("2/3 Writing wr_game_kinetics.parquet...")
    game_kinetics.write_parquet(OUTPUT_PARQUET)

    print("3/3 Computing Combine-to-NFL Kinetic Translation correlations...")
    if not ADV_KINETICS_PARQUET.exists() or not TARGETS_PARQUET.exists():
        raise FileNotFoundError(
            "Run src/03_wr_soe_target.py and src/05_wr_advanced_kinetics.py first."
        )
    adv_df = pl.read_parquet(ADV_KINETICS_PARQUET)
    targets_df = pl.read_parquet(TARGETS_PARQUET)
    translation = compute_translation_report(game_kinetics, adv_df, targets_df, counts)
    OUTPUT_TRANSLATION_JSON.write_text(json.dumps(translation, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s. Report JSON written to {OUTPUT_TRANSLATION_JSON}")


if __name__ == "__main__":
    main()
