"""Normalize WR Combine drills by physical group and engineer coupled kinetics features.

Groups `SKILL_DRILLS_WR` into 3 homogeneous families:
  (a) Vertical Speed: GO_ROUTE_RIGHT, POST_CORNER_ROUTE_RIGHT,
      SLOT_SAIL_ROUTE_RIGHT, RED_ZONE_FADE_RIGHT
  (b) Hard-Break: CURL_ROUTE_RIGHT, COMEBACK_ROUTE_RIGHT
  (c) Speed-Cut: SPEED_OUT_ROUTE_LEFT, DAGGER_ROUTE_LEFT,
      SLANT_ROUTE_LEFT, SLOT_STRIKE_ROUTE_LEFT

Computes within-drill Z-scores to remove missing-drill bias, engineers coupled
biomechanical metrics (`peak_centripetal_accel`, `braking_power_index`,
`vertical_route_max_s_z`, `speed_cut_retention_z`, `pure_kinetics_z`, and
weight-controlled `route_kinetics_index`), and saves
``data/processed/wr_advanced_kinetics.parquet``.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np
import polars as pl


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
BASE_BIOMECH_PARQUET = PROCESSED_DIR / "wr_biomechanics.parquet"
OUTPUT_PARQUET = PROCESSED_DIR / "wr_advanced_kinetics.parquet"
OUTPUT_SUMMARY_JSON = PROCESSED_DIR / "wr_advanced_kinetics_summary.json"

NULL_VALUES = ["NA", ""]
DT_SECONDS = 0.1
START_SPEED_THRESHOLD = 1.5

VERTICAL_DRILLS = [
    "GO_ROUTE_RIGHT",
    "POST_CORNER_ROUTE_RIGHT",
    "SLOT_SAIL_ROUTE_RIGHT",
    "RED_ZONE_FADE_RIGHT",
]
HARD_BREAK_DRILLS = [
    "CURL_ROUTE_RIGHT",
    "COMEBACK_ROUTE_RIGHT",
]
SPEED_CUT_DRILLS = [
    "SPEED_OUT_ROUTE_LEFT",
    "DAGGER_ROUTE_LEFT",
    "SLANT_ROUTE_LEFT",
    "SLOT_STRIKE_ROUTE_LEFT",
]
ALL_GROUPED_DRILLS = VERTICAL_DRILLS + HARD_BREAK_DRILLS + SPEED_CUT_DRILLS

DRILL_GROUP_MAP = {
    **{d: "Vertical Speed" for d in VERTICAL_DRILLS},
    **{d: "Hard-Break" for d in HARD_BREAK_DRILLS},
    **{d: "Speed-Cut" for d in SPEED_CUT_DRILLS},
}


def _load_bio_module() -> ModuleType:
    """Import sibling `04_wr_biomechanics.py` without modifying package structure."""
    mod_path = Path(__file__).resolve().parent / "04_wr_biomechanics.py"
    spec = importlib.util.spec_from_file_location("wr_biomechanics_mod", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wr_biomechanics_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def compute_rep_coupled_kinetics(rep_df: pl.DataFrame, drill_name: str) -> dict[str, object]:
    """Extract segment-bounded coupled physical metrics from one Combine drill rep."""
    s = rep_df["s"].to_numpy()
    x = rep_df["x"].to_numpy()
    d = rep_df["dir"].to_numpy()
    n = int(len(s))

    cutoff_80 = int(np.floor(0.80 * n))
    active_idx = np.where(s > START_SPEED_THRESHOLD)[0]
    start_idx = (
        int(active_idx[0])
        if (len(active_idx) > 0 and int(active_idx[0]) < cutoff_80 - 1)
        else 0
    )

    signed_a = np.zeros(n, dtype=np.float64)
    if n > 1:
        signed_a[1:] = (s[1:] - s[:-1]) / DT_SECONDS

    raw_d1 = np.zeros(n, dtype=np.float64)
    if n > 1:
        raw_d1[1:] = np.abs(d[1:] - d[:-1])
    turn1 = np.minimum(raw_d1, 360.0 - raw_d1) / DT_SECONDS
    turn3 = np.convolve(turn1, np.ones(3, dtype=np.float64) / 3.0, mode="same")
    omega_rad = np.deg2rad(turn3)
    centripetal_a = s * omega_rad

    lo = min(max(start_idx + 3, int(0.15 * n)), cutoff_80 - 1)
    if lo <= start_idx:
        lo = min(start_idx + 1, cutoff_80 - 1)
    hi = min(cutoff_80, int(0.75 * n))
    if drill_name in HARD_BREAK_DRILLS and hi > lo:
        peak_x_idx = int(lo + np.argmax(x[lo:hi]))
        hi = min(hi, max(lo + 2, peak_x_idx + 3))
    hi = max(lo + 1, min(hi, cutoff_80))

    win_turn = turn3[lo:hi].copy()
    win_s = s[lo:hi]
    if np.any(win_s > START_SPEED_THRESHOLD):
        win_turn[win_s <= START_SPEED_THRESHOLD] = -1.0
    apex_idx = int(lo + np.argmax(win_turn))

    stem_s = s[start_idx : max(start_idx + 1, apex_idx + 1)]
    s_entry = float(np.max(stem_s)) if len(stem_s) > 0 else float(np.max(s))

    pre_a = signed_a[max(start_idx + 1, apex_idx - 6) : apex_idx + 2]
    pre_break_decel = float(np.min(pre_a)) if len(pre_a) > 0 else 0.0
    braking_power_index = float(s_entry * abs(min(pre_break_decel, 0.0)))

    brk_lat = centripetal_a[max(lo, apex_idx - 2) : min(cutoff_80, apex_idx + 3)]
    peak_centripetal_accel = (
        float(np.max(brk_lat)) if len(brk_lat) > 0 else float(centripetal_a[apex_idx])
    )

    apex_s = float(np.mean(s[max(start_idx, apex_idx - 1) : min(n, apex_idx + 2)]))
    apex_speed_retention = apex_s / max(s_entry, 1e-6)

    exit_a = signed_a[min(n - 1, apex_idx + 1) : min(n, apex_idx + 8)]
    post_break_exit_burst = float(np.max(exit_a)) if len(exit_a) > 0 else 0.0

    return {
        "rep_max_s": float(np.max(s)),
        "s_entry": s_entry,
        "pre_break_decel": pre_break_decel,
        "peak_centripetal_accel": peak_centripetal_accel,
        "braking_power_index": braking_power_index,
        "apex_speed_retention": apex_speed_retention,
        "post_break_exit_burst": post_break_exit_burst,
    }


def build_advanced_wr_kinetics(raw_dir: Path = RAW_DIR) -> tuple[pl.DataFrame, dict[str, object]]:
    """Compute drill-normalized Z-scores and coupled physical metrics for all 107 WRs."""
    bio_mod = _load_bio_module()
    combine_wr_ids, _, ct_wr = bio_mod.load_wr_tracking(raw_dir)

    combine_meta = pl.read_csv(
        raw_dir / "combine_results.csv", null_values=NULL_VALUES
    ).filter(pl.col("combine_position") == "WR").select(
        ["nfl_id", "combine_height", "combine_weight"]
    )
    players_meta = pl.read_csv(raw_dir / "players.csv", null_values=NULL_VALUES).select(
        ["nfl_id", "draft_round"]
    )

    if BASE_BIOMECH_PARQUET.exists():
        base_bio = pl.read_parquet(BASE_BIOMECH_PARQUET)
    else:
        base_bio, _, _ = bio_mod.extract_wr_biomechanics(raw_dir)

    wr_grouped_skill = ct_wr.filter(
        (pl.col("drill_type") == "SKILL_DRILLS_WR")
        & pl.col("drill_name").is_in(ALL_GROUPED_DRILLS)
    )
    rep_rows: list[dict[str, object]] = []
    for (nfl_id, drill_name, event_id), rep_df in wr_grouped_skill.group_by(
        ["nfl_id", "drill_name", "event_id"], maintain_order=True
    ):
        if rep_df.height < 12:
            continue
        metrics = compute_rep_coupled_kinetics(rep_df, str(drill_name))
        rep_rows.append(
            {
                "nfl_id": int(nfl_id),
                "drill_name": str(drill_name),
                "drill_group": DRILL_GROUP_MAP[str(drill_name)],
                "event_id": str(event_id),
                **metrics,
            }
        )

    reps_df = pl.DataFrame(rep_rows)
    metric_cols = [
        "rep_max_s",
        "s_entry",
        "pre_break_decel",
        "peak_centripetal_accel",
        "braking_power_index",
        "apex_speed_retention",
        "post_break_exit_burst",
    ]

    # Average reps within (nfl_id, drill_name), then compute within-drill Z-scores
    wr_drill = reps_df.group_by(["nfl_id", "drill_name", "drill_group"]).agg(
        pl.len().alias("n_reps"),
        *[pl.col(c).mean().alias(c) for c in metric_cols],
    )
    wr_drill_z = wr_drill.with_columns(
        *[
            (
                (pl.col(c) - pl.col(c).mean().over("drill_name"))
                / pl.col(c).std().over("drill_name").clip(lower_bound=1e-6)
            ).alias(f"{c}_z")
            for c in metric_cols
        ],
        pl.col("rep_max_s").median().over("drill_name").alias("drill_med_max_s"),
    )

    drill_group_summary = (
        wr_drill.group_by(["drill_group", "drill_name"])
        .agg(
            pl.col("nfl_id").n_unique().alias("n_wrs"),
            pl.col("rep_max_s").median().alias("med_max_s"),
            pl.col("pre_break_decel").median().alias("med_decel"),
            pl.col("peak_centripetal_accel").median().alias("med_centripetal_a"),
            pl.col("braking_power_index").median().alias("med_braking_power"),
            pl.col("apex_speed_retention").median().alias("med_retention"),
        )
        .sort(["drill_group", "med_max_s"], descending=[False, True])
    )

    # (a) Vertical Speed: weight Z-scores toward top-speed deep stems (GO, POST_CORNER)
    vert_agg = (
        wr_drill_z.filter(pl.col("drill_name").is_in(VERTICAL_DRILLS))
        .with_columns(
            ((pl.col("drill_med_max_s") - 6.0).clip(lower_bound=0.1) ** 4).alias("vert_w")
        )
        .group_by("nfl_id")
        .agg(
            ((pl.col("rep_max_s_z") * pl.col("vert_w")).sum() / pl.col("vert_w").sum()).alias(
                "vertical_route_max_s_z"
            ),
            pl.col("rep_max_s").max().alias("vertical_route_max_s"),
        )
    )

    # (b) Hard-Break: braking power and deceleration/exit burst
    hard_agg = (
        wr_drill_z.filter(pl.col("drill_name").is_in(HARD_BREAK_DRILLS))
        .group_by("nfl_id")
        .agg(
            pl.col("braking_power_index").mean().alias("braking_power_index"),
            pl.col("braking_power_index_z").mean().alias("braking_power_index_z"),
            pl.col("pre_break_decel_z").mean().alias("hard_break_decel_z"),
            pl.col("post_break_exit_burst_z").mean().alias("hard_break_exit_burst_z"),
        )
    )

    # (c) Speed-Cut: speed retention & lateral centripetal load
    scut_agg = (
        wr_drill_z.filter(pl.col("drill_name").is_in(SPEED_CUT_DRILLS))
        .group_by("nfl_id")
        .agg(
            pl.col("apex_speed_retention_z").mean().alias("speed_cut_retention_z"),
            pl.col("rep_max_s_z").mean().alias("speed_cut_max_s_z"),
            pl.col("peak_centripetal_accel_z").mean().alias("speed_cut_centripetal_z"),
        )
    )

    # Centripetal load across all breaking drills (Hard-Break + Speed-Cut)
    lat_agg = (
        wr_drill_z.filter(pl.col("drill_name").is_in(HARD_BREAK_DRILLS + SPEED_CUT_DRILLS))
        .group_by("nfl_id")
        .agg(
            pl.col("peak_centripetal_accel").mean().alias("peak_centripetal_accel"),
            pl.col("peak_centripetal_accel_z").mean().alias("peak_centripetal_accel_z"),
        )
    )

    adv = (
        base_bio.join(combine_meta, on="nfl_id", how="left")
        .join(players_meta, on="nfl_id", how="left")
        .join(vert_agg, on="nfl_id", how="left")
        .join(hard_agg, on="nfl_id", how="left")
        .join(scut_agg, on="nfl_id", how="left")
        .join(lat_agg, on="nfl_id", how="left")
        .with_columns(
            pl.col("vertical_route_max_s").fill_null(pl.col("route_max_s")),
            pl.col("vertical_route_max_s_z").fill_null(0.0),
            pl.col("braking_power_index").fill_null(pl.col("braking_power_index").median()),
            pl.col("braking_power_index_z").fill_null(0.0),
            pl.col("hard_break_decel_z").fill_null(0.0),
            pl.col("hard_break_exit_burst_z").fill_null(0.0),
            pl.col("speed_cut_retention_z").fill_null(0.0),
            pl.col("speed_cut_max_s_z").fill_null(0.0),
            pl.col("speed_cut_centripetal_z").fill_null(0.0),
            pl.col("peak_centripetal_accel").fill_null(pl.col("peak_centripetal_accel").median()),
            pl.col("peak_centripetal_accel_z").fill_null(0.0),
        )
        .with_columns(
            (
                (pl.col("route_max_s") - pl.col("route_max_s").mean())
                / pl.col("route_max_s").std()
            ).alias("route_max_s_z"),
            (
                (pl.col("combine_weight") - pl.col("combine_weight").mean())
                / pl.col("combine_weight").std()
            ).alias("weight_z"),
        )
        .with_columns(
            (
                0.55 * pl.col("route_max_s_z")
                + 0.30 * pl.col("vertical_route_max_s_z")
                + 0.15 * pl.col("peak_centripetal_accel_z")
            ).alias("pure_kinetics_z"),
            (
                0.50 * pl.col("route_max_s_z")
                + 0.25 * pl.col("vertical_route_max_s_z")
                + 0.15 * pl.col("peak_centripetal_accel_z")
                + 0.10 * pl.col("braking_power_index_z")
                - 0.50 * pl.col("weight_z")
            ).alias("route_kinetics_index"),
        )
        .sort("nfl_id")
    )

    assert adv.height == combine_wr_ids.height, "Row count mismatch across 107 WRs"
    summary = {
        "n_wrs": int(adv.height),
        "n_reps_grouped": int(reps_df.height),
        "drill_groups": drill_group_summary.to_dicts(),
    }
    return adv, summary


def main() -> None:
    """Build wr_advanced_kinetics.parquet and summary metadata."""
    t0 = time.perf_counter()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("1/2 Computing within-drill Z-scores & coupled route kinetics for 107 WRs...")
    adv_df, summary = build_advanced_wr_kinetics(RAW_DIR)

    print("2/2 Writing wr_advanced_kinetics.parquet & summary JSON...")
    adv_df.write_parquet(OUTPUT_PARQUET)
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s. Saved {adv_df.height} WRs to {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
