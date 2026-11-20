"""Unit tests for WR SOE target construction and Combine drill biomechanics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TARGETS_PATH = PROCESSED_DIR / "wr_targets.parquet"
BIOMECH_PATH = PROCESSED_DIR / "wr_biomechanics.parquet"


def _load_module(name: str, path: Path) -> ModuleType:
    """Import a script whose filename begins with digits via importlib."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_no_abnormal_nans_in_targets_and_biomechanics() -> None:
    """Verify processed WR Parquets have 0 nulls/NaNs/Infs and valid physical ranges."""
    assert TARGETS_PATH.exists(), f"Missing {TARGETS_PATH}"
    assert BIOMECH_PATH.exists(), f"Missing {BIOMECH_PATH}"

    targets = pl.read_parquet(TARGETS_PATH)
    biomech = pl.read_parquet(BIOMECH_PATH)

    assert targets.height == 83
    assert targets["nfl_id"].n_unique() == 83
    assert biomech.height == 107
    assert biomech["nfl_id"].n_unique() == 107

    target_cols = [
        "soe_all",
        "soe_man",
        "soe_break",
        "raw_soe_all",
        "raw_soe_man",
        "raw_soe_break",
    ]
    for col in target_cols:
        assert targets[col].null_count() == 0, f"Nulls found in {col}"
        arr = targets[col].to_numpy()
        assert np.all(np.isfinite(arr)), f"NaN/Inf found in {col}"

    biomech_cols = [
        "route_max_s",
        "functional_speed_ratio",
        "pre_break_decel",
        "pre_break_jerk",
        "apex_speed_retention",
        "break_curvature",
        "post_break_exit_burst",
        "gauntlet_speed_retention",
        "gauntlet_path_straightness",
    ]
    for col in biomech_cols:
        assert biomech[col].null_count() == 0, f"Nulls found in {col}"
        arr = biomech[col].to_numpy()
        assert np.all(np.isfinite(arr)), f"NaN/Inf found in {col}"

    # Physical range invariants
    assert bool((biomech["route_max_s"] > 5.0).all())
    assert bool(((biomech["functional_speed_ratio"] > 0.5) & (biomech["functional_speed_ratio"] < 1.3)).all())
    assert bool((biomech["pre_break_decel"] < 0.0).all())
    assert bool((biomech["pre_break_jerk"] < 0.0).all())
    assert bool(((biomech["apex_speed_retention"] > 0.0) & (biomech["apex_speed_retention"] <= 1.2)).all())
    assert bool((biomech["break_curvature"] > 0.0).all())
    assert bool((biomech["post_break_exit_burst"] > 0.0).all())
    assert bool(((biomech["gauntlet_speed_retention"] > 0.0) & (biomech["gauntlet_speed_retention"] <= 1.0)).all())
    assert bool(((biomech["gauntlet_path_straightness"] > 0.5) & (biomech["gauntlet_path_straightness"] <= 1.0001)).all())


def test_no_season_leakage_in_soe_oof() -> None:
    """Ensure perturbing a holdout season's target does not change its own OOF predictions."""
    soe_mod = _load_module("wr_soe_target", SRC_DIR / "03_wr_soe_target.py")
    plays = soe_mod.load_wr_separation_plays()
    seasons = plays["season"].to_numpy()

    oof_base, _ = soe_mod.fit_predict_oof_by_season(plays)

    for holdout_season in (2023, 2025):
        perturbed = plays.with_columns(
            pl.when(pl.col("season") == holdout_season)
            .then(pl.col("separation_at_pass_forward") + 50.0)
            .otherwise(pl.col("separation_at_pass_forward"))
            .alias("separation_at_pass_forward")
        )
        oof_perturbed, _ = soe_mod.fit_predict_oof_by_season(perturbed)

        mask = seasons == holdout_season
        assert np.allclose(oof_base[mask], oof_perturbed[mask], atol=1e-12), (
            f"Leakage detected: season {holdout_season} OOF predictions changed when its own targets shifted."
        )
        assert not np.allclose(oof_base[~mask], oof_perturbed[~mask], atol=1e-6), (
            f"Sanity check failed: shifting season {holdout_season} did not alter other folds."
        )


def test_break_apex_not_in_last_20_percent_frames() -> None:
    """Verify Break Apex stays strictly before the final 20% walk-off frames on all reps."""
    bio_mod = _load_module("wr_biomechanics", SRC_DIR / "04_wr_biomechanics.py")

    biomech = pl.read_parquet(BIOMECH_PATH)
    assert bool((biomech["max_apex_rel"] < 0.80).all())

    _, _, ct_wr = bio_mod.load_wr_tracking()
    reps_df = bio_mod.segment_all_wr_reps(ct_wr)
    assert reps_df.height > 1000

    assert bool((reps_df["start_idx"] >= 0).all())
    assert bool((reps_df["start_idx"] < reps_df["apex_idx"]).all())
    assert bool((reps_df["apex_idx"] < reps_df["cutoff_80"]).all())
    assert bool((reps_df["apex_rel"] < 0.80).all())

    break_reps = reps_df.filter(pl.col("drill_name").is_in(bio_mod.CORE_BREAK_DRILLS))
    assert break_reps.height > 500
    assert bool((break_reps["apex_rel"] < 0.80).all())

    # Check raw frame speed at start_idx on sample breaking-route reps
    wr_skill = ct_wr.filter(
        (pl.col("drill_type") == "SKILL_DRILLS_WR")
        & pl.col("drill_name").is_in(bio_mod.CORE_BREAK_DRILLS)
    )
    checked = 0
    for (_, drill_name, _), rep_df in wr_skill.group_by(
        ["nfl_id", "drill_name", "event_id"], maintain_order=True
    ):
        if rep_df.height < 12:
            continue
        seg = bio_mod.segment_wr_rep(rep_df, str(drill_name))
        s_arr = rep_df["s"].to_numpy()
        assert s_arr[int(seg["start_idx"])] > bio_mod.START_SPEED_THRESHOLD
        assert int(seg["apex_idx"]) < int(np.floor(0.80 * rep_df.height))
        checked += 1
        if checked >= 30:
            break
    assert checked == 30


def test_advanced_kinetics_and_game_kinetics_parquets() -> None:
    """Verify wr_advanced_kinetics.parquet and wr_game_kinetics.parquet integrity & physics."""
    adv_path = PROCESSED_DIR / "wr_advanced_kinetics.parquet"
    game_path = PROCESSED_DIR / "wr_game_kinetics.parquet"
    assert adv_path.exists(), f"Missing {adv_path}"
    assert game_path.exists(), f"Missing {game_path}"

    adv = pl.read_parquet(adv_path)
    game = pl.read_parquet(game_path)

    assert adv.height == 107
    assert adv["nfl_id"].n_unique() == 107
    assert game.height == 83
    assert game["nfl_id"].n_unique() == 83

    adv_cols = [
        "peak_centripetal_accel",
        "braking_power_index",
        "vertical_route_max_s_z",
        "speed_cut_retention_z",
        "pure_kinetics_z",
        "route_kinetics_index",
    ]
    for col in adv_cols:
        assert adv[col].null_count() == 0, f"Nulls found in {col}"
        assert np.all(np.isfinite(adv[col].to_numpy())), f"NaN/Inf in {col}"

    assert bool((adv["peak_centripetal_accel"] > 0.0).all())
    assert bool((adv["braking_power_index"] >= 0.0).all())

    game_cols = [
        "nfl_route_p95_s",
        "nfl_route_max_s",
        "nfl_all_centripetal_p95",
        "nfl_all_decel_p05",
        "nfl_break_p95_s",
        "nfl_break_max_s",
        "nfl_break_centripetal_p95",
        "nfl_break_decel_p05",
    ]
    for col in game_cols:
        assert game[col].null_count() == 0, f"Nulls found in {col}"
        assert np.all(np.isfinite(game[col].to_numpy())), f"NaN/Inf in {col}"

    assert bool((game["nfl_route_max_s"] >= game["nfl_route_p95_s"]).all())
    assert bool((game["nfl_route_p95_s"] > 5.0).all())
    assert bool((game["nfl_break_decel_p05"] < 0.0).all())


def test_no_draft_class_leakage_in_lodco_cv() -> None:
    """Ensure perturbing holdout draft class targets does not alter its LODCO predictions."""
    lodco_mod = _load_module("wr_lodco_mod", SRC_DIR / "07_wr_modeling_lodco.py")
    df_all = lodco_mod.load_modeling_dataset().filter(pl.col("n_all") >= 50)
    draft_years = df_all["draft_year"].to_numpy()

    oof_base, _ = lodco_mod.fit_predict_lodco(
        df_all, lodco_mod.MODEL_C_COLS, "soe_break", estimator_name="ElasticNetCV"
    )

    for holdout_yr in (2023, 2025):
        perturbed = df_all.with_columns(
            pl.when(pl.col("draft_year") == holdout_yr)
            .then(pl.col("soe_break") + 50.0)
            .otherwise(pl.col("soe_break"))
            .alias("soe_break")
        )
        oof_pert, _ = lodco_mod.fit_predict_lodco(
            perturbed, lodco_mod.MODEL_C_COLS, "soe_break", estimator_name="ElasticNetCV"
        )
        mask = draft_years == holdout_yr
        assert np.allclose(oof_base[mask], oof_pert[mask], atol=1e-12), (
            f"Draft class {holdout_yr} leakage detected in LODCO CV."
        )
        assert not np.allclose(oof_base[~mask], oof_pert[~mask], atol=1e-6), (
            f"Sanity check failed: shifting draft class {holdout_yr} had no effect on other folds."
        )


def test_robustness_audit_optout_and_no_weight_mixing() -> None:
    """Verify opt-out flags, size-adjusted orthogonality, and clean audit report."""
    adv_path = PROCESSED_DIR / "wr_advanced_kinetics.parquet"
    audit_report = PROJECT_ROOT / "wr_robustness_audit.md"
    assert adv_path.exists(), f"Missing {adv_path}"
    assert audit_report.exists() and audit_report.stat().st_size > 1000, f"Missing/empty {audit_report}"

    adv = pl.read_parquet(adv_path)
    required_cols = [
        "n_route_drills_ran",
        "has_go_route",
        "has_post_corner_right",
        "route_speed_z",
        "size_adjusted_route_kinetics",
        "size_adjusted_route_speed_z",
    ]
    for col in required_cols:
        assert col in adv.columns, f"Missing column {col}"
        assert adv[col].null_count() == 0, f"Nulls in {col}"

    for col in ("route_speed_z", "size_adjusted_route_kinetics", "size_adjusted_route_speed_z"):
        assert np.all(np.isfinite(adv[col].to_numpy())), f"NaN/Inf in {col}"

    # Verify Dont'e Thornton opt-out flags explicitly
    thornton = adv.filter(pl.col("display_name") == "Dont'e Thornton")
    assert thornton.height == 1
    assert bool(thornton["has_go_route"][0]) is False
    assert bool(thornton["has_post_corner_right"][0]) is False

    # Verify size_adjusted_route_kinetics is orthogonal to combine_weight
    corr_wt = float(
        np.corrcoef(
            adv["size_adjusted_route_kinetics"].to_numpy(),
            adv["combine_weight"].cast(pl.Float64).to_numpy(),
        )[0, 1]
    )
    assert abs(corr_wt) < 1e-9, f"Residual not orthogonal to weight: r={corr_wt}"


