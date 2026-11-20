"""Discover preliminary Combine-to-NFL signals for OL, WR, and DL tracks.

Normalizes NFL Regular Season play data and Combine Skill Drills tracking data
using Polars, measures valid Regular Season sample coverage (>= 50 snaps),
computes target distributions, and compares Pearson/Spearman correlations of
4 Skill Drill kinematic features against Combine baselines (`ngs_athleticism_score`
and `forty`). Writes a concise report to ``signal_report.md``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import numpy as np
import polars as pl
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "signal_report.md"

NULL_VALUES = ["NA", ""]
MIN_SNAPS = 50
MIN_PRESSURES_SENSITIVITY = 10

BREAK_ROUTES = ["OUT", "IN", "HITCH", "SLANT", "CORNER"]
DRILL_NAME_MAP = {
    "PASS_PRO-MIRROR_DRILL": "PASS_PRO_MIRROR_DRILL",
    "SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT",
}

OL_SPECIFIC_DRILLS = ["PASS_PRO_MIRROR_DRILL", "PASS_RUSH_DROP"]
WR_BREAK_DRILLS = [
    "CURL_ROUTE_RIGHT",
    "COMEBACK_ROUTE_RIGHT",
    "SLANT_ROUTE_LEFT",
    "SPEED_OUT_ROUTE_LEFT",
    "DAGGER_ROUTE_LEFT",
]
DL_RUSH_DRILLS = ["PASS_RUSH_DRILL", "RUN_THE_HOOP_DRILL"]

FEATURE_META = [
    ("max_s", "Skill Drill `max_s` (yd/s)"),
    ("max_a", "Skill Drill `max_a` (yd/s²)"),
    ("min_a", "Skill Drill `min_a` (signed ds/dt, yd/s²)"),
    ("mean_abs_d_dir", "Skill Drill `mean_abs_d_dir` (deg/s)"),
    ("ngs_athleticism_score", "Baseline `ngs_athleticism_score`"),
    ("forty", "Baseline `forty` (s)"),
]


def md_escape(value: object) -> str:
    """Format a scalar for safe inclusion in a Markdown table cell."""
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
        lines.append("| " + " | ".join(md_escape(cell) for cell in row) + " |")
    return lines


def fmt_num(value: float | None, decimals: int = 3) -> str:
    """Format a float with fixed decimals, or return a dash when missing."""
    if value is None or np.isnan(value):
        return "—"
    return f"{value:.{decimals}f}"


def fmt_pct(value: float | None, decimals: int = 2) -> str:
    """Format a proportion in [0, 1] as a percentage string."""
    if value is None or np.isnan(value):
        return "—"
    return f"{value * 100:.{decimals}f}%"


def describe_series(df: pl.DataFrame, col: str) -> dict[str, float | int | None]:
    """Compute descriptive statistics on non-null values of a column."""
    s = df.select(pl.col(col).cast(pl.Float64)).drop_nulls()[col]
    n = int(s.len())
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "p25": None,
            "median": None,
            "p75": None,
            "min": None,
            "max": None,
        }
    return {
        "n": n,
        "mean": float(s.mean()),
        "std": float(s.std()) if n > 1 else 0.0,
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def render_distribution_table(rows_spec: list[tuple[str, dict[str, float | int | None], bool]]) -> list[str]:
    """Format multiple distribution summaries into a single Markdown table."""
    table_rows: list[list[object]] = []
    for label, d, is_pct in rows_spec:
        fmt = fmt_pct if is_pct else fmt_num
        table_rows.append(
            [
                label,
                f"{int(d['n']):,}",
                fmt(d["mean"]),
                fmt(d["std"]),
                fmt(d["p25"]),
                fmt(d["median"]),
                fmt(d["p75"]),
                fmt(d["min"]),
                fmt(d["max"]),
            ]
        )
    return format_markdown_table(
        ["Chỉ tiêu", "N", "Mean", "Std", "P25", "Median", "P75", "Min", "Max"],
        table_rows,
        ["left", "right", "right", "right", "right", "right", "right", "right", "right"],
    )


def load_and_normalize_data(raw_dir: Path) -> tuple[dict[str, pl.DataFrame], dict[str, int]]:
    """Load raw CSVs, filter REG & non-nullified plays, normalize LA->LAR and drill names."""
    games_raw = pl.read_csv(raw_dir / "games.csv", null_values=NULL_VALUES)
    players = pl.read_csv(raw_dir / "players.csv", null_values=NULL_VALUES).select(
        ["nfl_id", "display_name", "draft_year", "nfl_position"]
    )
    combine_results = pl.read_csv(
        raw_dir / "combine_results.csv", null_values=NULL_VALUES
    ).select(["nfl_id", "draft_year", "combine_position", "ngs_athleticism_score", "forty"])

    la_games_before = int(
        games_raw.filter(
            (pl.col("home_team_abbr") == "LA") | (pl.col("visitor_team_abbr") == "LA")
        ).height
    )
    games_reg = (
        games_raw.filter(pl.col("season_type") == "REG")
        .with_columns(
            pl.col("home_team_abbr").replace({"LA": "LAR"}),
            pl.col("visitor_team_abbr").replace({"LA": "LAR"}),
        )
        .select(["game_id", "season", "week", "home_team_abbr", "visitor_team_abbr"])
    )
    unique_teams_games = int(
        pl.concat(
            [
                games_reg.select(pl.col("home_team_abbr").alias("team")),
                games_reg.select(pl.col("visitor_team_abbr").alias("team")),
            ]
        )
        .unique()
        .height
    )

    pp_cols = [
        "game_id",
        "play_id",
        "nfl_id",
        "team_abbr",
        "possession_team",
        "yardline_side",
        "play_nullified_by_penalty",
        "lined_up_position",
        "pass_rushers_encountered",
        "peak_pressure_probability_allowed",
        "pressure_allowed",
        "route_ran",
        "separation_at_pass_forward",
        "target",
        "player_get_off",
        "time_to_pressure",
        "quick_pressure",
        "unblocked_pressure",
    ]
    pp_raw = (
        pl.scan_csv(
            raw_dir / "player_play.csv",
            null_values=NULL_VALUES,
            infer_schema_length=None,
        )
        .select(pp_cols)
        .collect()
    )
    pp_total_rows = int(pp_raw.height)
    pp_non_nullified = pp_raw.filter(pl.col("play_nullified_by_penalty") == "N")
    pp_reg = pp_non_nullified.join(
        games_reg.select(["game_id", "season"]),
        on="game_id",
        how="inner",
    )

    la_pp_rows = int(
        pp_reg.filter(
            (pl.col("team_abbr") == "LA")
            | (pl.col("possession_team") == "LA")
            | (pl.col("yardline_side") == "LA")
        ).height
    )
    pp_reg = (
        pp_reg.with_columns(
            pl.col("team_abbr").replace({"LA": "LAR"}),
            pl.col("possession_team").replace({"LA": "LAR"}),
            pl.col("yardline_side").replace({"LA": "LAR"}),
        )
        .join(combine_results.select(["nfl_id", "combine_position"]), on="nfl_id", how="left")
        .join(players.select(["nfl_id", "nfl_position"]), on="nfl_id", how="left")
    )
    unique_teams_pp = int(pp_reg.select("team_abbr").drop_nulls().unique().height)

    ct_raw = pl.read_csv(raw_dir / "combine_tracking.csv", null_values=NULL_VALUES)
    ct_total_rows = int(ct_raw.height)
    ct_player = ct_raw.filter(pl.col("entity_type") == "PLAYER")

    mirror_old_rows = int(ct_player.filter(pl.col("drill_name") == "PASS_PRO-MIRROR_DRILL").height)
    mirror_std_rows = int(ct_player.filter(pl.col("drill_name") == "PASS_PRO_MIRROR_DRILL").height)
    speed_old_rows = int(ct_player.filter(pl.col("drill_name") == "SPEED_OUT_LEFT").height)
    speed_std_rows = int(ct_player.filter(pl.col("drill_name") == "SPEED_OUT_ROUTE_LEFT").height)

    ct_norm = (
        ct_player.with_columns(pl.col("drill_name").replace(DRILL_NAME_MAP))
        .join(combine_results.select(["nfl_id", "combine_position"]), on="nfl_id", how="left")
    )

    norm_stats = {
        "games_raw": int(games_raw.height),
        "games_reg": int(games_reg.height),
        "la_games_before": la_games_before,
        "unique_teams_games": unique_teams_games,
        "pp_raw": pp_total_rows,
        "pp_non_nullified": int(pp_non_nullified.height),
        "pp_reg": int(pp_reg.height),
        "la_pp_rows": la_pp_rows,
        "unique_teams_pp": unique_teams_pp,
        "ct_raw": ct_total_rows,
        "ct_player": int(ct_norm.height),
        "ct_ball": ct_total_rows - int(ct_norm.height),
        "mirror_old_rows": mirror_old_rows,
        "mirror_std_rows": mirror_std_rows,
        "speed_old_rows": speed_old_rows,
        "speed_std_rows": speed_std_rows,
        "unique_drills_before": int(ct_player.select("drill_name").n_unique()),
        "unique_drills_after": int(ct_norm.select("drill_name").n_unique()),
    }
    tables = {
        "games_reg": games_reg,
        "players": players,
        "combine_results": combine_results,
        "player_play_reg": pp_reg,
        "combine_tracking": ct_norm,
    }
    return tables, norm_stats


def add_frame_kinematics(ct_player: pl.DataFrame) -> pl.DataFrame:
    """Compute per-frame dt, signed tangential acceleration, and circular angular rate."""
    return (
        ct_player.with_columns(
            pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f").alias("ts")
        )
        .sort(["nfl_id", "event_id", "ts"])
        .with_columns(
            ((pl.col("ts") - pl.col("ts").shift(1)).dt.total_milliseconds() / 1000.0)
            .over(["nfl_id", "event_id"])
            .alias("dt"),
            (pl.col("s") - pl.col("s").shift(1))
            .over(["nfl_id", "event_id"])
            .alias("ds"),
            (pl.col("dir") - pl.col("dir").shift(1))
            .abs()
            .over(["nfl_id", "event_id"])
            .alias("raw_d_dir"),
        )
        .with_columns(
            pl.when(pl.col("dt") > 0)
            .then(pl.col("ds") / pl.col("dt"))
            .otherwise(None)
            .alias("signed_a"),
            pl.when(pl.col("dt") > 0)
            .then(pl.min_horizontal(pl.col("raw_d_dir"), 360.0 - pl.col("raw_d_dir")) / pl.col("dt"))
            .otherwise(None)
            .alias("abs_d_dir"),
        )
    )


def aggregate_player_kinematics(
    ct_kin: pl.DataFrame,
    combine_results: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate peak speed, peak accel, peak signed decel, and mean turn rate per player."""
    feats = ct_kin.group_by("nfl_id").agg(
        pl.len().alias("drill_frames"),
        pl.col("event_id").n_unique().alias("drill_reps"),
        pl.col("s").max().alias("max_s"),
        pl.col("a").max().alias("max_a"),
        pl.col("signed_a").min().alias("min_a"),
        pl.col("abs_d_dir").mean().alias("mean_abs_d_dir"),
    )
    return feats.join(
        combine_results.select(["nfl_id", "ngs_athleticism_score", "forty"]),
        on="nfl_id",
        how="left",
    )


def extract_all_group_features(
    ct_kin: pl.DataFrame,
    combine_results: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Extract all-skill-drill and role-specific-drill kinematics for OL, WR, and DL."""
    ol_all_ct = ct_kin.filter(pl.col("drill_type") == "SKILL_DRILLS_OL")
    ol_spec_ct = ol_all_ct.filter(pl.col("drill_name").is_in(OL_SPECIFIC_DRILLS))

    wr_all_ct = ct_kin.filter(pl.col("drill_type") == "SKILL_DRILLS_WR")
    wr_spec_ct = wr_all_ct.filter(pl.col("drill_name").is_in(WR_BREAK_DRILLS))

    dl_all_ct = ct_kin.filter(
        (pl.col("combine_position") == "DL")
        & pl.col("drill_type").is_in(["SKILL_DRILLS_DL", "SKILL_DRILLS_LB"])
    )
    dl_spec_ct = dl_all_ct.filter(pl.col("drill_name").is_in(DL_RUSH_DRILLS))

    return {
        "OL_all": aggregate_player_kinematics(ol_all_ct, combine_results),
        "OL_spec": aggregate_player_kinematics(ol_spec_ct, combine_results),
        "WR_all": aggregate_player_kinematics(wr_all_ct, combine_results),
        "WR_spec": aggregate_player_kinematics(wr_spec_ct, combine_results),
        "DL_all": aggregate_player_kinematics(dl_all_ct, combine_results),
        "DL_spec": aggregate_player_kinematics(dl_spec_ct, combine_results),
    }


def compute_correlations(
    df: pl.DataFrame,
    target_col: str,
    features: list[tuple[str, str]] | None = None,
) -> list[dict[str, object]]:
    """Compute Pearson and Spearman correlations between feature columns and target_col."""
    if features is None:
        features = FEATURE_META
    results: list[dict[str, object]] = []
    for feat_col, feat_label in features:
        pair = df.select([feat_col, target_col]).drop_nulls()
        n = int(pair.height)
        if n < 3:
            results.append(
                {
                    "feature": feat_col,
                    "label": feat_label,
                    "n": n,
                    "pearson_r": None,
                    "pearson_p": None,
                    "spearman_rho": None,
                    "spearman_p": None,
                }
            )
            continue

        x = pair[feat_col].cast(pl.Float64).to_numpy()
        y = pair[target_col].cast(pl.Float64).to_numpy()
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            results.append(
                {
                    "feature": feat_col,
                    "label": feat_label,
                    "n": n,
                    "pearson_r": None,
                    "pearson_p": None,
                    "spearman_rho": None,
                    "spearman_p": None,
                }
            )
            continue

        pr = stats.pearsonr(x, y)
        sr = stats.spearmanr(x, y)
        results.append(
            {
                "feature": feat_col,
                "label": feat_label,
                "n": n,
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
                "spearman_rho": float(sr.statistic),
                "spearman_p": float(sr.pvalue),
            }
        )
    return results


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


def render_correlation_table(corr_rows: list[dict[str, object]]) -> list[str]:
    """Render a correlation table comparing Skill Drills vs Combine baselines."""
    rows: list[list[object]] = []
    for r in corr_rows:
        pr = r["pearson_r"]
        pp = r["pearson_p"]
        sr = r["spearman_rho"]
        sp = r["spearman_p"]
        pr_str = f"{pr:+.3f}{sig_star(pp)}" if pr is not None else "—"
        sr_str = f"{sr:+.3f}{sig_star(sp)}" if sr is not None else "—"
        pp_str = f"{pp:.4f}" if pp is not None else "—"
        sp_str = f"{sp:.4f}" if sp is not None else "—"
        rows.append([r["label"], f"{int(r['n']):,}", pr_str, pp_str, sr_str, sp_str])
    return format_markdown_table(
        ["Đặc trưng", "N", "Pearson r", "p-val (P)", "Spearman ρ", "p-val (S)"],
        rows,
        ["left", "right", "right", "right", "right", "right"],
    )


def analyze_ol(
    pp_reg: pl.DataFrame,
    ol_feats: pl.DataFrame,
    ol_spec_feats: pl.DataFrame,
) -> dict[str, object]:
    """Analyze OL REG coverage (`pass_rushers_encountered > 0`) and Combine signals."""
    ol_plays = pp_reg.filter(
        (pl.col("combine_position") == "OL") & (pl.col("pass_rushers_encountered") > 0)
    )
    ol_players_all = ol_plays.group_by("nfl_id").agg(
        pl.col("nfl_position").first().alias("nfl_position"),
        pl.len().alias("valid_snaps"),
        pl.col("peak_pressure_probability_allowed").mean().alias("peak_pressure_prob_allowed"),
        pl.col("pressure_allowed").cast(pl.Float64).mean().alias("pressure_allowed_rate"),
    )
    ol_cohort = ol_players_all.filter(pl.col("valid_snaps") >= MIN_SNAPS)
    ol_pos_breakdown = (
        ol_players_all.group_by("nfl_position")
        .agg(
            pl.len().alias("total_players"),
            (pl.col("valid_snaps") >= MIN_SNAPS).sum().alias("players_ge_50"),
            pl.col("valid_snaps").sum().alias("total_snaps"),
        )
        .sort("total_snaps", descending=True)
    )

    ol_merged = ol_cohort.join(ol_feats, on="nfl_id", how="inner")
    ol_spec_merged = ol_cohort.join(ol_spec_feats, on="nfl_id", how="inner")

    return {
        "plays": ol_plays,
        "players_all": ol_players_all,
        "cohort": ol_cohort,
        "pos_breakdown": ol_pos_breakdown,
        "merged": ol_merged,
        "corr_peak_prob": compute_correlations(ol_merged, "peak_pressure_prob_allowed"),
        "corr_pressure_rate": compute_correlations(ol_merged, "pressure_allowed_rate"),
        "corr_spec_peak_prob": compute_correlations(ol_spec_merged, "peak_pressure_prob_allowed"),
        "corr_spec_pressure_rate": compute_correlations(ol_spec_merged, "pressure_allowed_rate"),
    }


def analyze_wr(
    pp_reg: pl.DataFrame,
    wr_feats: pl.DataFrame,
    wr_spec_feats: pl.DataFrame,
) -> dict[str, object]:
    """Analyze WR REG coverage across routes and break routes, plus Combine signals."""
    wr_plays = pp_reg.filter(
        (pl.col("combine_position") == "WR") & pl.col("route_ran").is_not_null()
    )
    wr_break_plays = wr_plays.filter(pl.col("route_ran").is_in(BREAK_ROUTES))

    per_route_player_snaps = wr_plays.group_by(["route_ran", "nfl_id"]).len()
    wrs_ge_50_by_route = (
        per_route_player_snaps.filter(pl.col("len") >= MIN_SNAPS)
        .group_by("route_ran")
        .agg(pl.len().alias("wrs_ge_50"))
    )

    route_table = (
        wr_plays.group_by("route_ran")
        .agg(
            pl.len().alias("route_snaps"),
            pl.col("separation_at_pass_forward").is_not_null().sum().alias("sep_snaps"),
            pl.col("nfl_id").n_unique().alias("unique_wrs"),
            pl.col("separation_at_pass_forward").mean().alias("sep_mean"),
            pl.col("separation_at_pass_forward").std().alias("sep_std"),
            pl.col("separation_at_pass_forward").median().alias("sep_median"),
            pl.col("separation_at_pass_forward").quantile(0.25).alias("sep_p25"),
            pl.col("separation_at_pass_forward").quantile(0.75).alias("sep_p75"),
            pl.col("target").cast(pl.Float64).mean().alias("target_rate"),
        )
        .join(wrs_ge_50_by_route, on="route_ran", how="left")
        .with_columns(
            pl.col("wrs_ge_50").fill_null(0),
            pl.col("route_ran").is_in(BREAK_ROUTES).alias("is_break_route"),
        )
        .sort("route_snaps", descending=True)
    )

    wr_all_agg = wr_plays.group_by("nfl_id").agg(
        pl.len().alias("route_snaps"),
        pl.col("separation_at_pass_forward").is_not_null().sum().alias("sep_snaps"),
        pl.col("separation_at_pass_forward").mean().alias("mean_sep_all"),
        pl.col("target").cast(pl.Float64).mean().alias("target_rate_all"),
    )
    wr_break_agg = wr_break_plays.group_by("nfl_id").agg(
        pl.len().alias("break_route_snaps"),
        pl.col("separation_at_pass_forward").is_not_null().sum().alias("break_sep_snaps"),
        pl.col("separation_at_pass_forward").mean().alias("mean_sep_break"),
        pl.col("target").cast(pl.Float64).mean().alias("target_rate_break"),
    )

    wr_cohort_sep_all = wr_all_agg.filter(pl.col("sep_snaps") >= MIN_SNAPS)
    wr_cohort_tgt_all = wr_all_agg.filter(pl.col("route_snaps") >= MIN_SNAPS)
    wr_cohort_sep_break = wr_break_agg.filter(pl.col("break_sep_snaps") >= MIN_SNAPS)
    wr_cohort_tgt_break = wr_break_agg.filter(pl.col("break_route_snaps") >= MIN_SNAPS)

    return {
        "plays": wr_plays,
        "break_plays": wr_break_plays,
        "route_table": route_table,
        "all_agg": wr_all_agg,
        "break_agg": wr_break_agg,
        "cohort_sep_all": wr_cohort_sep_all,
        "cohort_tgt_all": wr_cohort_tgt_all,
        "cohort_sep_break": wr_cohort_sep_break,
        "cohort_tgt_break": wr_cohort_tgt_break,
        "corr_sep_all": compute_correlations(
            wr_cohort_sep_all.join(wr_feats, on="nfl_id", how="inner"), "mean_sep_all"
        ),
        "corr_tgt_all": compute_correlations(
            wr_cohort_tgt_all.join(wr_feats, on="nfl_id", how="inner"), "target_rate_all"
        ),
        "corr_sep_break": compute_correlations(
            wr_cohort_sep_break.join(wr_feats, on="nfl_id", how="inner"), "mean_sep_break"
        ),
        "corr_tgt_break": compute_correlations(
            wr_cohort_tgt_break.join(wr_feats, on="nfl_id", how="inner"), "target_rate_break"
        ),
        "corr_spec_sep_break": compute_correlations(
            wr_cohort_sep_break.join(wr_spec_feats, on="nfl_id", how="inner"), "mean_sep_break"
        ),
        "corr_spec_tgt_break": compute_correlations(
            wr_cohort_tgt_break.join(wr_spec_feats, on="nfl_id", how="inner"), "target_rate_break"
        ),
    }


def analyze_dl(
    pp_reg: pl.DataFrame,
    dl_feats: pl.DataFrame,
    dl_spec_feats: pl.DataFrame,
) -> dict[str, object]:
    """Analyze DL (EDGE & DT/IDL) REG pass-rush coverage and Combine signals."""
    dl_plays = pp_reg.filter(pl.col("combine_position") == "DL").with_columns(
        pl.when(pl.col("nfl_position").is_in(["DE", "OLB"]))
        .then(pl.lit("EDGE"))
        .otherwise(pl.lit("DT/IDL"))
        .alias("dl_role"),
        pl.col("unblocked_pressure").is_not_null().alias("is_pass_rush"),
    )
    dl_pass_rush_plays = dl_plays.filter(pl.col("is_pass_rush"))

    dl_agg = dl_plays.group_by("nfl_id").agg(
        pl.col("nfl_position").first().alias("nfl_position"),
        pl.col("dl_role").first().alias("dl_role"),
        pl.len().alias("total_snaps"),
        pl.col("player_get_off").is_not_null().sum().alias("all_get_off_snaps"),
        pl.col("is_pass_rush").sum().alias("pass_rush_snaps"),
        pl.col("player_get_off")
        .filter(pl.col("is_pass_rush"))
        .is_not_null()
        .sum()
        .alias("pass_rush_get_off_snaps"),
        pl.col("player_get_off")
        .filter(pl.col("is_pass_rush"))
        .mean()
        .alias("mean_player_get_off"),
        pl.col("player_get_off").mean().alias("mean_player_get_off_all"),
        pl.col("time_to_pressure").is_not_null().sum().alias("pressure_count"),
        pl.col("time_to_pressure").mean().alias("mean_time_to_pressure"),
        pl.col("quick_pressure")
        .filter(pl.col("is_pass_rush"))
        .fill_null(False)
        .sum()
        .alias("quick_pressure_count"),
        pl.col("quick_pressure")
        .cast(pl.Float64)
        .mean()
        .alias("quick_pressure_share"),
    ).with_columns(
        (pl.col("quick_pressure_count") / pl.col("pass_rush_snaps")).alias("quick_pressure_rate"),
        (pl.col("pressure_count") / pl.col("pass_rush_snaps")).alias("pressure_rate"),
    )

    dl_cohort = dl_agg.filter(pl.col("pass_rush_snaps") >= MIN_SNAPS)
    dl_cohort_ttp10 = dl_cohort.filter(pl.col("pressure_count") >= MIN_PRESSURES_SENSITIVITY)
    dl_edge = dl_cohort.filter(pl.col("dl_role") == "EDGE")
    dl_dt = dl_cohort.filter(pl.col("dl_role") == "DT/IDL")

    dl_merged = dl_cohort.join(dl_feats, on="nfl_id", how="inner")
    dl_ttp10_merged = dl_cohort_ttp10.join(dl_feats, on="nfl_id", how="inner")
    dl_edge_merged = dl_edge.join(dl_feats, on="nfl_id", how="inner")
    dl_dt_merged = dl_dt.join(dl_feats, on="nfl_id", how="inner")
    dl_spec_merged = dl_cohort.join(dl_spec_feats, on="nfl_id", how="inner")

    return {
        "plays": dl_plays,
        "pass_rush_plays": dl_pass_rush_plays,
        "agg": dl_agg,
        "cohort": dl_cohort,
        "cohort_ttp10": dl_cohort_ttp10,
        "edge_cohort": dl_edge,
        "dt_cohort": dl_dt,
        "corr_get_off": compute_correlations(dl_merged, "mean_player_get_off"),
        "corr_ttp": compute_correlations(dl_merged, "mean_time_to_pressure"),
        "corr_ttp10": compute_correlations(dl_ttp10_merged, "mean_time_to_pressure"),
        "corr_qp_rate": compute_correlations(dl_merged, "quick_pressure_rate"),
        "corr_qp_share": compute_correlations(dl_ttp10_merged, "quick_pressure_share"),
        "corr_edge_get_off": compute_correlations(dl_edge_merged, "mean_player_get_off"),
        "corr_dt_get_off": compute_correlations(dl_dt_merged, "mean_player_get_off"),
        "corr_edge_ttp": compute_correlations(dl_edge_merged, "mean_time_to_pressure"),
        "corr_dt_ttp": compute_correlations(dl_dt_merged, "mean_time_to_pressure"),
        "corr_edge_qp_rate": compute_correlations(dl_edge_merged, "quick_pressure_rate"),
        "corr_dt_qp_rate": compute_correlations(dl_dt_merged, "quick_pressure_rate"),
        "corr_spec_get_off": compute_correlations(dl_spec_merged, "mean_player_get_off"),
        "corr_spec_ttp": compute_correlations(dl_spec_merged, "mean_time_to_pressure"),
        "corr_spec_qp_rate": compute_correlations(dl_spec_merged, "quick_pressure_rate"),
    }


def best_corr_summary(
    corr_rows: list[dict[str, object]],
    feature_filter: set[str],
    metric_key: str = "spearman_rho",
) -> tuple[str, float | None, float | None]:
    """Return (feature_name, corr_value, p_value) with largest |correlation| in feature_filter."""
    p_key = "spearman_p" if metric_key == "spearman_rho" else "pearson_p"
    candidates = [
        r
        for r in corr_rows
        if r["feature"] in feature_filter and r[metric_key] is not None
    ]
    if not candidates:
        return "—", None, None
    best = max(candidates, key=lambda r: abs(float(r[metric_key])))
    return str(best["feature"]), float(best[metric_key]), float(best[p_key])


def format_best_badge(feat: str, r_val: float | None, p_val: float | None) -> str:
    """Format a compact `feature: r (p)` summary cell."""
    if r_val is None or p_val is None:
        return "—"
    return f"`{feat}`: {r_val:+.3f}{sig_star(p_val)} (p={p_val:.3f})"


def build_report(
    tables: dict[str, pl.DataFrame],
    norm_stats: dict[str, int],
    ol: dict[str, object],
    wr: dict[str, object],
    dl: dict[str, object],
) -> str:
    """Assemble the complete signal_report.md content."""
    lines: list[str] = [
        "# NFL Big Data Bowl 2027 — Preliminary Signal Discovery (OL vs WR vs DL)",
        "",
        f"- **Thời gian tạo**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- **Ngưỡng mẫu kiểm định**: Tối thiểu **{MIN_SNAPS} snaps hợp lệ** ở NFL Regular Season (2023–2025).",
        "- **Ký hiệu mức ý nghĩa thống kê**: `**` (*p < 0.01*), `*` (*p < 0.05*), `.` (*p < 0.10*).",
        "",
        "## 1. Chuẩn hóa dữ liệu & Định nghĩa đặc trưng vật lý",
        "",
        "### 1.1. Các bước làm sạch và chuẩn hóa",
        "",
    ]

    norm_rows = [
        ["`games.csv` (`season_type == \"REG\"`)", f"{norm_stats['games_raw']:,} → **{norm_stats['games_reg']:,}** trận", "Chỉ giữ Regular Season (272 trận × 3 mùa)"],
        ["Chuẩn hóa mã đội `LA` → `LAR`", f"{norm_stats['la_games_before']:,} trận (`games`); {norm_stats['la_pp_rows']:,} dòng (`player_play`)", f"Còn đúng **{norm_stats['unique_teams_games']}** đội ở `games` và **{norm_stats['unique_teams_pp']}** đội ở `player_play`"],
        ["`player_play.csv` (`play_nullified_by_penalty == \"N\"` + `REG`)", f"{norm_stats['pp_raw']:,} → {norm_stats['pp_non_nullified']:,} (N) → **{norm_stats['pp_reg']:,}** (REG)", "Loại bỏ các play bị penalty hủy và các trận PRE/POST"],
        ["`combine_tracking.csv` (`entity_type == \"PLAYER\"`)", f"{norm_stats['ct_raw']:,} → **{norm_stats['ct_player']:,}** dòng", f"Loại bỏ {norm_stats['ct_ball']:,} dòng theo dõi bóng (`BALL`)"],
        ["Gộp tên drill OL bị lệch gạch nối", f"`PASS_PRO-MIRROR_DRILL` ({norm_stats['mirror_old_rows']:,}) + `PASS_PRO_MIRROR_DRILL` ({norm_stats['mirror_std_rows']:,})", "Gộp thành `PASS_PRO_MIRROR_DRILL`"],
        ["Gộp tên drill WR bị thiếu chữ `ROUTE`", f"`SPEED_OUT_LEFT` ({norm_stats['speed_old_rows']:,}) + `SPEED_OUT_ROUTE_LEFT` ({norm_stats['speed_std_rows']:,})", f"Gộp thành `SPEED_OUT_ROUTE_LEFT` (tổng số `drill_name`: {norm_stats['unique_drills_before']} → {norm_stats['unique_drills_after']})"],
    ]
    lines.extend(
        format_markdown_table(
            ["Bước chuẩn hóa", "Số lượng trước → sau", "Ghi chú"],
            norm_rows,
            ["left", "left", "left"],
        )
    )
    lines.extend(
        [
            "",
            "### 1.2. Định nghĩa 4 đặc trưng vật lý từ Skill Drills (`combine_tracking.csv`)",
            "",
            "- **Lưu ý kỹ thuật về gia tốc `a`**: Cột `a` gốc trong `combine_tracking.csv` chỉ là độ lớn vô hướng (`a >= 0.00`, `max = 10.72 yd/s²`). Do đó, trong mỗi `(nfl_id, event_id)` sắp xếp theo `time` (bước lấy mẫu cố định `dt = 0.1s` ở 100% khung hình liên tiếp), ta tính gia tốc tiếp tuyến có dấu `signed_a = (s_t - s_{t-1}) / dt` để đo lực hãm phanh/giảm tốc thực tế.",
            "- **`max_s`** (`yd/s`): `max(s)` trên toàn bộ khung hình Skill Drills của cầu thủ.",
            "- **`max_a`** (`yd/s²`): `max(a)` từ đại lượng gia tốc tracking.",
            "- **`min_a`** (`yd/s²`): `min(signed_a)` — gia tốc âm cực đại phản ánh khả năng giảm tốc/hãm phanh đột ngột (giá trị càng âm tức phanh càng gắt).",
            "- **`mean_abs_d_dir`** (`deg/s`): Trung bình tốc độ đổi hướng góc `min(|dir_t - dir_{t-1}|, 360 - |dir_t - dir_{t-1}|) / dt` có xử lý xoay vòng `0°/360°`.",
            "",
            "---",
            "",
            "## 2. Độ phủ mẫu thực tế ở NFL Regular Season & Phân phối biến mục tiêu",
            "",
            "### 2.1. Nhóm OL (`combine_position == \"OL\"`, lọc `pass_rushers_encountered > 0`)",
            "",
        ]
    )

    ol_plays: pl.DataFrame = ol["plays"]  # type: ignore[assignment]
    ol_all: pl.DataFrame = ol["players_all"]  # type: ignore[assignment]
    ol_cohort: pl.DataFrame = ol["cohort"]  # type: ignore[assignment]
    ol_pos: pl.DataFrame = ol["pos_breakdown"]  # type: ignore[assignment]

    lines.append(
        f"- **Độ phủ mẫu**: **{ol_plays.height:,}** snaps hợp lệ từ **{ol_all.height}** cầu thủ OL có dữ liệu NFL REG; trong đó **{ol_cohort.height} / 121** cầu thủ Combine OL (57.9% toàn khóa Combine, 78.7% nhóm có ra sân REG) đạt ngưỡng **>= 50 snaps**."
    )
    lines.append("")
    lines.extend(
        format_markdown_table(
            ["Vị trí (`nfl_position`)", "Có >= 1 snap", "Đạt >= 50 snaps", "Tổng snaps"],
            [
                [r["nfl_position"], f"{r['total_players']:,}", f"{r['players_ge_50']:,}", f"{r['total_snaps']:,}"]
                for r in ol_pos.iter_rows(named=True)
            ],
            ["left", "right", "right", "right"],
        )
    )
    lines.append("")
    lines.extend(
        render_distribution_table(
            [
                ("Play-level `peak_pressure_probability_allowed`", describe_series(ol_plays, "peak_pressure_probability_allowed"), False),
                ("Play-level `pressure_allowed` (tỷ lệ)", describe_series(ol_plays, "pressure_allowed"), True),
                ("Player-level (>=50 snaps) `valid_snaps`", describe_series(ol_cohort, "valid_snaps"), False),
                ("Player-level (>=50 snaps) `peak_pressure_prob_allowed`", describe_series(ol_cohort, "peak_pressure_prob_allowed"), False),
                ("Player-level (>=50 snaps) `pressure_allowed_rate`", describe_series(ol_cohort, "pressure_allowed_rate"), True),
            ]
        )
    )
    lines.append("")

    # 2.2 WR
    wr_plays: pl.DataFrame = wr["plays"]  # type: ignore[assignment]
    wr_break_plays: pl.DataFrame = wr["break_plays"]  # type: ignore[assignment]
    wr_route_tbl: pl.DataFrame = wr["route_table"]  # type: ignore[assignment]
    wr_all_agg: pl.DataFrame = wr["all_agg"]  # type: ignore[assignment]
    wr_cohort_sep_all: pl.DataFrame = wr["cohort_sep_all"]  # type: ignore[assignment]
    wr_cohort_sep_break: pl.DataFrame = wr["cohort_sep_break"]  # type: ignore[assignment]

    lines.extend(
        [
            "### 2.2. Nhóm WR (`combine_position == \"WR\"`, lọc `route_ran` không null)",
            "",
            f"- **Độ phủ mẫu (tất cả route)**: **{wr_plays.height:,}** route snaps (**{wr_plays.filter(pl.col('separation_at_pass_forward').is_not_null()).height:,}** snaps có `separation_at_pass_forward`) từ **{wr_all_agg.height}** WR; trong đó **{wr_cohort_sep_all.height} / 107** WR Combine đạt **>= 50 snaps** có đo separation.",
            f"- **Độ phủ mẫu (nhóm Break Routes: `{', '.join(BREAK_ROUTES)}`)**: **{wr_break_plays.height:,}** route snaps (**{wr_break_plays.filter(pl.col('separation_at_pass_forward').is_not_null()).height:,}** snaps có separation); **{wr_cohort_sep_break.height} / 107** WR đạt **>= 50 snaps** có separation.",
            "",
        ]
    )
    lines.extend(
        format_markdown_table(
            [
                "route_ran",
                "Break?",
                "Route Snaps",
                "Sep Snaps",
                "WRs",
                "WRs (>=50)",
                "Sep Mean",
                "Sep Std",
                "Sep Median",
                "Target Rate",
            ],
            [
                [
                    r["route_ran"],
                    "Yes" if r["is_break_route"] else "No",
                    f"{r['route_snaps']:,}",
                    f"{r['sep_snaps']:,}",
                    f"{r['unique_wrs']:,}",
                    f"{r['wrs_ge_50']:,}",
                    fmt_num(r["sep_mean"]),
                    fmt_num(r["sep_std"]),
                    fmt_num(r["sep_median"]),
                    fmt_pct(r["target_rate"]),
                ]
                for r in wr_route_tbl.iter_rows(named=True)
            ],
            ["left", "center", "right", "right", "right", "right", "right", "right", "right", "right"],
        )
    )
    lines.append("")
    lines.extend(
        render_distribution_table(
            [
                ("Play-level `separation_at_pass_forward` (All routes)", describe_series(wr_plays, "separation_at_pass_forward"), False),
                ("Play-level `separation_at_pass_forward` (Break routes)", describe_series(wr_break_plays, "separation_at_pass_forward"), False),
                ("Player-level (>=50 sep snaps) `mean_sep_all`", describe_series(wr_cohort_sep_all, "mean_sep_all"), False),
                ("Player-level (>=50 sep snaps) `mean_sep_break`", describe_series(wr_cohort_sep_break, "mean_sep_break"), False),
                ("Player-level (>=50 route snaps) `target_rate_all`", describe_series(wr["cohort_tgt_all"], "target_rate_all"), True),  # type: ignore[arg-type]
                ("Player-level (>=50 break snaps) `target_rate_break`", describe_series(wr["cohort_tgt_break"], "target_rate_break"), True),  # type: ignore[arg-type]
            ]
        )
    )
    lines.append("")

    # 2.3 DL
    dl_plays: pl.DataFrame = dl["plays"]  # type: ignore[assignment]
    dl_pr_plays: pl.DataFrame = dl["pass_rush_plays"]  # type: ignore[assignment]
    dl_agg: pl.DataFrame = dl["agg"]  # type: ignore[assignment]
    dl_cohort: pl.DataFrame = dl["cohort"]  # type: ignore[assignment]
    dl_cohort_ttp10: pl.DataFrame = dl["cohort_ttp10"]  # type: ignore[assignment]
    dl_edge: pl.DataFrame = dl["edge_cohort"]  # type: ignore[assignment]
    dl_dt: pl.DataFrame = dl["dt_cohort"]  # type: ignore[assignment]

    dl_pr_get_off = dl_pr_plays.filter(pl.col("player_get_off").is_not_null())
    dl_all_get_off = dl_plays.filter(pl.col("player_get_off").is_not_null())
    dl_ttp_plays = dl_plays.filter(pl.col("time_to_pressure").is_not_null())
    dl_qp_plays = dl_pr_plays.with_columns(
        pl.col("quick_pressure").fill_null(False).cast(pl.Float64).alias("quick_pressure_rate_play")
    )

    lines.extend(
        [
            "### 2.3. Nhóm DL (`combine_position == \"DL\"`: EDGE `DE/OLB` và DT/IDL `DT/NT`)",
            "",
            "- **Lưu ý cấu trúc dữ liệu DL**:",
            f"  - Tổng snaps DL ở REG: **{dl_plays.height:,}** snaps ({dl_agg.height} cầu thủ); trong đó có **{dl_pr_plays.height:,}** pass-rush snaps (`unblocked_pressure` non-null) và **{dl_all_get_off.height:,}** snaps có `player_get_off` ({dl_pr_get_off.height:,} trên pass-rush snaps; {dl_agg.filter(pl.col('all_get_off_snaps') >= MIN_SNAPS).height} cầu thủ có >= 50 get-off snaps tổng hợp).",
            f"  - Với ngưỡng **>= 50 pass-rush snaps**: có **{dl_cohort.height} / 118** cầu thủ DL (gồm **{dl_edge.height} EDGE** và **{dl_dt.height} DT/IDL**).",
            f"  - Biến `time_to_pressure` và `quick_pressure` chỉ ghi nhận khi cầu thủ thực sự tạo áp lực ({dl_ttp_plays.height:,} lượt pressure / {dl_pr_plays.height:,} pass-rush snaps = {dl_ttp_plays.height / dl_pr_plays.height * 100:.2f}%). Để tránh sai lệch mẫu số, báo cáo tính đồng thời **`quick_pressure_rate`** = `số quick_pressure / tổng pass_rush_snaps` (tỷ lệ tạo áp lực nhanh thực tế) và **`quick_pressure_share`** = `P(quick_pressure | tạo được pressure)` ở nhóm có `>= 10 pressures` (**{dl_cohort_ttp10.height}** cầu thủ).",
            "",
        ]
    )
    lines.extend(
        render_distribution_table(
            [
                ("Play-level `player_get_off` (Pass-rush snaps)", describe_series(dl_pr_get_off, "player_get_off"), False),
                ("Play-level `time_to_pressure` (When pressure > 0)", describe_series(dl_ttp_plays, "time_to_pressure"), False),
                ("Play-level `quick_pressure_rate` (Per pass-rush snap)", describe_series(dl_qp_plays, "quick_pressure_rate_play"), True),
                ("Player-level (>=50 PR snaps) `mean_player_get_off`", describe_series(dl_cohort, "mean_player_get_off"), False),
                ("Player-level (>=50 PR snaps) `mean_time_to_pressure` (N=77 có >=1 PR)", describe_series(dl_cohort, "mean_time_to_pressure"), False),
                ("Player-level (>=50 PR snaps, >=10 press) `mean_time_to_pressure`", describe_series(dl_cohort_ttp10, "mean_time_to_pressure"), False),
                ("Player-level (>=50 PR snaps) `quick_pressure_rate`", describe_series(dl_cohort, "quick_pressure_rate"), True),
                ("Player-level (>=50 PR snaps, >=10 press) `quick_pressure_share`", describe_series(dl_cohort_ttp10, "quick_pressure_share"), True),
                ("EDGE (>=50 PR snaps) `mean_player_get_off`", describe_series(dl_edge, "mean_player_get_off"), False),
                ("DT/IDL (>=50 PR snaps) `mean_player_get_off`", describe_series(dl_dt, "mean_player_get_off"), False),
                ("EDGE (>=50 PR snaps) `quick_pressure_rate`", describe_series(dl_edge, "quick_pressure_rate"), True),
                ("DT/IDL (>=50 PR snaps) `quick_pressure_rate`", describe_series(dl_dt, "quick_pressure_rate"), True),
            ]
        )
    )
    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Kiểm tra tín hiệu sơ bộ (Skill Drills vs `ngs_athleticism_score` & `forty`)",
            "",
            "### 3.1. Nhóm OL (`valid_snaps >= 50`)",
            "",
            "#### Target: `peak_pressure_prob_allowed` (càng thấp càng tốt cho OL)",
            "",
        ]
    )
    lines.extend(render_correlation_table(ol["corr_peak_prob"]))  # type: ignore[arg-type]
    lines.extend(
        [
            "",
            "#### Target: `pressure_allowed_rate` (càng thấp càng tốt cho OL)",
            "",
        ]
    )
    lines.extend(render_correlation_table(ol["corr_pressure_rate"]))  # type: ignore[arg-type]

    lines.extend(
        [
            "",
            "### 3.2. Nhóm WR (`>= 50` snaps)",
            "",
            "#### Target: `mean_sep_all` — Separation trung bình (Tất cả route)",
            "",
        ]
    )
    lines.extend(render_correlation_table(wr["corr_sep_all"]))  # type: ignore[arg-type]
    lines.extend(
        [
            "",
            f"#### Target: `mean_sep_break` — Separation trung bình (Break routes: `{', '.join(BREAK_ROUTES)}`)",
            "",
        ]
    )
    lines.extend(render_correlation_table(wr["corr_sep_break"]))  # type: ignore[arg-type]
    lines.extend(
        [
            "",
            "#### Target: `target_rate_all` — Tỷ lệ được target (Tất cả route)",
            "",
        ]
    )
    lines.extend(render_correlation_table(wr["corr_tgt_all"]))  # type: ignore[arg-type]
    lines.extend(
        [
            "",
            "#### Target: `target_rate_break` — Tỷ lệ được target (Break routes)",
            "",
        ]
    )
    lines.extend(render_correlation_table(wr["corr_tgt_break"]))  # type: ignore[arg-type]

    lines.extend(
        [
            "",
            "### 3.3. Nhóm DL (`pass_rush_snaps >= 50`)",
            "",
            "#### Target: `mean_player_get_off` — Thời gian xuất phát sau snap (càng thấp càng nhanh)",
            "",
        ]
    )
    lines.extend(render_correlation_table(dl["corr_get_off"]))  # type: ignore[arg-type]
    lines.extend(
        [
            "",
            "#### Target: `mean_time_to_pressure` — Thời gian gây áp lực (tất cả DL có >= 1 pressure)",
            "",
        ]
    )
    lines.extend(render_correlation_table(dl["corr_ttp"]))  # type: ignore[arg-type]
    lines.extend(
        [
            "",
            f"#### Target: `mean_time_to_pressure` — Kiểm tra độ nhạy trên nhóm có >= {MIN_PRESSURES_SENSITIVITY} pressures",
            "",
        ]
    )
    lines.extend(render_correlation_table(dl["corr_ttp10"]))  # type: ignore[arg-type]
    lines.extend(
        [
            "",
            "#### Target: `quick_pressure_rate` — Tỷ lệ tạo quick pressure trên mỗi pass-rush snap",
            "",
        ]
    )
    lines.extend(render_correlation_table(dl["corr_qp_rate"]))  # type: ignore[arg-type]
    lines.extend(
        [
            "",
            f"#### Target: `quick_pressure_share` — Tỷ lệ quick pressure có điều kiện (>= {MIN_PRESSURES_SENSITIVITY} pressures)",
            "",
        ]
    )
    lines.extend(render_correlation_table(dl["corr_qp_share"]))  # type: ignore[arg-type]

    # 3.3b EDGE vs DT/IDL split check
    lines.extend(
        [
            "",
            "#### Kiểm tra tách nhóm DL: `EDGE` (`DE/OLB`) vs `DT/IDL` (`DT/NT`)",
            "",
        ]
    )
    skill_set = {"max_s", "max_a", "min_a", "mean_abs_d_dir"}
    base_set = {"ngs_athleticism_score", "forty"}
    split_specs = [
        ("EDGE — `mean_player_get_off`", dl["corr_edge_get_off"]),
        ("DT/IDL — `mean_player_get_off`", dl["corr_dt_get_off"]),
        ("EDGE — `mean_time_to_pressure`", dl["corr_edge_ttp"]),
        ("DT/IDL — `mean_time_to_pressure`", dl["corr_dt_ttp"]),
        ("EDGE — `quick_pressure_rate`", dl["corr_edge_qp_rate"]),
        ("DT/IDL — `quick_pressure_rate`", dl["corr_dt_qp_rate"]),
    ]
    split_rows: list[list[object]] = []
    for label, crows in split_specs:
        c_list: list[dict[str, object]] = crows  # type: ignore[assignment]
        n_val = max((int(r["n"]) for r in c_list), default=0)
        sb_p = best_corr_summary(c_list, skill_set, "pearson_r")
        sb_s = best_corr_summary(c_list, skill_set, "spearman_rho")
        bb_s = best_corr_summary(c_list, base_set, "spearman_rho")
        split_rows.append(
            [
                label,
                f"{n_val:,}",
                format_best_badge(*sb_p),
                format_best_badge(*sb_s),
                format_best_badge(*bb_s),
            ]
        )
    lines.extend(
        format_markdown_table(
            ["Phân nhóm & Target", "N", "Best Skill (Pearson)", "Best Skill (Spearman)", "Best Baseline (Spearman)"],
            split_rows,
            ["left", "right", "left", "left", "left"],
        )
    )

    # 3.4 Drill-specific sanity check
    lines.extend(
        [
            "",
            "### 3.4. Kiểm chứng nhanh: Toàn bộ Skill Drills vs Nhóm Drill chuyên biệt theo bài toán",
            "",
            f"- **OL chuyên biệt**: `{', '.join(OL_SPECIFIC_DRILLS)}`.",
            f"- **WR chuyên biệt (Break/Route)**: `{', '.join(WR_BREAK_DRILLS)}`.",
            f"- **DL chuyên biệt (Pass Rush)**: `{', '.join(DL_RUSH_DRILLS)}`.",
            "",
        ]
    )
    sanity_specs = [
        ("OL: `peak_pressure_prob_allowed`", ol["corr_peak_prob"], ol["corr_spec_peak_prob"]),
        ("OL: `pressure_allowed_rate`", ol["corr_pressure_rate"], ol["corr_spec_pressure_rate"]),
        ("WR: `mean_sep_break`", wr["corr_sep_break"], wr["corr_spec_sep_break"]),
        ("WR: `target_rate_break`", wr["corr_tgt_break"], wr["corr_spec_tgt_break"]),
        ("DL: `mean_player_get_off`", dl["corr_get_off"], dl["corr_spec_get_off"]),
        ("DL: `mean_time_to_pressure`", dl["corr_ttp"], dl["corr_spec_ttp"]),
        ("DL: `quick_pressure_rate`", dl["corr_qp_rate"], dl["corr_spec_qp_rate"]),
    ]
    sanity_rows: list[list[object]] = []
    for label, all_c, spec_c in sanity_specs:
        all_list: list[dict[str, object]] = all_c  # type: ignore[assignment]
        spec_list: list[dict[str, object]] = spec_c  # type: ignore[assignment]
        a_s = best_corr_summary(all_list, skill_set, "spearman_rho")
        sp_s = best_corr_summary(spec_list, skill_set, "spearman_rho")
        b_s = best_corr_summary(all_list, base_set, "spearman_rho")
        sanity_rows.append(
            [
                label,
                format_best_badge(*a_s),
                format_best_badge(*sp_s),
                format_best_badge(*b_s),
            ]
        )
    lines.extend(
        format_markdown_table(
            ["Hướng & Target", "All Skill Drills (Spearman)", "Drill chuyên biệt (Spearman)", "Baseline (Spearman)"],
            sanity_rows,
            ["left", "left", "left", "left"],
        )
    )

    # Section 4: Overall comparison
    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Tổng hợp so sánh tính khả thi 3 hướng đề tài (OL vs WR vs DL)",
            "",
        ]
    )
    summary_specs = [
        ("OL", "`peak_pressure_prob_allowed`", ol["cohort"].height, ol_plays.height, ol["corr_peak_prob"]),  # type: ignore[union-attr]
        ("OL", "`pressure_allowed_rate`", ol["cohort"].height, ol_plays.height, ol["corr_pressure_rate"]),  # type: ignore[union-attr]
        ("WR (All)", "`mean_sep_all`", wr_cohort_sep_all.height, wr_plays.height, wr["corr_sep_all"]),
        ("WR (Break)", "`mean_sep_break`", wr_cohort_sep_break.height, wr_break_plays.height, wr["corr_sep_break"]),
        ("WR (Break)", "`target_rate_break`", wr["cohort_tgt_break"].height, wr_break_plays.height, wr["corr_tgt_break"]),  # type: ignore[union-attr]
        ("DL (All)", "`mean_player_get_off`", dl_cohort.height, dl_pr_plays.height, dl["corr_get_off"]),
        ("DL (All)", "`mean_time_to_pressure`", dl_cohort.filter(pl.col("pressure_count") > 0).height, dl_ttp_plays.height, dl["corr_ttp"]),
        ("DL (All)", "`quick_pressure_rate`", dl_cohort.height, dl_pr_plays.height, dl["corr_qp_rate"]),
    ]
    comp_rows: list[list[object]] = []
    for track, tgt, n_players, n_plays, crows in summary_specs:
        clist: list[dict[str, object]] = crows  # type: ignore[assignment]
        best_skill_p = best_corr_summary(clist, skill_set, "pearson_r")
        best_skill_s = best_corr_summary(clist, skill_set, "spearman_rho")
        best_base_s = best_corr_summary(clist, base_set, "spearman_rho")
        comp_rows.append(
            [
                track,
                tgt,
                f"{n_players:,}",
                f"{n_plays:,}",
                format_best_badge(*best_skill_p),
                format_best_badge(*best_skill_s),
                format_best_badge(*best_base_s),
            ]
        )
    lines.extend(
        format_markdown_table(
            ["Nhóm", "Target chính", "Cầu thủ (>=50 snaps)", "Snaps REG", "Best Skill (Pearson)", "Best Skill (Spearman)", "Best Baseline (Spearman)"],
            comp_rows,
            ["left", "left", "right", "right", "left", "left", "left"],
        )
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run data normalization, feature extraction, NFL coverage & signal discovery."""
    t0 = time.perf_counter()
    print("1/4 Loading and normalizing raw data...")
    tables, norm_stats = load_and_normalize_data(RAW_DIR)

    print("2/4 Computing Combine Skill Drills frame kinematics & player features...")
    ct_kin = add_frame_kinematics(tables["combine_tracking"])
    feats = extract_all_group_features(ct_kin, tables["combine_results"])

    print("3/4 Auditing NFL REG sample coverage & correlations (OL, WR, DL)...")
    ol_res = analyze_ol(tables["player_play_reg"], feats["OL_all"], feats["OL_spec"])
    wr_res = analyze_wr(tables["player_play_reg"], feats["WR_all"], feats["WR_spec"])
    dl_res = analyze_dl(tables["player_play_reg"], feats["DL_all"], feats["DL_spec"])

    print("4/4 Writing signal_report.md...")
    report = build_report(tables, norm_stats, ol_res, wr_res, dl_res)
    OUTPUT_PATH.write_text(report, encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s. Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
