"""Audit all raw CSV datasets for NFL Big Data Bowl 2027 using Polars.

Inspects file sizes, row/column counts, inferred schemas, missing value rates,
categorical/key unique values, and player ID overlap between Combine and NFL
season tables, writing the full report to ``data_overview.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
import time

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data_overview.md"

NULL_VALUES = ["NA", ""]
FULL_INFER_THRESHOLD_BYTES = 250 * 1024 * 1024
LARGE_INFER_SCHEMA_LENGTH = 200_000


def scan_csv_lazy(path: Path, full_infer: bool = False) -> pl.LazyFrame:
    """Create a Polars LazyFrame for a competition CSV file."""
    infer_len: int | None = None
    if not full_infer and path.stat().st_size >= FULL_INFER_THRESHOLD_BYTES:
        infer_len = LARGE_INFER_SCHEMA_LENGTH
    return pl.scan_csv(
        path,
        null_values=NULL_VALUES,
        infer_schema_length=infer_len,
    )


def collect_with_fallback(
    path: Path,
    build_lf: Callable[[pl.LazyFrame], pl.LazyFrame],
) -> pl.DataFrame:
    """Collect a LazyFrame query in streaming mode, retrying with full schema inference if needed."""
    try:
        return build_lf(scan_csv_lazy(path, full_infer=False)).collect(engine="streaming")
    except pl.exceptions.ComputeError:
        return build_lf(scan_csv_lazy(path, full_infer=True)).collect(engine="streaming")


def md_escape(value: object) -> str:
    """Format a scalar for safe inclusion inside a Markdown table cell."""
    if value is None:
        return "`<NULL>`"
    text = str(value).replace("\r\n", " ").replace("\n", " ").replace("|", "\\|")
    return text


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


def format_pct(numerator: int, denominator: int) -> str:
    """Format a percentage safely with two decimal places."""
    if denominator <= 0:
        return "0.00%"
    return f"{(numerator / denominator) * 100:.2f}%"


def audit_file_summary_and_schema(
    path: Path,
) -> tuple[dict[str, object], list[tuple[str, str, int, float]]]:
    """Compute row count, column count, schema dtypes, and per-column null counts."""
    size_mb = path.stat().st_size / (1024 * 1024)

    def _run(full_infer: bool) -> tuple[pl.Schema, pl.DataFrame]:
        lf = scan_csv_lazy(path, full_infer=full_infer)
        schema = lf.collect_schema()
        exprs = [pl.len().alias("__row_count__")] + [
            pl.col(col).null_count().alias(col) for col in schema.names()
        ]
        counts_df = lf.select(exprs).collect(engine="streaming")
        return schema, counts_df

    try:
        schema, counts_df = _run(full_infer=False)
    except pl.exceptions.ComputeError:
        schema, counts_df = _run(full_infer=True)

    n_rows = int(counts_df["__row_count__"][0])
    n_cols = len(schema)
    schema_rows: list[tuple[str, str, int, float]] = []
    for col_name, dtype in schema.items():
        null_cnt = int(counts_df[col_name][0])
        missing_pct = (null_cnt / n_rows * 100.0) if n_rows > 0 else 0.0
        schema_rows.append((col_name, str(dtype), null_cnt, missing_pct))

    summary = {
        "file": path.name,
        "size_mb": size_mb,
        "n_rows": n_rows,
        "n_cols": n_cols,
    }
    return summary, schema_rows


def get_value_counts(
    path: Path,
    column: str,
    *,
    sort_by_value: bool = False,
    limit: int | None = None,
) -> tuple[pl.DataFrame, int]:
    """Return value counts DataFrame (`value`, `count`) and total distinct non-null values."""
    df = collect_with_fallback(
        path,
        lambda lf: lf.select(pl.col(column).alias("value"))
        .group_by("value")
        .agg(pl.len().alias("count")),
    )
    n_unique_non_null = int(df.filter(pl.col("value").is_not_null()).height)

    if sort_by_value:
        df = df.sort("value", nulls_last=True)
    else:
        df = df.sort(["count", "value"], descending=[True, False], nulls_last=True)

    if limit is not None:
        df = df.head(limit)
    return df, n_unique_non_null


def render_value_counts_block(
    path: Path,
    column: str,
    total_rows: int,
    *,
    sort_by_value: bool = False,
    limit: int | None = None,
    title_note: str | None = None,
) -> list[str]:
    """Build Markdown lines for a column's unique values and counts."""
    df, n_unique = get_value_counts(path, column, sort_by_value=sort_by_value, limit=limit)
    lines: list[str] = []
    note = f" ({title_note})" if title_note else ""
    showing = f" — hiển thị top {limit}" if limit is not None and n_unique > limit else ""
    lines.append(f"#### Cột `{column}` — {n_unique:,} giá trị duy nhất (non-null){note}{showing}")
    lines.append("")

    rows = [
        [row["value"], f"{row['count']:,}", format_pct(int(row["count"]), total_rows)]
        for row in df.iter_rows(named=True)
    ]
    lines.extend(
        format_markdown_table(
            ["Value", "Count", "% of Rows"],
            rows,
            ["left", "right", "right"],
        )
    )
    lines.append("")
    return lines


def get_key_stats(path: Path, key_col: str = "nfl_id") -> tuple[int, int, int]:
    """Return (total_rows, unique_non_null_keys, null_key_rows)."""
    df = collect_with_fallback(
        path,
        lambda lf: lf.select(
            pl.len().alias("total_rows"),
            pl.col(key_col).drop_nulls().n_unique().alias("unique_non_null"),
            pl.col(key_col).null_count().alias("null_rows"),
        ),
    )
    return (
        int(df["total_rows"][0]),
        int(df["unique_non_null"][0]),
        int(df["null_rows"][0]),
    )


def get_unique_id_set(path: Path, column: str = "nfl_id") -> set[int]:
    """Extract the exact set of non-null integer IDs from a CSV file."""
    df = collect_with_fallback(
        path,
        lambda lf: lf.select(pl.col(column)).drop_nulls().unique(),
    )
    return {int(v) for v in df[column].to_list()}


def audit_players_categoricals(raw_dir: Path, row_counts: dict[str, int]) -> list[str]:
    """Audit key integrity and categorical columns for players.csv."""
    path = raw_dir / "players.csv"
    total, uniq, nulls = get_key_stats(path, "nfl_id")
    lines = [
        "### `players.csv`",
        "",
        f"- Khóa `nfl_id`: **{uniq:,}** unique non-null / **{total:,}** dòng (null: {nulls:,}).",
        "",
    ]
    lines.extend(render_value_counts_block(path, "draft_year", row_counts["players.csv"], sort_by_value=True))
    lines.extend(render_value_counts_block(path, "nfl_position", row_counts["players.csv"]))
    lines.extend(render_value_counts_block(path, "draft_round", row_counts["players.csv"], sort_by_value=True))
    lines.extend(render_value_counts_block(path, "college_conference", row_counts["players.csv"]))
    lines.extend(
        render_value_counts_block(
            path,
            "college_name",
            row_counts["players.csv"],
            limit=20,
        )
    )
    return lines


def audit_combine_results_categoricals(raw_dir: Path, row_counts: dict[str, int]) -> list[str]:
    """Audit key integrity and categorical columns for combine_results.csv."""
    path = raw_dir / "combine_results.csv"
    total, uniq, nulls = get_key_stats(path, "nfl_id")
    lines = [
        "### `combine_results.csv`",
        "",
        f"- Khóa `nfl_id`: **{uniq:,}** unique non-null / **{total:,}** dòng (null: {nulls:,}).",
        "",
    ]
    lines.extend(
        render_value_counts_block(path, "draft_year", row_counts["combine_results.csv"], sort_by_value=True)
    )
    lines.extend(render_value_counts_block(path, "combine_position", row_counts["combine_results.csv"]))
    return lines


def audit_combine_tracking_categoricals(raw_dir: Path, row_counts: dict[str, int]) -> list[str]:
    """Audit keys, entity types, draft years, drills, and attempts for combine_tracking.csv."""
    path = raw_dir / "combine_tracking.csv"
    key_df = collect_with_fallback(
        path,
        lambda lf: lf.select(
            pl.len().alias("total_rows"),
            pl.col("nfl_id").drop_nulls().n_unique().alias("unique_nfl_id"),
            pl.col("nfl_id").filter(pl.col("entity_type") == "PLAYER").drop_nulls().n_unique().alias("unique_player_id"),
            pl.col("nfl_id").filter(pl.col("entity_type") == "BALL").drop_nulls().n_unique().alias("unique_ball_id"),
            pl.col("event_id").drop_nulls().n_unique().alias("unique_event_id"),
        ),
    )
    r = key_df.row(0, named=True)
    lines = [
        "### `combine_tracking.csv`",
        "",
        f"- Tổng số dòng: **{int(r['total_rows']):,}**",
        f"- Unique `nfl_id` (toàn bảng): **{int(r['unique_nfl_id']):,}**",
        f"- Unique `nfl_id` (`entity_type == \"PLAYER\"`): **{int(r['unique_player_id']):,}**",
        f"- Unique `nfl_id` (`entity_type == \"BALL\"`): **{int(r['unique_ball_id']):,}**",
        f"- Unique `event_id`: **{int(r['unique_event_id']):,}**",
        "",
    ]

    def _grouped_player_table(
        col: str,
        *,
        sort_cols: list[str],
        descending: list[bool],
        heading: str,
    ) -> list[str]:
        select_cols = list(dict.fromkeys([col, "nfl_id", "entity_type"]))
        df = collect_with_fallback(
            path,
            lambda lf: lf.select(select_cols)
            .group_by(col)
            .agg(
                pl.len().alias("rows"),
                pl.col("nfl_id").drop_nulls().n_unique().alias("unique_nfl_id"),
                pl.col("nfl_id")
                .filter(pl.col("entity_type") == "PLAYER")
                .drop_nulls()
                .n_unique()
                .alias("unique_player_nfl_id"),
            )
            .sort(sort_cols, descending=descending, nulls_last=True),
        )
        n_uniq = int(df.filter(pl.col(col).is_not_null()).height)
        block = [f"#### {heading} — {n_uniq:,} giá trị duy nhất", ""]
        table_rows = [
            [
                row[col],
                f"{int(row['rows']):,}",
                f"{int(row['unique_nfl_id']):,}",
                f"{int(row['unique_player_nfl_id']):,}",
            ]
            for row in df.iter_rows(named=True)
        ]
        block.extend(
            format_markdown_table(
                [col, "Rows", "Unique nfl_id", "Unique nfl_id (PLAYER)"],
                table_rows,
                ["left", "right", "right", "right"],
            )
        )
        block.append("")
        return block

    lines.extend(
        _grouped_player_table(
            "draft_year",
            sort_cols=["draft_year"],
            descending=[False],
            heading="Cột `draft_year`",
        )
    )
    lines.extend(
        _grouped_player_table(
            "entity_type",
            sort_cols=["rows", "entity_type"],
            descending=[True, False],
            heading="Cột `entity_type`",
        )
    )
    lines.extend(
        _grouped_player_table(
            "drill_type",
            sort_cols=["unique_player_nfl_id", "rows", "drill_type"],
            descending=[True, True, False],
            heading="Cột `drill_type`",
        )
    )
    lines.extend(
        _grouped_player_table(
            "drill_name",
            sort_cols=["unique_player_nfl_id", "rows", "drill_name"],
            descending=[True, True, False],
            heading="Cột `drill_name`",
        )
    )
    lines.extend(
        render_value_counts_block(
            path,
            "attempt",
            row_counts["combine_tracking.csv"],
            sort_by_value=True,
        )
    )
    return lines


def audit_games_categoricals(raw_dir: Path, row_counts: dict[str, int]) -> list[str]:
    """Audit keys, seasons, weeks, and team abbreviations for games.csv."""
    path = raw_dir / "games.csv"
    df_keys = collect_with_fallback(
        path,
        lambda lf: lf.select(
            pl.len().alias("total_rows"),
            pl.col("game_id").drop_nulls().n_unique().alias("unique_game_id"),
            pl.col("game_key").drop_nulls().n_unique().alias("unique_game_key"),
        ),
    )
    kr = df_keys.row(0, named=True)
    lines = [
        "### `games.csv`",
        "",
        f"- Unique `game_id`: **{int(kr['unique_game_id']):,}** / {int(kr['total_rows']):,} dòng; Unique `game_key`: **{int(kr['unique_game_key']):,}**.",
        "",
    ]

    lines.extend(render_value_counts_block(path, "season", row_counts["games.csv"], sort_by_value=True))
    lines.extend(render_value_counts_block(path, "season_type", row_counts["games.csv"]))

    combo_df = collect_with_fallback(
        path,
        lambda lf: lf.select(["season", "season_type", "week", "game_id"])
        .group_by(["season", "season_type"])
        .agg(
            pl.len().alias("games"),
            pl.col("week").min().alias("min_week"),
            pl.col("week").max().alias("max_week"),
        )
        .sort(["season", "season_type"]),
    )
    lines.append("#### Phân bố `season` × `season_type`")
    lines.append("")
    lines.extend(
        format_markdown_table(
            ["season", "season_type", "Games", "Week Range"],
            [
                [r["season"], r["season_type"], f"{int(r['games']):,}", f"{r['min_week']}..{r['max_week']}"]
                for r in combo_df.iter_rows(named=True)
            ],
            ["right", "left", "right", "center"],
        )
    )
    lines.append("")

    lines.extend(render_value_counts_block(path, "week", row_counts["games.csv"], sort_by_value=True))

    teams_df = collect_with_fallback(
        path,
        lambda lf: lf.select(["home_team_abbr", "visitor_team_abbr"]),
    )
    home_teams = sorted(v for v in teams_df["home_team_abbr"].drop_nulls().unique().to_list())
    visitor_teams = sorted(v for v in teams_df["visitor_team_abbr"].drop_nulls().unique().to_list())
    lines.append("#### Các đội bóng (`home_team_abbr` & `visitor_team_abbr`)")
    lines.append("")
    lines.append(f"- `home_team_abbr` ({len(home_teams)} đội): {', '.join(f'`{t}`' for t in home_teams)}")
    lines.append(f"- `visitor_team_abbr` ({len(visitor_teams)} đội): {', '.join(f'`{t}`' for t in visitor_teams)}")
    lines.append("")
    return lines


def audit_career_successes_categoricals(raw_dir: Path, row_counts: dict[str, int]) -> list[str]:
    """Audit key integrity and accolade columns for player_career_successes.csv."""
    path = raw_dir / "player_career_successes.csv"
    total, uniq, nulls = get_key_stats(path, "nfl_id")
    lines = [
        "### `player_career_successes.csv`",
        "",
        f"- Khóa `nfl_id`: **{uniq:,}** unique non-null / **{total:,}** dòng (null: {nulls:,}).",
        "",
    ]
    for col in ("ap_all_pro_1st_team", "ap_all_pro_2nd_team", "pro_bowl_original_ballot"):
        lines.extend(
            render_value_counts_block(
                path,
                col,
                row_counts["player_career_successes.csv"],
                sort_by_value=True,
            )
        )
    return lines


def audit_player_play_categoricals(raw_dir: Path, row_counts: dict[str, int]) -> list[str]:
    """Audit keys, season coverage, and categorical columns for player_play.csv."""
    path = raw_dir / "player_play.csv"
    games_path = raw_dir / "games.csv"

    key_df = collect_with_fallback(
        path,
        lambda lf: lf.select(
            pl.len().alias("total_rows"),
            pl.col("nfl_id").drop_nulls().n_unique().alias("unique_nfl_id"),
            pl.col("game_id").drop_nulls().n_unique().alias("unique_game_id"),
            pl.struct(["game_id", "play_id"]).n_unique().alias("unique_plays"),
        ),
    )
    kr = key_df.row(0, named=True)
    lines = [
        "### `player_play.csv`",
        "",
        f"- Tổng số dòng: **{int(kr['total_rows']):,}**",
        f"- Unique `nfl_id`: **{int(kr['unique_nfl_id']):,}**",
        f"- Unique `game_id`: **{int(kr['unique_game_id']):,}**",
        f"- Unique `(game_id, play_id)`: **{int(kr['unique_plays']):,}**",
        "",
    ]

    games_lf = scan_csv_lazy(games_path).select(["game_id", "season", "season_type"])
    season_df = collect_with_fallback(
        path,
        lambda lf: lf.select(["game_id", "nfl_id"])
        .join(games_lf, on="game_id", how="left")
        .group_by(["season", "season_type"])
        .agg(
            pl.len().alias("rows"),
            pl.col("game_id").n_unique().alias("unique_games"),
            pl.col("nfl_id").drop_nulls().n_unique().alias("unique_nfl_id"),
        )
        .sort(["season", "season_type"], nulls_last=True),
    )
    lines.append("#### Độ phủ theo mùa giải (join `game_id` với `games.csv`)")
    lines.append("")
    lines.extend(
        format_markdown_table(
            ["season", "season_type", "Rows", "Unique game_id", "Unique nfl_id"],
            [
                [
                    r["season"],
                    r["season_type"],
                    f"{int(r['rows']):,}",
                    f"{int(r['unique_games']):,}",
                    f"{int(r['unique_nfl_id']):,}",
                ]
                for r in season_df.iter_rows(named=True)
            ],
            ["right", "left", "right", "right", "right"],
        )
    )
    lines.append("")

    categorical_cols = [
        "lined_up_position",
        "team_abbr",
        "play_direction",
        "play_nullified_by_penalty",
        "offense_formation",
        "receiver_alignment",
        "pass_result",
        "team_coverage_man_zone",
        "team_coverage_type",
        "route_ran",
        "coverage_assignment",
        "coverage_assignment_at_snap",
    ]
    for col in categorical_cols:
        lines.extend(render_value_counts_block(path, col, row_counts["player_play.csv"]))
    return lines


def audit_game_tracking_categoricals(raw_dir: Path, filename: str) -> list[str]:
    """Audit keys, season coverage, and tracking events for a game_tracking_*.csv file."""
    path = raw_dir / filename
    games_df = collect_with_fallback(
        raw_dir / "games.csv",
        lambda lf: lf.select(["game_id", "season", "season_type"]),
    )

    stats_df = collect_with_fallback(
        path,
        lambda lf: lf.select(
            pl.len().alias("total_rows"),
            pl.col("nfl_id").drop_nulls().n_unique().alias("unique_nfl_id"),
            pl.col("nfl_id").null_count().alias("null_nfl_id_rows"),
            pl.col("game_id").drop_nulls().n_unique().alias("unique_game_id"),
            pl.struct(["game_id", "play_id"]).n_unique().alias("unique_plays"),
        ),
    )
    sr = stats_df.row(0, named=True)
    total_rows = int(sr["total_rows"])

    gp_unique_df = collect_with_fallback(
        path,
        lambda lf: lf.select(["game_id", "nfl_id"]).unique(),
    )
    season_df = (
        gp_unique_df.join(games_df, on="game_id", how="left")
        .group_by(["season", "season_type"])
        .agg(
            pl.col("game_id").n_unique().alias("unique_games"),
            pl.col("nfl_id").drop_nulls().n_unique().alias("unique_nfl_id"),
        )
        .sort(["season", "season_type"], nulls_last=True)
    )

    lines = [
        f"### `{filename}`",
        "",
        f"- Tổng số dòng: **{total_rows:,}**",
        f"- Unique non-null `nfl_id`: **{int(sr['unique_nfl_id']):,}** (dòng `nfl_id` null: {int(sr['null_nfl_id_rows']):,})",
        f"- Unique `game_id`: **{int(sr['unique_game_id']):,}**",
        f"- Unique `(game_id, play_id)`: **{int(sr['unique_plays']):,}**",
        "",
        "#### Độ phủ theo mùa giải (join `game_id` với `games.csv`)",
        "",
    ]
    lines.extend(
        format_markdown_table(
            ["season", "season_type", "Unique game_id", "Unique nfl_id"],
            [
                [
                    r["season"],
                    r["season_type"],
                    f"{int(r['unique_games']):,}",
                    f"{int(r['unique_nfl_id']):,}",
                ]
                for r in season_df.iter_rows(named=True)
            ],
            ["right", "left", "right", "right"],
        )
    )
    lines.append("")
    lines.extend(render_value_counts_block(path, "event", total_rows))
    return lines


def audit_player_overlap(raw_dir: Path) -> list[str]:
    """Compute and format nfl_id overlap statistics across Combine and NFL season files."""
    table_files = [
        ("players", "players.csv"),
        ("combine_results", "combine_results.csv"),
        ("combine_tracking", "combine_tracking.csv"),
        ("player_play", "player_play.csv"),
        ("game_tracking_2023", "game_tracking_2023.csv"),
        ("game_tracking_2024", "game_tracking_2024.csv"),
        ("game_tracking_2025", "game_tracking_2025.csv"),
        ("player_career_successes", "player_career_successes.csv"),
    ]
    id_sets: dict[str, set[int]] = {
        label: get_unique_id_set(raw_dir / fname, "nfl_id")
        for label, fname in table_files
    }

    ct_player_df = collect_with_fallback(
        raw_dir / "combine_tracking.csv",
        lambda lf: lf.filter(pl.col("entity_type") == "PLAYER")
        .select("nfl_id")
        .drop_nulls()
        .unique(),
    )
    combine_tracking_player = {int(v) for v in ct_player_df["nfl_id"].to_list()}
    ct_ball_only = id_sets["combine_tracking"] - combine_tracking_player

    combine_any = id_sets["combine_results"] | id_sets["combine_tracking"]
    combine_both = id_sets["combine_results"] & id_sets["combine_tracking"]
    game_tracking_all = (
        id_sets["game_tracking_2023"]
        | id_sets["game_tracking_2024"]
        | id_sets["game_tracking_2025"]
    )
    nfl_season_any = id_sets["player_play"] | game_tracking_all
    nfl_season_both = id_sets["player_play"] & game_tracking_all

    lines = [
        "## 4. Khóa liên kết `nfl_id` & Mức độ trùng khớp giữa Combine và mùa giải NFL",
        "",
        "### 4.1. Tổng số cầu thủ duy nhất (`nfl_id`) theo từng bảng và nhóm nguồn",
        "",
    ]

    summary_groups: list[tuple[str, str, set[int]]] = [
        ("`players.csv`", "Danh sách cầu thủ gốc", id_sets["players"]),
        ("`combine_results.csv`", "Chỉ số đo lường & điểm Combine", id_sets["combine_results"]),
        ("`combine_tracking.csv` (tất cả `entity_type`)", "Tracking Combine (toàn bảng)", id_sets["combine_tracking"]),
        ("`combine_tracking.csv` (`PLAYER` only)", "Tracking Combine (chỉ `entity_type == \"PLAYER\"`)", combine_tracking_player),
        ("`combine_any` (`results ∪ tracking`)", "Có ít nhất 1 nguồn Combine", combine_any),
        ("`combine_both` (`results ∩ tracking`)", "Có cả Combine results & Combine tracking", combine_both),
        ("`player_play.csv`", "Thống kê theo play NFL (2023–2025)", id_sets["player_play"]),
        ("`game_tracking_2023.csv`", "Tracking trận đấu 2023", id_sets["game_tracking_2023"]),
        ("`game_tracking_2024.csv`", "Tracking trận đấu 2024", id_sets["game_tracking_2024"]),
        ("`game_tracking_2025.csv`", "Tracking trận đấu 2025", id_sets["game_tracking_2025"]),
        ("`game_tracking_all` (`2023 ∪ 2024 ∪ 2025`)", "Có tracking trận đấu ít nhất 1 mùa", game_tracking_all),
        ("`nfl_season_any` (`player_play ∪ game_tracking_all`)", "Có mặt trong dữ liệu mùa giải NFL", nfl_season_any),
        ("`nfl_season_both` (`player_play ∩ game_tracking_all`)", "Có cả `player_play` và `game_tracking`", nfl_season_both),
        ("`player_career_successes.csv`", "Tổng hợp thành tích sự nghiệp NFL", id_sets["player_career_successes"]),
    ]
    lines.extend(
        format_markdown_table(
            ["Nguồn / Nhóm", "Mô tả", "Unique `nfl_id`"],
            [[name, desc, f"{len(s):,}"] for name, desc, s in summary_groups],
            ["left", "left", "right"],
        )
    )
    lines.append("")
    lines.append(
        f"- Kiểm tra `combine_tracking.csv`: tập `nfl_id` có `entity_type == \"PLAYER\"` gồm **{len(combine_tracking_player):,}** ID; "
        f"số ID chỉ xuất hiện với `BALL` (không có dòng `PLAYER`) là **{len(ct_ball_only):,}**."
    )
    lines.append("")

    # 4.2 Combine <-> NFL Season Matching Table
    lines.append("### 4.2. Bảng đối chiếu trùng khớp Combine ↔ Mùa giải NFL")
    lines.append("")

    comparisons: list[tuple[str, set[int], set[int]]] = [
        ("`combine_results` ↔ `combine_tracking`", id_sets["combine_results"], id_sets["combine_tracking"]),
        ("`combine_results` ↔ `player_play`", id_sets["combine_results"], id_sets["player_play"]),
        ("`combine_results` ↔ `game_tracking_2023`", id_sets["combine_results"], id_sets["game_tracking_2023"]),
        ("`combine_results` ↔ `game_tracking_2024`", id_sets["combine_results"], id_sets["game_tracking_2024"]),
        ("`combine_results` ↔ `game_tracking_2025`", id_sets["combine_results"], id_sets["game_tracking_2025"]),
        ("`combine_results` ↔ `game_tracking_all`", id_sets["combine_results"], game_tracking_all),
        ("`combine_results` ↔ `nfl_season_any`", id_sets["combine_results"], nfl_season_any),
        ("`combine_results` ↔ `player_career_successes`", id_sets["combine_results"], id_sets["player_career_successes"]),
        ("`combine_tracking` ↔ `player_play`", id_sets["combine_tracking"], id_sets["player_play"]),
        ("`combine_tracking` ↔ `game_tracking_2023`", id_sets["combine_tracking"], id_sets["game_tracking_2023"]),
        ("`combine_tracking` ↔ `game_tracking_2024`", id_sets["combine_tracking"], id_sets["game_tracking_2024"]),
        ("`combine_tracking` ↔ `game_tracking_2025`", id_sets["combine_tracking"], id_sets["game_tracking_2025"]),
        ("`combine_tracking` ↔ `game_tracking_all`", id_sets["combine_tracking"], game_tracking_all),
        ("`combine_tracking` ↔ `nfl_season_any`", id_sets["combine_tracking"], nfl_season_any),
        ("`combine_tracking` ↔ `player_career_successes`", id_sets["combine_tracking"], id_sets["player_career_successes"]),
        ("`combine_any` ↔ `nfl_season_any`", combine_any, nfl_season_any),
        ("`combine_any` ↔ `player_career_successes`", combine_any, id_sets["player_career_successes"]),
        ("`combine_both` ↔ `nfl_season_both`", combine_both, nfl_season_both),
    ]

    comp_rows: list[list[object]] = []
    for label, set_a, set_b in comparisons:
        matched = len(set_a & set_b)
        only_a = len(set_a - set_b)
        only_b = len(set_b - set_a)
        comp_rows.append(
            [
                label,
                f"{len(set_a):,}",
                f"{len(set_b):,}",
                f"{matched:,}",
                format_pct(matched, len(set_a)),
                format_pct(matched, len(set_b)),
                f"{only_a:,}",
                f"{only_b:,}",
            ]
        )
    lines.extend(
        format_markdown_table(
            [
                "Comparison (A ↔ B)",
                "|A|",
                "|B|",
                "Matched |A ∩ B|",
                "% of A",
                "% of B",
                "Only in A",
                "Only in B",
            ],
            comp_rows,
            ["left", "right", "right", "right", "right", "right", "right", "right"],
        )
    )
    lines.append("")

    # 4.3 Breakdown by draft_year
    lines.append("### 4.3. Phân tách trùng khớp theo `draft_year` (từ `players.csv`)")
    lines.append("")

    players_df = collect_with_fallback(
        raw_dir / "players.csv",
        lambda lf: lf.select(["nfl_id", "draft_year"]),
    )
    draft_groups: dict[int, set[int]] = {}
    for row in players_df.iter_rows(named=True):
        dy = int(row["draft_year"])
        draft_groups.setdefault(dy, set()).add(int(row["nfl_id"]))

    dy_rows: list[list[object]] = []
    for dy in sorted(draft_groups):
        p_set = draft_groups[dy]
        dy_rows.append(
            [
                dy,
                f"{len(p_set):,}",
                f"{len(p_set & id_sets['combine_results']):,}",
                f"{len(p_set & id_sets['combine_tracking']):,}",
                f"{len(p_set & id_sets['player_play']):,}",
                f"{len(p_set & id_sets['game_tracking_2023']):,}",
                f"{len(p_set & id_sets['game_tracking_2024']):,}",
                f"{len(p_set & id_sets['game_tracking_2025']):,}",
                f"{len(p_set & game_tracking_all):,}",
                f"{len(p_set & id_sets['player_career_successes']):,}",
                f"{len(p_set & combine_any & nfl_season_any):,}",
                f"{len(p_set & combine_both & nfl_season_both):,}",
            ]
        )
    lines.extend(
        format_markdown_table(
            [
                "draft_year",
                "players",
                "combine_results",
                "combine_tracking",
                "player_play",
                "gt_2023",
                "gt_2024",
                "gt_2025",
                "gt_all",
                "career_successes",
                "combine_any ∩ nfl_any",
                "combine_both ∩ nfl_both",
            ],
            dy_rows,
            ["right"] * 12,
        )
    )
    lines.append("")

    # 4.4 Full 8x8 Pairwise Overlap Matrix
    lines.append("### 4.4. Ma trận giao `|Set_i ∩ Set_j|` giữa 8 bảng chứa `nfl_id`")
    lines.append("")

    labels = [label for label, _ in table_files]
    matrix_rows: list[list[object]] = []
    for row_lbl in labels:
        row_cells: list[object] = [f"`{row_lbl}`"]
        for col_lbl in labels:
            row_cells.append(f"{len(id_sets[row_lbl] & id_sets[col_lbl]):,}")
        matrix_rows.append(row_cells)
    lines.extend(
        format_markdown_table(
            ["Table"] + [f"`{lbl}`" for lbl in labels],
            matrix_rows,
            ["left"] + ["right"] * len(labels),
        )
    )
    lines.append("")

    # 4.5 ID Integrity / Orphan Check
    lines.append("### 4.5. Kiểm tra tính toàn vẹn khóa `nfl_id` so với `players.csv`")
    lines.append("")

    master_ids = id_sets["players"]
    integrity_rows: list[list[object]] = []
    for label, fname in table_files:
        if label == "players":
            continue
        cur_ids = id_sets[label]
        orphan_ids = sorted(cur_ids - master_ids)
        missing_from_table = sorted(master_ids - cur_ids)
        sample_orphans = ", ".join(str(x) for x in orphan_ids[:10]) if orphan_ids else "—"
        integrity_rows.append(
            [
                f"`{fname}`",
                f"{len(cur_ids):,}",
                f"{len(orphan_ids):,}",
                f"{len(missing_from_table):,}",
                sample_orphans,
            ]
        )
    lines.extend(
        format_markdown_table(
            [
                "File",
                "Unique `nfl_id`",
                "Orphan IDs (ngoài `players.csv`)",
                "ID trong `players.csv` vắng mặt",
                "Ví dụ Orphan IDs",
            ],
            integrity_rows,
            ["left", "right", "right", "right", "left"],
        )
    )
    lines.append("")
    return lines


def main() -> None:
    """Run full EDA audit over raw CSV files and write data_overview.md."""
    t0 = time.perf_counter()
    csv_files = sorted(RAW_DIR.glob("*.csv"), key=lambda p: p.name)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    print(f"Auditing {len(csv_files)} CSV files in {RAW_DIR}...")

    summaries: list[dict[str, object]] = []
    schemas: dict[str, list[tuple[str, str, int, float]]] = {}
    row_counts: dict[str, int] = {}

    for path in csv_files:
        print(f"  - Auditing schema & counts: {path.name}")
        summary, schema_rows = audit_file_summary_and_schema(path)
        summaries.append(summary)
        schemas[path.name] = schema_rows
        row_counts[path.name] = int(summary["n_rows"])

    total_size_mb = sum(float(s["size_mb"]) for s in summaries)
    total_rows = sum(int(s["n_rows"]) for s in summaries)

    lines: list[str] = [
        "# NFL Big Data Bowl 2027 — Data Audit Overview",
        "",
        f"- **Thời gian tạo báo cáo**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- **Thư mục dữ liệu**: `{RAW_DIR}`",
        "- **Xử lý giá trị thiếu**: Đã cấu hình `null_values=[\"NA\", \"\"]` trong `polars.scan_csv` để nhận diện chính xác chuỗi `\"NA\"` và chuỗi rỗng là missing values (giữ nguyên các nhãn hợp lệ như `\"NONE\"`, `\"NO_ASSIGNMENT\"`).",
        "",
        "## 1. Tổng quan các file dữ liệu",
        "",
    ]

    summary_rows: list[list[object]] = [
        [
            f"`{s['file']}`",
            f"{float(s['size_mb']):.3f}",
            f"{int(s['n_rows']):,}",
            f"{int(s['n_cols']):,}",
        ]
        for s in summaries
    ]
    summary_rows.append(
        [
            f"**Tổng cộng ({len(summaries)} files)**",
            f"**{total_size_mb:.3f}**",
            f"**{total_rows:,}**",
            "—",
        ]
    )
    lines.extend(
        format_markdown_table(
            ["File", "Size (MB)", "Rows", "Columns"],
            summary_rows,
            ["left", "right", "right", "right"],
        )
    )
    lines.append("")

    lines.extend(
        [
            "## 2. Schema và tỷ lệ Missing Values từng file",
            "",
        ]
    )
    for s in summaries:
        fname = str(s["file"])
        lines.append(
            f"### `{fname}` ({float(s['size_mb']):.3f} MB | {int(s['n_rows']):,} dòng | {int(s['n_cols']):,} cột)"
        )
        lines.append("")
        schema_table_rows = [
            [
                idx,
                f"`{col_name}`",
                f"`{dtype}`",
                f"{null_cnt:,}",
                f"{missing_pct:.2f}%",
            ]
            for idx, (col_name, dtype, null_cnt, missing_pct) in enumerate(schemas[fname], start=1)
        ]
        lines.extend(
            format_markdown_table(
                ["#", "Column", "Dtype", "Missing Count", "Missing (%)"],
                schema_table_rows,
                ["right", "left", "left", "right", "right"],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## 3. Giá trị duy nhất của các cột phân loại & khóa liên kết",
            "",
        ]
    )
    print("  - Auditing categorical & key columns...")
    lines.extend(audit_players_categoricals(RAW_DIR, row_counts))
    lines.extend(audit_combine_results_categoricals(RAW_DIR, row_counts))
    lines.extend(audit_combine_tracking_categoricals(RAW_DIR, row_counts))
    lines.extend(audit_games_categoricals(RAW_DIR, row_counts))
    lines.extend(audit_career_successes_categoricals(RAW_DIR, row_counts))
    lines.extend(audit_player_play_categoricals(RAW_DIR, row_counts))
    for gt_file in (
        "game_tracking_2023.csv",
        "game_tracking_2024.csv",
        "game_tracking_2025.csv",
    ):
        if (RAW_DIR / gt_file).exists():
            print(f"  - Auditing tracking categoricals: {gt_file}")
            lines.extend(audit_game_tracking_categoricals(RAW_DIR, gt_file))

    print("  - Auditing Combine <-> NFL season nfl_id overlap...")
    lines.extend(audit_player_overlap(RAW_DIR))

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s. Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
