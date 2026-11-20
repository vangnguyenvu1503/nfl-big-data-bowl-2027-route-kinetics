"""End-to-end Kaggle/Local runner for the NFL Big Data Bowl 2027 WR Route Kinetics pipeline.

Environment variables:
    BDB_RAW_DIR: Input directory containing raw competition CSVs.
        Resolution order: $BDB_RAW_DIR -> /kaggle/input/nfl-big-data-bowl-2027
        -> shallow search for games.csv under /kaggle/input -> <repo>/data/raw.
    BDB_OUT_DIR: Writable root directory for processed outputs, audit reports, and PNGs.
        Resolution order: $BDB_OUT_DIR -> /kaggle/working -> <repo>.

Runtime dependencies (all standard in Kaggle Python images; zero pandas/pyarrow):
    polars>=1.0, numpy, scipy, scikit-learn, lightgbm, statsmodels, matplotlib.
    If an older container lacks polars>=1.0 (`pl.len()` / `how='full', coalesce=True`),
    attach a wheel dataset or run `pip install -q "polars>=1.0"`.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
_KAGGLE_INPUT = Path("/kaggle/input")
_KAGGLE_DEFAULT_RAW = _KAGGLE_INPUT / "nfl-big-data-bowl-2027"
_KAGGLE_WORK = Path("/kaggle/working")

PIPELINE_SCRIPTS = [
    "03_wr_soe_target.py",
    "04_wr_biomechanics.py",
    "05_wr_advanced_kinetics.py",
    "06_wr_game_tracking_bridge.py",
    "07_wr_modeling_lodco.py",
    "08_wr_robustness_audit.py",
    "09_generate_visuals.py",
]

REQUIRED_RAW_CSVS = [
    "games.csv",
    "combine_results.csv",
    "combine_tracking.csv",
    "player_play.csv",
    "players.csv",
    "player_career_successes.csv",
    "game_tracking_2023.csv",
    "game_tracking_2024.csv",
    "game_tracking_2025.csv",
]

EXPECTED_PROCESSED = [
    "wr_targets.parquet",
    "wr_oof_metrics.json",
    "wr_biomechanics.parquet",
    "wr_advanced_kinetics.parquet",
    "wr_advanced_kinetics_summary.json",
    "wr_game_kinetics.parquet",
    "wr_kinetic_translation.json",
]

EXPECTED_FIGURES = [
    "fig1_hero_palmer_vs_dike.png",
    "fig2_scouting_blindspot.png",
    "fig3_combine_to_nfl_bridge.png",
    "fig4_lodco_cv_performance.png",
    "fig5_draft_room_quadrant.png",
]


def _find_games_csv(root: Path, max_depth: int = 3) -> Path | None:
    """Locate a directory containing `games.csv` up to `max_depth` under `root`."""
    if not root.exists():
        return None
    for current, dirs, files in os.walk(root):
        cur_path = Path(current)
        depth = len(cur_path.relative_to(root).parts)
        if "games.csv" in files and "combine_tracking.csv" in files:
            return cur_path
        if depth >= max_depth:
            dirs[:] = []
    return None


def resolve_raw_dir() -> Path:
    """Resolve read-only competition raw CSV directory."""
    env = os.environ.get("BDB_RAW_DIR")
    if env:
        return Path(env).resolve()
    if (_KAGGLE_DEFAULT_RAW / "games.csv").exists():
        return _KAGGLE_DEFAULT_RAW
    if _KAGGLE_INPUT.exists():
        found = _find_games_csv(_KAGGLE_INPUT, max_depth=3)
        if found is not None:
            return found
    return (PROJECT_ROOT / "data" / "raw").resolve()


def resolve_out_dir() -> Path:
    """Resolve writable output directory."""
    env = os.environ.get("BDB_OUT_DIR")
    if env:
        return Path(env).resolve()
    if _KAGGLE_WORK.exists():
        return _KAGGLE_WORK
    return PROJECT_ROOT.resolve()


def check_dependencies() -> None:
    """Verify all required packages exist and enforce Polars >= 1.0 compatibility."""
    packages = [
        ("polars", "polars"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("lightgbm", "lightgbm"),
        ("statsmodels", "statsmodels"),
        ("matplotlib", "matplotlib"),
    ]
    versions: list[str] = []
    missing: list[str] = []
    for mod_name, pip_name in packages:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            versions.append(f"{pip_name}=={ver}")
            if mod_name == "polars" and ver != "unknown":
                major = int(str(ver).split(".")[0])
                if major < 1:
                    raise RuntimeError(
                        f"polars>={1}.0 required (found {ver}). Upgrade via: pip install -q 'polars>=1.0'"
                    )
        except ImportError:
            missing.append(pip_name)
    if missing:
        raise RuntimeError(
            f"Missing required libraries: {missing}. Install via: pip install -q {' '.join(missing)}"
        )
    print("Preflight libraries OK:", ", ".join(versions))


def check_raw_inputs(raw_dir: Path) -> None:
    """Ensure all 9 required competition CSVs exist inside raw_dir."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")
    missing = [f for f in REQUIRED_RAW_CSVS if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required raw CSV(s) in {raw_dir}: {missing}")
    print(f"Raw inputs verified in {raw_dir} ({len(REQUIRED_RAW_CSVS)} CSVs present).")


def main() -> int:
    """Run pipeline scripts 03 -> 09 sequentially and validate outputs."""
    t_total = time.perf_counter()
    raw_dir = resolve_raw_dir()
    out_dir = resolve_out_dir()
    processed_dir = out_dir / "data" / "processed"
    fig_dir = out_dir / "figures"

    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== BDB 2027 WR Route Kinetics Pipeline ===")
    print(f"SCRIPT_DIR : {SCRIPT_DIR}")
    print(f"BDB_RAW_DIR: {raw_dir}")
    print(f"BDB_OUT_DIR: {out_dir}")

    check_dependencies()
    check_raw_inputs(raw_dir)

    child_env = os.environ.copy()
    child_env["BDB_RAW_DIR"] = str(raw_dir)
    child_env["BDB_OUT_DIR"] = str(out_dir)
    child_env["MPLBACKEND"] = "Agg"
    child_env["PYTHONUNBUFFERED"] = "1"

    for idx, script_name in enumerate(PIPELINE_SCRIPTS, start=1):
        script_path = SCRIPT_DIR / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Pipeline script missing: {script_path}")
        print(f"\n--- [{idx}/{len(PIPELINE_SCRIPTS)}] Running {script_name} ---")
        t0 = time.perf_counter()
        subprocess.run([sys.executable, str(script_path)], env=child_env, check=True)
        print(f"--- Finished {script_name} in {time.perf_counter() - t0:.2f}s ---")

    missing_proc = [
        f
        for f in EXPECTED_PROCESSED
        if not (processed_dir / f).exists() or (processed_dir / f).stat().st_size == 0
    ]
    missing_figs = [
        f
        for f in EXPECTED_FIGURES
        if not (fig_dir / f).exists() or (fig_dir / f).stat().st_size == 0
    ]
    if missing_proc or missing_figs:
        raise RuntimeError(
            f"Pipeline completed with missing outputs. processed={missing_proc}, figures={missing_figs}"
        )

    print("\n=== Pipeline Verification Complete ===")
    print(f"Processed artifacts ({len(EXPECTED_PROCESSED)}): {processed_dir}")
    print(f"Published figures  ({len(EXPECTED_FIGURES)}): {fig_dir}")
    print(f"Total runtime      : {time.perf_counter() - t_total:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
