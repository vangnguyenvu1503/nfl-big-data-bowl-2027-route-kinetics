"""Automated audit and execution verifier for the Kaggle submission notebook."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOTEBOOK_PATH = ROOT / "notebooks" / "submission_writeup.ipynb"
EXPECTED_FIGURES = [
    "fig1_hero_palmer_vs_dike.png",
    "fig2_scouting_blindspot.png",
    "fig3_combine_to_nfl_bridge.png",
    "fig4_lodco_cv_performance.png",
    "fig5_draft_room_quadrant.png",
]


def count_markdown_tables(text: str) -> int:
    """Count Markdown pipe tables using header + separator row pairs."""
    lines = [ln.strip() for ln in text.splitlines()]
    tables = 0
    i = 0
    sep_pat = re.compile(r"^\|?(\s*:?-{3,}:?\s*\|)+\s*:?-{0,}:?\s*$")
    while i + 1 < len(lines):
        if "|" in lines[i] and sep_pat.match(lines[i + 1]):
            tables += 1
            i += 2
            while i < len(lines) and "|" in lines[i]:
                i += 1
            continue
        i += 1
    return tables


def clean_markdown_words(text: str) -> list[str]:
    """Return word tokens excluding Markdown structural syntax and image links."""
    cleaned_lines: list[str] = []
    sep_pat = re.compile(r"^\s*\|?(\s*:?-{3,}:?\s*\|)+\s*$")
    for line in text.splitlines():
        if sep_pat.match(line):
            continue
        line = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", line)
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = line.replace("|", " ")
        cleaned_lines.append(line)
    joined = "\n".join(cleaned_lines)
    # Count standard word-like tokens (numbers/symbols glued to words count once)
    tokens = re.findall(r"[A-Za-z0-9Δρβ±+\-−=≤≥><.%/']+", joined)
    return [t for t in tokens if re.search(r"[A-Za-z0-9]", t)]


def count_prose_only_words(text: str) -> int:
    """Count whitespace-delimited words excluding table separator lines and pure symbols."""
    out: list[str] = []
    sep_pat = re.compile(r"^\s*\|?(\s*:?-{3,}:?\s*\|)+\s*$")
    for line in text.splitlines():
        if sep_pat.match(line):
            continue
        line = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", line)
        line = re.sub(r"^\s*#+\s*", "", line)
        for tok in line.replace("|", " ").split():
            stripped = tok.strip("*_`~()[]{},;:!?\"'")
            if stripped and stripped not in {"—", "–", "-", "→", "vs", "·"} or stripped == "vs":
                if any(ch.isalnum() for ch in stripped):
                    out.append(stripped)
    return len(out)


def main() -> int:
    notebook_path = DEFAULT_NOTEBOOK_PATH
    if len(sys.argv) > 1:
        arg_path = Path(sys.argv[1])
        notebook_path = arg_path if arg_path.is_absolute() else (ROOT / arg_path).resolve()

    if not notebook_path.exists():
        print(f"[FAIL] Notebook missing: {notebook_path}")
        return 1

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    md_text = "\n\n".join("".join(c.get("source", [])) for c in md_cells)
    clean_words = len(clean_markdown_words(md_text))
    prose_words = count_prose_only_words(md_text)
    raw_ws_words = len(md_text.split())

    n_tables = count_markdown_tables(md_text)

    # Verify figures referenced in code or markdown cells and present on disk
    all_sources = "\n".join("".join(c.get("source", [])) for c in cells)
    found_figs: list[str] = []
    missing_files: list[str] = []
    for fname in EXPECTED_FIGURES:
        if fname in all_sources:
            found_figs.append(fname)
        fpath = ROOT / "figures" / fname
        if not fpath.exists() or fpath.stat().st_size == 0:
            missing_files.append(str(fpath))

    # Verify hidden input metadata on all code cells
    unhidden_indices = [
        idx
        for idx, c in enumerate(cells)
        if c.get("cell_type") == "code" and c.get("metadata", {}).get("_kg_hide-input") is not True
    ]

    # Check inline PNG outputs in figure cells
    fig_output_count = sum(
        1
        for c in code_cells
        for out in c.get("outputs", [])
        if out.get("output_type") == "display_data" and "image/png" in out.get("data", {})
    )

    # Execute every code cell from notebooks/ working directory
    import os
    import io
    import contextlib

    old_cwd = Path.cwd()
    os.chdir(notebook_path.parent)
    exec_ns: dict[str, object] = {}
    exec_errors: list[str] = []
    try:
        # Mock IPython.display.Image/display if IPython isn't running inside a kernel
        for idx, cell in enumerate(code_cells):
            src = "".join(cell.get("source", []))
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(src, f"<cell-{idx}>", "exec"), exec_ns)
            except Exception as exc:  # noqa: BLE001
                exec_errors.append(f"Code cell {idx}: {type(exc).__name__}: {exc}")
            out_txt = buf.getvalue().strip()
            if out_txt:
                print(f"[cell {idx} stdout]\n{out_txt}\n")
    finally:
        os.chdir(old_cwd)

    total_visuals = len(found_figs) + n_tables
    checks = {
        "prose_words_in_1650_1850": 1650 <= prose_words <= 1850,
        "clean_words_under_2000": clean_words < 2000,
        "raw_tokens_under_2000": raw_ws_words < 2000,
        "exact_5_figures_referenced": len(found_figs) == 5,
        "exact_5_png_outputs_embedded": fig_output_count == 5,
        "exact_2_markdown_tables": n_tables == 2,
        "total_visuals_eq_7_under_10": total_visuals == 7 and total_visuals < 10,
        "all_figure_files_exist": len(missing_files) == 0,
        "all_code_cells_kg_hide_input": len(unhidden_indices) == 0,
        "all_code_cells_execute_cleanly": len(exec_errors) == 0,
    }

    print("=== SUBMISSION AUDIT SUMMARY ===")
    print(f"Markdown cells       : {len(md_cells)}")
    print(f"Code cells           : {len(code_cells)}")
    print(f"Prose word count     : {prose_words} (target: 1,650 - 1,850)")
    print(f"Clean token count    : {clean_words} (< 2,000)")
    print(f"Raw whitespace count : {raw_ws_words} (< 2,000)")
    print(f"Figures referenced   : {len(found_figs)} / 5 ({found_figs})")
    print(f"Embedded PNG outputs : {fig_output_count} / 5")
    print(f"Markdown tables      : {n_tables} / 2")
    print(f"Total visuals        : {total_visuals} (< 10 ceiling)")
    print(f"Missing figure files : {missing_files}")
    print(f"Unhidden code cells  : {unhidden_indices}")
    print(f"Execution errors     : {exec_errors}")
    print("\nChecks:")
    for k, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
