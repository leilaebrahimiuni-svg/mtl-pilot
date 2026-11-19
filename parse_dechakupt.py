"""
Extract the 199-row antimicrobial-release dataset from Dechakupt et al. 2026's
Word-embedded supplementary table (Table S2: "Preprocessing data for model
training"), delivered inside the ScienceDirect supplementary ZIP as
`1-s2.0-S2666534426000061-mmc1.docx`.

Usage:
    python src/parse_dechakupt.py

Input:  data/raw/1-s2.0-S2666534426000061-mmc1.docx
Output: data/dechakupt_antimicrobial_data.csv

Requires `pandoc` on PATH (used to convert the .docx table to Markdown, which is
parsed directly rather than relying on a brittle python-docx table-object walk).

Exits with a non-zero status if the expected row count (199) is not recovered.
"""
import csv
import subprocess
import sys
from pathlib import Path

RAW_DOCX = Path("data/raw/1-s2.0-S2666534426000061-mmc1.docx")
OUT_CSV = Path("data/dechakupt_antimicrobial_data.csv")
EXPECTED_ROWS = 199
N_FEATURES = 21
EXPECTED_COLUMNS = 1 + N_FEATURES + 1 + 3  # row_id + x1..x21 + y + L1 + L2 + L3


def extract_markdown(docx_path: Path) -> list:
    # "-t gfm" (GitHub-Flavored Markdown), not plain "-t markdown": pandoc's
    # default markdown writer renders Word tables as multi-line "grid tables"
    # (+---+---+ borders, wrapped cell text spread across several lines),
    # which the pipe-based parser below cannot read. GFM renders the same
    # tables as single-line "| a | b | c |" pipe tables instead.
    result = subprocess.run(
        ["pandoc", "-t", "gfm", str(docx_path)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.split("\n")


def parse_rows(lines: list) -> list:
    data_lines = []
    capture = False
    for line in lines:
        if line.strip().startswith("| **Row**"):
            capture = True
            continue
        if line.strip().startswith("| --- "):
            continue
        if capture:
            if not line.strip().startswith("|"):
                break
            data_lines.append(line)

    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) != EXPECTED_COLUMNS:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    return rows


def main() -> None:
    if not RAW_DOCX.exists():
        sys.exit(
            f"ERROR: expected source file not found at {RAW_DOCX}\n"
            "Download the ScienceDirect supplementary ZIP for Dechakupt et al. 2026 "
            "(Biomater. Biosyst. 21, 100132), extract it, and place mmc1.docx there."
        )

    rows = parse_rows(extract_markdown(RAW_DOCX))

    if len(rows) != EXPECTED_ROWS:
        sys.exit(
            f"ERROR: parsed {len(rows)} rows, expected exactly {EXPECTED_ROWS}. "
            "Do not proceed with downstream analysis on an unverified extraction."
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = ["row"] + [f"x{i}" for i in range(1, N_FEATURES + 1)] + ["y", "L1", "L2", "L3"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"OK: wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
