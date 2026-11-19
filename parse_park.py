"""
Extract the 637-row cytotoxicity dataset from Park et al. 2025's row-level
Supporting Information spreadsheet (Table S1), `ng5c00035_si_002.xlsx`.
https://doi.org/10.1021/acsnanoscienceau.5c00035

Usage:
    python src/parse_park.py

Input:  data/raw/ng5c00035_si_002.xlsx
Output: data/park_cytotoxicity_data.csv

Exits with a non-zero status if the expected row count (637) or the expected
number of distinct cell lines (110) is not recovered.
"""
import csv
import sys
from pathlib import Path

from openpyxl import load_workbook

RAW_XLSX = Path("data/raw/ng5c00035_si_002.xlsx")
OUT_CSV = Path("data/park_cytotoxicity_data.csv")
EXPECTED_ROWS = 637
EXPECTED_CELL_LINES = 110
CELL_LINE_COLUMN_INDEX = 8  # 0-based index of "Cell-identification" in the header row


def main() -> None:
    if not RAW_XLSX.exists():
        sys.exit(
            f"ERROR: expected source file not found at {RAW_XLSX}\n"
            "Download it from the ACS Nanoscience Au Supporting Information for "
            "https://doi.org/10.1021/acsnanoscienceau.5c00035 and place it there."
        )

    wb = load_workbook(RAW_XLSX, read_only=True)
    if "Table S1" not in wb.sheetnames:
        sys.exit(f"ERROR: sheet 'Table S1' not found; sheets present: {wb.sheetnames}")

    ws = wb["Table S1"]
    all_rows = list(ws.iter_rows(values_only=True))

    header_idx = None
    for i, row in enumerate(all_rows):
        if row and row[0] == "Core":
            header_idx = i
            break
    if header_idx is None:
        sys.exit("ERROR: could not locate the header row (expected first cell 'Core').")

    header = all_rows[header_idx]
    data_rows = [r for r in all_rows[header_idx + 1:] if r and r[0] is not None]
    n_cell_lines = len({r[CELL_LINE_COLUMN_INDEX] for r in data_rows})

    if len(data_rows) != EXPECTED_ROWS:
        sys.exit(
            f"ERROR: parsed {len(data_rows)} rows, expected exactly {EXPECTED_ROWS}. "
            "Do not proceed with downstream analysis on an unverified extraction."
        )
    if n_cell_lines != EXPECTED_CELL_LINES:
        sys.exit(
            f"ERROR: parsed {n_cell_lines} distinct cell lines, expected exactly "
            f"{EXPECTED_CELL_LINES}."
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data_rows)

    print(f"OK: wrote {len(data_rows)} rows across {n_cell_lines} cell lines to {OUT_CSV}")


if __name__ == "__main__":
    main()
