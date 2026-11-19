"""
Extract the 1,847-row migration dataset from Vaferi et al. 2025's supplementary PDF
(Table S1: "Utilized databank for the models' development and predicted values by
the GBR"). https://doi.org/10.1038/s41598-025-92459-x

Usage:
    python src/parse_vaferi.py

Input:  data/raw/41598_2025_92459_MOESM1_ESM.pdf
Output: data/vaferi_migration_data.csv

Exits with a non-zero status and a clear error message if the expected row count
(1,847) is not recovered, so a corrupted download or a changed PDF layout is caught
immediately rather than silently producing wrong downstream numbers.
"""
import re
import csv
import subprocess
import sys
from pathlib import Path

RAW_PDF = Path("data/raw/41598_2025_92459_MOESM1_ESM.pdf")
OUT_CSV = Path("data/vaferi_migration_data.csv")
EXPECTED_ROWS = 1847
EXPECTED_MATERIAL_CODES = 19

ROW_PATTERN = re.compile(r'^\s*([\d.]+)\s+(.+?)\s+(\d+)\s+([\-\d.]+)\s+(\d+)\s+([\-\d.]+)\s+([\-\d.]+)\s*$')


def extract_text(pdf_path: Path) -> list:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.split("\n")


def parse_rows(lines: list) -> list:
    start_idx = end_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("0.0") and "Carbohydrates" in line:
            start_idx = i
            break
    for i, line in enumerate(lines):
        if "Nosratabadi" in line:  # first line of the References section
            end_idx = i
            break
    if start_idx is None or end_idx is None:
        raise RuntimeError(
            "Could not locate the data table in the PDF (markers not found). "
            "The source PDF's layout may have changed."
        )

    rows = []
    for line in lines[start_idx:end_idx]:
        line = line.rstrip("\n")
        if not line.strip() or line.strip().startswith("\x0c"):
            continue
        m = ROW_PATTERN.match(line)
        if m:
            etoh, material, code, logkow, temp, target, gbr = m.groups()
            rows.append((float(etoh), material.strip(), int(code),
                         float(logkow), int(temp), float(target), float(gbr)))
    return rows


def main() -> None:
    if not RAW_PDF.exists():
        sys.exit(
            f"ERROR: expected source file not found at {RAW_PDF}\n"
            "Download it from https://doi.org/10.1038/s41598-025-92459-x "
            "(Supplementary Information) and place it there."
        )

    rows = parse_rows(extract_text(RAW_PDF))
    n_materials = len(set(r[2] for r in rows))

    if len(rows) != EXPECTED_ROWS:
        sys.exit(
            f"ERROR: parsed {len(rows)} rows, expected exactly {EXPECTED_ROWS}. "
            "Do not proceed with downstream analysis on an unverified extraction."
        )
    if n_materials != EXPECTED_MATERIAL_CODES:
        sys.exit(
            f"ERROR: parsed {n_materials} distinct material codes, expected exactly "
            f"{EXPECTED_MATERIAL_CODES}."
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["etoh_eq", "material", "material_code", "logKow", "temp",
                    "target_logK", "gbr_pred"])
        w.writerows(rows)

    print(f"OK: wrote {len(rows)} rows across {n_materials} material codes to {OUT_CSV}")


if __name__ == "__main__":
    main()
