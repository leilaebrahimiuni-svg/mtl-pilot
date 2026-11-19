# Task-sharing pilot (H1) for the multi-task framework proposal

Code and results for the preliminary test of H1 (does hard parameter sharing across three
unrelated datasets change per-task error?) in the manuscript *Toward Multi-Task Prediction of
Nanoparticle Migration and Toxicity in Food-Contact Packaging: A Framework Proposal and a
Preliminary Task-Sharing Test* (submitted to Frontiers in Toxicology, Hypothesis and Theory).

## What the code does

1. `src/parse_*.py` extract row-level tables from the three source papers' supplementary files
   (1,847 migration rows, 637 cytotoxicity rows, 199 antimicrobial rows) and stop with an error if
   the row count differs from the expected one.
2. `src/mtl_manuscript_protocol.py` runs 20 group-aware splits (70/15/15 by migrant, nanoparticle
   or experimental condition; seeds 1000-1019). For each task it trains (a) a gradient-boosted
   regressor (reference only), (b) a single-task network and (c) a shared-trunk multi-task network
   (one Adam step per epoch on the summed loss; 1,000 epochs; test RMSE read at each task's best
   validation epoch). Networks are written in NumPy with manual backpropagation; the gradients are
   checked against finite differences at the start of each run.
3. `src/analyze_results.py` computes the statistics in the manuscript (paired differences,
   Nadeau-Bengio corrected t-test, Wilcoxon test, Holm adjustment, 95% intervals).

## Data (not included)

Download the following files from the publishers and place them in `data/raw/` under these names.
All three source papers are CC BY-NC-ND 4.0, so the data and any tables derived from them are not
redistributed here.

| Task | Source | File |
|---|---|---|
| Migration | Vaferi et al. (2025), Sci. Rep. 15, 7806, https://doi.org/10.1038/s41598-025-92459-x | `41598_2025_92459_MOESM1_ESM.pdf` |
| Antimicrobial | Dechakupt et al. (2026), Biomater. Biosyst. 21, 100132, https://doi.org/10.1016/j.bbiosy.2026.100132 | `1-s2.0-S2666534426000061-mmc1.docx` |
| Cytotoxicity | Park et al. (2025), ACS Nanosci. Au 5, 344-352, https://doi.org/10.1021/acsnanoscienceau.5c00035 | `ng5c00035_si_002.xlsx` |

## Requirements

Python 3.12 (tested with 3.12.3), `pip install -r requirements.txt`, plus the command-line tools
`pdftotext` (poppler) and `pandoc` used by the parsers. Tested versions: NumPy 2.4.4, SciPy 1.17.1,
scikit-learn 1.8.0, pandas 3.0.2.

## Run

```bash
python src/parse_vaferi.py
python src/parse_dechakupt.py
python src/parse_park.py
python src/mtl_manuscript_protocol.py 20     # writes results/results_manuscript_protocol.json
python src/analyze_results.py                # writes results/summary_statistics.json
```

`results/results_manuscript_protocol.json` is the output reported in the manuscript. Running the
scripts again on the same input files reproduces it byte for byte (all seeds are fixed).

## Limitations

The pilot tests only hard parameter sharing between proxy tasks that share no mechanism. It does not
test the physics penalty, conformal prediction or active learning described in the manuscript, and
none of the datasets contains nanoparticle migration measurements. Networks use linear encoders, one
trunk width and untuned hyperparameters.

## License

Code: MIT (see `LICENSE`). The license covers the code only, not the source datasets.
