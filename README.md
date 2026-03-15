# Zebrafish Cardiac Motion Analysis

This repository analyzes zebrafish MUSCLEMOTION results to summarize heartbeat variability and estimate arrhythmia likelihood for each sample.

## What this code does

- Reads sample folders from `MM_Results/` (for example `Phe_50_1.1-Contr-Results`).
- Uses each sample’s `contraction.txt` signal to detect heartbeat peaks.
- Measures time between beats and computes summary metrics.
- Writes results to `output/hrv_summary.csv` and saves per-sample plots in `output/`.
- Provides an optional Streamlit dashboard (`python/gui_app.py`) for interactive review.

## How arrhythmia probability is estimated

The script creates an `arrhythmia_probability` score between **0** and **1** from heartbeat timing irregularity:

- **Overall irregularity**: how spread out the beat-to-beat timings are.
- **Peak-to-peak interval**: how much neighboring beats change from one beat to the next.
- **Unusual beats**: how often beat intervals are far from the sample’s typical interval.

Each of these three signals is converted to a 0-to-1 score and then averaged into one final probability-like value.

- A value closer to **1** means more irregular rhythm.
- A value closer to **0** means more regular rhythm.
- The `Arrhymia` column is `True` when `arrhythmia_probability > 0.4`, otherwise `False`.

If there are too few detected beats to evaluate reliably, the probability is set to `0.0`.

> This output is a research-oriented signal quality/rhythm irregularity estimate, not a clinical diagnosis.

## How to run

From `python/` (inside a venv or Conda environment):

```bash
pip install -r requirements.txt
python data_analysis.py --results_dir ../MM_Results --output_dir ../output
```

Optional dashboard:

```bash
streamlit run gui_app.py
```
