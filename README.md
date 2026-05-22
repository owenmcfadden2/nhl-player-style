# NHL Player Style: Clustering & Position Prediction

Discovering hockey player "styles" from on-ice statistics, and testing whether a
player's statistical fingerprint alone can predict their listed position.

This is a course project for **DSC 148: Introduction to Data Mining** (UC San Diego).

## Overview

Hockey players are listed by position (center, wing, defense), but those labels
describe where a player lines up, not necessarily how they *play*. This project
asks two connected questions:

1. **Unsupervised:** If we cluster skaters purely by their statistical fingerprint
   (shots, hits, blocks, takeaways, giveaways, scoring rates), what natural
   "styles" emerge? Do the discovered clusters line up with listed positions, or
   cut across them?
2. **Supervised:** Can we predict a skater's listed position from that fingerprint
   alone? The players the model gets *wrong* are the interesting ones: skaters
   whose statistical style doesn't match their position label.

## Data

Source: the public NHL Stats API (`api.nhle.com/stats/rest/en`), skater `summary`
and `realtime` report tables.

- **Unit of analysis:** one row per (player, game) for the supervised task; one row
  per (player, season) aggregate for clustering.
- **Volume:** ~24,000 skater-games per season; multiple seasons used to exceed
  50,000 instances.
- **Label:** `positionCode` (C / L / R / D), grouped to forward vs. defense or kept
  fine-grained depending on the experiment.

## Reproduce

```bash
pip install requests pandas
python pull_nhl_data.py
```

This writes two files:

- `data/player_games.csv` — supervised dataset (one row per player-game)
- `data/player_seasons.csv` — clustering dataset (one row per player-season)

Edit the `SEASONS` list at the top of `pull_nhl_data.py` to change which seasons
are pulled.

## Repository structure

```
.
├── pull_nhl_data.py     # ingestion pipeline (NHL API -> CSVs)
├── data/                # generated datasets
├── notebooks/           # EDA, clustering, and modeling notebooks
├── src/                 # reusable code factored out of notebooks
├── figures/             # exported plots used in the report
└── report/              # ACM-format write-up (PDF)
```

## Method

*(filled in as the project develops)*

- **Features:** per-60-minute rate stats so players are compared on style rather
  than ice time.
- **Clustering:** _TBD (K-Means / Gaussian Mixture)_, projected to 2D for
  visualization.
- **Prediction:** baselines (logistic regression, naive Bayes) vs. proposed model
  (gradient boosting).
- **Evaluation:** accuracy, F1 (classes are imbalanced), with an ablation over
  feature groups.

## Results

*(to be added)*

| Model | Accuracy | F1 |
|-------|----------|-----|
| Logistic Regression (baseline) | — | — |
| Naive Bayes (baseline) | — | — |
| Gradient Boosting (proposed) | — | — |

## Notes

The NHL Stats API is undocumented; column names and behavior are inferred from
community reference material and may change.
