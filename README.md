# NHL Player Style: Clustering & Position Prediction

Discovering hockey player "styles" from on-ice statistics, and testing whether a
player's statistical fingerprint alone can predict their listed position.

Course project for **DSC 148: Introduction to Data Mining** (UC San Diego).

## Overview

This project asks two connected questions:

1. **Supervised:** Can a skater's listed position be predicted from per-60
   counting statistics alone?
2. **Unsupervised:** When forwards are clustered purely by their statistical
   fingerprint, do recognizable "styles" emerge?

**Headline findings:**
- Forward vs. defense classification reaches ~97% accuracy. Logistic regression
  outperforms gradient boosting, indicating the boundary is essentially linear.
- The 4-class task (C/L/R/D) fails specifically on left vs. right wings,
  revealing that wing designation is not a statistically distinct style.
- Unsupervised K-Means on forward profiles recovers four interpretable
  archetypes — stars, middle-six, fourth-liners, and enforcers — that map onto
  recognizable NHL roles. The algorithm rediscovers role hierarchy without
  access to any labels.
- The model's "misclassifications" identify stylistic outliers: offensive
  defensemen (e.g., Dougie Hamilton) and defensive forwards (e.g., Barclay
  Goodrow) whose role diverges from their listed position.

The full write-up is in `report/nhl_player_style.pdf`.

## Data

Source: public NHL Stats API (`api.nhle.com/stats/rest/en`), skater `summary`
and `realtime` report tables.

- **Volume:** ~94,000 skater-game records across the 2022-23 and 2023-24
  regular seasons.
- **Unit of analysis (supervised):** player-game records, 1,086 train / 357 test
  after a grouped split by `playerId` to prevent leakage.
- **Unit of analysis (clustering):** 1,443 player-season profiles after
  filtering to players with at least 200 minutes of ice time.
- **Features:** 11 per-60 rate statistics covering scoring (goals, assists,
  shots) and defensive/physical play (blocks, hits, takeaways, giveaways).

## Reproduce

```bash
pip install requests pandas scikit-learn matplotlib seaborn jupyter
python pull_nhl_data.py
jupyter notebook nhl_eda.ipynb         # builds player_fingerprints.csv
jupyter notebook nhl_modeling.ipynb    # trains models, produces tables
```

`pull_nhl_data.py` writes `data/player_games.csv` and `data/player_seasons.csv`.
The EDA notebook writes `data/player_fingerprints.csv`. The modeling notebook
writes the three result CSVs into `tables/`.

## Results summary

| Task | Best Model | Accuracy | F1 |
|------|------------|----------|-----|
| Forward vs. defense | Logistic Regression | 0.966 | 0.974 |
| 4-class (C/L/R/D)   | Logistic Regression | 0.700 | 0.571 (macro) |

| Forward archetype | n | Defining stat |
|-------------------|----|---------------|
| Middle-six        | 477 | balanced production |
| Stars             | 283 | high shots/goals/assists |
| Enforcers         | 120 | 14.5 hits per 60 |
| Fourth-line       | 55 | low offense |

## Repository structure
.
├── pull_nhl_data.py       # NHL Stats API ingestion (team-by-team paginated)
├── nhl_eda.ipynb          # EDA + feature engineering
├── nhl_modeling.ipynb     # classification, ablation, clustering, case studies
├── data/                  # raw pulls and processed datasets
├── figures/               # plots used in the report
├── tables/                # numeric result tables (CSV)
└── report/                # compiled PDF write-up

## Notes

The NHL Stats API is undocumented; the pull script paginates team-by-team to
work around a ~10,000-row offset ceiling. Column names and behavior are
inferred from community reference material and may change.
