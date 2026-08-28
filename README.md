# 🍺 IPA Beer Classifier

A binary classification model predicting whether a given beer recipe is an IPA (India Pale Ale), based on its physicochemical parameters.

## Problem

A classification task built on homebrewing recipe data — the goal was to build a model with the highest possible F1-score on an imbalanced and partially noisy dataset.

## Data

~37,000 craft beer recipes from Brewer's Friend, with the following fields:

| Column | Description |
|---|---|
| `IsIPA` | Target — is the beer an IPA? |
| `OG` | Specific gravity of wort before fermentation |
| `FG` | Specific gravity of wort after fermentation |
| `ABV` | Alcohol By Volume |
| `IBU` | International Bittering Units |
| `Color` | Standard Reference Method (light to dark, e.g. 40 = black) |
| `BoilGravity` | Specific gravity of wort before the boil |
| `PitchRate` | Yeast pitched per gravity unit (M cells/ml/°P) |
| `Efficiency` | Mash extraction efficiency |
| `UserId` | Brewer's Friend user ID who shared the recipe (dropped — not predictive) |

## Methodology

1. **Column audit** — dropped only `UserId` (identifier, no predictive value)
2. **Cleaning** — clipped physically impossible values (e.g. `OG = 15.897`)
3. **Feature engineering** — 19 additional features (ratios, interactions, log transforms, missing-value flags) → 28 features total
4. **Modeling** — a bag of 3 LightGBM models (different seeds), each trained with 5-fold stratified cross-validation (leakage-safe across folds)
5. **Threshold tuning** on out-of-fold (OOF) predictions

## Results

- **F1-score (OOF): ≈ 0.798**
- **F1-score (test): 0.7958**
- Also tested XGBoost, CatBoost, 7-seed bagging, and ensembling — all plateaued at a similar ceiling (~0.795–0.798)

## Conclusions

About 14% of recipes fall in an ambiguous probability band (0.3–0.7) — their features overlap between IPA and non-IPA. Pushing past the 0.80 threshold reliably would require additional information beyond the dataset (e.g. hop variety, yeast strain, recipe text).

## Repository structure

```
├── data/
│   ├── IPA.csv
│   ├── IPA_test.csv
│   └── IPA_description.txt
├── notebook/
│   └── The_great_group_IPA_code.ipynb
├── prediction/
│   └── IPA_predictions.csv
├── requirements.txt
└── README.md
```

## Running the project

```bash
pip install -r requirements.txt
jupyter notebook notebooks/IPA_code.ipynb
```
