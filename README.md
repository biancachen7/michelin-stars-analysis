# Is a High Price a Prerequisite for Three MICHELIN Stars?

A data wrangling and data science analysis of **19,720 MICHELIN Guide restaurants**, built from a Northeastern University MISM 6405 (Data Wrangling for Business) group project and extended with statistical modelling, machine learning and text analytics.

**[Explore the interactive map →](https://biancachen7.github.io/michelin-stars-analysis/)**

## Question
From the perspective of an *affordable* restaurant seeking MICHELIN recognition: can a restaurant earn stars without premium prices, and what besides price separates starred restaurants from the rest?

## Key findings
- **Three stars are a premium phenomenon:** 95.7% of three-star restaurants are priced `$$$$`; none are `$` or `$$`.
- **Star share climbs steeply with price:** 0.1% (`$`) → 1.2% (`$$`) → 17.9% (`$$$`) → 66.0% (`$$$$`). Chi-square p < 0.001, Cramér's V = 0.45.
- **Price is the strongest, but not the only, signal:** a gradient-boosted classifier reaches ROC AUC 0.88 on price alone and 0.93 with country, cuisine and facilities; without price it still reaches 0.86.
- **Inspectors reward craft:** a TF-IDF model on inspector descriptions predicts stars with AUC 0.90; star language centres on *chef, team, ingredients, tasting menu, creativity*, while non-star language centres on *bistro, grill, steak, terrace, popular*.
- **Star-potential model:** out-of-fold scores are well calibrated (Brier 0.08); about 800 unstarred restaurants score above 50% and form a watch-list of restaurants that resemble starred ones, each explained with SHAP. Affordable high scorers cluster in Cantonese and Jiangsu cuisine in Greater China and Singapore.
- **99 affordable starred restaurants exist** (97 with one star, 2 with two, none with three), concentrated in Greater China, Singapore, Mexico and the US, in specialised cuisines such as Cantonese, Sichuan and street food.

## Methods
pandas data cleaning and feature engineering (price symbols → ordinal levels, location parsing, multi-value facility encoding) · chi-square, Cramér's V, Spearman, Kruskal-Wallis · logistic regression with controls (statsmodels) · HistGradientBoosting with feature ablation and permutation importance (scikit-learn) · TF-IDF + logistic regression text model · out-of-fold calibrated probabilities with SHAP explanations · interactive D3 map.

## Repository
| File | Contents |
|---|---|
| `michelin_price_vs_stars.ipynb` | Full analysis with outputs |
| `data/michelin_my_maps.csv` | Raw data ([Kaggle, Jerry Ng](https://www.kaggle.com/datasets/ngshiheng/michelin-guide-restaurants-2021)) |
| `data/michelin_clean.csv` | Cleaned, analysis-ready table (also used for Excel/Tableau) |
| `data/star_candidates.csv` | Star-potential score for every unstarred restaurant |
| `index.html` | Interactive world map with award filters and the star-potential layer |
| `data/map_data.json` | Map data exported by the notebook: out-of-fold star probability and top-3 SHAP reasons per restaurant |
| `michelin_cleaning_excel.xlsx` | The same cleaning pipeline as Excel formulas, with a validation sheet |
| `michelin_code_only.py` | The notebook as a plain Python script |
| `requirements.txt` | Python dependencies |

## Run it
```bash
pip install -r requirements.txt
jupyter notebook michelin_price_vs_stars.ipynb
```

## Limitations
Price symbols are relative to each local market, not absolute amounts; the data only covers restaurants already in the Guide; results are associations from a single snapshot, not causal effects.
