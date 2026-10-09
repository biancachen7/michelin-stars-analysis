# Michelin: price vs stars — code only
# Install packages once in Terminal (not here):
#   python -m pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn shap

# %%
import re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
plt.rcParams["text.parse_math"] = False  # price labels contain "$"
pd.set_option("display.max_columns", 50)
sns.set_theme(style="whitegrid", context="notebook")
RANDOM_STATE = 42
AWARD_ORDER = ["Selected Restaurants", "Bib Gourmand", "1 Star", "2 Stars", "3 Stars"]
AWARD_COLORS = ["#c9c9c9", "#e9a23b", "#d9534f", "#a52a2a", "#5c0f0f"]

# %%
raw = pd.read_csv("data/michelin_my_maps.csv")
print(raw.shape)
print(raw.head(3).T)

# %%
audit = pd.DataFrame({
    "dtype": raw.dtypes.astype(str),
    "n_unique": raw.nunique(),
    "n_missing": raw.isna().sum(),
    "pct_missing": (raw.isna().mean() * 100).round(1),
})
print(audit)

# %%
print("Price values:\n", raw["Price"].value_counts(), "\n")
print("Award values:\n", raw["Award"].value_counts(), "\n")
print("Duplicate URLs:", raw["Url"].duplicated().sum(),
      "| Duplicate name+address:", raw.duplicated(["Name", "Address"]).sum(),
      "| Repeated names (different addresses):", raw["Name"].duplicated().sum())

# %%
df = raw.copy()

# 2.1 Price -> numeric level + currency
df = df[df["Price"].str.lower() != "none"].copy()
df["price_level"] = df["Price"].str.len().astype(int)
df["currency"] = df["Price"].str[0].map({"$": "USD-style $", "€": "EUR", "¥": "JPY/CNY", "฿": "THB"})
df["price_label"] = df["price_level"].map({1: "$", 2: "$$", 3: "$$$", 4: "$$$$"})

# 2.2 Location -> city / country
parts = df["Location"].str.split(",")
df["city"] = parts.str[0].str.strip()
df["country"] = parts.str[-1].str.strip().replace({"Dubai": "UAE", "Abu Dhabi": "UAE"})

# 2.3 Award -> ordinal targets
df["Award"] = pd.Categorical(df["Award"], categories=AWARD_ORDER, ordered=True)
df["stars"] = df["Award"].map({"1 Star": 1, "2 Stars": 2, "3 Stars": 3}).astype(float).fillna(0).astype(int)
df["starred"] = (df["stars"] > 0).astype(int)

# 2.4 Facilities -> multi-hot
fac = df["FacilitiesAndServices"].fillna("").str.get_dummies(sep=",")
fac.columns = ["fac_" + re.sub(r"\W+", "_", c.lower()).strip("_") for c in fac.columns]
df = df.join(fac)
df["n_facilities"] = fac.sum(axis=1)

# 2.5 Phone as text, cuisine/country grouping for modelling
df["PhoneNumber"] = df["PhoneNumber"].apply(lambda x: "" if pd.isna(x) else f"+{int(x)}")
top_cuisine = df["Cuisine"].value_counts().index[:15]
df["cuisine_group"] = np.where(df["Cuisine"].isin(top_cuisine), df["Cuisine"], "Other")
top_country = df["country"].value_counts().index[:15]
df["country_group"] = np.where(df["country"].isin(top_country), df["country"], "Other")
df["has_website"] = df["WebsiteUrl"].notna().astype(int)

print(df.shape)
print(df[["Name", "Price", "price_level", "currency", "city", "country", "Award", "stars", "starred", "n_facilities"]].head())

# %%
# Sanity checks after wrangling
assert df["price_level"].between(1, 4).all()
assert df["stars"].isin([0, 1, 2, 3]).all()
assert df["Url"].is_unique
print("Rows removed:", len(raw) - len(df))
print("Facility columns created:", fac.shape[1])
print("Countries:", df["country"].nunique(), "| Cities:", df["city"].nunique())

# Export a tidy file for Excel / Tableau teammates
export_cols = ["Name", "city", "country", "Latitude", "Longitude", "Cuisine", "cuisine_group",
               "Price", "price_level", "price_label", "currency", "Award", "stars", "starred",
               "GreenStar", "n_facilities", "has_website"] + list(fac.columns)
df[export_cols].to_csv("data/michelin_clean.csv", index=False)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
df["Award"].value_counts().reindex(AWARD_ORDER).plot.barh(ax=axes[0], color=AWARD_COLORS)
axes[0].set_title("Restaurants by award"); axes[0].set_xlabel("Count")
df["price_label"].value_counts().reindex(["$", "$$", "$$$", "$$$$"]).plot.bar(ax=axes[1], color="#4a6fa5")
axes[1].set_title("Restaurants by price level"); axes[1].set_ylabel("Count"); axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.show()

# %%
counts = pd.crosstab(df["price_label"], df["Award"]).reindex(["$", "$$", "$$$", "$$$$"])
share = pd.crosstab(df["price_label"], df["Award"], normalize="index").reindex(["$", "$$", "$$$", "$$$$"])
print(counts)
print((share * 100).round(1))

# %%
ax = (share * 100).plot.bar(stacked=True, color=AWARD_COLORS, figsize=(10, 5.5), width=0.75)
ax.set_title("Award mix within each price level (row %)"); ax.set_ylabel("% of restaurants"); ax.set_xlabel("Price level")
ax.tick_params(axis="x", rotation=0); ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout(); plt.show()

# %%
starred = df[df["starred"] == 1]
where_stars = pd.crosstab(starred["Award"].astype(str), starred["price_label"], normalize="index")[["$", "$$", "$$$", "$$$$"]]
ax = (where_stars * 100).plot.barh(stacked=True, figsize=(10, 3.6), color=["#d8e3f0", "#9fb8d8", "#4a6fa5", "#1d3557"])
ax.set_title("Where do starred restaurants sit on price? (share of each star tier)"); ax.set_xlabel("%")
ax.legend(title="Price", bbox_to_anchor=(1.01, 1), loc="upper left"); plt.tight_layout(); plt.show()
print((where_stars * 100).round(1))

# %%
ct = pd.crosstab(df["price_level"], df["Award"])
chi2, p, dof, expected = stats.chi2_contingency(ct)
n = ct.values.sum()
cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
print(f"Chi-square = {chi2:,.0f}, df = {dof}, p = {p:.2e}, Cramér's V = {cramers_v:.3f}")

rho_all = stats.spearmanr(df["price_level"], df["stars"])
rho_starred = stats.spearmanr(starred["price_level"], starred["stars"])
print(f"Spearman (all restaurants, price vs stars 0-3): rho = {rho_all.statistic:.3f}, p = {rho_all.pvalue:.1e}")
print(f"Spearman (starred only, price vs 1-3 stars):    rho = {rho_starred.statistic:.3f}, p = {rho_starred.pvalue:.1e}")

kw = stats.kruskal(*[g["price_level"] for _, g in starred.groupby("stars")])
print(f"Kruskal-Wallis (price level across 1/2/3 stars): H = {kw.statistic:.1f}, p = {kw.pvalue:.1e}")

# %%
fac_keep = ["fac_interesting_wine_list", "fac_counter_dining", "fac_great_view",
            "fac_valet_parking", "fac_terrace", "fac_car_park", "fac_wheelchair_access"]
formula = ("starred ~ C(price_level) + C(country_group, Treatment('Other')) "
           "+ C(cuisine_group, Treatment('Other')) + GreenStar + " + " + ".join(fac_keep))
logit = smf.logit(formula, data=df).fit(disp=0)
print(f"McFadden pseudo R² = {logit.prsquared:.3f}   n = {int(logit.nobs):,}")

ci = np.exp(logit.conf_int())
odds = pd.DataFrame({"odds_ratio": np.exp(logit.params), "ci_low": ci[0], "ci_high": ci[1], "p_value": logit.pvalues})
show = odds.loc[[i for i in odds.index if "price_level" in i or i in fac_keep + ["GreenStar"]]]
print(show.round(3))

# %%
plot_df = show.drop([i for i in show.index if "price_level" in i]).sort_values("odds_ratio")
plot_df.index = [i.replace("fac_", "").replace("_", " ") for i in plot_df.index]
fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(plot_df["odds_ratio"], plot_df.index, xerr=[plot_df["odds_ratio"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["odds_ratio"]],
            fmt="o", color="#1d3557", capsize=3)
ax.axvline(1, color="grey", ls="--"); ax.set_xscale("log")
ax.set_title("Odds of being starred (non-price factors, controlling for price, country, cuisine)")
ax.set_xlabel("Odds ratio (log scale)"); plt.tight_layout(); plt.show()

# %%
X = (pd.get_dummies(df[["country_group", "cuisine_group"]])
       .join(df[["price_level", "GreenStar", "n_facilities", "has_website"] + list(fac.columns)])
       .astype(float))
y = df["starred"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
feature_sets = {
    "Price only": ["price_level"],
    "Everything except price": [c for c in X.columns if c != "price_level"],
    "All features": list(X.columns),
}
results = {name: cross_val_score(HistGradientBoostingClassifier(random_state=RANDOM_STATE), X[cols], y, cv=cv, scoring="roc_auc")
           for name, cols in feature_sets.items()}
auc_table = pd.DataFrame({k: [v.mean(), v.std()] for k, v in results.items()}, index=["ROC AUC (mean)", "std"]).T.round(3)
print(auc_table)

# %%
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
gb = HistGradientBoostingClassifier(random_state=RANDOM_STATE).fit(X_tr, y_tr)
print("Hold-out ROC AUC:", round(roc_auc_score(y_te, gb.predict_proba(X_te)[:, 1]), 3))

pi = permutation_importance(gb, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).head(12)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
RocCurveDisplay.from_estimator(gb, X_te, y_te, ax=axes[0], name="Gradient boosting"); axes[0].plot([0, 1], [0, 1], "--", color="grey")
axes[0].set_title("ROC curve (hold-out set)")
imp[::-1].plot.barh(ax=axes[1], color="#4a6fa5"); axes[1].set_title("Permutation importance (drop in AUC)")
plt.tight_layout(); plt.show()

# %%
text = (df["Description"].fillna("").str.lower()
          .str.replace(r"\b(michelin|stars?|starred)\b", " ", regex=True))
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=10, max_features=20000)
T = tfidf.fit_transform(text)
text_auc = cross_val_score(LogisticRegression(max_iter=2000, C=2), T, y, cv=cv, scoring="roc_auc")
print(f"Text-only model ROC AUC: {text_auc.mean():.3f} ± {text_auc.std():.3f}")

lr_text = LogisticRegression(max_iter=2000, C=2).fit(T, y)
coefs = pd.Series(lr_text.coef_[0], index=tfidf.get_feature_names_out()).sort_values()
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
coefs.tail(15).plot.barh(ax=axes[0], color="#a52a2a"); axes[0].set_title("Words that signal a STARRED restaurant")
coefs.head(15)[::-1].plot.barh(ax=axes[1], color="#9fb8d8"); axes[1].set_title("Words that signal a NON-starred restaurant")
plt.tight_layout(); plt.show()

# %%
gems = df[(df["starred"] == 1) & (df["price_level"] <= 2)]
print(f"{len(gems)} affordable starred restaurants ({len(gems) / df['starred'].sum():.1%} of all starred)")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
gems["country"].value_counts().head(10)[::-1].plot.barh(ax=axes[0], color="#e9a23b"); axes[0].set_title("Affordable starred restaurants by country")
gems["Cuisine"].value_counts().head(10)[::-1].plot.barh(ax=axes[1], color="#e9a23b"); axes[1].set_title("Affordable starred restaurants by cuisine")
plt.tight_layout(); plt.show()
print(gems[["Name", "Location", "Cuisine", "Price", "Award"]].sort_values(["Price", "Award"]).head(15))

# %%
fig, ax = plt.subplots(figsize=(13, 6))
ax.scatter(df["Longitude"], df["Latitude"], s=2, color="#d0d0d0", label="All restaurants")
ax.scatter(starred["Longitude"], starred["Latitude"], s=4, color="#a52a2a", alpha=0.5, label="Starred")
ax.scatter(gems["Longitude"], gems["Latitude"], s=30, color="#e9a23b", edgecolor="black", label="Starred at $ / $$")
ax.set_title("Where the affordable starred restaurants are"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.legend(loc="lower left"); plt.tight_layout(); plt.show()

# %%
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

oof = cross_val_predict(HistGradientBoostingClassifier(random_state=RANDOM_STATE), X, y, cv=cv, method="predict_proba")[:, 1]
df["star_prob"] = oof
print(f"Out-of-fold ROC AUC = {roc_auc_score(y, oof):.3f} | Brier score = {brier_score_loss(y, oof):.3f}")

frac_pos, mean_pred = calibration_curve(y, oof, n_bins=10)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
ax.plot(mean_pred, frac_pos, "o-", color="#a52a2a", label="Model")
ax.set_xlabel("Predicted probability of a star"); ax.set_ylabel("Observed share starred")
ax.set_title("Calibration: can the scores be read as probabilities?"); ax.legend(); plt.tight_layout(); plt.show()

# %%
unstarred = df[df["starred"] == 0]
thresholds = pd.DataFrame({"threshold": [0.3, 0.5, 0.7],
                           "unstarred restaurants above it": [(unstarred["star_prob"] >= t).sum() for t in [0.3, 0.5, 0.7]]})
print(thresholds)
cand = unstarred[unstarred["star_prob"] >= 0.5]
print("Candidates (p >= 0.5) by award:", cand["Award"].value_counts().to_dict())
print("Candidates by price:", cand["price_label"].value_counts().to_dict())
print(cand.sort_values("star_prob", ascending=False)[["Name", "Location", "Cuisine", "Price", "Award", "star_prob"]].head(15).round(3))

# %%
# The project's real audience: affordable restaurants. Which $ / $$ restaurants score highest?
affordable = unstarred[unstarred["price_level"] <= 2].sort_values("star_prob", ascending=False)
print(affordable[["Name", "Location", "Cuisine", "Price", "Award", "star_prob"]].head(12).round(3))

# %%
import shap
gb_full = HistGradientBoostingClassifier(random_state=RANDOM_STATE).fit(X, y)
sv = shap.TreeExplainer(gb_full).shap_values(X)
mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False).head(12)
ax = mean_abs[::-1].plot.barh(figsize=(8, 4.5), color="#4a6fa5")
ax.set_title("Mean |SHAP value| — average impact on the log-odds of a star"); ax.set_xlabel("mean |SHAP|")
plt.tight_layout(); plt.show()

i = cand["star_prob"].idxmax(); row = df.index.get_loc(i)
contrib = pd.Series(sv[row], index=X.columns)
contrib = contrib[(X.iloc[row] != 0) | contrib.index.isin(["price_level", "n_facilities"])].sort_values(key=np.abs, ascending=False).head(6)
print(f"Top candidate: {df.loc[i, 'Name']} ({df.loc[i, 'Location']}), p = {df.loc[i, 'star_prob']:.2f}")
print(contrib.round(3))

# %%
cols = ["Name", "city", "country", "Cuisine", "Price", "Award", "star_prob"]
df.loc[df["starred"] == 0, cols].sort_values("star_prob", ascending=False).to_csv("data/star_candidates.csv", index=False)

# %%
# Turn each model column into the plain-language phrase shown on the card
fac_names = df["FacilitiesAndServices"].fillna("").str.get_dummies(sep=",").columns
FAC_LABEL = {"fac_" + re.sub(r"\W+", "_", n.lower()).strip("_"): n for n in fac_names}

def feature_label(name, value):
    if name == "price_level":
        return "Price tier " + "$" * int(value)
    if name == "n_facilities":
        return f"{int(value)} facilities listed"
    if name.startswith("country_group_"):
        c = name[len("country_group_"):]
        return "Located outside the top-15 countries" if c == "Other" else f"Located in {c}"
    if name.startswith("cuisine_group_"):
        c = name[len("cuisine_group_"):]
        return "Less common cuisine" if c == "Other" else f"Cuisine: {c}"
    if name == "GreenStar":
        return "MICHELIN Green Star"
    if name == "has_website":
        return "Has its own website"
    return FAC_LABEL[name]

ALWAYS_SHOW = {"price_level", "n_facilities"}   # numeric features: meaningful at any value
feature_names, X_values = list(X.columns), X.values

def top_reasons(row, k=3):
    order = np.argsort(-np.abs(sv[row]))          # biggest impact first, whether up or down
    reasons = []
    for j in order:
        name, value = feature_names[j], X_values[row, j]
        if value == 0 and name not in ALWAYS_SHOW:
            continue                               # skip features this restaurant doesn't have
        reasons.append((feature_label(name, value), "raises" if sv[row, j] > 0 else "lowers"))
        if len(reasons) == k:
            break
    return reasons

df["reasons"] = [top_reasons(r) for r in range(len(df))]

# Example: the card for Aquavit (New York, two stars)
row = df.index.get_loc(df.index[df["Name"] == "Aquavit"][0])
print(f"{df.iloc[row]['Name']}: star probability {df.iloc[row]['star_prob']:.0%}")
for label, direction in df.iloc[row]["reasons"]:
    print(f"  {'▲' if direction == 'raises' else '▼'} {direction} {label}")


# %%
# Export everything the map needs to data/map_data.json
import json

m = df.assign(award_rank=df["Award"].cat.codes).sort_values("award_rank", kind="stable")  # stars drawn on top
vocab = {}
def phrase_id(s):
    return vocab.setdefault(s, len(vocab))   # store each phrase once to keep the file small

map_data = {
    "n": m["Name"].tolist(),
    "c": m["city"].tolist(),
    "k": m["country"].tolist(),
    "q": m["Cuisine"].tolist(),
    "p": m["price_level"].tolist(),
    "a": m["award_rank"].tolist(),                       # 0 = Selected ... 4 = 3 Stars
    "x": m["Longitude"].round(4).tolist(),
    "y": m["Latitude"].round(4).tolist(),
    "g": m["GreenStar"].tolist(),
    "u": m["Url"].str.replace("https://guide.michelin.com/", "", regex=False).tolist(),
    "r": (m["star_prob"] * 100).round().astype(int).tolist(),
    "w": [[v for label, d in rs for v in (phrase_id(label), int(d == "raises"))] for rs in m["reasons"]],
}
map_data["V"] = list(vocab)

with open("data/map_data.json", "w", encoding="utf-8") as f:
    json.dump(map_data, f, ensure_ascii=False, separators=(",", ":"))
print(f"Wrote {len(map_data['n']):,} restaurants and {len(vocab)} reason phrases to data/map_data.json")

