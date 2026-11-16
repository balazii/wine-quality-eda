"""Exported code for Wine Quality: Comparative Exploratory Data Analysis."""

# Recommended execution environment: Google Colab.


# Wine Quality — Comparative Exploratory Data Analysis
from pathlib import Path
from zipfile import ZipFile
import warnings
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 30)

# A. Dataset setup and integrity checks — Questions 1–6
DATA_DIR = Path("/content/wine_quality_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = DATA_DIR / "wine_quality.zip"
URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
if not ZIP_PATH.exists():
    response = requests.get(URL, timeout=90)
    response.raise_for_status()
    ZIP_PATH.write_bytes(response.content)
with ZipFile(ZIP_PATH) as archive:
    archive.extractall(DATA_DIR)

def standardize_columns(frame):
    """Return a copy whose headers use lowercase snake_case."""
    out = frame.copy()
    out.columns = out.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    return out

red = standardize_columns(pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";"))
white = standardize_columns(pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";"))
red["wine_type"] = "red"
white["wine_type"] = "white"
combined = pd.concat([red, white], ignore_index=True)
FEATURES = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
            "ph", "sulphates", "alcohol"]
TARGET = "quality"

print("Red:", red.shape, "White:", white.shape, "Combined:", combined.shape)
for name, frame in [("red", red), ("white", white)]:
    print(name, "first 10 rows")
    display(frame.head(10))
    print(name, "last 10 rows")
    display(frame.tail(10))

# A. Dataset setup and integrity checks — Questions 1–6
def audit(frame, name):
    """Report dimensions, dtypes, memory, missingness, and exact duplicates."""
    print(f"{name}: shape={frame.shape}, memory={frame.memory_usage(deep=True).sum()/1e6:.2f} MB, "
          f"missing={frame.isna().sum().sum()}, duplicate_rows={frame.duplicated().sum()}")
    display(frame.dtypes.rename("dtype").to_frame())
    display(pd.DataFrame({"missing_count": frame.isna().sum(),
                          "missing_pct": 100 * frame.isna().mean()}))
for name, frame in [("red", red), ("white", white), ("combined", combined)]:
    audit(frame, name)

dictionary = pd.DataFrame([
    ("fixed_acidity", "Fixed acids; g/dm3"), ("volatile_acidity", "Volatile acids; g/dm3"),
    ("citric_acid", "Citric acid; g/dm3"), ("residual_sugar", "Residual sugar; g/dm3"),
    ("chlorides", "Sodium chloride; g/dm3"), ("free_sulfur_dioxide", "Free sulfur dioxide; mg/dm3"),
    ("total_sulfur_dioxide", "Total sulfur dioxide; mg/dm3"), ("density", "Wine density; g/cm3"),
    ("ph", "Acidity/alkalinity measure"), ("sulphates", "Potassium sulphate; g/dm3"),
    ("alcohol", "Alcohol; percent by volume"), ("quality", "Ordinal sensory score, 0–10"),
    ("wine_type", "Red or white source dataset"),
], columns=["feature", "meaning"])
dictionary["data_type"] = ["categorical" if x == "wine_type" else "ordinal integer" if x == "quality" else "continuous" for x in dictionary["feature"]]
dictionary["preprocessing"] = "No unit conversion; inspect range and skew before modelling."
display(dictionary)
display(combined[FEATURES + [TARGET]].describe().T)
display(pd.DataFrame({
    "dataset": ["red", "white", "combined"],
    "rows": [len(red), len(white), len(combined)],
    "duplicate_rows": [red.duplicated().sum(), white.duplicated().sum(), combined.duplicated().sum()],
}))
print("Retain exact repeated measurements in the main analysis: the data contain no sample ID to prove they are erroneous duplicates.")

# B. Univariate distributions — Questions 7–12
display(combined.groupby("wine_type")["quality"].agg(
    count="size", mean="mean", median="median", mode=lambda s: s.mode().tolist()))
display(combined["quality"].value_counts().sort_index().rename("count").to_frame()
        .assign(percent=lambda x: 100*x["count"]/len(combined)))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.countplot(data=combined, x="quality", hue="wine_type", ax=axes[0])
axes[0].set_title("Quality counts by wine type")
quality_pct = pd.crosstab(combined["quality"], combined["wine_type"], normalize="columns") * 100
quality_pct.plot(kind="bar", ax=axes[1])
axes[1].set_title("Quality-score percentage within each type")
axes[1].set_ylabel("Percent")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for wine_type, group in combined.groupby("wine_type"):
    sns.histplot(group["alcohol"], stat="density", kde=True, element="step",
                 fill=False, label=wine_type, ax=axes[0, 0])
axes[0, 0].legend()
axes[0, 0].set_title("Alcohol distribution")
shape_features = ["fixed_acidity", "volatile_acidity", "citric_acid",
                  "residual_sugar", "chlorides", "free_sulfur_dioxide",
                  "total_sulfur_dioxide", "density", "ph", "sulphates"]
for feature, ax in zip(shape_features, axes.flat[1:]):
    sns.histplot(data=combined, x=feature, hue="wine_type", kde=True, element="step", ax=ax)
    ax.set_title(feature.replace("_", " ").title())
for ax in axes.flat[len(shape_features)+1:]:
    ax.remove()
plt.tight_layout()
plt.show()
shape_table = combined[shape_features].agg(["mean", "median", "std", "min", "max"]).T
shape_table["skewness"] = combined[shape_features].skew()
display(shape_table.sort_values("skewness", key=lambda s: s.abs(), ascending=False))

# C. Red versus white wine — Questions 13–18
def distribution_summary(frame, features):
    """Calculate comparable statistics and IQR outlier rates by wine type."""
    rows = []
    for wine_type, group in frame.groupby("wine_type"):
        for feature in features:
            values = group[feature].dropna()
            q1, q3 = values.quantile([.25, .75])
            iqr = q3 - q1
            outlier = (values < q1-1.5*iqr) | (values > q3+1.5*iqr)
            rows.append({"wine_type": wine_type, "feature": feature, "n": len(values),
                         "mean": values.mean(), "median": values.median(), "std": values.std(),
                         "iqr": iqr, "range": values.max()-values.min(),
                         "skew": values.skew(), "iqr_outlier_pct": 100*outlier.mean()})
    return pd.DataFrame(rows)
type_summary = distribution_summary(combined, FEATURES + [TARGET])
display(type_summary.round(3))

fig, axes = plt.subplots(4, 3, figsize=(16, 16))
for feature, ax in zip(FEATURES, axes.flat):
    sns.boxplot(data=combined, x="wine_type", y=feature, showfliers=False, ax=ax)
    ax.set_title(feature.replace("_", " ").title())
for ax in axes.flat[len(FEATURES):]:
    ax.remove()
plt.tight_layout()
plt.show()
display((pd.crosstab(combined["quality"], combined["wine_type"], normalize="columns")*100).round(2))

# D. Correlations and chemical relationships — Questions 19–25
numeric_features = FEATURES + [TARGET]
for label, frame in [("Red", red), ("White", white), ("Combined", combined)]:
    corr = frame[numeric_features].corr(method="spearman")
    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", square=True)
    plt.title(f"{label} wine Spearman correlation matrix")
    plt.tight_layout()
    plt.show()
    display(corr[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
            .rename(f"{label}: association with quality").to_frame())

pairs = [("alcohol", "density"), ("fixed_acidity", "ph"),
         ("free_sulfur_dioxide", "total_sulfur_dioxide"),
         ("residual_sugar", "density"), ("volatile_acidity", "citric_acid")]
sample = combined.sample(min(2500, len(combined)), random_state=42)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for (x, y), ax in zip(pairs, axes.flat):
    sns.scatterplot(data=sample, x=x, y=y, hue="wine_type", alpha=.35, s=18, ax=ax)
    sns.regplot(data=combined, x=x, y=y, scatter=False, color="black", ax=ax)
    rho = combined[[x, y]].corr(method="spearman").iloc[0, 1]
    ax.set_title(f"{x} vs {y}: Spearman rho={rho:.2f}")
axes.flat[-1].remove()
plt.tight_layout()
plt.show()

for wine_type, group in combined.groupby("wine_type"):
    print(wine_type, "volatile acidity × quality:",
          round(group["volatile_acidity"].corr(group["quality"], method="spearman"), 3),
          "| alcohol × quality:",
          round(group["alcohol"].corr(group["quality"], method="spearman"), 3))

# E. Quality associations — Questions 26–30
combined["quality_band"] = pd.cut(combined["quality"], [-1, 4, 6, 10],
    labels=["low (0–4)", "medium (5–6)", "high (7–10)"])
key_features = ["alcohol", "volatile_acidity", "sulphates", "chlorides"]
display(combined.groupby(["wine_type", "quality"], observed=True)[key_features]
        .agg(["count", "mean", "median"]))
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for feature, ax in zip(key_features, axes.flat):
    sns.boxplot(data=combined, x="quality_band", y=feature, hue="wine_type",
                showfliers=False, ax=ax)
    ax.set_title(f"{feature.replace('_', ' ').title()} by quality band")
    ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.show()
quality_corr = []
for wine_type, group in combined.groupby("wine_type"):
    for feature in FEATURES:
        quality_corr.append({"wine_type": wine_type, "feature": feature,
            "pearson_r": group[feature].corr(group[TARGET], method="pearson"),
            "spearman_rho": group[feature].corr(group[TARGET], method="spearman")})
display(pd.DataFrame(quality_corr).sort_values(["wine_type", "spearman_rho"],
    key=lambda s: s.abs() if s.name == "spearman_rho" else s, ascending=[True, False]))

# F. Outliers, hypothesis tests, and combined association — Questions 31–35
def outlier_rates(series):
    """Calculate IQR and absolute z-score outlier counts for numeric data."""
    values = series.dropna()
    q1, q3 = values.quantile([.25, .75])
    iqr = q3-q1
    iqr_flag = (values < q1-1.5*iqr) | (values > q3+1.5*iqr)
    z_flag = ((values-values.mean())/values.std(ddof=0)).abs() > 3
    return {"n": len(values), "iqr_count": int(iqr_flag.sum()),
            "iqr_pct": 100*iqr_flag.mean(), "z_gt_3_count": int(z_flag.sum()),
            "z_gt_3_pct": 100*z_flag.mean()}
outlier_rows = []
for wine_type, group in combined.groupby("wine_type"):
    for feature in FEATURES:
        outlier_rows.append({"wine_type": wine_type, "feature": feature,
                             **outlier_rates(group[feature])})
display(pd.DataFrame(outlier_rows).round(3))

def bh_adjust(p_values):
    """Apply the Benjamini–Hochberg false-discovery adjustment."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum.accumulate((ranked*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(adj)
    result[order] = np.clip(adj, 0, 1)
    return result
test_rows = []
for feature in FEATURES + [TARGET]:
    r, w = red[feature].dropna(), white[feature].dropna()
    welch = stats.ttest_ind(r, w, equal_var=False)
    mw = stats.mannwhitneyu(r, w, alternative="two-sided")
    pooled_sd = np.sqrt(((len(r)-1)*r.var(ddof=1)+(len(w)-1)*w.var(ddof=1))/(len(r)+len(w)-2))
    test_rows.append({"feature": feature, "red_mean": r.mean(), "white_mean": w.mean(),
        "mean_diff": r.mean()-w.mean(), "median_diff": r.median()-w.median(),
        "cohen_d": (r.mean()-w.mean())/pooled_sd,
        "welch_p": welch.pvalue, "mannwhitney_p": mw.pvalue})
tests = pd.DataFrame(test_rows)
tests["welch_p_fdr"] = bh_adjust(tests["welch_p"])
tests["mannwhitney_p_fdr"] = bh_adjust(tests["mannwhitney_p"])
display(tests.sort_values("mannwhitney_p_fdr").round(5))

quality_test = stats.mannwhitneyu(red["quality"], white["quality"], alternative="two-sided")
print(f"Quality distributions: U={quality_test.statistic:.0f}, p={quality_test.pvalue:.3g}; "
      f"red median={red.quality.median():.1f}, white median={white.quality.median():.1f}.")

model_data = combined[FEATURES+["wine_type", TARGET]].dropna()
X = pd.get_dummies(model_data[FEATURES+["wine_type"]], columns=["wine_type"], drop_first=True, dtype=float)
y = model_data[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
full = LinearRegression().fit(X_train, y_train)
full_pred = full.predict(X_test)
single = []
for feature in FEATURES:
    one = LinearRegression().fit(X_train[[feature]], y_train)
    pred = one.predict(X_test[[feature]])
    single.append({"feature": feature, "test_r2": r2_score(y_test, pred),
                   "test_mae": mean_absolute_error(y_test, pred)})
single = pd.DataFrame(single).sort_values("test_r2", ascending=False)
display(single)
print(f"Combined model: test R2={r2_score(y_test, full_pred):.3f}, "
      f"MAE={mean_absolute_error(y_test, full_pred):.3f}; "
      f"best single feature={single.iloc[0]['feature']} (R2={single.iloc[0]['test_r2']:.3f}).")
print("This model checks joint linear signal, not causation; nonlinear effects may remain.")

# Optional bonus challenges
bonus = []
for wine_type, group in combined.groupby("wine_type"):
    bonus.append({"wine_type": wine_type,
        "volatile_acidity_quality_rho": group["volatile_acidity"].corr(group["quality"], method="spearman"),
        "residual_sugar_quality_rho": group["residual_sugar"].corr(group["quality"], method="spearman"),
        "alcohol_density_rho": group["alcohol"].corr(group["density"], method="spearman"),
        "free_total_so2_rho": group["free_sulfur_dioxide"].corr(group["total_sulfur_dioxide"], method="spearman"),
        "alcohol_quality_pearson": group["alcohol"].corr(group["quality"], method="pearson")})
display(pd.DataFrame(bonus).set_index("wine_type").round(3))
sugar_bins = pd.qcut(combined["residual_sugar"], q=8, duplicates="drop")
display(combined.groupby(sugar_bins, observed=True).agg(
    n=("quality", "size"), median_sugar=("residual_sugar", "median"),
    mean_quality=("quality", "mean"), median_quality=("quality", "median")))

profile = combined[FEATURES].dropna()
scaled = StandardScaler().fit_transform(profile)
sample_idx = np.random.RandomState(42).choice(len(scaled), min(2500, len(scaled)), replace=False)
clusters = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(scaled)
combined.loc[profile.index, "chemistry_cluster"] = clusters
print("Four-cluster silhouette on sample:", round(silhouette_score(scaled[sample_idx], clusters[sample_idx]), 3))
display(combined.groupby("chemistry_cluster", observed=True)[FEATURES+[TARGET]].mean().round(3))
display(pd.crosstab(combined["chemistry_cluster"], combined["wine_type"], normalize="index").round(3))

anomaly_summary = []
for wine_type, group in combined.groupby("wine_type"):
    scaled_group = StandardScaler().fit_transform(group[FEATURES])
    flag = IsolationForest(contamination="auto", random_state=42, n_jobs=-1).fit_predict(scaled_group) == -1
    anomaly_summary.append({"wine_type": wine_type, "flagged_pct": 100*flag.mean(),
        "flagged_quality_median": group.loc[flag, "quality"].median(),
        "other_quality_median": group.loc[~flag, "quality"].median()})
display(pd.DataFrame(anomaly_summary).round(3))
print("Clusters and anomaly flags are exploratory; they are not quality grades or proof of defective samples.")

# Conclusion and submission checklist
quality_counts = combined["quality"].value_counts().sort_index()
quality_mode = int(quality_counts.idxmax())
type_quality = combined.groupby("wine_type").quality.agg(["count", "mean", "median", "std"])
within_type_assoc = pd.DataFrame(quality_corr)
strongest = within_type_assoc.loc[within_type_assoc.groupby("wine_type").spearman_rho.apply(lambda s: s.abs().idxmax())]
print(f"Across {len(combined):,} observations, the most common quality score is {quality_mode} "
      f"({quality_counts.max():,} samples, {100*quality_counts.max()/len(combined):.1f}%).")
display(type_quality.round(3))
for _, row in strongest.iterrows():
    print(f"Within {row.wine_type} wine, {row.feature} has the strongest absolute Spearman association "
          f"with quality (rho={row.spearman_rho:.3f}); this is an association, not a causal effect.")
print("Red and white samples differ in chemistry and quality distributions; compare medians, IQRs, "
      "and within-type percentages because sample counts are unequal and many features are skewed.")
print("Formal tests and outlier flags are sensitive to sample size, distribution shape, and the selected rule; "
      "review effect sizes and adjusted p-values together.")
