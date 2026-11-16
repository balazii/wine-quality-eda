# Project Report: Wine Quality: Comparative Exploratory Data Analysis

**Student:** Podisetty Balaji  
**Course:** Artificial Intelligence — Mini Project 1  
**Project artifact:** `notebooks/wine_quality_eda.ipynb`  
**Executed notebook:** [Open the Google Colab notebook](https://colab.research.google.com/drive/18nJ0ceRJXOpIon5ofzBn4OshtfI6AJGY)

## 1. Executive summary

This project compares the physicochemical profiles and ordinal quality scores of red and white Vinho Verde wine samples. It checks data quality, explores feature distributions, tests group differences, measures quality associations, and uses clustering and anomaly screening as exploratory tools.

## 2. Problem statement and objectives

The assignment asks for a reproducible analysis of wine chemistry and quality. The notebook:

- Loads and combines the red and white source files and labels wine type.
- Audits shape, data types, missing values, repeated rows, and feature ranges.
- Describes score and chemistry distributions with summary tables and visualizations.
- Compares red and white wines with within-type percentages, medians, IQRs, ranges, skewness, and outlier rates.
- Measures Pearson and Spearman relationships, including quality associations within each wine type.
- Uses Welch and Mann–Whitney tests, effect sizes, and Benjamini–Hochberg adjusted p-values.
- Applies K-means clustering and Isolation Forest for exploratory pattern screening.

## 3. Data source and scope

The source is the UCI Wine Quality dataset (red and white Vinho Verde wines), dataset 186: [UCI dataset page](https://archive.ics.uci.edu/dataset/186/wine+quality). The combined sample has 6,497 records. It is observational and describes the supplied dataset, not every wine or a current population.

## 4. Approach and implementation

The notebook downloads the official UCI ZIP archive, reads its semicolon-delimited CSV files, normalizes column names, adds a `wine_type` label, and combines the two tables. It reports repeated rows rather than dropping them because there is no unique bottle identifier.

Exploration covers the quality score and 11 physicochemical inputs. Statistical comparisons include mean and median differences, effect sizes, two kinds of tests, and multiple-comparison adjustment. Standardized chemistry features are used for K-means; a reproducible sample silhouette score evaluates the exploratory clusters. Isolation Forest flags unusual profiles for review.

## 5. Outcomes

- Red wine: **1,599 records**. White wine: **4,898 records**. Combined: **6,497 observations and 13 columns** after adding the type label.
- The audit found **no missing values**. It counted **1,177 repeated rows** across the combined data (240 red and 937 white); they were retained because repetition alone does not identify an erroneous record.
- Quality score **6** was the most common overall (2,836 records, 43.7%). The median was 6 for both types; mean scores were 5.636 for red and 5.878 for white.
- Alcohol had the strongest absolute within-type Spearman association with quality (red rho **0.479**, white rho **0.440**). Volatile acidity was negatively associated with quality (red rho **-0.381**, white rho **-0.197**).
- The exploratory four-cluster chemistry sample had a silhouette score of **0.245**.

These correlations are associations, not causal effects. Cluster and anomaly outputs are exploratory and do not represent wine grades or verified defects.

## 6. Tools and reproducibility

Python 3, Google Colab/Jupyter, pandas, NumPy, requests, Matplotlib, Seaborn, SciPy, and scikit-learn. Fixed random seeds are used for reproducible sample/model steps. The notebook ran top to bottom in Colab; the linked version contains saved results. The repository notebook is a clean source copy without embedded outputs and can be rerun with **Runtime → Run all**.

## 7. Limitations and interpretation

- The source is observational; chemistry associations do not establish causes of a quality score.
- Quality is an ordinal sensory rating. Mean values and correlations are summaries, not evidence that the rating is continuous.
- The red and white sample sizes differ; comparisons use within-type percentages where appropriate.
- Test results, outlier cutoffs, clusters, and anomaly flags depend on sample size and modeling choices and should be interpreted as exploratory.

## 8. Conclusion

The two wine types share a median quality score of 6, and alcohol has the strongest observed within-type association with quality in this analysis. The modest cluster silhouette suggests limited separation under this four-cluster setup. The results describe the supplied sample and should not be interpreted as causal quality rules.

## 9. References

- [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
- Course assignment brief provided through Training Central.
