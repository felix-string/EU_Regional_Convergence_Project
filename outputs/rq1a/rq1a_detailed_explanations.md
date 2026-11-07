# RQ1A — Is the GDP spread narrowing year by year?

## Scope and measurement

This section measures **sigma convergence**: whether the cross-sectional dispersion of GDP per capita (PPS index) across NUTS 2 regions decreases over time. It does **not** test whether initially poorer regions subsequently grow faster (beta convergence), explain industrial or geographical differences, or reconcile the two convergence tests. Those questions are reserved for other workstreams.

The dataset contains **2,196 region-year observations**, **244 unique regions**, and **9 years (2016–2024)**. Each region receives equal weight. The standard deviation uses the sample formula (ddof=1). CV is SD divided by the cross-sectional mean, multiplied by 100. P90–P10 subtracts the annual 10th percentile from the annual 90th percentile. SD and percentile gaps are in **PPS index points, not euros**. Negative changes in dispersion indicate narrowing; positive changes indicate widening. These descriptive statistics do not identify causes.

## Figure 1. Indexed dispersion indicators

![Figure 1. Three dispersion indicators indexed to the first year = 100](figure_1_indexed_dispersion.png)

**What the figure shows.** The coefficient of variation, standard deviation and P90–P10 gap are rescaled to **2016 = 100** to permit comparison of their proportional movement despite different original units. An index below 100 indicates less dispersion than in the base year; above 100 indicates more. Indexing does not alter the underlying values or their direction of change.

**Results.** The coefficient of variation narrowed from 39.77% in 2016 to 37.96% in 2024 (change: -1.82%; -4.57% relative to the starting value). Standard deviation narrowed from 37.56 index points in 2016 to 35.69 index points in 2024 (change: -1.87 index points; -4.99% relative to the starting value). The P90–P10 gap narrowed from 88.70 index points in 2016 to 75.70 index points in 2024 (change: -13.00 index points; -14.66% relative to the starting value).

**Interpretation.** All three measures finish below their initial values, supporting **overall narrowing over the full period**, but this is not equivalent to uninterrupted annual convergence. The indicators show different intermediate movements, particularly during 2019–2022: the CV and SD increased while the P90–P10 gap fell. This suggests that changes outside the central P10–P90 range and/or changes in the distribution's shape matter for the aggregate dispersion statistics. The figure alone cannot identify which regions drove those differences.

**Limitation.** Because these are annual cross-sectional measures of a relative GDP index, they should not be interpreted as differences in GDP expressed in currency or as proof of causal economic integration.

## Figure 2. Annual distribution boxplots

![Figure 2. Distribution of regional GDP per capita PPS index by year](figure_2_annual_boxplots.png)

**What the figure shows.** Each box represents the middle 50% of regional GDP index observations (25th to 75th percentiles); the horizontal line is the median. Whiskers follow the conventional 1.5×IQR rule, and observations beyond them are plotted individually. These dots are **statistical outliers, not necessarily data errors**.

**Results.** The interquartile range changed from **48.25** to **39.25 index points** (-9.00). The median changed from **88.00** to **87.00 index points**. The boxplots should be read alongside the percentile gap: a narrower central box does not imply that every part of the distribution narrowed.

**Interpretation.** The declining IQR supports a reduction in dispersion among the middle half of regions. The persistence of distant observations provides a reason to examine whether the overall CV is sensitive to unusually high GDP index values. The boxplots provide a distributional explanation, not evidence about which specific regions gained or lost ground.

**Limitation.** The region-level observations are equally weighted; the boxes do not represent the income distribution of individuals or population-weighted inequality.

## Figure 3. Sensitivity to high-index regions

![Figure 3. Coefficient of variation with and without the annual highest 1 percent of GDP index observations](figure_3_outlier_sensitivity.png)

**What the figure shows.** The main annual CV is compared with a recalculated CV after removing observations above the **year-specific 99th percentile**. This is a robustness test, not a preferred replacement for the full sample.

**Results.** The full-sample CV changed by **-1.82 percentage points** between 2016 and 2024. The trimmed CV changed by **-2.71 percentage points**. Both show a decline over the full period.

**Interpretation.** The overall narrowing conclusion does not depend solely on the most prosperous observations. However, the magnitude of convergence changes when the upper tail is excluded. Because the 99th-percentile cutoff is recalculated every year, the identities of the removed regions can change; therefore this is a **distributional sensitivity check**, not a fixed-region comparison.

**Limitation.** Trimming may remove genuine economic variation, not errors. Do not use the trimmed estimates as the headline series or infer that outlying regions caused the observed time pattern.

## Period comparisons

The three periods are descriptive windows, not causal regimes or formal structural-break tests. Adjacent windows share an endpoint year, so their changes should not be treated as independent observations.

- **2016–2019:** CV -1.21 percentage points; SD -1.32 index points; P90–P10 -3.00 index points. These are endpoint changes over 3 years, not annualised rates.
- **2019–2022:** CV +1.36 percentage points; SD +1.44 index points; P90–P10 -1.70 index points. These are endpoint changes over 3 years, not annualised rates.
- **2022–2024:** CV -1.97 percentage points; SD -1.99 index points; P90–P10 -8.30 index points. These are endpoint changes over 2 years, not annualised rates.

**Interpretation.** The 2019–2022 disagreement is especially important: the CV and SD imply widening while P90–P10 implies narrowing. This is not a computational contradiction; each statistic summarises a different aspect of dispersion. The analysis establishes **when and in what sense** the spread changed; it does not establish the economic drivers of those changes.

## Year-by-year changes

- **2017:** CV -0.27 percentage points; SD -0.31 index points; P90–P10 +1.40 index points. Indicators disagreed about the direction of change.
- **2018:** CV -0.24 percentage points; SD -0.27 index points; P90–P10 +0.60 index points. Indicators disagreed about the direction of change.
- **2019:** CV -0.70 percentage points; SD -0.74 index points; P90–P10 -5.00 index points. All three indicators narrowed.
- **2020:** CV +0.86 percentage points; SD +0.92 index points; P90–P10 +0.00 index points. Indicators disagreed about the direction of change.
- **2021:** CV +0.26 percentage points; SD +0.20 index points; P90–P10 -2.70 index points. Indicators disagreed about the direction of change.
- **2022:** CV +0.24 percentage points; SD +0.32 index points; P90–P10 +1.00 index points. All three indicators widened.
- **2023:** CV -1.90 percentage points; SD -1.86 index points; P90–P10 -4.60 index points. All three indicators narrowed.
- **2024:** CV -0.07 percentage points; SD -0.12 index points; P90–P10 -3.70 index points. All three indicators narrowed.

**Reading rule.** A negative CV change is measured in **percentage points**, not percent; SD and P90–P10 changes are in **GDP index points**. Years where all three indicators decreased: **2019, 2023, 2024**. Years with mixed directional signals: **2017, 2018, 2020, 2021**.

## Data quality, flags and comparability

The panel is **balanced** in its observed region-year structure, with **0 missing GDP index values** and no duplicate region-year combinations. The flag field contains **433 provisional (p)**, **38 break-marked (b)** and **33 estimated (e)** observations; **504 observations** are flagged in total. The files `gdp_flags_by_year.csv` and `unflagged_only_sensitivity.csv` permit transparent review. A flagged observation is not automatically invalid; removing all flagged observations also changes the annual sample and may bias comparisons.

**Critical unresolved check.** Stable region codes and a balanced panel do **not** prove harmonisation to a single NUTS classification. Before final submission, the team should verify the source vintage, whether historical figures were backcast to the same NUTS boundaries, and the meaning of the original flags against Eurostat metadata. Until that is documented, describe the panel as *balanced by observed codes*, not as definitively boundary-harmonised.

## Conclusion and hand-off to the other research questions

**RQ1A answer:** Dispersion in regional GDP per capita PPS index **narrowed between 2016 and 2024**, according to the CV, SD and P90–P10 gap, but **not year after year**. Changes differ across periods and across statistics, so the conclusion is conditional on the measure and time window. The decline also persists in an upper-tail-trimmed CV check, though its size changes.

**For RQ2:** These results document sigma convergence but do not test beta convergence; compare this descriptive time pattern with growth-on-initial-income findings. **For RQ3:** The distributional differences motivate investigation of regional groupings, without assigning mechanisms here. **For RQ4:** The periods where dispersion widens or measures disagree provide useful cases for reconciling sigma and beta convergence. No explanatory or causal claims are made in RQ1A.

## Reproducibility

The narrative and figures are generated from the same input panel by `rq1a_enhanced_with_explanations.py`. Re-running the script regenerates the numerical statements in this Markdown file and the supporting CSVs. It is suitable for inclusion in the team's Quarto project; verify figure paths relative to the final QMD file.
