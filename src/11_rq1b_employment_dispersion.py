"""
RQ1B: Are employment gaps between EU regions narrowing?

Sigma convergence test for regional employment rates (total, age 20-64, %)
across EU NUTS 2 regions, 2016-2024.

Three samples are used so that changes in dispersion are not caused by
regions entering or leaving the data:

    master     - every region with an employment value in that year
                 (237 to 244 regions; the region count is reported per year)
    constant   - master-panel regions with employment data in all nine years
                 (236 regions)
    balanced   - the team's balanced panel (224 regions), used as the
                 sensitivity check required by the project plan

Outputs are written to outputs/rq1b/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

root = Path(__file__).resolve().parents[1]
processed = root / "data" / "processed"
outputs = root / "outputs" / "rq1b"
figures = outputs / "figures"

outputs.mkdir(parents=True, exist_ok=True)
figures.mkdir(parents=True, exist_ok=True)

VARIABLE = "employment_rate"
YEARS = list(range(2016, 2025))

# Colour slots and text colours (validated categorical palette)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

master = pd.read_csv(processed / "eu_nuts2_master_panel.csv")
balanced = pd.read_csv(processed / "eu_nuts2_balanced_panel.csv")

master = master[master["year"].isin(YEARS)]
balanced = balanced[balanced["year"].isin(YEARS)]

years_per_region = (
    master.dropna(subset=[VARIABLE])
    .groupby("region_code")["year"]
    .nunique()
)
complete_regions = years_per_region[years_per_region == len(YEARS)].index
constant = master[master["region_code"].isin(complete_regions)]

samples = {
    "master": master,
    "constant": constant,
    "balanced": balanced,
}


# ---------------------------------------------------------------------------
# 1. Coverage: how many regions have employment data each year
# ---------------------------------------------------------------------------

all_regions = master["region_code"].nunique()

coverage = (
    master.groupby("year")[VARIABLE]
    .agg(regions_with_data="count")
    .reset_index()
)
coverage["regions_in_master_panel"] = all_regions
coverage["regions_missing"] = (
    coverage["regions_in_master_panel"] - coverage["regions_with_data"]
)

missing_lists = (
    master[master[VARIABLE].isna()]
    .groupby("year")["region_code"]
    .apply(lambda codes: ", ".join(sorted(codes)))
)
coverage["missing_region_codes"] = (
    coverage["year"].map(missing_lists).fillna("")
)
coverage["regions_in_balanced_panel"] = (
    balanced.groupby("year")[VARIABLE].count().values
)

coverage.to_csv(outputs / "rq1b_employment_coverage.csv", index=False)


# ---------------------------------------------------------------------------
# 2. Annual dispersion measures for each sample
# ---------------------------------------------------------------------------

def dispersion_by_year(data, sample_name):
    rows = []

    for year, group in data.dropna(subset=[VARIABLE]).groupby("year"):
        values = group[VARIABLE]
        mean = values.mean()
        sd = values.std(ddof=1)
        p10 = values.quantile(0.10)
        p50 = values.quantile(0.50)
        p90 = values.quantile(0.90)

        rows.append({
            "sample": sample_name,
            "year": year,
            "regions": len(values),
            "mean": mean,
            "median": p50,
            "std_dev": sd,
            "coef_variation_pct": sd / mean * 100,
            "p10": p10,
            "p90": p90,
            "p90_minus_p10": p90 - p10,
            "min": values.min(),
            "max": values.max(),
            "max_minus_min": values.max() - values.min(),
        })

    return pd.DataFrame(rows)


dispersion = pd.concat(
    [dispersion_by_year(data, name) for name, data in samples.items()],
    ignore_index=True,
)

# Year-on-year change in the CV, to explain individual years
dispersion["cv_change_from_previous_year"] = (
    dispersion.groupby("sample")["coef_variation_pct"].diff()
)
dispersion["direction"] = np.select(
    [
        dispersion["cv_change_from_previous_year"] < 0,
        dispersion["cv_change_from_previous_year"] > 0,
    ],
    ["narrowing", "widening"],
    default="",
)

dispersion.round(3).to_csv(
    outputs / "rq1b_employment_dispersion_by_year.csv", index=False
)


# ---------------------------------------------------------------------------
# 3. Trend test: is the fall in dispersion statistically meaningful?
# ---------------------------------------------------------------------------

trend_rows = []

for sample_name, table in dispersion.groupby("sample"):
    for measure in ["coef_variation_pct", "std_dev", "p90_minus_p10"]:
        fit = stats.linregress(table["year"], table[measure])
        t_critical = stats.t.ppf(0.975, len(table) - 2)

        first = table.loc[table["year"] == YEARS[0], measure].iloc[0]
        last = table.loc[table["year"] == YEARS[-1], measure].iloc[0]

        tau, tau_p = stats.kendalltau(table["year"], table[measure])

        trend_rows.append({
            "sample": sample_name,
            "measure": measure,
            "value_2016": first,
            "value_2024": last,
            "change_2016_2024": last - first,
            "pct_change_2016_2024": (last - first) / first * 100,
            "ols_slope_per_year": fit.slope,
            "slope_ci95_low": fit.slope - t_critical * fit.stderr,
            "slope_ci95_high": fit.slope + t_critical * fit.stderr,
            "ols_p_value": fit.pvalue,
            "r_squared": fit.rvalue ** 2,
            "kendall_tau": tau,
            "kendall_p_value": tau_p,
            "years_narrowing": int(
                (table["cv_change_from_previous_year"] < 0).sum()
            ) if measure == "coef_variation_pct" else np.nan,
        })

trend = pd.DataFrame(trend_rows)
trend.round(4).to_csv(outputs / "rq1b_employment_trend_tests.csv", index=False)

# Break-in-series check. In 2021 Eurostat introduced the new Labour Force
# Survey framework (IESS regulation), so almost every region carries a "b"
# flag that year. If the narrowing were only a methodology artefact it would
# appear as a single jump at 2021. Comparing change within each period shows
# whether dispersion fell under both methodologies.
break_flags = (
    master.assign(
        has_break=master[f"{VARIABLE}_flag"]
        .fillna("").astype(str).str.contains("b")
    )
    .groupby("year")["has_break"].sum()
)

periods = {
    "2016-2020 (old LFS method)": (2016, 2020),
    "2020-2021 (break year)": (2020, 2021),
    "2021-2024 (new LFS method)": (2021, 2024),
}

period_rows = []
for sample_name, table in dispersion.groupby("sample"):
    cv = table.set_index("year")["coef_variation_pct"]
    for label, (start, end) in periods.items():
        period_rows.append({
            "sample": sample_name,
            "period": label,
            "cv_start": cv[start],
            "cv_end": cv[end],
            "cv_change": cv[end] - cv[start],
            "cv_change_per_year": (cv[end] - cv[start]) / (end - start),
        })

period_check = pd.DataFrame(period_rows)
period_check.round(3).to_csv(
    outputs / "rq1b_break_in_series_check.csv", index=False
)
break_flags.rename("regions_flagged_b").to_csv(
    outputs / "rq1b_break_flags_by_year.csv"
)


# ---------------------------------------------------------------------------
# 4. Between-country vs within-country dispersion (balanced panel)
#    Total variance across regions = variance of country means (between)
#    + average variance of regions around their own country mean (within).
# ---------------------------------------------------------------------------

decomposition_rows = []

for year, group in balanced.groupby("year"):
    values = group[VARIABLE]
    country_mean = group.groupby("country_code")[VARIABLE].transform("mean")

    total = values.var(ddof=0)
    between = country_mean.var(ddof=0)
    within = ((values - country_mean) ** 2).mean()

    decomposition_rows.append({
        "year": year,
        "total_variance": total,
        "between_country_variance": between,
        "within_country_variance": within,
        "between_country_share_pct": between / total * 100,
        "within_country_share_pct": within / total * 100,
    })

decomposition = pd.DataFrame(decomposition_rows)
decomposition.round(3).to_csv(
    outputs / "rq1b_between_within_country.csv", index=False
)


# ---------------------------------------------------------------------------
# 5. Which regions moved: change 2016-2024 (balanced panel)
# ---------------------------------------------------------------------------

change = (
    balanced[balanced["year"].isin([YEARS[0], YEARS[-1]])]
    .pivot_table(
        index=["region_code", "region_name", "country_code"],
        columns="year",
        values=VARIABLE,
    )
    .reset_index()
)
change.columns.name = None
change = change.rename(columns={
    YEARS[0]: "employment_2016",
    YEARS[-1]: "employment_2024",
})
change["change_pp"] = change["employment_2024"] - change["employment_2016"]
change["quintile_2016"] = pd.qcut(
    change["employment_2016"], 5,
    labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
)
change = change.sort_values("change_pp", ascending=False)
change.round(2).to_csv(outputs / "rq1b_region_change_2016_2024.csv", index=False)

quintile_change = (
    change.groupby("quintile_2016", observed=True)
    .agg(
        regions=("region_code", "count"),
        mean_2016=("employment_2016", "mean"),
        mean_2024=("employment_2024", "mean"),
        mean_change_pp=("change_pp", "mean"),
    )
    .reset_index()
)
quintile_change.round(2).to_csv(
    outputs / "rq1b_change_by_starting_quintile.csv", index=False
)

catch_up = stats.linregress(change["employment_2016"], change["change_pp"])


# ---------------------------------------------------------------------------
# 6. Charts
# ---------------------------------------------------------------------------

def finish(fig, name):
    fig.tight_layout()
    fig.savefig(figures / name, dpi=200)
    plt.close(fig)


def label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(
        text, xy=(x, y), xytext=(6, dy), textcoords="offset points",
        va="center", fontsize=9, color=TEXT,
    )
    ax.plot(x, y, "o", color=color, markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5)


# Figure 1: CV over time, three samples
fig, ax = plt.subplots(figsize=(8, 4.6))
series = [
    ("balanced", "Balanced panel (224)", BLUE),
    ("constant", "Constant sample (236)", AQUA),
    ("master", "Master panel (237-244)", ORANGE),
]
offsets = {"balanced": 4, "constant": -8, "master": 10}

for key, label, color in series:
    table = dispersion[dispersion["sample"] == key]
    ax.plot(table["year"], table["coef_variation_pct"],
            color=color, linewidth=2, label=label)
    last = table.iloc[-1]
    label_end(ax, last["year"], last["coef_variation_pct"],
              label.split(" (")[0], color, offsets[key])

ax.axvspan(2019.5, 2020.5, color=GRID, alpha=0.6, linewidth=0)
ax.text(2020, ax.get_ylim()[1], "COVID-19", ha="center", va="top",
        fontsize=8.5, color=TEXT_SECONDARY)
ax.set_xticks(YEARS)
ax.set_xlim(2015.7, 2025.6)
ax.set_ylabel("Coefficient of variation (%)")
ax.set_title("Employment rates across EU regions became more similar",
             loc="left", fontsize=12, color=TEXT)
ax.legend(loc="lower left", frameon=False, fontsize=9)
finish(fig, "fig1_employment_cv_by_year.png")


# Figure 2: P10, median, P90 (balanced panel)
table = dispersion[dispersion["sample"] == "balanced"]
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.fill_between(table["year"], table["p10"], table["p90"],
                color=BLUE, alpha=0.12, linewidth=0)
ax.plot(table["year"], table["p90"], color=BLUE, linewidth=2)
ax.plot(table["year"], table["median"], color=BLUE, linewidth=1.2,
        linestyle="--")
ax.plot(table["year"], table["p10"], color=BLUE, linewidth=2)

first, last = table.iloc[0], table.iloc[-1]
label_end(ax, last["year"], last["p90"],
          f"90th percentile (+{last['p90'] - first['p90']:.1f} pp)", BLUE)
label_end(ax, last["year"], last["median"], "Median", BLUE)
label_end(ax, last["year"], last["p10"],
          f"10th percentile (+{last['p10'] - first['p10']:.1f} pp)", BLUE)

for row, offset, align in [(first, 8, "left"), (last, -8, "right")]:
    ax.annotate(
        f"gap {row['p90_minus_p10']:.1f} pp",
        xy=(row["year"], (row["p10"] + row["median"]) / 2),
        xytext=(offset, 0), textcoords="offset points",
        ha=align, va="center", fontsize=9, color=TEXT_SECONDARY,
    )

ax.set_xticks(YEARS)
ax.set_xlim(2015.6, 2027.2)
ax.set_ylabel("Employment rate, age 20-64 (%)")
ax.set_title("The weakest regions gained more than the strongest",
             loc="left", fontsize=12, color=TEXT)
finish(fig, "fig2_employment_p10_p90_band.png")


# Figure 3: between- and within-country variance (stacked bars)
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.set_axisbelow(True)
ax.bar(decomposition["year"], decomposition["between_country_variance"],
       width=0.7, color=BLUE, edgecolor=SURFACE, linewidth=2,
       label="Between countries")
ax.bar(decomposition["year"], decomposition["within_country_variance"],
       bottom=decomposition["between_country_variance"],
       width=0.7, color=ORANGE, edgecolor=SURFACE, linewidth=2,
       label="Within countries")

for _, row in decomposition.iterrows():
    ax.text(row["year"], row["between_country_variance"] / 2,
            f"{row['between_country_share_pct']:.0f}%",
            ha="center", va="center", fontsize=8.5, color="white")

ax.set_xticks(YEARS)
ax.set_ylabel("Variance of regional employment rates (pp²)")
ax.set_title("Most of the narrowing came from gaps between countries",
             loc="left", fontsize=12, color=TEXT)
ax.legend(loc="upper right", frameon=False, fontsize=9)
finish(fig, "fig3_between_within_country_variance.png")


# Figure 4: starting level vs change (balanced panel)
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.scatter(change["employment_2016"], change["change_pp"],
           s=22, color=BLUE, alpha=0.75,
           edgecolor=SURFACE, linewidth=0.8)

x_line = np.linspace(change["employment_2016"].min(),
                     change["employment_2016"].max(), 50)
ax.plot(x_line, catch_up.intercept + catch_up.slope * x_line,
        color=TEXT_SECONDARY, linewidth=1.5)

to_label = pd.concat([change.head(1), change.tail(2)])
for _, row in to_label.iterrows():
    ax.annotate(
        f"{row['region_name']} ({row['region_code']})",
        xy=(row["employment_2016"], row["change_pp"]),
        xytext=(6, 0), textcoords="offset points",
        fontsize=8, color=TEXT, va="center",
    )

ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
ax.set_xlabel("Employment rate in 2016 (%)")
ax.set_ylabel("Change 2016-2024 (percentage points)")
ax.set_title(
    f"Regions that started lower gained more "
    f"(slope {catch_up.slope:.2f}, R² {catch_up.rvalue ** 2:.2f})",
    loc="left", fontsize=12, color=TEXT,
)
finish(fig, "fig4_start_level_vs_change.png")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

print("RQ1B employment dispersion completed.\n")
print("Regions with employment data per year (master panel):")
print(coverage[["year", "regions_with_data", "regions_missing"]]
      .to_string(index=False))

print("\nCoefficient of variation (%) by year:")
print(
    dispersion.pivot(index="year", columns="sample",
                     values="coef_variation_pct")
    .round(2).to_string()
)

print("\nTrend in the CV:")
print(
    trend[trend["measure"] == "coef_variation_pct"][[
        "sample", "value_2016", "value_2024", "pct_change_2016_2024",
        "ols_slope_per_year", "ols_p_value", "kendall_p_value",
    ]].round(4).to_string(index=False)
)

print("\nCV change within each LFS methodology period:")
print(period_check.round(3).to_string(index=False))

print("\nBetween-country share of variance (balanced panel):")
print(decomposition[["year", "between_country_share_pct"]]
      .round(1).to_string(index=False))

print(f"\nCatch-up slope (change on 2016 level): {catch_up.slope:.3f}, "
      f"p = {catch_up.pvalue:.2g}, R² = {catch_up.rvalue ** 2:.2f}")
print(f"\nFiles written to {outputs}")