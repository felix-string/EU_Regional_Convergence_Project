"""RQ3: which EU NUTS2 regions behave alike, and is it geography or industrial structure?

Reads  data/processed/eu_nuts2_balanced_panel.csv (and sector_shares.csv if it exists)
Writes outputs/rq3/
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parents[1]
PANEL = root / "data" / "processed" / "eu_nuts2_balanced_panel.csv"
SECTORS = root / "data" / "processed" / "sector_shares.csv"
OUT = root / "outputs" / "rq3"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
K = 3  # set to None to let the silhouette score pick k
K_RANGE = range(2, 9)

# 1. one row per region with a few numbers that describe how it behaves.
# gdp is an index (EU27 = 100) so "growth" is the change in relative position.
# no volatility feature, the index is rounded to whole numbers so it would just be noise
df = pd.read_csv(PANEL).sort_values(["region_code", "year"])
first, last = df["year"].min(), df["year"].max()
wide = lambda col: df.pivot(index="region_code", columns="year", values=col)
gdp, emp, une = wide("gdp_pps_index"), wide("employment_rate"), wide("unemployment_rate")

X = pd.DataFrame({
    "gdp_level_log": np.log(gdp).mean(axis=1),
    "gdp_rel_growth": (np.log(gdp[last]) - np.log(gdp[first])) / (last - first) * 100,
    "emp_level": emp.mean(axis=1),
    "emp_change": emp[last] - emp[first],
    "unemp_level": une.mean(axis=1),
    "unemp_change": une[last] - une[first],
})
meta = df.drop_duplicates("region_code").set_index("region_code")[["region_name", "country_code"]]
X.to_csv(OUT / "rq3_features_raw.csv")

# 2. standardise, otherwise the variables with big numbers dominate the distances
Z = pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=X.columns)

# 3. try different k and look at elbow + silhouette
rows = []
for k in K_RANGE:
    km = KMeans(k, n_init=50, random_state=SEED).fit(Z)
    rows.append((k, km.inertia_, silhouette_score(Z, km.labels_)))
sel = pd.DataFrame(rows, columns=["k", "inertia", "silhouette"])
sel.to_csv(OUT / "rq3_choose_k.csv", index=False)
print(sel.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(sel.k, sel.inertia, "o-")
ax[0].set_title("Elbow")
ax[1].plot(sel.k, sel.silhouette, "o-")
ax[1].set_title("Silhouette")
for a in ax:
    a.set_xlabel("number of clusters (k)")
plt.tight_layout()
plt.savefig(OUT / "rq3_fig1_choose_k.png", dpi=200)
plt.close()

k_final = K or int(sel.loc[sel.silhouette.idxmax(), "k"])
print(f"\nUsing k = {k_final}")

# 4. final clusters, then check them with a second method and with subsamples
km = KMeans(k_final, n_init=50, random_state=SEED).fit(Z)
hc = AgglomerativeClustering(k_final, linkage="ward").fit(Z)
print("k-means vs Ward (ARI):", round(adjusted_rand_score(km.labels_, hc.labels_), 2))

rng = np.random.default_rng(SEED)
stab = []
for _ in range(100):
    idx = rng.choice(len(Z), int(0.8 * len(Z)), replace=False)
    sub = KMeans(k_final, n_init=10, random_state=int(rng.integers(1e6))).fit(Z.iloc[idx])
    stab.append(adjusted_rand_score(km.labels_[idx], sub.labels_))
print("stability over 100 random 80%% subsamples (mean ARI): %.2f" % np.mean(stab))

# number the clusters from richest (1) to poorest
X["cluster"] = km.labels_
order = X.groupby("cluster")["gdp_level_log"].mean().sort_values(ascending=False).index
X["cluster"] = X["cluster"].map({old: new + 1 for new, old in enumerate(order)})

# 5. what does each cluster look like (real units)
prof = X.groupby("cluster").mean()
prof["n_regions"] = X.groupby("cluster").size()
prof["gdp_level_(index)"] = np.exp(prof["gdp_level_log"])
prof.round(2).to_csv(OUT / "rq3_cluster_profiles.csv")
print("\n", prof.round(2).to_string())

pca = PCA(2).fit(Z)
P = pca.transform(Z)
fig, ax = plt.subplots(figsize=(7, 5.5))
sc = ax.scatter(P[:, 0], P[:, 1], c=X["cluster"], cmap="tab10", s=25)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
ax.legend(*sc.legend_elements(), title="cluster")
ax.set_title("EU NUTS2 regions by cluster (PCA of the standardised features)")
plt.tight_layout()
plt.savefig(OUT / "rq3_fig2_clusters_pca.png", dpi=200)
plt.close()

# 6. geography: are the clusters just countries?
X = X.join(meta)
pd.crosstab(X["country_code"], X["cluster"]).to_csv(OUT / "rq3_cluster_by_country.csv")
ari = adjusted_rand_score(X["cluster"], X["country_code"])
nmi = normalized_mutual_info_score(X["cluster"], X["country_code"])
multi = X.groupby("country_code").filter(lambda g: len(g) > 1)  # leave out one-region countries
mixed = (pd.crosstab(multi["country_code"], multi["cluster"]) > 0).sum(axis=1)
print(f"\nGEOGRAPHY: cluster vs country ARI={ari:.2f}, NMI={nmi:.2f}")
print(f"countries with 2+ regions that end up in 2+ clusters: {(mixed > 1).sum()} of {len(mixed)}")

# how much of each variable is explained by country alone (eta squared)
eta = {}
for c in Z.columns:
    z = Z.loc[multi.index, c]
    ssw = z.groupby(multi["country_code"]).apply(lambda s: ((s - s.mean()) ** 2).sum()).sum()
    eta[c] = 1 - ssw / ((z - z.mean()) ** 2).sum()
eta = pd.Series(eta).round(2)
eta.to_csv(OUT / "rq3_country_eta_squared.csv", header=["share_explained_by_country"])
print("\nshare of variance explained by country (eta^2):\n", eta.to_string())

# 7. industrial structure: cluster on sector shares only and compare
if SECTORS.exists():
    S = pd.read_csv(SECTORS).set_index("region_code").reindex(X.index)
    S = S[["share_agri", "share_industry", "share_services"]].dropna()
    struct = KMeans(k_final, n_init=50, random_state=SEED).fit_predict(
        StandardScaler().fit_transform(S))
    common = X.loc[S.index, "cluster"]
    print("\nSTRUCTURE: performance clusters vs structure clusters ARI = %.2f"
          % adjusted_rand_score(common, struct))
    print("STRUCTURE: structure clusters vs country ARI = %.2f"
          % adjusted_rand_score(struct, X.loc[S.index, "country_code"]))
    mix = X.join(S).groupby("cluster")[["share_agri", "share_industry", "share_services"]].mean()
    mix.round(1).to_csv(OUT / "rq3_cluster_sector_mix.csv")
    print("\naverage sector mix per cluster (% of jobs):\n", mix.round(1).to_string())
else:
    print("\nno sector_shares.csv found, structure test skipped")

# 8. save the cluster of every region
X.to_csv(OUT / "rq3_region_clusters.csv")
print("\nsaved to", OUT)
