"""
=============================================================================
  THESIS CLUSTERING COMPARISON HARNESS — Final
  Bootstrapped DPMM (RU-DPMM) vs. Baseline / Competing Models
  Metrics : Silhouette · DB · CH · BIC (GMM) · ELBO (DPMM) · Demand CV
  Plots   : Bar Charts · Fuzzy Grid · Stability Lines · Capacity Curve
=============================================================================

=============================================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────
from sklearn.metrics import (silhouette_score,
                             davies_bouldin_score,
                             calinski_harabasz_score,
                             adjusted_rand_score)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
from scipy.spatial.distance import cdist
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")


try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("\n[WARNING] hdbscan not installed — skipped.\n"
                  "Fix: pip install hdbscan\n")

warnings.filterwarnings("ignore")


# =============================================================================
# ██████  CONFIG
# =============================================================================

DATASET_PATH = "online_retail.csv"
CLUSTERED_OUT_PATH = "online_retail_clustered.csv"

DEMAND_COL = None            # Override for demand column name.
# None → auto-detect from keywords (demand/sales/qty/volume).

# Column containing cluster IDs in CLUSTERED_OUT_PATH.
CLUSTER_COL = "Cluster_Label"
# Confirmed from uploaded CSV: column is "Cluster_Label".

UNCERTAINTY_COL = "Uncertainty_Score"
# Pre-computed per-point uncertainty column in your
# DPMM output CSV (values in [0, 1]; 0 = certain,
# 1 = fully uncertain).  This is read directly —
# no proxy calculation needed.
# Set to None to fall back to 1 − max(predict_proba).

MAX_K = 10
HDBSCAN_MIN_SIZE = 15
STABILITY_RUNS = 10
N_BOOTSTRAP = 500    # [NEW-2] Bootstrap iterations for O'Hagan GMM
USER_RUNTIME = 0.598   # Replace with your pipeline's measured runtime (s)

# [FIX-9] USER_BIC removed — BIC is invalid for a Bayesian DPMM model.
#         The ELBO from the base DPMM fit is used instead (auto-computed below).

# =============================================================================
# 1. Data helpers
# =============================================================================


def smart_load(path: str) -> pd.DataFrame:
    # 1) robust read: any encoding (DataCo is Latin-1). Reuses robust_load if present.
    try:
        from robust_load import robust_load
        df = robust_load(path)
    except Exception:
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            df = pd.read_csv(path, encoding="latin-1")

    # 2) drop the same non-feature columns your model drops (incl. geography),
    #    so the baselines cluster on the SAME economic features -> fair comparison
    drop_hints = ['id', 'index', 'idx', 'uuid', 'serial', 'customer_no',
                  'latitude', 'longitude', 'zipcode', 'zip', 'risk', 'status', 'Channel', 'Region']
    drop_cols = [c for c in df.columns if any(
        h in c.lower() for h in drop_hints)]
    df = df.drop(columns=drop_cols, errors="ignore")

    # 3) scoped NaN handling: kill empty/near-empty columns BEFORE dropping rows,
    #    or DataCo's blank columns (Product Description, Order Zipcode) zero the frame
    df = df.dropna(axis=1, how="all")
    df = df.select_dtypes(include=[np.number])
    df = df.loc[:, df.isna().mean() < 0.5]
    df = df.dropna().reset_index(drop=True)

    print(f"[Data]  {len(df):,} rows × {len(df.columns)} numeric features")
    return df


def scale(df: pd.DataFrame):
    # Use the SAME transform selector as the model so baselines and RU-DPMM
    # share a feature space. On low-skew data this picks StandardScaler (no
    # change); on skewed data it picks Yeo-Johnson, matching the model.
    try:
        from select_transform import select_scaler
        sc, _which, _ = select_scaler(df.values, verbose=False)
    except Exception:
        sc = StandardScaler().fit(df.values)
    return sc.transform(df.values), sc


def infer_k(X: np.ndarray, max_k: int = 10) -> int:
    m = BayesianGaussianMixture(
        n_components=max_k,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1e-2,
        max_iter=300, random_state=0
    ).fit(X)
    # [FIX-A] Mass-based active-cluster count.  The old `weights_ > 1e-3`
    # threshold counted the stick-breaking posterior's tail of tiny "junk"
    # components (which soak up outliers/noise), inflating k -- e.g. k=7 on
    # data with 2 real clusters.  Count the smallest set of components that
    # together hold 95% of the total weight instead.
    w = np.sort(m.weights_)[::-1]
    k = max(2, int(np.searchsorted(np.cumsum(w), 0.95) + 1))
    print(f"[Auto-k] DPMM suggests {k} active clusters (upper bound: {max_k})")
    return k


def detect_demand_col(df: pd.DataFrame) -> str | None:
    """[NEW-3] Auto-detect demand column; DEMAND_COL config overrides."""
    if DEMAND_COL and DEMAND_COL in df.columns:
        return DEMAND_COL
    keywords = ['demand', 'volume', 'sales', 'qty', 'quantity']
    col = next((c for c in df.columns
                if any(kw in c.lower() for kw in keywords)), None)
    if col:
        print(f"[Demand] Auto-detected demand column: '{col}'")
    else:
        print("[Demand] No demand column found — Mean Demand CV will be NaN.")
    return col


# =============================================================================
# 2. BootstrapGMM  [NEW-1] — O'Hagan et al. (2019) bootstrapped GMM competitor
# =============================================================================

class BootstrapGMM:
    """
    Python reimplementation of O'Hagan et al. (2019) MclustBootstrap.
    This is the most direct published competitor to RU-DPMM because it also
    uses bootstrap resampling on a mixture model.

    Paper : https://doi.org/10.1007/s00180-019-00897-9
    R ref : mclust::MclustBootstrap()

    CC.py dropped this entirely.  Its absence weakens the thesis comparison
    because the key claim of RU-DPMM is that bootstrapped DPMM outperforms
    bootstrapped GMM.  Without O'Hagan, that claim is unsupported.
    """

    def __init__(self, n_components=4, n_bootstrap=500, random_state=42):
        self.n_components = n_components
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.base_model_ = None
        self.boot_means_ = None
        self.mean_std_ = None

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        self.base_model_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            n_init=5, random_state=self.random_state
        ).fit(X)
        base_means = self.base_model_.means_

        boot_means = []
        for b in range(self.n_bootstrap):
            X_b = resample(X, replace=True, n_samples=len(X),
                           random_state=int(rng.integers(1_000_000)))
            try:
                gm_b = GaussianMixture(
                    n_components=self.n_components,
                    covariance_type="full",
                    n_init=2, random_state=b
                ).fit(X_b)
                matched = gm_b.means_[
                    np.argmin(cdist(base_means, gm_b.means_), axis=1)]
                boot_means.append(matched)
            except Exception:
                continue

        self.boot_means_ = np.array(boot_means)
        self.mean_std_ = self.boot_means_.std(axis=0)
        return self

    def predict(self, X):
        return self.base_model_.predict(X)

    def bic(self, X):
        return self.base_model_.bic(X)

    def uncertainty_per_point(self, X):
        """Per-point uncertainty: 1 − max posterior membership probability."""
        return 1.0 - self.base_model_.predict_proba(X).max(axis=1)

    def bootstrap_uncertainty_summary(self):
        return np.linalg.norm(self.mean_std_, axis=1)


# =============================================================================
# 3. Model runners — all return (labels, elapsed, bic, uncert)
#    Exception: run_vanilla_dpmm returns (labels, elapsed, nan, uncert, elbo, model)
# =============================================================================

def run_kmeans(X, k, seed=42):
    t0 = time.time()
    m = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    # K-Means has no probabilistic uncertainty — zeros mean "fully assigned"
    # log_lik = NaN: KMeans minimises inertia, not a likelihood function
    return m.labels_, time.time() - t0, np.nan, np.zeros(len(X)), np.nan


def run_gmm(X, k, seed=42):
    t0 = time.time()
    m = GaussianMixture(n_components=k, covariance_type="full",
                        n_init=3, random_state=seed).fit(X)
    uncert = 1.0 - m.predict_proba(X).max(axis=1)
    # m.score(X) = mean per-sample log-likelihood (nats); directly comparable
    # to DPMM's score(X) and derivable from BIC: log_L=(k·log(n)−BIC)/2
    return m.predict(X), time.time() - t0, m.bic(X), uncert, m.score(X)


def run_vanilla_dpmm(X, max_k, seed=42):
    """
    [FIX-1] BIC approximation removed.  Returns (labels, elapsed, nan,
    uncert, elbo, model) — 6 values.  Callers that don't need elbo/model
    use _ placeholders.  Stability test constructs DPMM inline to avoid
    cascading the 6-tuple into every loop iteration.
    """
    t0 = time.time()
    m = BayesianGaussianMixture(
        n_components=max_k,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1e-2,
        covariance_type="full",
        max_iter=500, random_state=seed
    ).fit(X)
    uncert = 1.0 - m.predict_proba(X).max(axis=1)
    print(f"  [DPMM]  ELBO (lower_bound_, ↑ better): {m.lower_bound_:.4f}")
    # m.score(X) = mean per-sample log-likelihood — same unit as GMM score,
    # allowing direct cross-model comparison in the Log-Lik/sample column.
    # ELBO ≤ log p(X), so score(X) ≥ lower_bound_ (ELBO is a lower bound).
    return m.predict(X), time.time() - t0, np.nan, uncert, m.lower_bound_, m, m.score(X)


def run_bootstrap_gmm(X, k, n_bootstrap):
    """[NEW-1] O'Hagan et al. Bootstrap GMM — restored from v4."""
    t0 = time.time()
    model = BootstrapGMM(n_components=k, n_bootstrap=n_bootstrap).fit(X)
    labels = model.predict(X)
    elapsed = time.time() - t0
    uncert = model.uncertainty_per_point(X)
    boot_sig = model.bootstrap_uncertainty_summary()
    print(f"  [Bootstrap GMM] Per-cluster uncertainty (L2 of bootstrap σ):")
    for i, u in enumerate(boot_sig):
        print(f"    Cluster {i}: {u:.5f}")
    # log_lik from base GMM — same scale as run_gmm's score, comparable
    return labels, elapsed, model.bic(X), uncert, model.base_model_.score(X)


def run_hdbscan(X, min_cluster_size):
    if not HDBSCAN_AVAILABLE:
        return np.full(len(X), -1), 0.0, np.nan, np.ones(len(X)), np.nan
    t0 = time.time()
    m = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                        prediction_data=True).fit(X)
    # [KEPT] m.probabilities_ is HDBSCAN's native membership confidence —
    # more principled than predict_proba for density-based methods.
    uncert = 1.0 - m.probabilities_
    # HDBSCAN has no parametric likelihood → log_lik = NaN
    return m.labels_, time.time() - t0, np.nan, uncert, np.nan


def run_agglomerative(X, k):
    t0 = time.time()
    labels = AgglomerativeClustering(
        n_clusters=k, linkage="ward").fit_predict(X)
    # Agglomerative is deterministic and non-probabilistic → log_lik = NaN
    return labels, time.time() - t0, np.nan, np.zeros(len(X)), np.nan


# =============================================================================
# 4. Unified evaluator
# =============================================================================

def evaluate(X, labels, model_name, elapsed, bic, elbo=np.nan,
             log_lik=np.nan, demand_array=None):
    """
    Unified evaluator.  log_lik = mean per-sample log-likelihood from
    model.score(X), defined for GMM, Bootstrap GMM, and DPMM.  This is
    the only column where BIC-family and ELBO-family models can be compared
    on the same numerical scale:
      • GMM/Bootstrap GMM:  frequentist MLE log-likelihood
      • Vanilla DPMM:       approximate posterior-predictive log-likelihood
      • K-Means/Agglomerative/HDBSCAN: NaN (no likelihood defined)

    Relationship to other columns:
      BIC  = −2·log_L + k·log(n)  →  log_L = (k·log(n) − BIC) / 2
      ELBO ≤ log_L                 →  log_lik ≥ ELBO (ELBO is lower bound)
    """
    mask = labels >= 0
    n_clust = len(set(labels[mask]))

    if n_clust < 2 or mask.sum() < 10:
        return {
            "Model": model_name, "Clusters": n_clust,
            "Silhouette ↑": np.nan, "Davies-Bouldin ↓": np.nan,
            "Calinski-Harabasz ↑": np.nan,
            "BIC ↓": np.nan, "ELBO ↑": np.nan,
            "Log-Lik/sample ↑": np.nan,
            "Mean Demand CV ↓": np.nan,
            "Runtime (s)": round(elapsed, 3),
            "Note": "< 2 clusters or < 10 valid pts"
        }

    sil = silhouette_score(X[mask], labels[mask])
    db = davies_bouldin_score(X[mask], labels[mask])
    ch = calinski_harabasz_score(X[mask], labels[mask])

    mean_cv = np.nan
    if demand_array is not None:
        tmp = pd.DataFrame(
            {"demand": demand_array[mask], "label": labels[mask]})
        cv = (tmp.groupby("label")["demand"].std() /
              (tmp.groupby("label")["demand"].mean() + 1e-10))
        mean_cv = cv.mean()

    return {
        "Model":                  model_name,
        "Clusters":               n_clust,
        "Silhouette ↑":           round(sil, 4),
        "Davies-Bouldin ↓":       round(db,  4),
        "Calinski-Harabasz ↑":    round(ch,  2),
        "BIC ↓":                  round(bic,     1) if not np.isnan(bic) else np.nan,
        "ELBO ↑":                 round(elbo,    4) if not np.isnan(elbo) else np.nan,
        "Log-Lik/sample ↑":       round(log_lik, 4) if not np.isnan(log_lik) else np.nan,
        "Mean Demand CV ↓":       round(mean_cv, 4) if not np.isnan(mean_cv) else np.nan,
        "Runtime (s)":            round(elapsed, 3),
    }


# =============================================================================
# 5. Visualisation suite
# =============================================================================

def plot_bar_charts(results_df, path="metrics_barcharts.png"):
    print(f"\n[Plot] Bar charts → {path}")
    metrics = ["Clusters", "Silhouette ↑", "Davies-Bouldin ↓",
               "Calinski-Harabasz ↑", "BIC ↓", "ELBO ↑",
               "Log-Lik/sample ↑", "Mean Demand CV ↓", "Runtime (s)"]
    # 9 metrics fills the 3×3 grid exactly — no hidden panels needed

    plot_df = results_df.copy()
    for m in ["BIC ↓", "ELBO ↑", "Log-Lik/sample ↑"]:
        plot_df[m] = pd.to_numeric(plot_df[m], errors="coerce")

    fig, axes = plt.subplots(3, 3, figsize=(17, 14))
    axes = axes.flatten()
    names = plot_df["Model"].apply(lambda x: x.split(" (")[0]).values
    n_m = len(names)

    # [FIX-10] .colors removed; use indexing instead
    # [FIX-B] cm.get_cmap is deprecated and removed in matplotlib 3.11.
    cmap_obj = matplotlib.colormaps["tab10"].resampled(n_m)
    colors = [cmap_obj(i) for i in range(n_m)]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        vals = plot_df[metric].values
        bars = ax.bar(names, vals, color=colors,
                      edgecolor="black", linewidth=1.3)
        ax.set_title(metric, fontweight="bold", fontsize=11)
        ax.set_xticks(range(n_m))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.margins(y=0.3)

        for bar in bars:
            yval = bar.get_height()
            if not (np.isnan(yval) or yval == 0):
                y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
                offset = 0.02 * y_span
                va = "bottom" if yval >= 0 else "top"
                y_pos = yval + offset if yval >= 0 else yval - offset
                label_t = f"{yval:.2f}" if abs(yval) < 1000 else f"{yval:.0f}"
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                        label_t, ha="center", va=va, fontsize=8, fontweight="bold")

    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")


def plot_comparison_fuzzy(X_scaled, names, labels_list, uncert_list,
                          path="fuzzy_comparison.png"):
    """
    [FIX-8] PC axis labels are now factually correct variance percentages.
    CC.py hard-coded semantic labels ("Operational Scale Factor",
    "Demand Volatility") which PCA does not guarantee.
    """
    print(f"[Plot] Fuzzy grid → {path}")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    vx = pca.explained_variance_ratio_[0] * 100
    vy = pca.explained_variance_ratio_[1] * 100
    tot = vx + vy

    cols = 3
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows))
    axes = axes.flatten()

    for idx, (name, labels, uncerts) in enumerate(zip(names, labels_list, uncert_list)):
        ax = axes[idx]
        is_ru = "RU-DPMM" in name
        # [FIX-B] cm.get_cmap removed in matplotlib 3.11; use colormaps registry.
        cmap_p = matplotlib.colormaps["tab10"].resampled(
            max(len(set(labels)), 2))

        if is_ru:
            # ── RU-DPMM: smooth gradual opacity from pipeline uncertainty ─
            # alpha = 1 − Uncertainty_Score so:
            #   score ≈ 0.04  →  alpha ≈ 0.96  (solid, very certain)
            #   score ≈ 0.38  →  alpha ≈ 0.62  (medium, cluster edge)
            #   score ≈ 0.99  →  alpha ≈ 0.05  (ghost, deep boundary)
            # Minimum alpha of 0.05 keeps even the most uncertain dots
            # faintly visible — they show WHERE uncertainty is concentrated.
            alphas = np.clip(1.0 - uncerts, 0.05, 1.0)
            for i, cid in enumerate(sorted(set(labels))):
                if cid == -1:
                    continue
                mask = labels == cid
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                           c=[cmap_p(i)], s=40, alpha=alphas[mask],
                           edgecolors="none")

            # Three-tier legend communicating the opacity scale
            from matplotlib.lines import Line2D
            legend_elems = [
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#333", markersize=9,
                       alpha=0.95, label="Low  (score ≈ 0.05)"),
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#333", markersize=9,
                       alpha=0.55, label="Med  (score ≈ 0.45)"),
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#333", markersize=9,
                       alpha=0.08, label="High (score ≈ 0.95)"),
            ]
            ax.legend(handles=legend_elems, fontsize=7, loc="lower right",
                      framealpha=0.85, handlelength=1.2,
                      title="Uncertainty_Score", title_fontsize=7)
            ax.set_title("RU-DPMM (Uncertainty Sets)\n"
                         r"Opacity $\propto$ 1 $-$ Uncertainty_Score",
                         fontsize=9, fontweight="bold")

        else:
            # ── All other models: continuous opacity from their own uncerts ─
            alphas = np.clip(1.0 - uncerts, 0.02, 1.0)
            for i, cid in enumerate(sorted(set(labels))):
                if cid == -1:
                    continue
                mask = labels == cid
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                           c=[cmap_p(i)], s=25, alpha=alphas[mask],
                           edgecolors="none")
            ax.set_title(name.split(" (")[0], fontsize=11, fontweight="bold")

        ax.set_xlabel(f"PC1 ({vx:.1f}% variance)", fontsize=9)
        ax.set_ylabel(f"PC2 ({vy:.1f}% variance)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(len(names), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Uncertainty Set Mapping — All Models\n"
        f"Opacity ∝ certainty  ·  PCA total variance: {tot:.1f}%",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")


def run_stability_test(X, k, max_k, user_labels=None, dpmm_model=None,
                       ru_model=None, ru_X=None,
                       stability_path="stability_line_chart.png"):
    """
    [FIX-7] Monte Carlo label-flip simulation removed.
    [FIX-11] HDBSCAN added to stability test.

    User model stability now uses the fitted DPMM model's predict() on each
    80% subsample, then compares those predictions to the user's full-data
    labels for those same rows (DPMM proxy).  Rationale: RU-DPMM uses DPMM
    as its core estimator, so DPMM subsample prediction stability is a valid
    lower-bound proxy.  Clearly labelled as "(DPMM proxy)" on the plot.
    """
    print(f"\n[Stability] 80% subsample ARI test ({STABILITY_RUNS} iters) …")

    # Base labels on full data — [FIX-2] inline DPMM construction
    base_km = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    base_gmm = GaussianMixture(n_components=k, covariance_type="full",
                               n_init=3, random_state=42).fit(X).predict(X)
    base_dpmm = BayesianGaussianMixture(
        n_components=max_k, weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1e-2, max_iter=300, random_state=42
    ).fit(X).predict(X)
    # [FIX-C] Bootstrap GMM must be compared against its OWN full-data base,
    # not the plain-GMM base, so its stability line is apples-to-apples with
    # every other model (each compares base-vs-subsample of the same method).
    base_bgmm = BootstrapGMM(n_components=k, n_bootstrap=30,
                             random_state=42).fit(X).predict(X)
    base_agg = AgglomerativeClustering(
        n_clusters=k, linkage="ward").fit_predict(X)
    base_hdb = (hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_SIZE).fit_predict(X)
                if HDBSCAN_AVAILABLE else np.zeros(len(X), dtype=int))

    # [FIX-11] HDBSCAN added
    stability = {"K-Means": [], "GMM (EM)": [], "Bootstrap GMM": [],
                 "Vanilla DPMM": [], "Agglomerative": [], "HDBSCAN": []}
    # [REMOVED] RU-DPMM "(DPMM proxy)" series dropped -- not a real
    # resample-stability measure (see note below).

    rng = np.random.default_rng(42)
    n = len(X)
    n_ss = int(0.8 * n)

    for i in range(STABILITY_RUNS):
        seed_i = 42 + i
        idx = rng.choice(n, size=n_ss, replace=False)
        X_sub = X[idx]

        sub_km = KMeans(n_clusters=k, n_init=5,
                        random_state=seed_i).fit_predict(X_sub)
        sub_gmm = GaussianMixture(n_components=k, covariance_type="full",
                                  n_init=2, random_state=seed_i).fit(X_sub).predict(X_sub)
        sub_bgmm = BootstrapGMM(n_components=k, n_bootstrap=30,
                                random_state=seed_i).fit(X_sub).predict(X_sub)
        sub_dpmm = BayesianGaussianMixture(
            n_components=max_k, weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=1e-2, max_iter=200, random_state=seed_i
        ).fit(X_sub).predict(X_sub)
        sub_agg = AgglomerativeClustering(
            n_clusters=k, linkage="ward").fit_predict(X_sub)
        sub_hdb = (hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_SIZE).fit_predict(X_sub)
                   if HDBSCAN_AVAILABLE else np.zeros(n_ss, dtype=int))

        stability["K-Means"].append(adjusted_rand_score(base_km[idx],   sub_km))
        stability["GMM (EM)"].append(
            adjusted_rand_score(base_gmm[idx],  sub_gmm))
        stability["Bootstrap GMM"].append(
            adjusted_rand_score(base_bgmm[idx], sub_bgmm))   # [FIX-C]
        stability["Vanilla DPMM"].append(
            adjusted_rand_score(base_dpmm[idx], sub_dpmm))
        stability["Agglomerative"].append(
            adjusted_rand_score(base_agg[idx], sub_agg))

        # [FIX-11] HDBSCAN noise-aware ARI
        bh = base_hdb[idx]
        valid = (bh >= 0) & (sub_hdb >= 0)
        stability["HDBSCAN"].append(
            adjusted_rand_score(bh[valid], sub_hdb[valid]
                                ) if valid.sum() > 10 else np.nan
        )

        # [REMOVED] RU-DPMM "DPMM proxy" stability: compared the user's FIXED
        # labels to a FIXED Vanilla-DPMM prediction (~constant), which is label
        # agreement, not resample stability, and misleads on shared axes.

    # ── Genuine RU-DPMM assignment stability: refit the model's OWN DPMM core
    #    (cloned with its exact hyperparameters) on each 80% subsample of its
    #    clustered data, ARI vs its base labels. Measured exactly like every
    #    other line. ARI is permutation-invariant, so DPMM label-switching is
    #    irrelevant -- the line jitters with genuine partition instability.
    if ru_model is not None and ru_X is not None and len(ru_X) > 25:
        from sklearn.base import clone
        base_ru = ru_model.predict(ru_X)
        m = len(ru_X)
        m_ss = int(0.8 * m)
        rng_u = np.random.default_rng(42)
        ru_scores = []
        for _ in range(STABILITY_RUNS):
            jdx = rng_u.choice(m, size=m_ss, replace=False)
            try:
                est = clone(ru_model)
                try:
                    est.set_params(max_iter=300)      # cap iters for the refit
                except Exception:
                    pass
                sub = est.fit(ru_X[jdx]).predict(ru_X[jdx])
                ru_scores.append(adjusted_rand_score(base_ru[jdx], sub))
            except Exception:
                ru_scores.append(np.nan)
        stability["RU-DPMM"] = ru_scores

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, scores in stability.items():
        valid_s = [s for s in scores if not np.isnan(s)]
        if not valid_s:
            continue
        is_yours = "RU-DPMM" in model_name
        ax.plot(range(1, STABILITY_RUNS + 1), scores,
                marker="o",
                linewidth=2.5 if is_yours else 1.2,
                linestyle="-" if is_yours else "--",
                label=f"{model_name}  (avg={np.mean(valid_s):.2f})")

    ax.set_title("Cluster Assignment Stability — ARI across 80% Subsamples\n"
                 "Each model refit on every subsample; ARI vs its own full-data base",
                 fontweight="bold")
    ax.set_xlabel("Resample Iteration")
    ax.set_ylabel("Adjusted Rand Index  (1.0 = perfect)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(stability_path, dpi=300)
    print(f"  [Stability] → {stability_path}")

    print("\n  Stability Summary (mean ARI ± std):")
    for name, scores in stability.items():
        valid_s = [s for s in scores if not np.isnan(s)]
        if valid_s:
            print(
                f"    {name:<35s}  {np.mean(valid_s):.3f} ± {np.std(valid_s):.3f}")


def plot_capacity_stability(X, user_labels=None, max_test_k=15,
                            path="capacity_stability_plot.png"):
    """
    Sweeps k from 2 to max_test_k.  For each k:
      • K-Means, GMM (EM), and Agglomerative are forced to use exactly k clusters
        and their silhouette score is recorded — showing how sensitive each model
        is to a wrong capacity assumption.
      • Vanilla DPMM uses k only as an upper bound; active clusters ≤ k.
      • Bootstrap GMM is excluded: 200 bootstrap iterations × max_test_k fits
        would take several minutes and add no new information beyond plain GMM.
      • RU-DPMM is shown as a horizontal plateau at its discovered silhouette
        score, starting from its naturally selected k.  This is the thesis's
        key visual argument: RU-DPMM does not need a k to be specified at all.
    """
    print(f"\n[Plot] Capacity stability → {path}")
    k_range = list(range(2, max_test_k + 1))
    kmeans_scores = []
    gmm_scores = []
    dpmm_scores = []
    agg_scores = []

    for k in k_range:
        # K-Means
        km_lbl = KMeans(n_clusters=k, n_init=10,
                        random_state=42).fit_predict(X)
        kmeans_scores.append(silhouette_score(X, km_lbl))

        # GMM (EM) — added: directly comparable to K-Means (both need a fixed k)
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             n_init=3, random_state=42).fit(X)
        gm_lbl = gm.predict(X)
        gmm_scores.append(
            silhouette_score(X, gm_lbl) if len(set(gm_lbl)) > 1 else 0.0
        )

        # Vanilla DPMM base — uses k as upper bound only
        m_d = BayesianGaussianMixture(
            n_components=k, weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=1e-2, max_iter=300, random_state=42
        ).fit(X)
        d_lbl = m_d.predict(X)
        dpmm_scores.append(
            silhouette_score(X, d_lbl) if len(set(d_lbl)) > 1 else 0.0
        )

        # Agglomerative (Ward) — added: deterministic hierarchical baseline
        agg_lbl = AgglomerativeClustering(
            n_clusters=k, linkage="ward").fit_predict(X)
        agg_scores.append(silhouette_score(X, agg_lbl))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(k_range, kmeans_scores, label="K-Means (forced k)",
            marker="o", linestyle="--", color="#aec7e8", linewidth=1.5)
    ax.plot(k_range, gmm_scores,    label="GMM-EM (forced k)",
            marker="^", linestyle="--", color="#98df8a", linewidth=1.5)
    ax.plot(k_range, agg_scores,    label="Agglomerative Ward (forced k)",
            marker="v", linestyle="--", color="#ffbb78", linewidth=1.5)
    ax.plot(k_range, dpmm_scores,   label="Vanilla DPMM (k = upper bound)",
            marker="s", linewidth=2.2, color="#7f7f7f", alpha=0.8)

    if user_labels is not None:
        mask = user_labels >= 0
        user_k = len(set(user_labels[mask]))
        if user_k > 1:
            user_sil = silhouette_score(X[mask], user_labels[mask])
            px = [kv for kv in k_range if kv >= user_k]
            if not px:                        # RU-DPMM found more clusters than the swept range
                # draw the plateau at the right edge
                px = [k_range[-1]]
            py = [user_sil] * len(px)
            ax.plot(px, py, label="RU-DPMM — Bootstrapped DPMM",
                    marker="D", linewidth=3.0, color="#1f77b4", markersize=8)
            # clamp marker/label into the axes
            kx = min(user_k, k_range[-1])
            ax.axvline(x=kx, color="#1f77b4", linestyle=":", alpha=0.6)
            ax.text(kx + 0.15, user_sil + 0.01,
                    f"k = {user_k}\n(auto-selected)",
                    color="#1f77b4", fontsize=9, fontweight="bold")
            ax.annotate(
                "Horizontal plateau = RU-DPMM silhouette\nat its naturally discovered k\n"
                "(not re-evaluated at each capacity)",
                xy=(px[-1], user_sil),
                xytext=(max(px[-1] - 5, k_range[0]), user_sil - 0.10),
                fontsize=8, color="#444",
                arrowprops=dict(arrowstyle="->", color="#444", lw=0.8)
            )

    ax.set_xlabel(
        "Capacity Limit K  (number of clusters assumed)", fontweight="bold")
    ax.set_ylabel("Silhouette Score  (↑ better)", fontweight="bold")
    ax.set_title(
        "Sensitivity to Wrong Capacity Assumption\n"
        "Fixed-k models degrade when k ≠ true k;  "
        "RU-DPMM discovers k automatically",
        fontweight="bold"
    )
    ax.margins(y=0.18)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=300)


# =============================================================================
# 6. Main pipeline
# =============================================================================

def compare(data_path: str, user_clustered_path: str):
    import os
    # Prefix every output file with the input CSV's base name, so different
    # datasets never overwrite each other's outputs.
    out_prefix = os.path.splitext(os.path.basename(data_path))[0] + "_"
    df = smart_load(data_path)
    X, _ = scale(df)
    k = infer_k(X, MAX_K)

    demand_col = detect_demand_col(df)   # [NEW-3]
    demand_array = df[demand_col].values if demand_col else None

    # ── Fit baselines ──────────────────────────────────────────────────────
    print("\nFitting models …\n")

    lbl_km,   t_km,   bic_km,   unc_km,   ll_km = run_kmeans(X, k)
    lbl_gmm,  t_gmm,  bic_gmm,  unc_gmm,  ll_gmm = run_gmm(X, k)
    lbl_dpmm, t_dpmm, _,        unc_dpmm, elbo_d, dpmm_model, ll_dpmm = run_vanilla_dpmm(
        X, MAX_K)
    lbl_bgmm, t_bgmm, bic_bgmm, unc_bgmm, ll_bgmm = run_bootstrap_gmm(
        X, k, N_BOOTSTRAP)
    lbl_hdb,  t_hdb,  _,        unc_hdb,  ll_hdb = run_hdbscan(
        X, HDBSCAN_MIN_SIZE)
    lbl_agg,  t_agg,  _,        unc_agg,  ll_agg = run_agglomerative(X, k)

    model_runs = [
        # (name, labels, elapsed, bic, elbo, uncert, log_lik)
        ("K-Means",                         lbl_km,
         t_km,   bic_km,   np.nan, unc_km,   ll_km),
        ("GMM (EM)",                         lbl_gmm,
         t_gmm,  bic_gmm,  np.nan, unc_gmm,  ll_gmm),
        ("Vanilla DPMM (Blei & Jordan)",     lbl_dpmm,
         t_dpmm, np.nan,   elbo_d, unc_dpmm, ll_dpmm),
        ("Bootstrap GMM (O'Hagan et al.)",   lbl_bgmm,
         t_bgmm, bic_bgmm, np.nan, unc_bgmm, ll_bgmm),
        ("HDBSCAN (McInnes & Healy)",        lbl_hdb,
         t_hdb,  np.nan,   np.nan, unc_hdb,  ll_hdb),
        ("Agglomerative (Ward)",             lbl_agg,
         t_agg,  np.nan,   np.nan, unc_agg,  ll_agg),
    ]

    results = []
    all_names = []
    all_labels = []
    all_uncerts = []

    for name, labels, elapsed, bic, elbo, uncert, log_lik in model_runs:
        r = evaluate(X, labels, name, elapsed, bic,
                     elbo, log_lik, demand_array)
        results.append(r)
        all_names.append(name)
        all_labels.append(labels)
        all_uncerts.append(uncert)
        print(f"  {name:<44s}  k={r['Clusters']:<3d}  "
              f"Sil={r.get('Silhouette ↑', float('nan')):.4f}  "
              f"DB={r.get('Davies-Bouldin ↓', float('nan')):.4f}  "
              f"LL={r.get('Log-Lik/sample ↑', float('nan')):.3f}  "
              f"({r['Runtime (s)']} s)")

    # ── User model (RU-DPMM) ──────────────────────────────────────────────
    user_final_labels = None
    user_final_uncert = None
    ru_stab_model = None      # saved DPMM core for genuine RU-DPMM stability
    ru_stab_X = None

    try:
        your_df = pd.read_csv(user_clustered_path)

        # Detect whether the CSV carries a pre-computed uncertainty column
        has_uncert = bool(
            UNCERTAINTY_COL and UNCERTAINTY_COL in your_df.columns
        )

        # ── 3-tier alignment — labels AND uncertainty extracted together ──
        if len(your_df) == len(df):
            # Tier 1: row counts match — direct assignment
            print("\n  [Align] Tier 1: direct (row counts match)")
            user_labels_raw = your_df[CLUSTER_COL].values.astype(int)
            user_uncert_raw = (your_df[UNCERTAINTY_COL].values.astype(float)
                               if has_uncert else None)

        elif "__orig_idx__" in your_df.columns:
            # Tier 2: pipeline preserved the original row index
            print("\n  [Align] Tier 2: index-based (__orig_idx__ found)")
            orig_idx = your_df["__orig_idx__"].values.astype(int)
            user_labels_raw = np.full(len(df), -1, dtype=int)
            user_labels_raw[orig_idx] = your_df[CLUSTER_COL].values.astype(int)
            if has_uncert:
                user_uncert_raw = np.full(len(df), 1.0, dtype=float)
                user_uncert_raw[orig_idx] = (
                    your_df[UNCERTAINTY_COL].values.astype(float)
                )
            else:
                user_uncert_raw = None

        else:
            # Tier 3: feature-value merge
            print("\n  [Align] Tier 3: feature-value merge")
            # Join on shared numeric feature columns only —
            # exclude CLUSTER_COL and UNCERTAINTY_COL from the join keys
            excl = {CLUSTER_COL}
            if UNCERTAINTY_COL:
                excl.add(UNCERTAINTY_COL)
            shared = [c for c in df.columns
                      if c in your_df.columns and c not in excl]
            if not shared:
                raise ValueError(
                    "No shared feature columns for merge.  Either ensure the "
                    "CSVs share numeric columns, or add '__orig_idx__' to your "
                    "DPMM output before dropping outliers."
                )
            df_r = df[shared].round(4).copy()
            df_r["__idx__"] = np.arange(len(df))

            # Include UNCERTAINTY_COL in the right-hand side of the merge
            out_cols = shared + [CLUSTER_COL]
            if has_uncert:
                out_cols = out_cols + [UNCERTAINTY_COL]
            out_r = your_df[out_cols].copy()
            out_r[shared] = out_r[shared].round(4)   # round join keys only

            merged = (df_r.merge(out_r, on=shared, how="left")
                      .drop_duplicates(subset=["__idx__"], keep="first")
                      .sort_values("__idx__")
                      .reset_index(drop=True))
            user_labels_raw = merged[CLUSTER_COL].fillna(-1).astype(int).values
            user_uncert_raw = (
                merged[UNCERTAINTY_COL].fillna(1.0).values.astype(float)
                if has_uncert else None
            )

        user_final_labels = user_labels_raw

        # ── Uncertainty: use pipeline column or fall back to predict_proba ──
        if user_uncert_raw is not None:
            user_final_uncert = user_uncert_raw
            print(f"  [Uncertainty] Using '{UNCERTAINTY_COL}' from CSV  "
                  f"(range [{user_final_uncert.min():.3f}, "
                  f"{user_final_uncert.max():.3f}], "
                  f"mean {user_final_uncert.mean():.3f})")
        else:
            # Fallback: derive from the fitted DPMM's posterior probabilities
            user_final_uncert = 1.0 - dpmm_model.predict_proba(X).max(axis=1)
            print(f"  [Uncertainty] '{UNCERTAINTY_COL}' not found — "
                  f"falling back to 1 - max(predict_proba)")

        # ── Evaluate RU-DPMM on its OWN exported features, in a consistent
        #    transform (no merge, no space mismatch -> removes the spurious
        #    negative silhouette). ELBO/log-lik are the model's REAL values
        #    (from its saved .pkl), never copied from Vanilla DPMM.
        import os as _os
        import joblib as _joblib
        _feat = [c for c in df.columns if c in your_df.columns]
        ru_X_raw = your_df[_feat].apply(
            pd.to_numeric, errors="coerce").to_numpy()
        ru_labels = your_df[CLUSTER_COL].values.astype(int)
        _keep = ~np.isnan(ru_X_raw).any(axis=1)
        ru_X_raw, ru_labels = ru_X_raw[_keep], ru_labels[_keep]
        try:
            from select_transform import select_scaler
            _rusc, _, _ = select_scaler(ru_X_raw, verbose=False)
        except Exception:
            _rusc = StandardScaler().fit(ru_X_raw)
        ru_X = _rusc.transform(ru_X_raw)
        ru_demand = (your_df.loc[_keep, demand_col].to_numpy()
                     if (demand_col and demand_col in your_df.columns) else None)
        ru_elbo = ru_ll = np.nan
        _pkl = user_clustered_path.replace("_clustered.csv", "_model.pkl")
        if _os.path.exists(_pkl):
            try:
                _ck = _joblib.load(_pkl)
                ru_elbo = float(_ck["model"].lower_bound_)
                ru_ll = float(_ck["model"].score(
                    _ck["scaler"].transform(ru_X_raw)))
                # genuine stability
                ru_stab_model = _ck["model"]
                ru_stab_X = _ck["scaler"].transform(
                    ru_X_raw)   # genuine stability
            except Exception:
                pass
        your_row = evaluate(ru_X, ru_labels, "RU-DPMM",
                            USER_RUNTIME, np.nan, ru_elbo, ru_ll, ru_demand)
        results.append(your_row)
        all_names.append("RU-DPMM")
        all_labels.append(user_final_labels)
        all_uncerts.append(user_final_uncert)

        print(f"  {'RU-DPMM':<44s}  k={your_row['Clusters']:<3d}  "
              f"Sil={your_row.get('Silhouette ↑', float('nan')):.4f}  "
              f"DB={your_row.get('Davies-Bouldin ↓', float('nan')):.4f}")

    except Exception as e:
        print(f"\n[Error] Could not load/align user model: {e}")
        import traceback
        traceback.print_exc()

    # ── Results table ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 115)
    print(f"{'COMPARISON RESULTS':^115}")
    print("=" * 115)
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{out_prefix}comparison_results.csv", index=False)
    print(f"\n[Output] {out_prefix}comparison_results.csv saved.")

    # ── Visualisations ─────────────────────────────────────────────────────
    plot_bar_charts(results_df, path=f"{out_prefix}metrics_barcharts.png")
    plot_comparison_fuzzy(X, all_names, all_labels, all_uncerts,
                          path=f"{out_prefix}fuzzy_comparison.png")
    run_stability_test(X, k, MAX_K, user_final_labels, dpmm_model,
                       ru_model=ru_stab_model, ru_X=ru_stab_X,
                       stability_path=f"{out_prefix}stability_line_chart.png")
    plot_capacity_stability(X, user_labels=user_final_labels, max_test_k=MAX_K,
                            path=f"{out_prefix}capacity_stability_plot.png")


# =============================================================================
# 7. Entry point
# =============================================================================

if __name__ == "__main__":
    compare(DATASET_PATH, CLUSTERED_OUT_PATH)
