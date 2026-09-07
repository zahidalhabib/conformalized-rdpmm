"""
R-DPMM: Conformalized, Anisotropically-Calibrated Bootstrapped DPMM
=====================================================================
Extends Neofytou, Liu & Akartunali (2025) [Optimization 75(6):1183-1226].

Levers
------
Lever 1  Split-conformal calibration — distribution-free (1-alpha) coverage
Lever 2  Anisotropic bootstrap expansion via delta_k = sqrt(lambda_max)
Lever 3  l1 ∩ l2 ∩ l∞ geometry + compactness measurement vs paper baseline
Lever 4  Bayesian bootstrap Option B — Dirichlet-weighted sufficient stats,
         joint (mu_k, Sigma_k) draws, smoothed tau_k, no label switching

Bug fixes
---------
BUG-1  NameError on actual_clustering_runtime when loading persisted model
BUG-2  Timer end / runtime assigned inside the cluster loop
BUG-3  master_means dead parameter in _single_bootstrap (function replaced)
BUG-4  Redundant first joblib.dump without uncertainty_set_params
BUG-5  prefer="threads" for CPU-bound work
BUG-6  safe_jobs fallback ≈ 1 (budget_bytes/bytes_per_job tautology)
BUG-7  alpha_coverage=1.0 hardcoded in every call to calculate_adaptive_gamma

Additional fixes (this revision)
--------------------------------
FIX-1  Reproducible Bayesian bootstrap — each draw gets a fixed per-draw
       seed via np.random.default_rng(seed); bare np.random.seed() made the
       whole pipeline non-deterministic despite random_state=42 elsewhere.
       Draws remain independent; the full set is now identical across runs.
FIX-2  sigma_star_k is rescaled back to real units in the DOCX report. It
       was sqrt(diag(cov_mu_star)) measured in standardized space but was
       printed next to real-unit means — mismatched units. Multiplying by
       scaler.scale_ puts the spread in the same units as the centre.
FIX-3  High-dim visualization now derives the Lever-2 margin (delta_k) in
       the PCA-projected space instead of reusing the original-space value.
       The full centre covariance is carried in the params dict and
       projected via pca.components_ so the drawn boundary is unit-consistent.
"""

import os
import json
import datetime
import subprocess
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time

from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning

from robust_load import robust_load

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── Global tuning parameters ──────────────────────────────────────────────────
CONFORMAL_ALPHA = 0.10   # miscoverage rate → (1-alpha)=90% coverage guarantee
CAL_FRACTION = 0.20   # fraction of cleaned data held out for conformal cal
BOOTSTRAP_L = 500    # Bayesian bootstrap replicates (cheap with Option B)
MAX_COMPONENTS = 10     # DPMM upper bound on number of clusters
ACTIVE_THR = 0.05   # weight threshold for an "active" cluster


# =============================================================================
# SMART PREPROCESSING
# =============================================================================

def smart_preprocess(df):
    """
    Drops ID/sequential columns, label-encodes low-cardinality categoricals,
    and returns a fully numeric DataFrame ready for clustering.
    """
    print("  > Auto-detecting feature types...")
    working_df = df.copy()
    initial_cols = working_df.columns.tolist()
    cols_to_drop = []

    for col in initial_cols:
        col_lower = col.lower()
        drop_hints = ['id', 'index', 'idx', 'uuid', 'serial', 'customer_no',
                      'latitude', 'longitude', 'zipcode', 'zip', 'risk', 'status', 'Channel', 'Region']
        if any(t in col_lower for t in drop_hints):
            cols_to_drop.append(col)
            continue
        if (working_df[col].nunique() == len(working_df)
                and pd.api.types.is_integer_dtype(working_df[col])):
            cols_to_drop.append(col)
            continue
        if (working_df[col].dtype == 'object'
                or working_df[col].dtype.name == 'category'):
            n_unique = working_df[col].nunique()
            if n_unique <= 15:
                print(
                    f"  > Encoding categorical: '{col}' ({n_unique} classes)")
                le = LabelEncoder()
                working_df[col] = le.fit_transform(working_df[col].astype(str))
            else:
                cols_to_drop.append(col)

    working_df = (working_df.drop(columns=cols_to_drop)
                  .select_dtypes(include=[np.number]))
    dropped = set(initial_cols) - set(working_df.columns)
    if dropped:
        print(f"  > Dropped: {sorted(dropped)}")
    return working_df


# =============================================================================
# CORE DPMM WRAPPER
# =============================================================================

class VariationalDPMM:
    def __init__(self, max_components=MAX_COMPONENTS,
                 max_iter=2000, tol=1e-3, random_state=42):
        self.max_components = max_components
        self.model = BayesianGaussianMixture(
            weight_concentration_prior_type='dirichlet_process',
            n_components=max_components,
            max_iter=max_iter,
            tol=tol,
            reg_covar=1e-3,
            covariance_type='full',
            init_params='kmeans',
            random_state=random_state,
        )

    def fit(self, X):
        self.model.fit(X)
        return self.model


# =============================================================================
# LEVER 4 — BAYESIAN BOOTSTRAP (OPTION B) WITH PRECISION MATRICES
# =============================================================================

def _bayesian_bootstrap_step(X, R, reg=1e-3, seed=None):
    """
    Single Bayesian bootstrap draw (Rubin 1981).

    Draws Dirichlet(1,...,1) weights, applies them to the pre-computed
    soft-assignment matrix R, and computes weighted cluster means +
    covariances with no DPMM refit → no EM noise, no label switching.

    Parameters
    ----------
    X    : (n, d)  data — training fold, possibly row-capped
    R    : (n, K)  master model predict_proba output, fixed across all draws
    reg  : float   covariance regularisation added to diagonal
    seed : int     FIX-1: per-draw seed → reproducible yet independent draws.
                   A dedicated Generator replaces the global np.random state so
                   workers don't share/clobber each other's RNG.

    Returns
    -------
    boot_means : (K, d)
    boot_covs  : (K, d, d)
    valid      : (K,) bool   validity mask (BUG-C)
    """
    rng = np.random.default_rng(
        seed)              # FIX-1: isolated, seeded RNG
    n, d = X.shape
    K = R.shape[1]
    weights = rng.dirichlet(np.ones(n))
    eff = weights[:, None] * R
    mass = eff.sum(axis=0)
    boot_means = np.empty((K, d))
    boot_covs = np.empty((K, d, d))
    valid = np.empty(K, dtype=bool)           # BUG-C: validity mask
    for k in range(K):
        if mass[k] < 1e-9:
            boot_means[k] = np.zeros(d)
            boot_covs[k] = np.eye(d) * reg
            valid[k] = False                  # BUG-C
            continue
        mu_k = (eff[:, k, None] * X).sum(axis=0) / mass[k]
        diff = X - mu_k
        sqrt_eff = np.sqrt(eff[:, k])[:, None]
        weighted_diff = sqrt_eff * diff
        cov_k = (weighted_diff.T @ weighted_diff) / mass[k]
        boot_means[k] = mu_k
        boot_covs[k] = cov_k + np.eye(d) * reg
        valid[k] = True                       # BUG-C
    return boot_means, boot_covs, valid            # BUG-C: three-tuple now


def bootstrap_procedure(X, master_model, L=BOOTSTRAP_L,
                        max_sample_size=50_000, memory_per_job_mb=200):
    """
    Runs L Bayesian bootstrap draws in parallel and aggregates.

    Returns
    -------
    mu_star     : (K, d)     bootstrap mean of cluster centres
    cov_mu_star : (K, d, d)  bootstrap covariance of centres  (→ Lever 2)
    tau_star    : (K, d, d)  Cholesky precision from mean bootstrap Sigma_k
                             (→ Lever 4 smoothed precision matrix)
    """
    n_master = master_model.n_components
    n_features = X.shape[1]

    # Pre-compute soft assignments ONCE — all workers reuse the same R.
    # BUG-3 fix: old _single_bootstrap accepted master_means but never used it.
    R = master_model.predict_proba(X)    # (n, K)

    sample_size = min(len(X), max_sample_size)
    if sample_size < len(X):
        idx = np.random.choice(len(X), size=sample_size, replace=False)
        X_boot = X[idx]
        R_boot = R[idx]
        print(f"  Capped Bayesian bootstrap to {sample_size:,} rows.")
    else:
        X_boot, R_boot = X, R

    # BUG-6 fix: fallback uses a realistic 4 GB assumption instead of
    # budget_bytes/bytes_per_job which evaluated to ~1 by construction.
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        safe_jobs = max(1, int(available_mb / memory_per_job_mb))
    except ImportError:
        assumed_ram_mb = 4096
        safe_jobs = max(1, int(assumed_ram_mb / memory_per_job_mb))

    n_jobs = min(os.cpu_count() or 1, safe_jobs)
    print(f"Bayesian Bootstrap (Option B): L={L} | n_jobs={n_jobs}")

    # BUG-5 fix: prefer="processes" — Dirichlet sampling and matrix ops are
    # CPU-bound; threads cannot parallelise them due to the GIL.
    # FIX-1: pass a distinct, fixed seed to every draw so the full set of L
    # draws is reproducible run-to-run while staying mutually independent.
    results = joblib.Parallel(n_jobs=n_jobs, prefer="processes")(
        joblib.delayed(_bayesian_bootstrap_step)(X_boot, R_boot, seed=i)
        for i in range(L)
    )

    active = np.where(master_model.weights_ > ACTIVE_THR)[0]
    stored_means = {k: [] for k in active}
    stored_covs = {k: [] for k in active}

    for boot_means, boot_covs, valid in results:   # BUG-C: unpack third value
        for k in active:
            if valid[k]:                           # BUG-C: filter sentinels
                stored_means[k].append(boot_means[k])
                stored_covs[k].append(boot_covs[k])

    mu_star = master_model.means_.copy()
    cov_mu_star = np.zeros((n_master, n_features, n_features))
    tau_star = np.zeros((n_master, n_features, n_features))

    for k in active:
        if not stored_means[k]:                    # now meaningful after BUG-C fix
            tau_star[k] = np.eye(n_features)
            continue
        arr_means = np.array(stored_means[k])
        arr_covs = np.array(stored_covs[k])
        mu_star[k] = arr_means.mean(axis=0)
        if len(arr_means) > 1:
            C = np.cov(arr_means, rowvar=False)
            cov_mu_star[k] = (C + C.T) / 2        # BUG-B: enforce symmetry

        # Lever 4: smoothed precision — try increasing reg until Cholesky works
        mean_sigma_k = arr_covs.mean(axis=0)
        tau_star[k] = np.eye(n_features)
        for reg in (1e-3, 1e-2, 0.1, 1.0):
            try:
                tau_star[k] = np.linalg.cholesky(
                    np.linalg.inv(mean_sigma_k + np.eye(n_features) * reg))
                break
            except np.linalg.LinAlgError:
                continue

    print("Bayesian Bootstrap complete.")
    return mu_star, cov_mu_star, tau_star


# =============================================================================
# OUTLIER DETECTION  (gross-filter; de-emphasised now Lever 4 gives robust tau_k)
# =============================================================================

def outlier_detection(X, trained_model):
    """Adaptive 3-sigma log-probability filter kept as a gross-outlier guard."""
    print("Outlier detection (adaptive 3-sigma)...")
    log_probs = trained_model.score_samples(X)
    threshold = np.mean(log_probs) - 3.0 * np.std(log_probs)
    mask = log_probs > threshold
    n_out = int((~mask).sum())
    print(f"  Threshold={threshold:.4f} | removed {n_out} "
          f"({100*n_out/len(X):.1f}%) | kept {int(mask.sum())} points.")
    return X[mask], X[~mask]


# =============================================================================
# LEVER 1 — SPLIT-CONFORMAL CALIBRATION
# =============================================================================

def conformal_gamma(cal_points, mu_k, tau_k, alpha=CONFORMAL_ALPHA):
    """
    Split-conformal Gamma — distribution-free (1-alpha) coverage guarantee.

    Nonconformity score: whitened residual z = tau_k @ (x - mu_k).
    Gamma is set to the ceil((n+1)(1-alpha))/n quantile of l1, l2, l∞ norms
    of z over the held-out calibration set.

    Returns (gamma_1, gamma_2, gamma_inf) or (None, None, None) if
    fewer than 5 calibration points exist for this cluster.
    """
    if len(cal_points) < 5:
        return None, None, None

    diffs = (cal_points - mu_k) @ tau_k.T      # (n_cal, d)
    scores_1 = np.linalg.norm(diffs, ord=1,    axis=1)
    scores_2 = np.linalg.norm(diffs, ord=2,    axis=1)
    scores_inf = np.linalg.norm(diffs, ord=np.inf, axis=1)

    n = len(cal_points)
    q_idx = int(np.clip(int(np.ceil((n + 1) * (1 - alpha))) - 1, 0, n - 1))

    return (float(np.sort(scores_1)[q_idx]),
            float(np.sort(scores_2)[q_idx]),
            float(np.sort(scores_inf)[q_idx]))


# =============================================================================
# FALLBACK — IN-SAMPLE EMPIRICAL QUANTILE
# =============================================================================

def calculate_adaptive_gamma(cluster_data, mu_k, tau_k, alpha_coverage=None):
    """
    In-sample fallback when the calibration set is too small.

    BUG-7 fix: defaults to (1 - CONFORMAL_ALPHA) instead of 1.0, so the
    fallback targets the same nominal coverage as the conformal path.
    Returns triple (gamma_1, gamma_2, gamma_inf) matching conformal_gamma.
    """
    if alpha_coverage is None:
        alpha_coverage = 1.0 - CONFORMAL_ALPHA

    N_k = len(cluster_data)
    if N_k == 0:
        return 0.0, 0.0, 0.0

    diffs = (cluster_data - mu_k) @ tau_k.T
    dist_1 = np.linalg.norm(diffs, ord=1,    axis=1)
    dist_2 = np.linalg.norm(diffs, ord=2,    axis=1)
    dist_inf = np.linalg.norm(diffs, ord=np.inf, axis=1)

    q_idx = int(np.clip(int(np.round(N_k * alpha_coverage)) - 1, 0, N_k - 1))
    return (float(np.sort(dist_1)[q_idx]),
            float(np.sort(dist_2)[q_idx]),
            float(np.sort(dist_inf)[q_idx]))


# =============================================================================
# LEVER 3 — SET AREA COMPARISON
# =============================================================================

def compute_set_areas(gamma_1, gamma_2, gamma_inf, n_theta=500):
    """
    Polar-integration area of l1 ∩ l2 ∩ l∞ (new) and l1 ∩ l∞ (paper baseline)
    in whitened Z-space.  Returns (area_new, area_old, pct_tighter).
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    eps = 1e-12

    r_1 = gamma_1 / (np.abs(np.cos(theta)) + np.abs(np.sin(theta)) + eps)
    r_2 = np.full_like(theta, float(gamma_2))
    r_inf = gamma_inf / (np.maximum(np.abs(np.cos(theta)),
                                    np.abs(np.sin(theta))) + eps)
    r_new = np.minimum(np.minimum(r_1, r_inf), r_2)
    r_old = np.minimum(r_1, r_inf)

    def _shoelace(r):
        x, y = r * np.cos(theta), r * np.sin(theta)
        return 0.5 * abs(float(np.dot(x, np.roll(y, 1))
                               - np.dot(y, np.roll(x, 1))))

    area_new = _shoelace(r_new)
    area_old = _shoelace(r_old)
    pct = max(0.0, 100.0 * (area_old - area_new) / (area_old + eps))
    return area_new, area_old, pct


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def report_silhouette(X_scaled, labels, sample_size=90000):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print("  [Silhouette] Skipped: fewer than 2 clusters.")
        return None

    # Downsample if the dataset is massive to avoid O(N^2) hang
    if len(X_scaled) > sample_size:
        print(
            f"  [Silhouette] Computing on a random sample of {sample_size:,} points...")
        score = silhouette_score(
            X_scaled, labels, sample_size=sample_size, random_state=42)
    else:
        score = silhouette_score(X_scaled, labels)

    quality = ("excellent" if score > 0.7 else "good" if score > 0.5
               else "fair" if score > 0.25 else "poor")
    print(f"\n  Silhouette Score : {score:.4f}  ({quality})")
    print("  Scale: -1 (wrong cluster) ... 0 (overlapping) ... +1 (perfect)")
    return score


def calculate_feature_importance(X_scaled, labels, feature_names):
    print("\n--- FEATURE IMPORTANCE ---")
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        print("  Insufficient clusters.")
        return None

    overall_mean = np.mean(X_scaled, axis=0)
    importances = []
    for i in range(X_scaled.shape[1]):
        fv = X_scaled[:, i]
        ss_between = ss_within = 0.0
        for label in unique_labels:
            c = fv[labels == label]
            if len(c) < 2:
                continue
            cm = np.mean(c)
            ss_between += len(c) * (cm - overall_mean[i]) ** 2
            ss_within += np.sum((c - cm) ** 2)
        importances.append(ss_between / (ss_within + 1e-9))

    importances = np.array(importances)
    importances = importances / (importances.sum() + 1e-9) * 100.0
    df_imp = pd.DataFrame({'Feature': feature_names,
                           'Importance_Score': importances
                           }).sort_values('Importance_Score', ascending=False)
    for _, row in df_imp.iterrows():
        print(f"  {row['Feature']}: {row['Importance_Score']:.2f}%")
    return df_imp


# =============================================================================
# VISUALIZATION  (l1 ∩ l2 ∩ l∞ boundary — Lever 3)
# =============================================================================

def plot_uncertainty_sets(
    cleaned_X_scaled, outliers_X_scaled,
    final_labels, uncertainty_set_params,
    scaler, original_columns,
    output_img='uncertainty_set_visualization.png',
):
    dim = cleaned_X_scaled.shape[1]
    if dim == 1:
        print("\nVisualization skipped: 1-D not supported.")
        return

    print("\nGenerating visualization...")
    fig, ax = plt.subplots(figsize=(12, 8))

    is_high_dim = dim > 2
    pca = None
    if is_high_dim:
        print(f"  {dim}-D — projecting to 2D via PCA.")
        pca = PCA(n_components=2)
        pca.fit(cleaned_X_scaled)
        xlabel, ylabel = 'Principal Component 1', 'Principal Component 2'
    else:
        xlabel = original_columns[0] if original_columns else 'Feature A'
        ylabel = original_columns[1] if len(
            original_columns) > 1 else 'Feature B'

    if len(outliers_X_scaled) > 0:
        out_plot = (pca.transform(outliers_X_scaled) if is_high_dim
                    else scaler.inverse_transform(outliers_X_scaled))
        ax.scatter(out_plot[:, 0], out_plot[:, 1], marker='*',
                   color='darkblue', label='Outliers', alpha=0.8, s=40)

    cleaned_labelled = center_labelled = boundary_labelled = False

    for i, (k, params) in enumerate(uncertainty_set_params.items()):
        cluster_scaled = cleaned_X_scaled[final_labels == k]
        if len(cluster_scaled) == 0:
            continue

        if is_high_dim:
            cluster_plot = pca.transform(cluster_scaled)
            mu_plot = np.mean(cluster_plot, axis=0)
            cov_plot = np.cov(cluster_plot, rowvar=False) + np.eye(2) * 1e-6
            try:
                eigvals, eigvecs = np.linalg.eigh(cov_plot)
                # Symmetric inverse square root for whitening
                tau_plot = eigvecs @ np.diag(1.0 /
                                             np.sqrt(eigvals + 1e-9)) @ eigvecs.T

            except (np.linalg.LinAlgError, ValueError):
                tau_plot = np.eye(2)
            g1, g2, ginf = calculate_adaptive_gamma(
                cluster_plot, mu_plot, tau_plot)

            # FIX-3: derive the Lever-2 margin in the SAME PCA-projected space
            # as the rest of the boundary instead of reusing the original-space
            # delta_k.  The centre covariance is linear, so it projects as
            #   cov_pca = W . cov_mu_star_k . W^T,   W = pca.components_ (2 x d)
            # then whiten with tau_plot and take sqrt of the largest eigenvalue.
            cov_centre = params.get('cov_mu_star_k', None)
            if cov_centre is not None:
                W = pca.components_                       # (2, d)
                cov_pca = W @ np.asarray(cov_centre) @ W.T
                C_w = tau_plot @ cov_pca @ tau_plot.T
                dk = float(np.sqrt(max(float(np.linalg.eigvalsh(C_w).max()),
                                       0.0)))
            else:
                dk = 0.0                                  # no covariance → no pad
            gamma_1, gamma_2, gamma_inf = g1 + dk, g2 + dk, ginf + dk
            mu_scaled = None
        else:
            cluster_plot = scaler.inverse_transform(cluster_scaled)
            mu_plot = scaler.inverse_transform([params['mu_star']])[0]
            tau_plot = params['tau_k']
            gamma_1 = params['Gamma_1']
            gamma_2 = params['Gamma_2']
            gamma_inf = params['Gamma_inf']
            mu_scaled = params['mu_star']

        lbl_d = 'Cleaned Data' if not cleaned_labelled else None
        ax.scatter(cluster_plot[:, 0], cluster_plot[:, 1],
                   color='#0088cc', label=lbl_d, alpha=0.5, s=15)
        cleaned_labelled = True

        lbl_c = 'Cluster Centre (mu*)' if not center_labelled else None
        ax.scatter(mu_plot[0], mu_plot[1], marker='D', color='orange',
                   edgecolor='darkorange', s=80, zorder=6, label=lbl_c)
        center_labelled = True

        # Three-norm polar boundary (Lever 3)
        theta = np.linspace(0, 2 * np.pi, 500)
        eps = 1e-12
        r_1 = gamma_1 / (np.abs(np.cos(theta)) + np.abs(np.sin(theta)) + eps)
        r_2 = np.full_like(theta, float(gamma_2))
        r_inf = gamma_inf / (np.maximum(np.abs(np.cos(theta)),
                                        np.abs(np.sin(theta))) + eps)
        r_int = np.minimum(np.minimum(r_1, r_inf), r_2)

        Z = np.column_stack([r_int * np.cos(theta), r_int * np.sin(theta)])
        try:
            tau_inv = np.linalg.inv(tau_plot)
        except (np.linalg.LinAlgError, ValueError):    # BUG-A
            tau_inv = np.eye(2)

        U_plot = (Z @ tau_inv.T + mu_plot if is_high_dim
                  else scaler.inverse_transform(Z @ tau_inv.T + mu_scaled))

        lbl_b = 'l1∩l2∩l∞ Uncertainty Set' if not boundary_labelled else None
        ax.plot(U_plot[:, 0], U_plot[:, 1], color='purple',
                linewidth=2.0, alpha=0.9, label=lbl_b)
        boundary_labelled = True

        for xv in (U_plot[:, 0].min(), U_plot[:, 0].max()):
            ax.axvline(x=xv, color='black', linestyle='--', lw=0.5, alpha=0.4)
        for yv in (U_plot[:, 1].min(), U_plot[:, 1].max()):
            ax.axhline(y=yv, color='black', linestyle='--', lw=0.5, alpha=0.4)

        pct = params.get('area_pct_tighter', 0.0)
        if pct:
            ax.annotate(f"k={k}: {pct:.1f}% tighter than l1∩l∞",
                        xy=(mu_plot[0], mu_plot[1]),
                        xytext=(0, 14), textcoords='offset points',
                        fontsize=7, color='purple', ha='center')

    ax.set_title(f'Conformalized DPMM — l1∩l2∩l∞ Sets '
                 f'(alpha={CONFORMAL_ALPHA:.0%}, L={BOOTSTRAP_L})', fontsize=13)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Visualization saved to '{output_img}'.")
    plt.show()


# =============================================================================
# DOCX REPORT EXPORT
# =============================================================================

def export_docx_report(uncertainty_set_params, importance_df, silhouette_val,
                       final_labels, original_columns, scaler,
                       output_path='dpmm_cluster_report.docx'):
    clusters_payload = {}
    for k, params in uncertainty_set_params.items():
        mu_real = scaler.inverse_transform([params['mu_star']])[0]

        sigma_scaled = np.asarray(
            params.get('sigma_star_k', np.zeros_like(params['mu_star'])), dtype=float)
        scale_attr = getattr(scaler, "scale_", None)
        if scale_attr is not None:
            sigma_real = sigma_scaled * scale_attr
        else:
            mu_t = np.asarray(params['mu_star'], dtype=float)
            hi = scaler.inverse_transform([mu_t + sigma_scaled])[0]
            lo = scaler.inverse_transform([mu_t - sigma_scaled])[0]
            sigma_real = np.abs(hi - lo) / 2.0

        clusters_payload[str(k)] = {
            'n_points':         int((final_labels == k).sum()),
            'mu_star':          mu_real.tolist(),
            'sigma_star':       sigma_real.tolist(),
            'delta_k':          float(params.get('delta_k', 0.0)),
            'gamma_1':          float(params['Gamma_1']),
            'gamma_2':          float(params['Gamma_2']),
            'gamma_inf':        float(params['Gamma_inf']),
            'area_new':         float(params.get('area_new', 0.0)),
            'area_old':         float(params.get('area_old', 0.0)),
            'area_pct_tighter': float(params.get('area_pct_tighter', 0.0)),
            'n_cal_points':     int(params.get('n_cal_points', 0)),
        }

    fi_list = ([] if importance_df is None else
               [{'feature': str(r['Feature']), 'score': float(r['Importance_Score'])}
                for _, r in importance_df.iterrows()])

    payload = {
        'clusters':           clusters_payload,
        'feature_importance': fi_list,
        'silhouette_score':   float(silhouette_val) if silhouette_val else 0.0,
        'total_points':       int(len(final_labels)),
        'conformal_alpha':    CONFORMAL_ALPHA,
        'bootstrap_L':        BOOTSTRAP_L,
        'generated_at':       datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'original_columns':   original_columns,
    }

    json_path = 'dpmm_report_data.json'
    with open(json_path, 'w') as fh:
        json.dump(payload, fh, indent=2)

    js_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'generate_report.js')
    if not os.path.exists(js_path):
        print(f"  [WARN] '{js_path}' not found — DOCX skipped.")
        os.path.exists(json_path) and os.remove(json_path)
        return

    result = subprocess.run(['node', js_path, json_path, output_path],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr}")
    else:
        print(result.stdout.strip())
        print(f"  DOCX saved to '{output_path}'.")

    os.path.exists(json_path) and os.remove(json_path)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

if __name__ == "__main__":

    # Synthetic fallback -------------------------------------------------------
    input_filename = 'retail_fmcg_inventory.csv'
    if not os.path.exists(input_filename):
        print(f"'{input_filename}' not found. Generating synthetic 3-D data...")
        np.random.seed(42)
        c1 = np.random.multivariate_normal([100, 50_000, 200],
                                           [[50, 0, 0], [0, 10_000, 0], [0, 0, 150]], 150)
        c2 = np.random.multivariate_normal([250, 90_000, 400],
                                           [[80, 0, 0], [0, 15_000, 0], [0, 0, 200]], 100)
        out = np.random.uniform([0, 20_000, 0], [400, 120_000, 600], (15, 3))
        pd.DataFrame(np.vstack([c1, c2, out]),
                     columns=['Feature_A', 'Feature_B', 'Feature_C']
                     ).to_csv(input_filename, index=False)

    # Output paths -------------------------------------------------------------
    input_dir = os.path.dirname(os.path.abspath(input_filename))
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    def out_path(s): return os.path.join(input_dir, f"{base_name}{s}")
    output_csv = out_path('_clustered.csv')
    output_png = out_path('_uncertainty_sets.png')
    output_docx = out_path('_cluster_report.docx')
    output_model = out_path('_model.pkl')

    # Step 1: Load & preprocess ------------------------------------------------
    print(f"\n--- STEP 1: Loading '{input_filename}' ---")
    df = robust_load(input_filename)
    df = df.dropna(axis=1, how="all")
    print("\n--- STEP 1.5: Smart Preprocessing ---")
    df = smart_preprocess(df)
    df = df.loc[:, df.isna().mean() < 0.5]
    df = df.dropna().reset_index(drop=True)
    original_columns = df.columns.tolist()
    X_raw = df.to_numpy()

    # Step 2: Scale ------------------------------------------------------------
    print("\n--- STEP 2: Standardizing ---")
    from select_transform import select_scaler

    # was: scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_raw)
    # decides StandardScaler vs Yeo-Johnson
    scaler, which, _ = select_scaler(X_raw)
    X_scaled = scaler.transform(X_raw)
    print(f"--- STEP 2: Standardizing (auto-selected: {which}) ---")
    t_clustering_start = time.time()
    actual_clustering_runtime = 0.0

    if os.path.exists(output_model):
        # Load persisted model -------------------------------------------------
        print(f"\n--- STEP 3: Loading Persisted Model ---")
        ck = joblib.load(output_model)
        master_model = ck['model']
        scaler = ck['scaler']
        print("  Loaded.")
        print("\n--- STEP 4: Outlier Removal ---")
        cleaned_X_scaled, outliers_X_scaled = outlier_detection(
            X_scaled, master_model)
        print("\n--- STEP 4.5: Conformal Train / Cal Split ---")
        cleaned_fit_X, cal_X_scaled = train_test_split(
            cleaned_X_scaled, test_size=CAL_FRACTION, random_state=42)
        print(f"  fit={len(cleaned_fit_X):,}  |  cal={len(cal_X_scaled):,}")

    else:
        # Fresh fit ------------------------------------------------------------
        print("\n--- STEP 3: Fitting Initial Master DPMM ---")
        dpmm = VariationalDPMM(max_components=MAX_COMPONENTS)
        master_model = dpmm.fit(X_scaled)
        print("\n--- STEP 4: Outlier Removal ---")
        cleaned_X_scaled, outliers_X_scaled = outlier_detection(
            X_scaled, master_model)
        print("\n--- STEP 4.5: Conformal Train / Cal Split ---")
        cleaned_fit_X, cal_X_scaled = train_test_split(
            cleaned_X_scaled, test_size=CAL_FRACTION, random_state=42)
        print(f"  fit={len(cleaned_fit_X):,}  |  cal={len(cal_X_scaled):,}")
        print("\n--- STEP 5: Refitting on Clean Training Data ---")
        master_model = dpmm.fit(cleaned_fit_X)

    # Step 6: Bayesian Bootstrap -----------------------------------------------
    print("\n--- STEP 6: Bayesian Bootstrap (Lever 4 — Option B) ---")
    mu_star, cov_mu_star, tau_star = bootstrap_procedure(
        cleaned_fit_X, master_model, L=BOOTSTRAP_L)

    # Step 7: Conformalized Uncertainty Sets -----------------------------------
    print("\n--- STEP 7: Constructing Conformalized Uncertainty Sets ---")
    active_clusters = np.where(master_model.weights_ > ACTIVE_THR)[0]
    cal_labels = master_model.predict(cal_X_scaled)
    train_labels = master_model.predict(cleaned_fit_X)
    uncertainty_set_params = {}

    for k in active_clusters:
        tau_k = tau_star[k]                          # Lever 4 precision
        cal_cluster = cal_X_scaled[cal_labels == k]
        g1, g2, ginf = conformal_gamma(
            cal_cluster, mu_star[k], tau_k)  # Lever 1

        if g1 is None:
            train_cluster = cleaned_fit_X[train_labels == k]
            g1, g2, ginf = calculate_adaptive_gamma(
                train_cluster, mu_star[k], tau_k)
            print(
                f"  Cluster {k}: fallback (only {len(cal_cluster)} cal pts).")

        # Lever 2: anisotropic expansion
        C_w = tau_k @ cov_mu_star[k] @ tau_k.T
        delta_k = float(
            np.sqrt(max(float(np.linalg.eigvalsh(C_w).max()), 0.0)))
        eg1, eg2, eginf = g1+delta_k, g2+delta_k, ginf+delta_k

        # Lever 3: compactness
        area_new, area_old, pct = compute_set_areas(eg1, eg2, eginf)

        uncertainty_set_params[k] = {
            'mu_star':          mu_star[k],
            'tau_k':            tau_k,
            'Gamma_1':          eg1,
            'Gamma_2':          eg2,
            'Gamma_inf':        eginf,
            'delta_k':          delta_k,
            'sigma_star_k':     np.sqrt(np.maximum(np.diag(cov_mu_star[k]), 0)),
            # FIX-3: carry the full centre covariance so the high-dim
            # visualization can project the Lever-2 margin into PCA space.
            'cov_mu_star_k':    cov_mu_star[k],
            'area_new':         area_new,
            'area_old':         area_old,
            'area_pct_tighter': pct,
            'n_cal_points':     int(len(cal_cluster)),
        }
        print(f"  Cluster {k}: coverage={(1-CONFORMAL_ALPHA):.0%}  "
              f"Γ1/Γ2/Γ∞={eg1:.3f}/{eg2:.3f}/{eginf:.3f}  "
              f"δ_k={delta_k:.4f}  area {pct:.1f}% tighter\n")

    # BUG-2 fix: OUTSIDE the loop
    t_clustering_end = time.time()
    actual_clustering_runtime = t_clustering_end - t_clustering_start

    # BUG-4 fix: single save, after params are complete
    joblib.dump({'model': master_model, 'scaler': scaler}, output_model)
    print(f"  Model saved to '{output_model}'.")

    # Step 7.5: Silhouette -----------------------------------------------------
    print("\n--- STEP 7.5: Silhouette Score ---")
    final_labels = master_model.predict(cleaned_X_scaled)
    sil_score = report_silhouette(cleaned_X_scaled, final_labels)

    # Step 8: Feature importance -----------------------------------------------
    print("\n--- STEP 8: Feature Importance ---")
    importance_df = calculate_feature_importance(
        cleaned_X_scaled, final_labels, original_columns)

    # Step 9: Export CSV -------------------------------------------------------
    print("\n--- STEP 9: Exporting CSV ---")
    final_df = pd.DataFrame(scaler.inverse_transform(cleaned_X_scaled),
                            columns=original_columns)
    final_df['Cluster_Label'] = final_labels

    unc_scores = np.zeros(len(cleaned_X_scaled))
    for k, params in uncertainty_set_params.items():
        mask = final_labels == k
        if not mask.any():
            continue
        mu_k = params['mu_star']
        tau_k = params['tau_k']
        mb = max(float(params.get('Gamma_inf', 1.0)), 1e-9)
        pts = cleaned_X_scaled[mask]
        diffs = (pts - mu_k) @ tau_k.T             # BUG-E: vectorised
        unc_scores[mask] = np.minimum(
            np.linalg.norm(diffs, ord=np.inf, axis=1) / mb, 1.0)

    final_df['Uncertainty_Score'] = unc_scores
    final_df.to_csv(output_csv, index=False)
    print(f"  Exported to '{output_csv}'.")

    # Step 10: Visualization ---------------------------------------------------
    plot_uncertainty_sets(cleaned_X_scaled, outliers_X_scaled,
                          final_labels, uncertainty_set_params,
                          scaler, original_columns, output_img=output_png)

    print(f"\nTotal Clustering Runtime: {actual_clustering_runtime:.3f} s")

    # Step 11: DOCX ------------------------------------------------------------
    print("\n--- STEP 11: DOCX Report ---")
    export_docx_report(uncertainty_set_params, importance_df, sil_score,
                       final_labels, original_columns, scaler, output_docx)
    # --- Auto-label clusters by category + profile, then stamp into the report ---
    try:
        from cluster_labeler import load_and_align, build_labels, augment_docx
        merged, feats = load_and_align(
            input_filename, output_csv)   # original + clustered CSV
        labels, primary, _ = build_labels(merged, feats, 'Cluster_Label')
        augment_docx(output_docx, output_docx, labels,
                     primary)      # in-place augmentation
        print(f"  Cluster labels added (primary category column: {primary}).")
    except Exception as e:
        print(f"  [WARN] cluster labeling skipped: {e}")
