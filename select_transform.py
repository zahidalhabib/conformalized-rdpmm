"""
select_transform.py -- automatically choose StandardScaler vs Yeo-Johnson.

Implements the conditional rule "de-skew only when it actually helps" by letting
the data decide: fit the DPMM under each transform and keep whichever yields the
better clustering. This reproduces the observed behaviour automatically --
- already-separable data: Yeo-Johnson flattens the multimodal gaps, the DPMM
  over-segments, its score drops -> StandardScaler is kept;
- skewed/overlapping data: de-skewing exposes structure, its score rises
  -> Yeo-Johnson is kept.

Returns a FITTED scaler, a drop-in for the model's Step 2.
"""
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score


def select_scaler(X_raw, reference=None, max_eval=20000, n_components=10,
                  low_skew=0.5, random_state=42, verbose=True):
    """
    Parameters
    ----------
    X_raw      : (n,d) feature matrix (post feature-selection, pre-scaling)
    reference  : optional (n,) array of ground-truth labels (e.g. Category Name).
                 If given, transforms are scored by AMI to it -- chance-corrected,
                 so it is fair across differing cluster counts and measures the
                 goal directly. Otherwise silhouette (each transform's own space).
    low_skew   : if mean |skew| is below this, skip the search and use
                 StandardScaler (data not skewed enough to need de-skewing).

    Returns (fitted_scaler, name, scores_dict).
    """
    X_raw = np.asarray(X_raw, dtype=float)
    raw_skew = float(np.abs(skew(X_raw, axis=0, nan_policy="omit")).mean())

    # cheap short-circuit: low skew -> de-skewing only risks flattening structure
    if raw_skew < low_skew:
        if verbose:
            print(f"  [transform] raw |skew|={raw_skew:.2f} < {low_skew} "
                  f"-> StandardScaler (no de-skew needed)")
        return StandardScaler().fit(X_raw), "StandardScaler", {}

    rng = np.random.default_rng(random_state)
    n = len(X_raw)
    idx = rng.choice(n, max_eval, replace=False) if n > max_eval else np.arange(n)
    Xe = X_raw[idx]
    ref = None if reference is None else np.asarray(reference)[idx]

    candidates = {
        "StandardScaler": StandardScaler(),
        "Yeo-Johnson":    PowerTransformer(method="yeo-johnson", standardize=True),
    }
    scores = {}
    for name, sc in candidates.items():
        Xs = sc.fit_transform(Xe)
        lbl = BayesianGaussianMixture(
            n_components=n_components,
            weight_concentration_prior_type="dirichlet_process",
            covariance_type="full", max_iter=300,
            random_state=random_state).fit(Xs).predict(Xs)
        k = len(set(lbl))
        if ref is not None:
            score, metric = adjusted_mutual_info_score(ref.astype(str),
                                                        lbl.astype(str)), "AMI"
        else:
            score = silhouette_score(Xs, lbl) if k > 1 else -1.0
            metric = "silhouette"
        scores[name] = {"score": float(score), "clusters": int(k)}
        if verbose:
            print(f"  [transform] {name:14s} {metric}={score:+.3f}  clusters={k}")

    best = max(scores, key=lambda nm: scores[nm]["score"])
    if verbose:
        print(f"  [transform] raw |skew|={raw_skew:.2f}  ->  selected {best}")
    return candidates[best].fit(X_raw), best, scores


if __name__ == "__main__":
    # quick self-test on the two regimes
    rng = np.random.default_rng(0)
    # Regime A: three clean, well-separated blobs (mild skew) -> expect StandardScaler
    A = np.vstack([rng.normal(m, 0.5, (1200, 3)) for m in (0, 6, 12)])
    # Regime B: skewed, overlapping log-normal -> expect Yeo-Johnson
    B = np.vstack([np.exp(rng.normal(mu, 0.6, (1800, 3))) for mu in (1.0, 1.6)])
    print("Regime A (clean, separable):")
    _, a, _ = select_scaler(A)
    print("Regime B (skewed, overlapping):")
    _, b, _ = select_scaler(B)
    print(f"\nRESULT  A -> {a}   B -> {b}")
