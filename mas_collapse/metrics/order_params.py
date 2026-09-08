"""Five order parameters for a discussion cloud: m, sigma, d, k, chi.

The time grid and the estimation support are deliberately decoupled:

* ``m`` (cloud center) is computed on non-overlapping blocks of ``window_size``
  comments, so the resulting radial curve stays comparable with the existing
  ``sims_to_first`` trajectories.
* ``sigma`` / ``d`` / ``k`` are computed on a wider support window of
  ``support_size`` comments centred on the same block, because ten points are
  too few to estimate dispersion, effective dimension or mode count.
* ``chi`` is a semantic state, plotted against time. It does not use author
  identity. Official chi is the polar angle of the window center in this
  thread's 2-D semantic plane: ``0.5*(1-cos Δθ)`` versus the first window.
  A loop in high-D becomes a wave on the time axis; a freeze stays near 0.
  If the center sits on the origin, chi falls back to the window's principal
  axis (internal swap). The old 1024-D Hungarian cost is ``churn_config``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

from mas_collapse.metrics.semantic import participation_ratio


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def cloud_center(embs: np.ndarray) -> np.ndarray:
    """m: L2-normalized mean of the window's comment vectors."""
    return _unit(np.asarray(embs, dtype=np.float64).mean(axis=0))


def dispersion(embs: np.ndarray, center: np.ndarray | None = None) -> float:
    """sigma: mean cosine distance from each comment to the cloud center."""
    e = np.asarray(embs, dtype=np.float64)
    if len(e) < 2:
        return 0.0
    c = cloud_center(e) if center is None else center
    sims = np.clip(e @ np.asarray(c, dtype=np.float64), -1.0, 1.0)
    return float(np.mean(1.0 - sims))


def effective_dim(embs: np.ndarray) -> float:
    """d: eigenvalue participation ratio of the cosine kernel."""
    return participation_ratio(np.asarray(embs, dtype=np.float64))


def _self_tuning_spectrum(embs: np.ndarray, n_neighbors: int = 7):
    """Cosine kernel + Zelnik-Manor self-tuning affinity + normalized Laplacian.

    A dense ``(cos+1)/2`` affinity makes the graph fully connected, in which case
    the eigengap always reports k=1, so the local scale (distance to the
    ``n_neighbors``-th neighbour) is used instead.
    """
    e = np.asarray(embs, dtype=np.float64)
    n = len(e)
    sim = np.clip(e @ e.T, -1.0, 1.0)
    dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sim))
    np.fill_diagonal(dist, 0.0)
    kk = min(n_neighbors, n - 1)
    scale = np.maximum(np.sort(dist, axis=1)[:, kk], 1e-6)
    aff = np.exp(-(dist**2) / (scale[:, None] * scale[None, :]))
    np.fill_diagonal(aff, 0.0)
    deg = aff.sum(axis=1)
    if np.any(deg <= 1e-12):
        return sim, None, None
    dinv = 1.0 / np.sqrt(deg)
    lsym = np.eye(n) - (aff * dinv[:, None]) * dinv[None, :]
    evals, evecs = np.linalg.eigh(lsym)
    return sim, np.clip(evals, 0.0, None), evecs


def _silhouette_cosine(sim: np.ndarray, labels: np.ndarray) -> float:
    dist = 1.0 - sim
    n = len(labels)
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        same = labels == labels[i]
        other = ~same
        n_same = int(same.sum())
        if n_same <= 1 or not other.any():
            scores[i] = 0.0
            continue
        a = (dist[i, same].sum()) / (n_same - 1)
        b = float(dist[i, other].mean())
        denom = max(a, b)
        scores[i] = 0.0 if denom <= 1e-12 else (b - a) / denom
    return float(scores.mean())


def mode_count(embs: np.ndarray, max_k: int = 5, n_neighbors: int = 7) -> tuple[int, float]:
    """Legacy k: eigengap on the self-tuning Laplacian. Saturates at 1 on Reddit."""
    if len(embs) < 6:
        return 1, 0.0
    _, evals, _ = _self_tuning_spectrum(embs, n_neighbors=n_neighbors)
    if evals is None:
        return 1, 0.0
    top = min(max_k + 1, len(embs) - 1)
    gaps = np.diff(evals[: top + 1])
    if gaps.size == 0:
        return 1, 0.0
    idx = int(np.argmax(gaps))
    return idx + 1, float(gaps[idx])


def range_cluster_count(
    embs: np.ndarray,
    *,
    distance_threshold: float = 0.55,
    min_size: int = 2,
    max_k: int = 5,
    method: str = "average",
) -> tuple[int, np.ndarray]:
    """Count semantic camps by cutting a cosine-distance dendrogram.

    Two comments fall in the same cluster if their average-linkage cosine
    distance stays below ``distance_threshold`` (default 0.55 ≈ cosine ≥ 0.45).
    Isolated points are not camps. Returns ``(k, labels)`` with labels in
    ``0..k_raw-1`` (singletons included in labels, excluded from ``k``).
    """
    e = np.asarray(embs, dtype=np.float64)
    n = len(e)
    labels = np.zeros(n, dtype=int)
    if n < 2:
        return 1, labels
    sim = np.clip(e @ e.T, -1.0, 1.0)
    dist = np.maximum(0.0, 1.0 - sim)
    np.fill_diagonal(dist, 0.0)
    if n == 2:
        same = dist[0, 1] <= distance_threshold
        if same:
            return 1, labels
        labels[1] = 1
        return 1, labels
    try:
        condensed = squareform(dist, checks=False)
        z = linkage(condensed, method=method)
    except ValueError:
        return 1, labels
    raw = fcluster(z, t=distance_threshold, criterion="distance")
    labels = raw - raw.min()
    sizes = np.bincount(labels)
    n_camps = int((sizes >= min_size).sum())
    if n_camps <= 0:
        return 1, labels
    return min(n_camps, max_k), labels


def bimodality(embs: np.ndarray, n_neighbors: int = 7) -> tuple[float, float]:
    """Continuous stand-in for k: how cleanly the cloud splits into two lumps.

    The split is the sign of the Fiedler vector of the same self-tuning Laplacian
    used by :func:`mode_count`. Returns ``(silhouette, balance)`` under cosine
    distance: silhouette near 0 means one blob, higher means two separated lumps;
    balance is the fraction of points in the smaller lump, so a 1-vs-29 split can
    be discounted rather than read as a real fission.
    """
    if len(embs) < 6:
        return float("nan"), float("nan")
    sim, evals, evecs = _self_tuning_spectrum(embs, n_neighbors=n_neighbors)
    if evals is None or evecs is None or evecs.shape[1] < 2:
        return 0.0, 0.0
    fiedler = evecs[:, 1]
    labels = (fiedler > 0).astype(int)
    counts = np.bincount(labels, minlength=2)
    if counts.min() == 0:
        labels = (fiedler > np.median(fiedler)).astype(int)
        counts = np.bincount(labels, minlength=2)
        if counts.min() == 0:
            return 0.0, 0.0
    return _silhouette_cosine(sim, labels), float(counts.min() / len(embs))


def _centered_unit(embs: np.ndarray) -> np.ndarray:
    e = np.asarray(embs, dtype=np.float64)
    v = e - e.mean(axis=0, keepdims=True)
    norms = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return v / norms


def churn_shape(prev: np.ndarray, cur: np.ndarray) -> float:
    """Legacy 1024-D Hungarian matching. Saturates on distinct sentences.

    Kept only so new chi can be compared with the old series. Do not use as
    the official chi: it cannot wave or freeze on real Reddit embeddings.
    """
    if len(prev) < 2 or len(cur) < 2:
        return float("nan")
    p = _centered_unit(prev)
    c = _centered_unit(cur)
    cost = 1.0 - np.clip(p @ c.T, -1.0, 1.0)
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())


def _thread_pca_2d(embs: np.ndarray) -> np.ndarray:
    """Project comments onto this thread's first two principal directions.

    Used only to define semantic *places* around the cloud center. The chi
    figure itself stays time vs chi — this plane is not plotted.
    """
    e = np.asarray(embs, dtype=np.float64)
    e = e - e.mean(axis=0, keepdims=True)
    if len(e) < 2:
        return np.zeros((len(e), 2), dtype=np.float64)
    _, s, vt = np.linalg.svd(e, full_matrices=False)
    axes = vt[:2]
    if axes.shape[0] < 2:
        pad = np.zeros((2 - axes.shape[0], e.shape[1]), dtype=np.float64)
        axes = np.vstack([axes, pad])
    if s.size >= 2 and s[1] <= 1e-12:
        extra = np.zeros((len(e), 1), dtype=np.float64)
        return np.hstack([e @ axes[:1].T, extra])
    return e @ axes.T


def _axis_angle(pts2d: np.ndarray) -> float | None:
    p = np.asarray(pts2d, dtype=np.float64)
    if len(p) < 3:
        return None
    p = p - p.mean(axis=0, keepdims=True)
    cov = np.atleast_2d(np.cov(p.T))
    if cov.shape != (2, 2):
        return None
    evals, evecs = np.linalg.eigh(cov)
    if evals[-1] <= 1e-12:
        return None
    ratio = float(evals[-1] / max(evals[-2], 1e-12))
    if ratio < 2.5:
        return None
    vx, vy = evecs[:, -1]
    return float(np.arctan2(vy, vx))


def _axis_chi(a: float | None, b: float | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 0.5
    delta = abs(a - b) % np.pi
    delta = min(delta, np.pi - delta)
    return float(delta / (np.pi / 2.0))


def _polar_chi(mu_t: np.ndarray, mu_ref: np.ndarray, min_r: float) -> float | None:
    rt = float(np.linalg.norm(mu_t))
    rr = float(np.linalg.norm(mu_ref))
    if rt < min_r or rr < min_r:
        return None
    ang = np.arctan2(mu_t[1], mu_t[0]) - np.arctan2(mu_ref[1], mu_ref[0])
    return float(0.5 * (1.0 - np.cos(ang)))


def semantic_chi_series(
    comment_embs: np.ndarray,
    blocks_index: list[tuple[int, int]],
    n_bins: int = 6,
) -> tuple[list[float], list[float]]:
    """chi vs first window, and adjacent-step chi.

    Polar angle of the window center in this thread's 2-D semantic plane
    (0.5 * (1 - cos Δθ)). A loop around the origin becomes a wave on the
    time axis. If the center sits on the origin, fall back to the principal
    axis of the window (internal swap) or 0 if the cloud is round.
    """
    del n_bins
    proj = _thread_pca_2d(comment_embs)
    if len(proj) == 0:
        return [], []
    mus = [proj[a:b].mean(axis=0) for a, b in blocks_index]
    axes = [_axis_angle(proj[a:b]) for a, b in blocks_index]
    high_d = [cloud_center(comment_embs[a:b]) for a, b in blocks_index]
    spreads = []
    for a, b in blocks_index:
        pts = proj[a:b]
        spreads.append(float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean() + 1e-9))
    if not mus:
        return [], []

    def _at_origin(i: int) -> bool:
        return float(np.linalg.norm(mus[i])) < 2.0 * spreads[i]

    def _same_high_d(i: int, j: int) -> bool:
        return float(np.clip(np.dot(high_d[i], high_d[j]), -1.0, 1.0)) >= 0.92

    def pair(i: int, j: int) -> float:
        if _same_high_d(i, j):
            return 0.0
        if _at_origin(i) and _at_origin(j):
            if axes[i] is None or axes[j] is None:
                return 0.0
            return _axis_chi(axes[i], axes[j])
        polar = _polar_chi(mus[i], mus[j], min(2.0 * spreads[i], 2.0 * spreads[j]))
        if polar is not None:
            return polar
        if axes[i] is None or axes[j] is None:
            return 0.0
        return _axis_chi(axes[i], axes[j])

    chi = [0.0]
    chi_step = [0.0]
    for i in range(1, len(mus)):
        chi.append(pair(i, 0))
        chi_step.append(pair(i, i - 1))
    return chi, chi_step


def extract_order_params(
    comment_embs: np.ndarray,
    *,
    window_size: int = 10,
    support_size: int = 30,
    max_k: int = 5,
    cluster_distance: float = 0.55,
) -> dict[str, Any]:
    e = np.asarray(comment_embs, dtype=np.float32)
    n = len(e)
    if n == 0:
        raise ValueError("no comment embeddings")

    cuts = [(i, min(n, i + window_size)) for i in range(0, n, window_size)]
    cuts = [(a, b) for a, b in cuts if b > a]
    blocks = [e[a:b] for a, b in cuts]
    centers = np.stack([cloud_center(b) for b in blocks])
    t = len(blocks)

    sims_to_first = [float(np.clip(np.dot(centers[0], c), -1.0, 1.0)) for c in centers]
    adj_sims: list[float] = [1.0]
    churn_cent: list[float] = [0.0]
    churn_cfg: list[float] = [float("nan")]
    for i in range(1, t):
        s = float(np.clip(np.dot(centers[i - 1], centers[i]), -1.0, 1.0))
        adj_sims.append(s)
        churn_cent.append(1.0 - s)
        churn_cfg.append(churn_shape(blocks[i - 1], blocks[i]))

    chi, chi_step = semantic_chi_series(e, cuts)

    half = max(0, (support_size - window_size) // 2)
    sigma: list[float] = []
    eff: list[float] = []
    kmodes: list[int] = []
    keigen: list[int] = []
    kgap: list[float] = []
    bimod: list[float] = []
    balance: list[float] = []
    n_support: list[int] = []
    for i in range(t):
        end = min(n, i * window_size - half + support_size)
        start = max(0, end - support_size)
        sup = e[start:end]
        n_support.append(int(len(sup)))
        sigma.append(dispersion(sup))
        eff.append(effective_dim(sup))
        k_c, _ = range_cluster_count(sup, distance_threshold=cluster_distance, max_k=max_k)
        kmodes.append(k_c)
        if len(sup) < 6:
            keigen.append(1)
            kgap.append(0.0)
            bimod.append(float("nan"))
            balance.append(float("nan"))
            continue
        sim, evals, evecs = _self_tuning_spectrum(sup)
        if evals is None or evecs is None:
            keigen.append(1)
            kgap.append(0.0)
            bimod.append(float("nan"))
            balance.append(float("nan"))
            continue
        top = min(max_k + 1, len(sup) - 1)
        gaps = np.diff(evals[: top + 1])
        idx = int(np.argmax(gaps)) if gaps.size else 0
        keigen.append(idx + 1)
        kgap.append(float(gaps[idx]) if gaps.size else 0.0)
        fiedler = evecs[:, 1]
        labels = (fiedler > 0).astype(int)
        if np.bincount(labels, minlength=2).min() == 0:
            labels = (fiedler > np.median(fiedler)).astype(int)
        counts = np.bincount(labels, minlength=2)
        if counts.min() == 0:
            bimod.append(0.0)
            balance.append(0.0)
        else:
            bimod.append(_silhouette_cosine(sim, labels))
            balance.append(float(counts.min() / len(sup)))

    return {
        "n_comments_used": int(n),
        "n_windows": int(t),
        "window_size": int(window_size),
        "support_size": int(support_size),
        "sims_to_first": sims_to_first,
        "adj_sims": adj_sims,
        "sigma": sigma,
        "eff_dim": eff,
        "k_modes": kmodes,
        "k_eigengap": keigen,
        "k_cluster_distance": float(cluster_distance),
        "k_gap": kgap,
        "bimodality": bimod,
        "mode_balance": balance,
        "churn_centroid": churn_cent,
        "churn_config": churn_cfg,
        "chi": chi,
        "chi_step": chi_step,
        "n_support": n_support,
    }


def summarize_order_params(traj: dict[str, Any], late_windows: int = 3) -> dict[str, Any]:
    """Per-thread scalars: value at the start, at the end, and the end-minus-start delta."""

    def head(key: str) -> float:
        vals = [v for v in traj[key][:1] if v is not None and np.isfinite(v)]
        return float(vals[0]) if vals else float("nan")

    def tail(key: str) -> float:
        vals = [v for v in traj[key][-late_windows:] if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    out: dict[str, Any] = {
        "s_end": tail("sims_to_first"),
        "adj_sim_late": tail("adj_sims"),
        "churn_config_late": tail("churn_config"),
        "chi_late": tail("chi"),
        "chi_step_late": tail("chi_step"),
    }
    for key in ("sigma", "eff_dim", "k_modes", "bimodality", "chi"):
        first, last = head(key), tail(key)
        out[f"{key}_first"] = first
        out[f"{key}_late"] = last
        out[f"{key}_delta"] = float(last - first) if np.isfinite(first) and np.isfinite(last) else float("nan")
    mid = [v for v in traj["churn_config"] if v is not None and np.isfinite(v)]
    out["churn_config_mean"] = float(np.mean(mid)) if mid else float("nan")
    chi_vals = [v for v in traj.get("chi", []) if v is not None and np.isfinite(v)]
    step_vals = [v for v in traj.get("chi_step", [])[1:] if v is not None and np.isfinite(v)]
    out["chi_mean"] = float(np.mean(chi_vals)) if chi_vals else float("nan")
    out["chi_range"] = float(np.max(chi_vals) - np.min(chi_vals)) if chi_vals else float("nan")
    out["chi_step_mean"] = float(np.mean(step_vals)) if step_vals else float("nan")
    out["adj_sim_mean"] = float(np.mean(traj["adj_sims"][1:])) if len(traj["adj_sims"]) > 1 else float("nan")
    return out
