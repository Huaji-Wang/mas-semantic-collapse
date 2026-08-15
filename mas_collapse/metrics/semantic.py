from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class MetricResult:
    s_end: float
    slope: float
    t_tau: float | None
    vendi_mean: float | None
    vendi_first: float | None
    vendi_last: float | None
    lexical_unique_total: int
    lexical_growth: float
    n_windows: int
    sims_to_first: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def window_similarities_to_first(window_embs: np.ndarray) -> list[float]:
    if len(window_embs) == 0:
        return []
    first = window_embs[0]
    return [cosine(first, w) for w in window_embs]


def average_converge_distance(sims: Sequence[float]) -> float:
    """Mean cosine distance from window 1, excluding the first window itself."""
    if len(sims) < 2:
        return 0.0
    return float(np.mean([1.0 - float(s) for s in sims[1:]]))


def participation_ratio(embs: np.ndarray) -> float:
    """Effective dimensionality via eigenvalue participation ratio of the cosine kernel."""
    n = len(embs)
    if n < 2:
        return 1.0
    k = embs @ embs.T
    k = np.clip(k, -1.0, 1.0)
    evals = np.linalg.eigvalsh(k)
    evals = np.clip(evals, 0.0, None)
    s1 = float(evals.sum())
    s2 = float(np.square(evals).sum())
    if s2 <= 1e-12:
        return 0.0
    return float((s1 * s1) / s2)


def late_mean(values: Sequence[float], late_windows: int = 3) -> float:
    if not values:
        return float("nan")
    k = min(late_windows, len(values))
    return float(np.mean(values[-k:]))


def s_end_from_sims(sims: Sequence[float], late_windows: int = 3) -> float:
    if not sims:
        return float("nan")
    k = min(late_windows, len(sims))
    return float(np.mean(sims[-k:]))


def slope_from_sims(sims: Sequence[float]) -> float:
    if len(sims) < 2:
        return 0.0
    x = np.arange(len(sims), dtype=np.float64)
    y = np.asarray(sims, dtype=np.float64)
    # positive slope => becoming more similar to first window => converging
    return float(np.polyfit(x, y, 1)[0])


def time_to_threshold(sims: Sequence[float], tau: float) -> float | None:
    """First window index where sim >= tau. None if never."""
    for i, s in enumerate(sims):
        if s >= tau:
            return float(i)
    return None


def lexical_stats(window_texts: Sequence[str]) -> tuple[int, float]:
    seen: set[str] = set()
    counts: list[int] = []
    for text in window_texts:
        toks = [t.lower() for t in text.split() if t.strip()]
        seen.update(toks)
        counts.append(len(seen))
    if not counts:
        return 0, 0.0
    growth = float(counts[-1] - counts[0]) if len(counts) > 1 else float(counts[0])
    return int(counts[-1]), growth


def _vendi_from_embeddings(embs: np.ndarray) -> float:
    """Normalized Vendi-like score from eigenvalue entropy of cosine kernel."""
    n = len(embs)
    if n < 2:
        return 0.0
    # cosine kernel since embeddings are L2-normalized
    k = embs @ embs.T
    k = np.clip(k, -1.0, 1.0)
    # trace-normalize
    tr = float(np.trace(k))
    if tr <= 1e-12:
        return 0.0
    k = k / tr
    evals = np.linalg.eigvalsh(k)
    evals = np.clip(evals, 0.0, None)
    s = evals.sum()
    if s <= 1e-12:
        return 0.0
    p = evals / s
    p = p[p > 1e-12]
    entropy = float(-(p * np.log(p)).sum())
    # effective support = exp(entropy); normalize by n for [0,1]-ish scale
    eff = float(np.exp(entropy))
    return eff / n


def vendi_trajectory(
    utterance_embs_per_window: Sequence[np.ndarray],
    sample_size: int = 30,
    repeats: int = 20,
    rng: np.random.Generator | None = None,
) -> list[float]:
    rng = rng or np.random.default_rng(0)
    scores: list[float] = []
    for embs in utterance_embs_per_window:
        if len(embs) == 0:
            scores.append(0.0)
            continue
        vals = []
        m = min(sample_size, len(embs))
        for _ in range(repeats):
            idx = rng.choice(len(embs), size=m, replace=False) if len(embs) >= m else np.arange(len(embs))
            vals.append(_vendi_from_embeddings(embs[idx]))
        scores.append(float(np.mean(vals)))
    return scores


def compute_all_metrics(
    window_embs: np.ndarray,
    window_texts: Sequence[str],
    *,
    late_windows: int = 3,
    tau: float | None = None,
    utterance_embs_per_window: Sequence[np.ndarray] | None = None,
    vendi_sample_size: int = 30,
    vendi_repeats: int = 20,
    seed: int = 42,
) -> MetricResult:
    sims = window_similarities_to_first(window_embs)
    s_end = s_end_from_sims(sims, late_windows=late_windows)
    slope = slope_from_sims(sims)
    t_tau = time_to_threshold(sims, tau) if tau is not None else None
    lex_total, lex_growth = lexical_stats(window_texts)

    vendi_mean = vendi_first = vendi_last = None
    if utterance_embs_per_window is not None and len(utterance_embs_per_window) > 0:
        vt = vendi_trajectory(
            utterance_embs_per_window,
            sample_size=vendi_sample_size,
            repeats=vendi_repeats,
            rng=np.random.default_rng(seed),
        )
        vendi_mean = float(np.mean(vt)) if vt else None
        vendi_first = float(vt[0]) if vt else None
        vendi_last = float(vt[-1]) if vt else None

    return MetricResult(
        s_end=s_end,
        slope=slope,
        t_tau=t_tau,
        vendi_mean=vendi_mean,
        vendi_first=vendi_first,
        vendi_last=vendi_last,
        lexical_unique_total=lex_total,
        lexical_growth=lex_growth,
        n_windows=len(sims),
        sims_to_first=sims,
    )
