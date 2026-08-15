from __future__ import annotations

from typing import Any

import numpy as np

from mas_collapse.data.threads import Thread, linearize_comments
from mas_collapse.embed.backend import Embedder, _l2_normalize
from mas_collapse.metrics.semantic import compute_all_metrics


def _window_texts(bodies: list[str], window_size: int) -> list[str]:
    if not bodies:
        return []
    if len(bodies) < window_size:
        return ["\n\n".join(bodies)]
    return ["\n\n".join(bodies[i : i + window_size]) for i in range(0, len(bodies), window_size)]


def _mean_pool_windows(comment_embs: np.ndarray, window_size: int) -> np.ndarray:
    n = len(comment_embs)
    if n == 0:
        raise ValueError("no comments to pool")
    if n < window_size:
        return _l2_normalize(comment_embs.mean(axis=0, keepdims=True))
    means = [
        comment_embs[i : i + window_size].mean(axis=0)
        for i in range(0, n, window_size)
    ]
    return _l2_normalize(np.stack(means))


def score_comment_sequence(
    bodies: list[str],
    embedder: Embedder,
    *,
    window_size: int = 10,
    late_windows: int = 3,
    tau: float | None = None,
    vendi_sample_size: int = 30,
    vendi_repeats: int = 20,
    seed: int = 42,
    compute_vendi: bool = True,
    window_pool: str = "mean_comments",
) -> dict[str, Any]:
    windows = _window_texts(bodies, window_size)
    if not windows:
        raise ValueError("no windows to score")

    utterance_embs_per_window = None
    if window_pool == "mean_comments":
        comment_embs = embedder.embed(bodies)
        window_embs = _mean_pool_windows(comment_embs, window_size)
        if compute_vendi:
            utterance_embs_per_window = []
            n = len(comment_embs)
            step = window_size if n >= window_size else n
            for i in range(0, n, step):
                utterance_embs_per_window.append(comment_embs[i : i + step])
    elif window_pool == "concat":
        window_embs = embedder.embed(windows)
        if compute_vendi:
            utterance_embs_per_window = []
            for i in range(0, len(bodies), window_size):
                chunk = bodies[i : i + window_size]
                if chunk:
                    utterance_embs_per_window.append(embedder.embed(chunk))
    else:
        raise ValueError(f"Unknown window_pool: {window_pool}")

    result = compute_all_metrics(
        window_embs,
        windows,
        late_windows=late_windows,
        tau=tau,
        utterance_embs_per_window=utterance_embs_per_window,
        vendi_sample_size=vendi_sample_size,
        vendi_repeats=vendi_repeats,
        seed=seed,
    )
    out = result.to_dict()
    out["window_pool"] = window_pool
    return out


def score_thread(
    thread: Thread,
    embedder: Embedder,
    cfg: dict[str, Any],
    *,
    compute_vendi: bool = True,
) -> dict[str, Any]:
    comments = linearize_comments(thread)
    bodies = [c.body for c in comments]
    metrics = score_comment_sequence(
        bodies,
        embedder,
        window_size=int(cfg["windowing"]["window_size"]),
        late_windows=int(cfg["windowing"]["late_windows"]),
        tau=cfg["metrics"].get("tau"),
        vendi_sample_size=int(cfg["metrics"].get("vendi_sample_size", 30)),
        vendi_repeats=int(cfg["metrics"].get("vendi_repeats", 20)),
        seed=int(cfg["project"].get("seed", 42)),
        compute_vendi=compute_vendi,
        window_pool=str(cfg.get("windowing", {}).get("window_pool", "mean_comments")),
    )
    return {
        "post_id": thread.post_id,
        "subreddit": thread.subreddit,
        "title": thread.title,
        "n_comments": len(comments),
        "embedder": embedder.name,
        **metrics,
    }


def assign_labels(
    rows: list[dict[str, Any]],
    high_q: float = 0.70,
    low_q: float = 0.30,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    s_vals = np.asarray([r["s_end"] for r in rows], dtype=np.float64)
    hi = float(np.quantile(s_vals, high_q))
    lo = float(np.quantile(s_vals, low_q))
    out = []
    for r in rows:
        s = r["s_end"]
        if s >= hi:
            label = "converge"
        elif s <= lo:
            label = "non_converge"
        else:
            label = "ambiguous"
        rr = dict(r)
        rr["label_A"] = label
        rr["threshold_high"] = hi
        rr["threshold_low"] = lo
        out.append(rr)
    return out, {"high": hi, "low": lo, "tau_suggest": hi}
