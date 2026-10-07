# Motion-mixture-v1: operational scoring standard

## Scope and status

Each thread-side trajectory is represented by ten nonnegative weights summing to one. These are **normalized heuristic compatibility scores**, not calibrated class probabilities, fractions of comments, counts of people, or a learned mixture decomposition. All ten components remain present; zero raw evidence does not imply a zero softmax weight. No claim is made that these ten motions exhaust the possible dynamics.

The geometric vocabulary is an interpretive framework for semantic change. Five measured descriptors supply evidence; hand-specified rules map that evidence to familiar motion patterns. Coefficients, shape thresholds, bounds and temperature are methodological choices, not an externally validated classification standard. The implementation retains the nine existing motion rules and adds dimension-decline evidence for filamentation. The report and default pipeline share the same scoring function in `scripts/score_motion_mixtures.py`; feature extraction and the first nine rules are in `scripts/classify_thread_motions.py`.

## 1. Measurement

Comments are embedded with one fixed encoder. In the centred protocol, subtract the same human-corpus mean vector from every human and simulated comment vector, then normalize each vector to unit length. Ten consecutive comments define a time block. Sigma, dimension and modes use a wider support of 30 comments around that block; boundary support can be smaller. The final partial block is retained.

| Descriptor | Operational interpretation |
|---|---|
| m | Unit-normalized mean comment vector per block. Its cosine similarity to the first block gives `sims_to_first`; leaving the initial position is `1 - s_end`. |
| sigma | Dispersion around the support cloud centre (`sigma`). |
| d | Effective dimensionality, estimated by the participation ratio (`eff_dim`). |
| k | Number of nonsingleton clusters from average-linkage cosine-distance clustering at cutoff 0.55 (`k_modes`). |
| chi | Semantic angular change relative to the first block, `0.5 * (1 - cos(delta theta))`, in the trajectory's two-dimensional semantic plane; `chi_step` measures adjacent-block change. Degenerate cases follow the fallbacks in `mas_collapse/metrics/order_params.py`. |

The two sides have separate fitted semantic planes, so chi compares within-trajectory dynamics rather than a shared absolute angle. These descriptors do not identify a participant's stance or psychological state. Thread-level results must not be described as person counts.

## 2. Feature extraction and reference scales

For sigma, d, k and chi, compute delta as the mean of the last `min(3, n)` windows minus the mean of the first `max(1, floor(n/3))` windows. Early and late intervals can overlap for short trajectories. Also compute `m_leave = 1 - s_end`, the mean of `chi_step[1:]`, and the standard deviation of the last three chi values.

Fit seven scales from **human reference trajectories only**: delta sigma, delta d, delta k, delta chi, m_leave, mean chi_step and late chi standard deviation. Each uses `1.4826 * median(abs(x - median(x)))`. If this is at most 1e-6, use `max(population_std(x), 0.001)`; fewer than four reference rows give scale 1. The k scale has a floor of 0.35. The same scales are applied to both sides. The original report fitted the pooled 311 human trajectories; a new cohort defaults to its own human reference unless a frozen scale file is supplied.

For delta x and its scale a, define:

- `U(x) = sigmoid(delta x / a)`; `D(x) = sigmoid(-delta x / a)`.
- `S(x) = exp(-abs(delta x) / a)`.
- `M = clip(m_leave / (2 * a_m + 1e-9), 0, 1)` and `P = 1 - M`.
- `F = exp(-abs(mean chi_step) / a_step) * exp(-abs(late chi std) / a_chi_std)` and `L = 1 - F`.

U and D both equal 0.5 when delta is zero. They are smooth directional compatibility terms, not tests establishing an increase or decrease.

Shape evidence requires at least four windows:

- `Wm`: leave-and-return in initial-position similarity. The minimum must be interior and the initial drop at least 0.04; score is `clip(recovery/drop * clip(drop/0.12, 0, 1), 0, 1)`.
- `Wchi`: an interior chi peak at least 0.12; score is `clip((peak - mean(first,last))/peak, 0, 1)`.
- `Wsigma`: the same peak rule applied to `sigma - min(sigma)`.
- `Jm` and `Jchi`: for `1-similarity` and chi, respectively, take the largest absolute adjacent jump's share of total absolute change, multiplied by `exp(-std(tail)/(largest_jump+1e-9))`. A one-point tail has flatness 1; a constant curve scores 0.

With fewer than four windows all shape terms are 0 and `shape_features_resolved=false`. Positive orbit/breathing/cascade weights can still occur because of baseline terms and softmax. Short trajectories therefore do not establish the corresponding dynamic shapes.

## 3. Ten raw rules and fixed bounds

Subscripts name descriptors; U, D and S refer to their reference-scaled deltas.

| Motion | Raw compatibility score r | Fixed [lower, upper] |
|---|---|---|
| Translation | 1.4M + 0.6S_sigma + 0.4S_d + 0.4S_k + 0.8S_chi - 0.8Wm | [-0.8, 3.6] |
| Diffusion | 1.2U_sigma + U_chi + 0.4P + 0.3S_k | [0, 2.9] |
| Condensation | 1.2D_sigma + D_chi + 0.4P | [0, 2.6] |
| Fission | U_sigma + 1.4U_k + 0.8D_chi + 0.3P | [0, 3.5] |
| Crystallization | 1.6F + 0.5S_sigma + 0.4S_k + 0.4P - 0.6L | [-0.6, 2.9] |
| Vortex | 1.4L + 0.5P + 0.4S_sigma + 0.4S_k - 0.7Wchi - 0.5F | [-1.2, 2.7] |
| Orbit | 1.3Wm + 1.3Wchi + 0.3S_sigma | [0, 2.9] |
| Breathing | 1.2Wsigma + 1.2Wchi + 0.3P | [0, 2.7] |
| Cascade | 1.3Jm + 1.1Jchi + 0.3S_sigma | [0, 2.7] |
| Filamentation | Q * (2 + 0.5S_sigma + 0.5S_k), where Q = 1 - exp(-max(0, -delta d)/a_d) | [0, 3] |

These conservative algebraic bounds are fixed; they are not cohort minima or maxima. Filamentation has zero raw evidence unless effective dimension declines.

## 4. Normalize and aggregate

For each motion j, transform `c_j = (r_j - lower_j)/(upper_j - lower_j)`. Then

```text
w_j = exp((c_j - max(c))/T) / sum_l exp((c_l - max(c))/T)
T = 0.25
```

Every weight is nonnegative and the ten weights sum to one. Temperature controls concentration; it does not calibrate correctness. Preserve temperature and scales when comparing arms. The optional `dominant_motion` is argmax(w), but the default analysis never replaces w by this label.

For N trajectories on one side, the mean vector is `sum_i w_i / N`. The fractional thread-equivalent mass for motion j is `sum_i w_ij`; the ten masses sum to N. Within a topic, paired differences are `100 * mean_i(w_sim,i - w_human,i)` percentage points. Mean total variation is `mean_i(sum_j abs(w_sim,ij - w_human,ij) / 2)`. Each thread has equal weight regardless of its comment count.

Scorer outputs retain raw scores, normalized compatibilities, all weights, normalized entropy, short-trajectory flags, temperature, and method version. The separate summary records fitted scales, fixed bounds and source-file SHA-256 hashes. Replay measurements record source/settings fingerprints. A frozen scales file must come from the same embedding and measurement protocol; supplying one is an explicit choice, not automatic proof of comparability.

## 5. Validation and limits

Portable regression tests:

```bash
python -m unittest discover -s tests -p "test_motion_weights.py"
```

Tests cover weight conservation, human-only scale fitting, paired aggregation, malformed/unpaired input, shared topic assignment, replay alignment, and default pipeline routing. Hashing embeddings are deterministic smoke-test fixtures and must not support scientific findings.

For the local report cohort, the generic scorer can be checked against all 622 saved trajectory weight vectors in `docs/report_motion_mixtures.json`. That local artifact contains derived research material and is not required for portable tests. The new full-scale run has no numerical results until its measurement stage has actually completed.

Further research validation should include temperature/reference-scale sensitivity, four-or-more-window analyses, independent review of motion and topic interpretations, and paired uncertainty estimates. The present pipeline reports descriptive differences, without claiming significance, causal effects, calibrated probabilities or independent empirical validation of the motion taxonomy.
