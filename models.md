# Anomaly-detection models

The service implements direct univariate detectors and two-stage multivariate
pipelines. All model inputs are fully preprocessed, primed streams. Direct
univariate models evaluate the newest point in a retained event-time window.
Multivariate models form synchronized vectors, produce a scalar distance or
reconstruction error, and send that scalar through Z-score or MAD for the final
anomaly decision.

## Detrending and deseasonality scope

Holt detrending and sliding-DFT deseasonality are preprocessing stages for
univariate anomaly scoring, not multivariate feature engineering. A direct
univariate detector receives its metric residual after the configured stages.
In contrast, Mahalanobis, PCA, kernel PCA, and GSTA receive raw metric
coordinates after scaling and smoothing only. Their scalar output then becomes
a univariate score stream and receives Holt/DFT processing and the configured
smoother before Z-score or MAD.

This design retains relationships such as correlated daily levels for the
multivariate engine while removing predictable drift and periodicity only where
the final scalar anomaly score is normalized. A model result exposes both
`details.multivariate_score` (the engine's raw output) and
`details.decomposed_multivariate_score` (the residual before scalar smoothing)
and `details.smoothed_multivariate_score` (the value actually scored).

## Univariate models

The univariate detectors are in `src/univariate/`. They are window-local: their
reference distribution is recomputed from the current window after every
accepted update rather than learned once and retained indefinitely.

### Common behavior

Every `univariate` model configuration must have exactly one named stream
input, a supported algorithm (`z_score` or `mad`), and a finite positive
`thresholds.score`. Multiple models may evaluate the same stream.

On evaluation, a model returns the score for the latest timestamped processed
sample and marks it anomalous only when

```text
score > thresholds.score
```

Equality is not anomalous. Detection details include the input, latest
timestamp and value, score inputs, threshold, and sample count. The processed
window may include interpolation, scaling, Holt detrending, sliding-DFT
deseasonality, and smoothing, so scores and thresholds refer to that final
representation rather than necessarily to the source unit.

Because the candidate point is included in the reference window, a large point
can shift the mean, standard deviation, median, or MAD that scores it. This is
the implementation's current inclusive-window behavior; it is simple and
responsive, but less sensitive than leave-one-out or historical-baseline
scoring for a single isolated extreme.

### Z-score (`z_score`)

The Z-score detector assumes departures are meaningful relative to the window
mean and dispersion. For `n` processed values, it calculates

```text
mean = sum(xi) / n
standard_deviation = sqrt(sum((xi - mean)^2) / (n - ddof))
score = abs(x_latest - mean) / standard_deviation.
```

`ddof` is the degrees-of-freedom correction:

- `ddof: 0` (the default) uses population standard deviation.
- `ddof: 1` uses the familiar sample-standard-deviation denominator `n - 1`.

It must be a non-negative integer and smaller than `window.sample_count`.
At evaluation, there must be more than `ddof` actual samples.

The score is two-sided because of the absolute deviation: unusually high and
unusually low values are treated the same. Under an approximately normal,
stationary baseline, common starting thresholds are around 2--3 standard
deviations, but the appropriate value should be calibrated against observed
false-alert rates and the effects of preprocessing.

Z-score is efficient and interpretable, but its mean and standard deviation
are sensitive to outliers, skewness, trends, and unremoved seasonality. Use it
when the preprocessed residuals are roughly symmetric with stable variance; use
decomposition first when periodic structure would otherwise dominate the score.

If the computed standard deviation is zero, an exactly equal latest value has
score 0; a differing latest value has score `+infinity` and is anomalous for
any finite threshold.

Example:

```yaml
- id: temperature-z-score
  enabled: true
  kind: univariate
  algorithm: z_score
  inputs: [outdoor_temperature]
  parameters: { ddof: 1 }
  thresholds: { score: 3.0 }
```

### Median absolute deviation (`mad`)

MAD replaces mean and standard deviation with robust median-based estimates.
For the window values,

```text
center = median(xi)
raw_MAD = median(abs(xi - center))
smoothed_MAD(t) = alpha * raw_MAD(t) + (1 - alpha) * smoothed_MAD(t-1)
score = 0.6745 * abs(x_latest - center) / (smoothed_MAD + epsilon).
```

The median has a 50% breakdown point: fewer than half the observations can be
arbitrarily extreme without making the estimator arbitrarily extreme. The MAD
has the same robustness intuition, making this detector more reliable than a
Z-score when isolated spikes or heavy-tailed residuals are expected.

The factor `0.6745` puts this modified score approximately on a standard-normal
Z-score scale. `parameters.mad_ema_alpha` controls how quickly the denominator
tracks changing dispersion and defaults to `0.05`. `parameters.epsilon` is a
strictly positive noise floor (default `1e-6`) that keeps scores finite for
discretized or constant signals. Set epsilon in the units of the fully
preprocessed model input, typically around 5% to 10% of normal background
noise, and tune the threshold empirically for the complete preprocessing path.
Both settings are validated at startup: alpha must be in `(0, 1]` and epsilon
must be finite and positive.

The first evaluation after a model is primed (or after its input stream resets)
uses `raw_MAD` as the EMA baseline. Subsequent evaluations update the baseline
once per accepted stream point. The reset is intentional: carrying dispersion
from a discarded window into a newly primed operating regime would defeat the
noise-control benefit. A MAD detection includes `median_absolute_deviation`
(raw MAD), `smoothed_mad`, `stabilized_mad`, `mad_ema_alpha`, and `epsilon` in
its details so production tuning can distinguish a changing signal from a
changing denominator.

Example:

```yaml
- id: tank-level-mad
  enabled: true
  kind: univariate
  algorithm: mad
  inputs: [tank_level]
  parameters: { mad_ema_alpha: 0.05, epsilon: 0.1 }
  thresholds: { score: 4.0 }
```

### Choosing between the univariate models

| Situation | Prefer | Reason |
| --- | --- | --- |
| Stable, approximately Gaussian residuals | `z_score` | Mean and standard deviation are efficient and scores are familiar. |
| Spikes, heavy tails, or occasional contaminated points | `mad` | Median-based location and scale resist isolated extremes. |
| Strong seasonality or trend | Preprocess first, then either | Both are level/dispersion detectors, not seasonal models. |
| Long regime changes | Revisit window and preprocessing choices | A rolling reference adapts, so persistent shifts can become normal. |

For Holt tuning, increase `alpha` or `beta` when normal ramps are being
misclassified as residual anomalies; decrease them when brief excursions are
being absorbed too readily. For seasonal tuning, set period bounds using the
source cadence and expected cycle duration, then inspect
`detected_seasonal_period_samples` in multivariate result details to verify the
selected period. During the initial seasonal window warm-up there is no seasonal
subtraction, so calibrate alert latency with that stage included.

Neither model establishes causality or detects every form of anomaly. Their
usefulness depends on cadence, the window duration, missing-data policy,
preprocessing, and a threshold calibrated on representative normal and
abnormal operating data.

## Common multivariate pipeline

A multivariate entry lists inputs in vector-coordinate order. Kafka records for
those inputs arrive independently, so the model accepts a row only when:

1. every input stream is fully primed;
2. every input's latest timestamp is newer than the timestamp used for that
   coordinate in the prior accepted row; and
3. `max(timestamp) - min(timestamp) <= max_time_skew_ms`.

This is bounded synchronization without indefinite last-observation carry
forward. It naturally runs at the cadence of the slowest input. Input order is
stable and comes directly from `models[].inputs`.

The point-at-a-time multivariate engines use a fixed sample-count `window_size`.
They fill the baseline with `window_size` aligned vectors. Each later candidate is scored
against the preceding window and rolled into the window only afterward. This
candidate-exclusive order avoids reducing an extreme point's own score by
letting it alter the mean, covariance, or principal directions used to score
it.

GSTA instead uses `window_size` as the temporal length of each complete aligned
channel tensor and has a separate self-training warmup. When the score detector
is enabled, every engine's raw scalar is first passed through the configured
Holt detrending and sliding-DFT deseasonality, then the configured causal
smoother. The smoothed remainder then enters a separate inclusive
`score_window_size` window and is evaluated by `score_detector: z_score` or
`score_detector: mad`. Raw multivariate metric inputs themselves receive only
scaling and smoothing; decomposition never alters their coordinates. For
Mahalanobis/PCA the first final result appears after
`window_size + score_window_size` accepted aligned vectors. For GSTA it appears
after `warmup_steps + score_window_size` complete channel windows once source
windows are primed. `thresholds.score` applies to this second-stage Z-score/MAD,
not directly to distance or reconstruction error.

Set `parameters.score_detector_enabled: false` to skip this second stage. The
engine then emits as soon as its own baseline is ready, `Detection.score` and
Kafka `anomaly_score` contain the raw distance/reconstruction error, and
`thresholds.score` is compared directly with that raw value. Kafka reports
`score_algorithm: raw` for this mode.

The final `Detection` contains:

- `score`: the Z-score or MAD ratio used for the anomaly decision;
- `details.multivariate_score`: the raw Mahalanobis distance, reconstruction
  error, or GSTA MSE;
- `details.decomposed_multivariate_score`: the residual before scalar smoothing;
- `details.smoothed_multivariate_score`: the scalar value sent to the final
  score window;
- `details.multivariate_score_name`, `multivariate_algorithm`, input values,
  input timestamps, and (for RFF modes) the effective `rbf_gamma`.

An interpolation gap reset in any input clears the affected multivariate
baseline, automatic kernel calibration, timestamp synchronization state, and
score window. Process restart also starts these in-memory states cold.

## Mahalanobis distance (`mahalanobis`)

Mahalanobis distance measures how unusual a vector is relative to both the
variance of each coordinate and correlations between coordinates. For a
candidate vector `x`, baseline mean `mu`, and covariance `Sigma`,

```text
distance(x) = sqrt((x - mu)^T Sigma^-1 (x - mu)).
```

The implementation never forms `Sigma^-1`. It computes a Cholesky
factorization and solves `Sigma y = x - mu`, then evaluates
`sqrt((x - mu)^T y)`. This is more stable and avoids constructing an explicit
inverse.

### Rolling two-way Welford state

The detector retains a `VecDeque` only to identify the outgoing vector and
maintains sample count `n`, mean `mu`, and the unnormalized covariance sum
`M2`. Adding `x` uses:

```text
delta = x - mu
n' = n + 1
mu' = mu + delta / n'
M2' = M2 + delta (x - mu')^T.
```

When the window is full, removing its oldest vector reverses the state:

```text
delta = x_old - mu
n' = n - 1
mu' = mu - delta / n'
M2' = M2 - delta (x_old - mu')^T.
```

The sample covariance is `M2 / (n - 1)`. The code explicitly symmetrizes it to
remove small floating-point asymmetries accumulated by repeated updates.

### Shrinkage and regularization

Correlated or flat telemetry can make an empirical covariance singular. Before
Cholesky, the implementation applies spherical shrinkage and a positive ridge:

```text
v = trace(Sigma) / dimensions
Sigma_safe = (1 - shrinkage) Sigma
             + shrinkage v I
             + regularization I.
```

`shrinkage` is in `[0, 1]`; `regularization` must be finite and strictly
positive. Shrinkage stabilizes weakly estimated correlation structure, while
the ridge also handles a completely flat window where `v` is zero. A
non-positive-definite result is returned as a model error rather than panicking.

The Welford update is `O(d^2)` per accepted vector and does not scan historical
rows. Cholesky scoring is `O(d^3)` and the retained vector queue is
`O(window_size * d)`. The matrices keep fixed dimensions, though temporary
linear-algebra values may still be allocated during scoring.

```yaml
- id: weather-mahalanobis
  enabled: true
  kind: multivariate
  algorithm: mahalanobis
  inputs: [outdoor_temperature, outdoor_humidity]
  parameters:
    window_size: 16
    max_time_skew_ms: 5000
    regularization: 0.000001
    shrinkage: 0.10
    score_detector: mad
    score_window_size: 12
  thresholds: { score: 4.0 }
```

`window_size` must exceed the number of inputs so the unregularized sample
covariance has enough observations to reach full rank in non-degenerate data.

## Linear PCA (`pca`)

PCA models a lower-dimensional linear subspace. For the preceding baseline,
the detector builds a `window_size x input_dimensions` matrix, subtracts its
column mean, and computes

```text
X_centered = U S V^T.
```

The first `retained_components` columns of `V` form `V_k`. After centering the
candidate with the baseline mean, its squared reconstruction residual is

```text
error(x) = ||(x - mu) - V_k V_k^T (x - mu)||^2.
```

Large residuals indicate a violation of the usual cross-metric relationships,
even when individual metric values are not extreme. PCA does not standardize
coordinates internally; enable the service's per-stream scaling when raw units
or variances differ materially.

An SVD is recomputed from the current baseline for every raw score. This keeps
the implementation straightforward and correct for a rolling window, but its
cost grows roughly with `window_size`, working dimension, and their smaller
matrix dimension. Keep these values bounded for the target Kafka rate.

```yaml
- id: tank-linear-pca
  enabled: true
  kind: multivariate
  algorithm: pca
  inputs: [tank_level, tank_weight]
  parameters:
    window_size: 16
    max_time_skew_ms: 2500
    retained_components: 1
    score_detector: z_score
    score_window_size: 12
    score_ddof: 1
  thresholds: { score: 3.0 }
```

`retained_components` must be positive and smaller than both `window_size` and
the working dimension.

## Approximate RBF kernel PCA (`kernel_pca`)

The kernel mode reuses the identical rolling PCA and reconstruction path after
an RFF transform. For the RBF kernel

```text
k(x, y) = exp(-gamma ||x - y||^2),
```

it samples fixed frequencies and phases

```text
omega_j ~ Normal(0, 2 gamma I)
b_j ~ Uniform(0, 2 pi)
phi_j(x) = sqrt(2 / D) cos(omega_j^T x + b_j),
```

so `phi(x)^T phi(y)` approximates `k(x, y)` by Bochner's theorem. Linear PCA
over the `D = rff_dimension` explicit features is therefore a scalable
approximation to RBF kernel PCA with one downstream code path. It is not exact
kernel PCA: approximation quality depends on `rff_dimension` and `seed`, and
the reported residual is in RFF feature space rather than an input-space
pre-image reconstruction error.

With numeric `gamma`, the transform exists immediately. With `gamma: "auto"`,
the first `window_size` raw vectors form a calibration buffer. The detector
computes all pairwise squared distances, uses their median `m`, and sets
`gamma = 1 / m`. If calibration is flat (`m <= 1e-12`), it safely uses
`gamma = 1.0`. It then transforms those same calibration vectors to populate
the PCA baseline; no calibration history is discarded. Pairwise calibration
uses `O(window_size^2)` time and temporary memory.

RFF matrices remain fixed after activation. `seed` makes deployments and tests
reproducible; changing it changes the approximation and requires threshold
recalibration.

```yaml
- id: weather-kernel-pca
  enabled: true
  kind: multivariate
  algorithm: kernel_pca
  inputs: [outdoor_temperature, outdoor_humidity]
  parameters:
    window_size: 16
    max_time_skew_ms: 5000
    retained_components: 4
    rff_dimension: 32
    gamma: auto
    seed: 20260824
    score_detector: mad
    score_window_size: 12
  thresholds: { score: 4.0 }
```

## Gated Self-Training Autoencoder (`gsta`)

GSTA consumes a tensor shaped `[1, input_count, window_size]`. Input streams
must be positionally aligned. Every corresponding channel position must be
within `max_time_skew_ms`; otherwise that update waits for alignment.

The encoder applies `input_count -> 64 -> latent_channels` same-padded Conv1d
blocks. Graph attention treats the latent channels as fully connected nodes
whose features span the time window. The decoder applies
`latent_channels -> 64 -> input_count` transposed convolutions and produces a
linear reconstruction so negative standardized values remain representable.
The raw score is

```text
MSE = mean((reconstruction - input_window)^2).
```

The first `warmup_steps` complete windows update Adam unconditionally and do
not enter the final score pipeline. Afterward, MSE is measured with Burn's
non-autodiff view before any training. If `MSE <= gate_threshold`, a second
autodiff pass updates the weights at `learning_rate`. If it is larger, the
optimizer is skipped. Both clean and gated post-warmup MSE values still enter
the final configured MAD/Z-score window; the gate protects training and is not
itself the final anomaly decision.

```yaml
- id: weather-gsta
  enabled: true
  kind: multivariate
  algorithm: gsta
  inputs: [outdoor_temperature, outdoor_humidity]
  parameters:
    window_size: 12
    max_time_skew_ms: 5000
    latent_channels: 8
    attention_heads: 2
    gate_threshold: 0.05
    learning_rate: 0.001
    warmup_steps: 128
    seed: 20260824
    score_detector: mad
    score_window_size: 12
  thresholds: { score: 4.0 }
```

WGPU is the production backend and autodiff is enabled only for accepted
training passes. The model, optimizer, and attention parameters are protected
by the existing multivariate mutex. They are not checkpointed, so restart or a
stream reset begins a seeded fresh warmup.

## Multivariate configuration reference

| Parameter | Applies to | Constraint / meaning |
| --- | --- | --- |
| `window_size` | all | Fixed baseline count for point engines or temporal length for GSTA; at least 2, greater than input count for Mahalanobis, and must fit the event-time window for GSTA. |
| `max_time_skew_ms` | all | Maximum cross-channel skew in one vector, or at each temporal position for GSTA. |
| `score_detector_enabled` | all | Defaults to `true`. When false, emit and threshold the raw engine score directly; `score_detector`, `score_window_size`, and `score_ddof` are not required. |
| `score_detector` | score detector enabled | `z_score` or `mad`. |
| `score_window_size` | score detector enabled | Fixed raw-score window; at least 2. |
| `score_ddof` | Z-score stage | Defaults to 0 and must be less than `score_window_size`. |
| `mad_ema_alpha` | Direct `mad` model | Raw-MAD EMA weight; defaults to `0.05` and must be in `(0, 1]`. |
| `epsilon` | Direct `mad` model | Positive MAD denominator noise floor in the fully preprocessed input's units; defaults to `1e-6`. |
| `thresholds.score` | all | Finite positive threshold for the final Z-score/MAD ratio, or the raw engine score when `score_detector_enabled: false`. |
| `regularization` | Mahalanobis | Positive diagonal ridge; defaults to `1e-6`. |
| `shrinkage` | Mahalanobis | Spherical covariance shrinkage in `[0, 1]`; defaults to 0. |
| `retained_components` | PCA/kernel PCA | Positive count below `min(window_size, working_dimension)`. |
| `rff_dimension` | kernel PCA | Explicit feature count, at least 2. |
| `gamma` | kernel PCA | Positive number or `auto`. |
| `latent_channels` | GSTA | Convolution bottleneck width; defaults to 16 and must be at least 2. |
| `attention_heads` | GSTA | Attention head count; defaults to 2 and must divide `window_size`. |
| `gate_threshold` | GSTA | Positive raw MSE limit for post-warmup weight updates. |
| `learning_rate` | GSTA | Positive Adam learning rate; defaults to 0.001. |
| `warmup_steps` | GSTA | Unconditional training windows with output suppressed; defaults to 128 and must be positive. |
| `seed` | kernel PCA/GSTA | Deterministic RFF or neural initialization seed; defaults to 0. |

Stateful multivariate instances use cloneable `Arc<Mutex<_>>` handles. Lock
poisoning becomes a `ModelError`, and locks are never held over an async await.
The Kafka pipeline also serializes its overall mutable window state, so model
updates remain deterministic for consumer order.

## Choosing a multivariate model

| Situation | Prefer | Reason |
| --- | --- | --- |
| Approximately elliptical relationships and interpretable joint distance | Mahalanobis | Directly models covariance and correlation. |
| Mostly linear low-rank relationships | PCA | Residual measures departure from principal directions. |
| Curved or multimodal normal structure inadequately modeled by linear PCA | Kernel PCA/RFF | RBF features can linearize nonlinear relationships. |
| Evolving temporal and nonlinear cross-channel relationships with GPU capacity | GSTA | Conv1d and attention learn a reconstruction manifold while gating limits contamination. |
| Highly contaminated score history | MAD second stage | More robust than mean/standard deviation. |
| Stable, roughly Gaussian score history | Z-score second stage | Familiar scale and efficient estimate. |

Thresholds, window sizes, shrinkage, retained components, RFF dimension, and
gamma must be calibrated on representative `iot-sim` and production traces.
Adaptive rolling models can eventually absorb persistent regime changes, and
none of these scores identifies root cause by itself.
