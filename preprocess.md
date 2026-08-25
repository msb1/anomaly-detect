# Preprocessing

The preprocessing subsystem converts each stream's retained raw event-time
window into a derived window for anomaly detection. It is implemented in
`src/preprocess/` and coordinated by `Preprocessor` and `WindowStore`.

Every configured stream is processed independently. Therefore all fitted
quantities--minimum, mean, seasonal pattern, and smoothed values--are based
only on that stream's current window and never combine different units or
metrics.

## Pipeline

For each accepted source observation, the pipeline is:

```text
Kafka record matching a stream
  -> validate payload and event time
  -> bounded linear interpolation (optional)
  -> event-time insertion, lateness check, and window eviction
  -> scaling (optional)
  -> seasonal-trend decomposition (optional)
  -> smoothing (optional)
  -> processed sliding window supplied to models
```

The configured transform order is always **scaling -> decomposition ->
smoothing**. Interpolation occurs before a sample enters the window. After each
accepted update, the derived window is rebuilt from the retained raw window;
this is intentional, since expiry of an old sample changes window-local
statistics and fits.

A stream is usable by a model only when its raw event-time span reaches
`window.duration_ms`, it retains at least `window.minimum_samples`, and a
processed view exists. If decomposition is enabled but cannot be fitted, no
processed view is published.

## Event-time windowing and gaps

Raw samples are stored timestamp-ordered. The watermark is the greatest event
time observed, and points older than `watermark - window.duration_ms` are
evicted. A point older than `watermark - max_lateness_ms` is rejected as late.

For expected cadence `h = interval_ms`, successive forward observations at
times `t0` and `t1` have missing event time

```text
g = t1 - t0 - h
```

With interpolation enabled, the gap is filled only if
`g <= floor(window.duration_ms * max_gap_fraction)`. Configuration requires a
positive fraction no greater than 0.20. For an allowed gap, synthetic points
are inserted at cadence times `t0 + k h` using

```text
x(t0 + k h) = x0 + (k h / (t1 - t0)) * (x1 - x0).
```

This is the usual linear interpolation assumption: the unobserved signal is
approximated by a straight line between its two observed endpoints. It is a
reasonable repair for short, smooth outages, but can hide a short transient or
create misleading values for a step-like process. A gap beyond the limit clears
both raw and processed state, retains the new real observation as a fresh start,
and revokes readiness until the new full window is primed. Non-forward and
already-cadent observations are not interpolated.

## Scaling

Scaling is fitted to the current raw window.

### Min-max scaling

For window minimum `a`, maximum `b`, and configured output bounds `L` and `U`,
the transformed value is

```text
x' = L + ((x - a) / (b - a)) * (U - L).
```

It preserves the window's ordering and maps its endpoints to the configured
range. It is useful when bounded, comparable numerical inputs are needed, but
is sensitive to an extreme observation because that observation defines the
range. If `b - a <= epsilon`, every value is mapped to `(L + U) / 2` to avoid a
near-zero division.

### Standard scaling

With window mean `mu` and population standard deviation

```text
sigma = sqrt(sum((xi - mu)^2) / n),
```

the transform is `x' = (x - mu) / sigma`. This centers the current window at
zero and gives it unit population variance. It can make signals with different
scales easier to compare, but mean and standard deviation are themselves
sensitive to outliers. If `sigma <= epsilon`, all outputs are zero.

`epsilon` must be finite and positive; output bounds must be finite with
`output_min < output_max`.

## Additive decomposition

When enabled, decomposition models the scaled series as

```text
observed_t = trend_t + seasonal_t + remainder_t.
```

Only `remainder_t = observed_t - trend_t - seasonal_t` proceeds downstream.
The period in samples is the rounded-up conversion
`ceil(period_ms / interval_ms)`; the implementation requires at least two
samples per period and at least two full periods in the retained window.

### STL-style decomposition (`stl`)

The STL-style method estimates a smooth trend using local-linear LOESS across
time. Neighbors receive tricube distance weights

```text
w(d) = (1 - d^3)^3,  0 <= d <= 1,
```

combined with a robustness weight. After detrending, each seasonal phase
(`index mod period`) is smoothed with local-linear LOESS using `loess_span`.
The seasonal component is centered to mean zero so that the baseline belongs to
the trend rather than seasonality.

Robust passes reduce the effect of unusual residuals. Their Tukey-bisquare
weights are based on `c = 6 * median(|remainder|)`:

```text
u = |remainder| / c
robust_weight = (1 - u^2)^2  if u < 1; otherwise 0.
```

The procedure runs the initial fit plus `robust_iterations` reweighted fits.
`loess_span` must be odd and at least 3. This method accommodates a changing
trend and repeating seasonal behavior; its quality depends on having enough
periods and representative phase observations.

### Median seasonal decomposition (`twitter`)

The `twitter` method is a robust, simpler seasonal baseline. It uses the global
window median as a constant trend. For each phase of the period it takes the
median value after subtracting that level, centers the phase profile by its
median, repeats the profile across the window, and computes the remainder.

Medians have high resistance to isolated extremes, so this works well for a
stable-level signal with consistent periodic behavior. Unlike STL, its trend is
constant within a window; a genuine drift can therefore remain in the
remainder.

## Smoothing

Smoothing is causal within the retained window: an output at time `t` uses that
sample and earlier retained samples, never future samples.

`moving_average` is the trailing simple moving average over up to `m =
window_size` values:

```text
y_t = mean(x_{max(0, t-m+1)}, ..., x_t).
```

It reduces high-frequency noise but delays and attenuates abrupt changes.

`exponential_moving_average` sets `alpha = 2 / (m + 1)`, starts at the first
value, and recursively computes

```text
y_0 = x_0
y_t = alpha * x_t + (1 - alpha) * y_{t-1}.
```

EMA gives recent observations exponentially more influence and usually reacts
faster than a simple average of the same nominal size. Both methods smooth the
remainder when decomposition is active, otherwise the scaled series (or raw
series if scaling is disabled). `window_size` must be greater than zero.

## Configuration outline

```yaml
preprocessing:
  interpolation: { enabled: true, max_gap_fraction: 0.20 }
  scaling:
    enabled: true
    method: standard # min_max | standard
    output_min: 0.0  # used by min_max
    output_max: 1.0  # used by min_max
    epsilon: 1.0e-12
  decomposition:
    enabled: false
    method: stl      # stl | twitter
    period_ms: 86400000
    stl: { loess_span: 7, robust_iterations: 2 }
  smoothing:
    enabled: true
    method: moving_average # moving_average | exponential_moving_average
    window_size: 5
```

