# Preprocessing

The preprocessing subsystem converts each stream's retained raw event-time
window into a derived window for anomaly detection. It is implemented in
`src/preprocess/` and coordinated by `Preprocessor` and `WindowStore`.

Every configured stream is processed independently. Scaling and smoothing are
shared preprocessing stages. Trend and seasonal decomposition are confined to
univariate scoring: direct univariate metric models use them, multivariate
metric inputs bypass them, and a multivariate model's scalar output uses them
and the configured smoother before its final Z-score or MAD detector.

## Pipeline

For each accepted source observation, the pipeline is:

```text
Kafka record matching a stream
  -> validate payload and event time
  -> bounded linear interpolation (optional)
  -> event-time insertion, lateness check, and window eviction
  -> scaling (optional)
  +-> multivariate metric view: smoothing (optional)
  +-> univariate metric view: Holt detrending -> sliding-DFT deseasonality
                              -> smoothing (optional)
```

For direct univariate detection the order is always **scaling -> detrending ->
deseasonality -> smoothing**. For multivariate engine inputs it is **scaling ->
smoothing**. The multivariate engine's scalar output then follows **detrending
-> deseasonality -> smoothing -> Z-score/MAD**. Interpolation occurs before a sample enters
either view.

A stream is usable when it retains `window.sample_count` values. Seasonal
decomposition emits the detrended value while its spectral window warms up, so
it does not prevent metric-window priming.

## MAD decision stability

MAD is a detector-stage control rather than a preprocessing transform. Once a
direct univariate residual window is primed, its raw MAD is smoothed per model
with `parameters.mad_ema_alpha` (default `0.05`) and scored with
`0.6745 * abs(remainder - median) / (smoothed_mad + epsilon)`. The positive
`parameters.epsilon` floor (default `1e-6`) is in the units emitted by this
preprocessing pipeline; choose it from normal residual noise, commonly 5% to
10%. On a stream reset, the MAD EMA is reset along with the derived window, so
the next fully primed window establishes a fresh dispersion baseline.

## Event-time windowing and gaps

Raw samples are stored timestamp-ordered. The watermark is the greatest event
time observed; older arrivals are rejected and the oldest samples are evicted
when the deque exceeds `window.sample_count`.

For expected cadence `h = interval_ms`, successive forward observations at
times `t0` and `t1` have missing event time

```text
g = t1 - t0 - h
```

With interpolation enabled, the gap is filled only when the number of missing
samples does not exceed `round(window.sample_count * max_gap_fraction)`.
Configuration requires a positive fraction no greater than 0.20. For an
allowed gap, synthetic points are inserted at cadence times `t0 + k h` using

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

## Streaming decomposition

When enabled, decomposition models the scaled series as

```text
observed_t = trend_t + seasonal_t + remainder_t.
```

Only `remainder_t = observed_t - trend_t - seasonal_t` proceeds downstream.
Trend and seasonal stages have independent `enabled` flags.

### Holt linear detrending

Holt's double exponential smoother tracks a level `l` and slope `b`:

```text
l_t = alpha * x_t + (1 - alpha) * (l_(t-1) + b_(t-1))
b_t = beta * (l_t - l_(t-1)) + (1 - beta) * b_(t-1)
trend_t = l_t + b_t
```

Both coefficients must be in `(0, 1]`. Tracking velocity lets the trend move
with a ramp instead of introducing the fixed phase delay of SMA/EMA
detrending.

### Sliding DFT deseasonality

The detrended stream fills `window_size_samples`. RustFFT initializes the first
frequency spectrum. For every later sample the exact sliding DFT updates each
bin from only the discarded and incoming values, making steady-state spectrum
maintenance O(N) per point. The dominant non-DC frequency is searched within
the configured period bounds every `detection_interval_samples`. The matching
phase-aligned lagged detrended value is the seasonal estimate. Before the
spectral window is full, the seasonal estimate is zero.

For a window of `N` detrended values and frequency bin `k`, the initialized
spectrum is the ordinary DFT

```text
X_k = sum(x_j * exp(-i * 2π * k * j / N)),  j = 0..N-1.
```

When `x_old` leaves and `x_new` arrives, the implementation updates that bin
exactly as

```text
X'_k = exp(i * 2π * k / N) * (X_k + x_new - x_old).
```

This is why the steady-state update is O(N): it updates `N` retained bins, not
an O(N log N) transform over the whole history. The initial FFT is only used
when the spectral window first becomes full (and after a reset). Bin zero is
ignored because it represents the remaining DC level; candidate bins are
limited by the configured period bounds. If the selected bin is `k`, the
reported period is `round(N / k)` samples and the seasonal estimate for the
new point is the detrended observation one such period behind.

The algorithm is causal: it never revises earlier residuals using future data.
Consequently, period changes take one spectral window plus the configured
detection interval to settle. Very broad period bounds can choose noise; use
domain cadence and known cycle lengths to make them narrow.

## Scope and state ownership

`WindowStore` keeps one `StreamingDecomposer` per metric stream only when at
least one decomposition stage is enabled. It receives each newly scaled input
once, including bounded interpolation points. It is reset with the stream after
an excessive interpolation gap.

Each multivariate model owns a separate `StreamingDecomposer` inside its final
score pipeline. It sees only that model's scalar distance, reconstruction
error, or MSE, and resets whenever the multivariate baseline resets. This
separation is intentional: sharing seasonal state between raw metric streams
and model scores would make unrelated models affect one another.

Raw multivariate metric views never instantiate or advance decomposition state.
They use scaling and the configured smoother only.

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
    trend: { enabled: true, alpha: 0.2, beta: 0.1 }
    seasonal:
      enabled: true
      window_size_samples: 256
      detection_interval_samples: 1
      min_period_samples: 2
      max_period_samples: 128
  smoothing:
    enabled: true
    method: moving_average # moving_average | exponential_moving_average
    window_size: 5
```
