# anomaly-detect

Rust service for streaming univariate and multivariate anomaly detection.
The service consumes `iot-sim` telemetry from Kafka, filters records by Kafka
headers, preprocesses selected metric windows, and evaluates Z-score and MAD
detectors directly or as the final stage after rolling Mahalanobis distance,
linear PCA, Random-Fourier-Feature (RFF) kernel PCA, and a gated self-training
graph-attention autoencoder (GSTA).

## Run

The project uses the system `librdkafka`, matching the sibling `iot-sim`
project. On macOS with Homebrew:

```bash
brew install rust librdkafka pkg-config
cargo test
cargo run -- --config config/anomaly-detect.yaml
```

GSTA uses Burn 0.16's WGPU backend and therefore requires a working Metal,
Vulkan, DirectX 12, OpenGL, or WebGPU adapter. A configured GSTA model fails
startup with a model error when no WGPU adapter is available; it does not
silently change numerical backends.

## Anomaly results

After a model's final univariate scoring window is ready, every new final
detection is published as JSON to `kafka.output_topic` (default
`iot.anomaly.v1`). The Kafka record is keyed by the unique model ID. Input
offsets are committed only after the broker acknowledges all anomaly results
created from that input record.

Each result includes the most recent point's event-time `timestamp_ms` and
`anomaly_score`, whether that point is anomalous, and `anomalous_points` plus
`window_sample_count` for the final scoring window. `model_id`, `model_kind`,
`algorithm`, `score_algorithm`, and the configured input stream names link it
back to the model configuration. For a direct detector, `algorithm` and
`score_algorithm` are the same. For a multivariate detector, `algorithm` names
the distance/reconstruction engine and `score_algorithm` names its final MAD
or Z-score stage. The direct detector's scoring window is the processed
event-time stream window; for a multivariate detector, it is the configured
fixed-count `score_window_size` of distances or reconstruction errors. The
contract is documented by
[schemas/iot-anomaly-v1.schema.json](schemas/iot-anomaly-v1.schema.json).

JSON has no representation for an infinite score. In the zero-dispersion edge
case, `is_anomalous` and `anomalous_points` remain authoritative and
`anomaly_score` is encoded as `null`.

The checked-in brokers, topic, header values, and source cadences come from
`../iot-sim/config/simulation.yaml`. `ANOMALY_DETECT__` environment variables
with `__` between nested keys override YAML values.

## Ingestion and preprocessing

[config/anomaly-detect.yaml](config/anomaly-detect.yaml) defines named streams.
Each stream has exact Kafka header filters and an `interval_ms` matching its
source cadence. Records that match no configured stream are committed without
parsing their JSON payload. Selected values are validated against the headers,
then processed in this fixed order:

1. bounded linear interpolation;
2. scaling;
3. decomposition;
4. smoothing.

Each stage has a global `enabled` flag. The selected methods and their settings
apply to every metric, but fitting is performed independently per metric
window. This avoids mixing incomparable units such as degrees, percentages,
and kilograms into one set of statistics.

### Interpolation

For a stream with expected cadence `interval_ms`, missing event time is
`new_timestamp - previous_timestamp - interval_ms`. Interpolation is allowed
only when that missing duration is at most
`window.duration_ms * max_gap_fraction`. Configuration validation prevents
`max_gap_fraction` from exceeding `0.20`.

An allowed gap is filled at exact cadence timestamps by linear interpolation
between the prior real/sample value and the new real value. If the gap exceeds
20%, no values are imputed: that stream's raw and processed windows are cleared,
the new point becomes the first point of a fresh window, and anomaly readiness
is revoked until the complete window is primed again.

### Scaling

`min_max` rescales the current window into `output_min..output_max`. `standard`
uses the current window's population mean and standard deviation. Constant
min-max windows map to the output midpoint; constant standard-scaled windows
map to zero. `epsilon` controls the near-constant cutoff.

Scaling is refit over the retained raw window after every update. Statistics
from evicted values therefore cannot leak into the current window.

### Decomposition

Seasonality is configured as event time with `period_ms`, such as `86400000`
for a day or `604800000` for a week. Each stream converts this to a sample count
using its own `interval_ms`. The sliding window must cover at least two periods.

- `stl` performs additive seasonal-trend decomposition with local-linear,
  tricube-weighted LOESS for the trend and seasonal subseries. Tukey bisquare
  robustness weights are updated for `robust_iterations` passes.
- `twitter` performs a robust Twitter-style decomposition using a global median
  trend and median values for each seasonal phase.

The remainder (`observed - trend - seasonal`) becomes the downstream series.
Until two periods are present, no processed view is published and models remain
unready.

### Smoothing

`moving_average` uses a trailing `window_size` in samples.
`exponential_moving_average` derives `alpha = 2 / (window_size + 1)`. Smoothing
is applied to the decomposition remainder, or directly to scaled values when
decomposition is disabled.

## Sliding-window state

`WindowStore` holds an in-memory `BTreeMap<String, StreamWindow>`, one entry per
configured metric. Each entry contains:

- a timestamp-ordered `VecDeque<Sample>` containing raw and interpolated data;
- a derived `VecDeque<Sample>` containing the fully preprocessed values;
- the expected interval and the raw window's first-observed timestamp and
  event-time watermark.

On each accepted Kafka point, the service checks lateness and interpolation,
inserts the point(s) in timestamp order, advances the watermark, and removes raw
points older than `watermark - duration_ms`. It then rebuilds the processed
deque from the retained raw deque in the four-stage order above. This full
recalculation is important: window-local scaling, seasonal estimates, and
smoothing change when either a new point arrives or an old point expires.

A stream is primed only when it has observed at least `duration_ms`, retains at
least `minimum_samples`, and has a valid processed view. Every enabled
univariate model for a changed stream becomes ready with that stream. A
multivariate model waits for every input stream to be primed and only creates a
new vector when every input has advanced since its previous vector.
`parameters.max_time_skew_ms` bounds the timestamps within that vector. An
excessive interpolation gap resets the first-observed timestamp, both deques,
and any affected multivariate baseline and score window. State is currently
process memory and is rebuilt after restart.

## Models

Each `univariate` model has exactly one named stream input. Multiple model
entries can use the same stream, and other entries can target different
streams. The implemented univariate algorithms are:

- `z_score`: `abs(Xi - mean) / standard_deviation`; optional integer
  `parameters.ddof` defaults to `0` and must be less than
  `window.minimum_samples`.
- `mad`: `abs(Xi - median(X)) / median(abs(X - median(X)))`.

Both require a finite positive `thresholds.score`. A sample is anomalous when
its score is strictly greater than the threshold. A constant window scores
zero. For MAD, a non-median value in a zero-MAD window scores positive infinity
and is anomalous.

Each `multivariate` model has at least two inputs and uses one of:

- `mahalanobis`: a rolling, two-way Welford mean/covariance with configurable
  covariance shrinkage and mandatory diagonal regularization, solved by
  Cholesky decomposition without forming an inverse;
- `pca`: rolling linear PCA reconstruction error from an SVD;
- `kernel_pca`: the same PCA engine after a deterministic RFF approximation to
  the RBF kernel, with numeric `gamma` or `gamma: "auto"` median calibration.
- `gsta`: a Burn/WGPU gated self-training autoencoder. Same-padded Conv1d
  blocks extract local temporal features, multi-head graph attention learns
  dependencies among latent channels, and transposed convolutions reconstruct
  the complete channel window. Its raw score is reconstruction MSE.

The candidate is scored against the preceding multivariate baseline and is
rolled into it afterward. The resulting Mahalanobis distance or reconstruction
error enters its own configured `z_score` or `mad` score window. Only that
second-stage result is a final detection. Key parameters are `window_size`,
`max_time_skew_ms`, `score_detector`, `score_window_size`, and the
algorithm-specific values shown in
[config/anomaly-detect.yaml](config/anomaly-detect.yaml). RFF `seed` makes the
projection reproducible.

### GSTA streaming and gated training

GSTA forms tensors as `[batch=1, configured input streams, window_size]` from
fully preprocessed data. All of its input streams must have the same
`interval_ms`; each position across channels must also satisfy
`max_time_skew_ms`. The configured `window_size` must fit inside the global
event-time window. Incomplete or misaligned windows are held back rather than
padded or carried forward.

The first `warmup_steps` aligned windows train unconditionally and emit no GSTA
detection. This prevents a random model from exceeding its own gate forever.
After warmup, each window is first evaluated through Burn's non-autodiff model
view. If reconstruction MSE is no greater than `gate_threshold`, a separate
autodiff forward/backward pass updates the weights with Adam at `learning_rate`.
If it exceeds the gate, backpropagation is skipped, protecting the learned
normal baseline from abnormal windows. Either way, the post-warmup MSE enters
the configured `score_detector` and `score_window_size`; that MAD or Z-score is
the final Kafka `anomaly_score`.

`latent_channels` controls the bottleneck width, `attention_heads` must divide
`window_size`, and `seed` makes initialization reproducible. The checked-in
`weather-gsta` example uses the two 5-second weather streams, a 12-point window,
8 latent channels, 2 attention heads, and the approved 128-step warmup. GSTA
weights and Adam state are process-local, like the other detector state, so a
restart begins a new warmup.

Detectors consume fully processed, primed stream values. Implementations live
under [src/univariate.rs](src/univariate.rs) and
[src/multivariate.rs](src/multivariate.rs). See [models.md](models.md) for the
equations, warm-up stages, configuration reference, and operational caveats.

Use `RUST_LOG=anomaly_detect=debug` to see interpolation and model-readiness
events and normal scores; anomalies are logged at warning level. With
`invalid_message_policy: skip`, malformed selected messages are logged and
committed; `fail` stops before committing the offending record.
