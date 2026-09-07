# anomaly-detect

Rust service for streaming and dataset univariate and multivariate anomaly
detection. Streaming mode consumes `iot-sim` telemetry from Kafka; dataset mode
downloads a Parquet dataset from RustFS/S3 and replays it through the same
sliding windows. Both modes filter selected metric streams, preprocess their
windows, and evaluate Z-score and MAD
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

GSTA uses Burn 0.21's WGPU backend and therefore requires a working Metal,
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
Each stream has exact Kafka header filters. Source cadence is carried as
`interval_ms` in each Kafka payload. Records that match no configured stream are committed without
parsing their JSON payload. Selected values are validated against the headers,
then processed. Univariate metric windows use this fixed order:

1. bounded linear interpolation;
2. scaling;
3. Holt detrending followed by sliding-DFT deseasonality;
4. smoothing.

Multivariate metric inputs deliberately skip step 3: they receive only scaling
and smoothing. After a multivariate engine emits its scalar distance or
reconstruction error, that score is detrended, deseasonalized, and smoothed
before its configured final Z-score or MAD detector. Decomposition settings apply to both
of those univariate scoring paths, with independent state per stream/model.

### Interpolation

For a stream with expected cadence `interval_ms`, missing event time is
`new_timestamp - previous_timestamp - interval_ms`. Interpolation is allowed
only when the number of missing points is at most
`round(window.sample_count * max_gap_fraction)`. For example, a 128-point
window and `0.20` limit allow up to 26 missing points. Configuration validation prevents
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

Trend removal uses causal Holt linear double exponential smoothing. `alpha`
updates the level and `beta` updates its slope, avoiding the fixed phase lag of
a trailing average. Deseasonality keeps a fixed detrended sample window. The
first spectrum is initialized with an FFT; after that, the sliding DFT updates
all frequency bins from the incoming and outgoing points in O(N) per sample.
The dominant period is searched between `min_period_samples` and
`max_period_samples` at `detection_interval_samples`, and its phase-aligned
lagged value is removed. During spectral warm-up, the detrended value is the
remainder.

### Configure detrending and deseasonality

The stages are independent. Enable only Holt when a metric drifts without a
reliable cycle; enable only deseasonality for an already stationary periodic
signal; normally enable both for a drifting seasonal signal. The checked-in
configuration uses a 240-sample spectral window and searches on every sample:

```yaml
preprocessing:
  decomposition:
    trend:
      enabled: true
      alpha: 0.2
      beta: 0.1
    seasonal:
      enabled: true
      window_size_samples: 240
      detection_interval_samples: 1
      min_period_samples: 2
      max_period_samples: 120
```

`alpha` and `beta` must be in `(0, 1]`. Higher values adapt more rapidly but
let transient changes influence the trend estimate more strongly. Set
`window_size_samples` to several cycles of the shortest seasonal behavior that
matters; it is the state retained by the spectral estimator, independent of
the direct detector's `window.sample_count`. Restrict the period range to
plausible sample counts. For example, a 5-second signal with an expected
10-minute cycle uses a 120-sample period; set bounds around that value rather
than allowing every frequency bin to compete. `detection_interval_samples: 1`
re-evaluates the dominant frequency on every arrival; a larger value reduces
period-search work while holding the last detected period between searches.

This configuration is global. It has exactly two effects:

- A direct `univariate` model consumes the metric residual.
- A `multivariate` model consumes scaled/smoothed metric coordinates, then
  applies the same decomposition to its scalar raw score immediately before
  final Z-score/MAD scoring.

It never detrends or deseasonalizes the metric coordinates given to
Mahalanobis, PCA, kernel PCA, or GSTA. Removed `stl`, `twitter`, `method`, and
`period_samples` settings are invalid configuration fields.

For a detailed derivation and tuning guidance, see [preprocess.md](preprocess.md);
[architecture.md](architecture.md) shows state ownership and routing.

### Smoothing

`moving_average` uses a trailing `window_size` in samples.
`exponential_moving_average` derives `alpha = 2 / (window_size + 1)`. Smoothing
is applied to the decomposition remainder, or directly to scaled values when
decomposition is disabled.

## Sliding-window state

`WindowStore` holds an in-memory `BTreeMap<String, StreamWindow>`, one entry per
configured metric. Each entry contains:

- a timestamp-ordered `VecDeque<Sample>` containing raw and interpolated data;
- separate derived `VecDeque<Sample>` values for univariate and multivariate
  metric inputs;
- state for Holt detrending and the sliding DFT.

On each accepted Kafka point, the service checks lateness and interpolation,
inserts the point(s) in timestamp order, advances the watermark, and removes the
oldest raw points beyond `window.sample_count`. Scaling and smoothing remain
window-local; decomposition state advances causally for each arriving point.

A stream is primed when it retains `window.sample_count` points and has a valid
processed view. Every enabled
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
  `window.sample_count`.
- `mad`: `0.6745 * abs(Xi - median(X)) / (EMA(MAD) + epsilon)`;
  `parameters.mad_ema_alpha` defaults to `0.05` and positive
  `parameters.epsilon` defaults to `1e-6`.

Both require a finite positive `thresholds.score`. A sample is anomalous when
its score is strictly greater than the threshold. A constant window scores
zero. MAD's configured epsilon keeps a non-median value in a zero-MAD window
finite while preserving sensitivity to deviations from the flat baseline.

### MAD stability controls

For a direct MAD model, `parameters.mad_ema_alpha` is the raw-MAD weight in
the model-local EMA. The first fully primed evaluation initializes that EMA to
the raw MAD; later evaluations use `alpha * raw_mad + (1 - alpha) * prior`.
The default `0.05` has an approximately 20-sample adaptation time, limiting
threshold steps as the rolling median changes. `parameters.epsilon` is added
only to the denominator and must use the units of the fully preprocessed
stream; start at roughly 5% to 10% of normal background noise. Both values are
validated at startup (`0 < mad_ema_alpha <= 1`, `epsilon > 0`). A stream reset
also resets its MAD EMA, preventing an old operating regime from affecting the
newly primed window. Detection details expose `median_absolute_deviation`,
`smoothed_mad`, `stabilized_mad`, `mad_ema_alpha`, and `epsilon` for tuning.

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

By default, each raw multivariate score is normalized by its configured
`score_detector` (`z_score` or `mad`). Set
`parameters.score_detector_enabled: false` to publish and threshold the raw
Mahalanobis distance or reconstruction error directly; the output then uses
`score_algorithm: raw`.

The candidate is scored against the preceding multivariate baseline and is
rolled into it afterward. The resulting Mahalanobis distance or reconstruction
error is detrended and deseasonalized when those stages are enabled, then
smoothed with the configured causal smoother before it enters its configured
`z_score` or `mad` score window. Only that second-stage result
is a final detection. Key parameters are `window_size`,
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

The `kafka.logging` options in the YAML control full structured record logs:
`log_consumed_messages` logs telemetry after it passes the configured header
allow-list, and `log_produced_results` logs each successfully published anomaly
message, including `anomaly_score`. Both are enabled in the sample config and
can be disabled for lower log volume or to avoid recording telemetry values.
Use `RUST_LOG=anomaly_detect=debug` to also see interpolation and
model-readiness events and normal-score summaries; anomalies are logged at
warning level. With `invalid_message_policy: skip`, malformed selected messages
are logged and committed; `fail` stops before committing the offending record.
## Dataset mode

`anomaly-detect` has two exclusive input modes: `streaming` (the default) and
`dataset`. Streaming mode consumes `iot-sim` telemetry from Kafka and publishes
anomaly results to Kafka as before. Dataset mode downloads a single Parquet
object generated by `iot-sim` from RustFS/S3, reads the complete file into the
application, orders its metric rows by source timestamp, then feeds every
matching row through the same `Pipeline`, window store, preprocessing, and
models used by streaming mode. It does not consume or produce Kafka records.

Set the following in a deployment-specific configuration to run dataset mode:

```yaml
mode: "dataset"
dataset:
  parquet_key: "dataset/iot-telemetry-<start-epoch-ms>.parquet"
  s3_endpoint_url: "http://192.168.1.50:9000"
  s3_bucket_name: "iotsim"
  s3_access_key: "access"
  s3_secret_key: "secret"
```

The run exits after the full dataset has been evaluated. It writes every final
model result (normal and anomalous) to
`iotsim/results/<dataset-name>-results.parquet`, logs anomalous detections, and
reports final counts. The Parquet rows include model identity, input list,
window counts, anomaly state and score, and JSON-encoded details. Use `ANOMALY_DETECT__DATASET__S3_ACCESS_KEY` and
`ANOMALY_DETECT__DATASET__S3_SECRET_KEY` to keep credentials out of YAML.
