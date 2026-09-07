# Architecture

```text
Kafka topic
    |
    | decode four headers only
    v
Header allow-list (configured streams) -- no match --> commit/ignore
    |
    | selected payload only
    v
Telemetry validation
    |
    v
Per-stream raw event-time window (interpolate/reset, insert, evict)
    |
    v
scale
    +-----------------------------+-------------------------------+
    | Holt -> sliding DFT -> smooth| smooth only
    v                             v
Univariate metric view    Multivariate metric view
    |                             | every input primed, fresh, aligned
    v                             v
Z-score / MAD             vector synchronization
                                  |
                                  v
               Mahalanobis / PCA / RFF-PCA / GSTA
                                  |
                                  | scalar distance/error/MSE
                                  v
                  Holt -> sliding DFT -> smooth
                                  |
                                  v
                         score window -> Z-score / MAD
                                  |
                                  v
                           final detections
                                  |
                                  v
                    Kafka anomaly-result topic
```

`Pipeline` is transport-independent: Kafka extracts headers and bytes, while
the pipeline routes a decoded `TelemetryMessage` into `WindowStore`. Named
streams decouple physical sensor metadata from model inputs and allow multiple
models to share one retained window.

`WindowStore` owns a `BTreeMap<String, StreamWindow>`. Each `StreamWindow` owns
the raw `TimeWindow`, distinct univariate and multivariate processed deques,
and its causal decomposition state. `TimeWindow` tracks an event-time
watermark, rejects older records, keeps samples timestamp-sorted, and retains
the configured number of samples.

The two decomposition states have deliberately different owners:

| State | Input | Owner | Used by |
| --- | --- | --- | --- |
| Metric decomposer | newest scaled metric value | `StreamWindow` | direct univariate models only |
| Score decomposer | newest multivariate scalar score | one multivariate model | that model's final Z-score/MAD only |
| MAD EMA | raw MAD from the current score window | one `Mad` model | stabilized MAD denominator |

`Mad` keeps its EMA behind a small synchronous mutex because model evaluation
uses a shared model reference. Each evaluation computes the current raw MAD,
updates that model's EMA once, and scores with
`0.6745 * abs(value - median) / (smoothed_mad + epsilon)`. The epsilon floor
keeps a flat window's non-median scores finite. When an excessive input gap
resets a stream, the coordinator also clears the affected direct MAD EMA; a
multivariate score-pipeline reset clears its contained final-detector EMA.

The first spectral window is initialized through RustFFT. Afterwards the
streaming decomposer updates every DFT bin with the outgoing/incoming delta in
O(N) per point, then uses the dominant permitted frequency to select a
phase-aligned seasonal lag. Holt state (level and slope) is updated before this
seasonal stage. Neither state is present in the multivariate metric branch.

The previous newest raw point and incoming point drive bounded interpolation.
Missing points above 20% of the sample window clear that stream and restart
priming. Otherwise any interpolated points and the real point are inserted
before retention is enforced. Scaling and smoothing views are regenerated from
the retained raw deque; Holt and sliding-DFT state advance once per incoming or
interpolated point. Models never receive a partially transformed window.

Priming requires `window.sample_count` retained values. Every configured
univariate detector whose named stream changed is evaluated
against the latest value in that processed window. Multiple detectors may read
the same retained window, while detectors for other parameters use their own
windows. Each Kafka payload supplies its cadence as `interval_ms`.

For a multivariate model, `ModelCoordinator` presents inputs in configured
order. The stateful model accepts a vector only when every input timestamp is
newer than the corresponding timestamp in its previous accepted vector and the
newest/oldest timestamp difference is at most `max_time_skew_ms`. This bounded,
all-inputs-fresh synchronization prevents fast sensors from manufacturing
extra observations by repeatedly carrying a slow sensor value forward.

The multivariate engine owns a sample-count rolling baseline. A candidate is
scored before baseline insertion, avoiding self-masking. Mahalanobis maintains
mean and M2 with forward/reverse Welford updates; PCA rebuilds the centered
baseline matrix and performs an SVD; kernel PCA first maps values through its
fixed RFF transform. Each raw score is detrended and deseasonalized, then passed
through the configured causal smoother before it is appended to a separate
fixed-count score window and evaluated by the existing Z-score or MAD
implementation. The final
`Detection.score` is the second-stage univariate score;
`details.multivariate_score` retains the raw distance, reconstruction error,
or GSTA MSE.

GSTA is the temporal-window exception to the point-at-a-time multivariate
engines. It requires equal configured cadences and builds a channel-major Burn
tensor only when the latest `window_size` samples are positionally aligned
within `max_time_skew_ms`. Conv1d layers preserve the temporal axis while
compressing channels to the latent width. The graph-attention layer treats
latent channels as a fully connected graph and learns per-window dependency
weights. Its source/target additive attention calculation is equivalent to a
linear GAT attention vector without allocating a five-dimensional concatenated
pair tensor. Transposed convolutions reconstruct the temporal input, with a
linear final layer so signed standard-scaled values remain representable.

The WGPU model and Adam optimizer are mutable and live inside the existing
`SharedMultivariateModel` mutex. This preserves Kafka event order and prevents
concurrent inference/backpropagation from racing over weights. Gated windows
use Burn's non-autodiff model view and do not build a backward graph. Clean
windows run a second autodiff pass and optimizer step. The first 128 windows by
default are an unconditional, output-suppressed warmup; all later MSE values,
including gated ones, flow into the common final MAD/Z-score window.

Every final detector also counts all anomalous points in its current scoring
window. Kafka maps the detection and its model configuration to the
`iot.anomaly.v1` message contract, waits for broker acknowledgement, and only
then commits the source offset. This is not an exactly-once transaction: a
failure after an output acknowledgement but before the input commit can replay
a result, so downstream consumers must tolerate duplicates. Window and model
state also remain process-local and must warm up again after a restart.

For direct MAD models, `parameters.mad_ema_alpha` defaults to `0.05` and
`parameters.epsilon` defaults to `1e-6`; both are per-model. The detection
details retain raw and smoothed MAD values so denominator movement is
observable in production.

Kafka messages are keyed by `entity_id` in `iot-sim`, preserving ordering for
an entity. The lateness bound still protects state from replayed or anomalously
old events. State is currently in memory and is rebuilt after restart. An
excessive gap in any input clears the affected multivariate baseline, RFF auto
calibration, and score window. Durable window/model checkpoints can be
introduced later behind `WindowStore` and the model coordinator.

The preprocessing implementation is split under `src/preprocess`: interpolation,
scaling, streaming decomposition, and smoothing contain the algorithms, while
`src/preprocess.rs` builds the non-decomposition transforms and `WindowStore`
routes the two metric views. Univariate
models follow the same modern module layout: `src/univariate.rs` declares the
module and `src/univariate/z_score.rs` and `src/univariate/mad.rs` implement the
detectors. `src/multivariate.rs` owns aligned-vector and score-pipeline
coordination; `src/multivariate/mahalanobis.rs` and
`src/multivariate/pca.rs` contain the numerical engines. Stateful multivariate
models are held behind cloneable `Arc<std::sync::Mutex<_>>` handles. The lock is
never held across an `.await`; the outer `Pipeline` remains protected by its
Tokio mutex at the Kafka boundary.
## Dataset execution

Dataset execution is an alternative ingress path, selected by `mode: dataset`.
It downloads the configured `dataset/*.parquet` object from the same
RustFS-compatible S3 endpoint configured for `iot-sim`, materializes the whole
long-form telemetry file, sorts it chronologically, constructs the equivalent
identity headers from each Parquet row, and invokes `Pipeline::ingest`. The
pipeline therefore owns the same sliding windows, preprocessing, and model
evaluation in both dataset and Kafka streaming modes. Dataset mode performs no
Kafka consume or result-produce work. It writes every final model evaluation to
`results/<dataset-name>-results.parquet` in the same `iotsim` bucket. The
result Parquet schema preserves event time, model/algorithm identity, inputs,
window counts, anomaly state and score, with model details encoded as JSON.
