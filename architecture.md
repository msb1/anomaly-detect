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
scale -> decompose to remainder -> smooth
    |
    v
Per-stream processed window
    +-----------------------------+-------------------------------+
    | one primed input            | every input primed, fresh, aligned
    v                             v
Z-score / MAD             vector synchronization
                                  |
                                  v
               Mahalanobis / PCA / RFF-PCA / GSTA
                                  |
                                  | scalar distance/error/MSE
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
the raw `TimeWindow`, its expected cadence, and an optional processed
`VecDeque`. `TimeWindow` tracks an event-time watermark, accepts bounded late
records, keeps samples timestamp-sorted, and evicts data older than `watermark
- duration_ms`.

The previous newest raw point and incoming point drive bounded interpolation.
Missing duration above 20% of the time window clears that stream and restarts
priming. Otherwise any interpolated points and the real point are inserted
before retention is enforced. The processed deque is then regenerated from the
retained raw deque. Models never receive a partially transformed window.

Priming records observed history separately from retained values, so it cannot
occur before a complete duration elapses. It also requires `minimum_samples`
and, when decomposition is enabled, enough data for two seasonal periods.
Every configured univariate detector whose named stream changed is evaluated
against the latest value in that processed window. Multiple detectors may read
the same retained window, while detectors for other parameters use their own
windows. Different cadences are represented by per-stream `interval_ms`.

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
fixed RFF transform. Each raw score is appended to a separate fixed-count score
window and evaluated by the existing Z-score or MAD implementation. The final
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

Kafka messages are keyed by `entity_id` in `iot-sim`, preserving ordering for
an entity. The lateness bound still protects state from replayed or anomalously
old events. State is currently in memory and is rebuilt after restart. An
excessive gap in any input clears the affected multivariate baseline, RFF auto
calibration, and score window. Durable window/model checkpoints can be
introduced later behind `WindowStore` and the model coordinator.

The preprocessing implementation is split under `src/preprocess`: interpolation,
scaling, decomposition, and smoothing contain the algorithms, while
`src/preprocess.rs` enforces ordering and builds the derived window. Univariate
models follow the same modern module layout: `src/univariate.rs` declares the
module and `src/univariate/z_score.rs` and `src/univariate/mad.rs` implement the
detectors. `src/multivariate.rs` owns aligned-vector and score-pipeline
coordination; `src/multivariate/mahalanobis.rs` and
`src/multivariate/pca.rs` contain the numerical engines. Stateful multivariate
models are held behind cloneable `Arc<std::sync::Mutex<_>>` handles. The lock is
never held across an `.await`; the outer `Pipeline` remains protected by its
Tokio mutex at the Kafka boundary.
