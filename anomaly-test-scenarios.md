# Industrial anomaly test scenarios

This document describes the detector-side contract for
`config/anomaly-detect.yaml`. Its source telemetry is defined by
`../iot-sim/config/simulation.yaml` and documented in
`../iot-sim/simulated-scenarios.md`.

## Configuration policy

The stream allow-list contains only metrics used by the nine requested anomaly
tests. The configured preprocessing uses standard scaling, Holt detrending,
sliding-DFT deseasonality, and a trailing moving average for direct univariate
scoring. Raw multivariate input vectors use only the standard scaling and
moving average; their scalar outputs receive Holt/DFT processing before the
final MAD score. Bounded interpolation remains enabled for occasional
transport gaps.

The direct water-hammer MAD model uses a `0.05` raw-MAD EMA and a `0.1` noise
floor after preprocessing. Its score is therefore finite even if the pressure
residual window becomes perfectly flat. When evaluating normal chatter, record
the reported raw and `smoothed_mad` values as well as alerts; abrupt changes in
the raw value should be reflected gradually in the smoothed denominator.

The shared metric window retains 840 samples. Multivariate models add
candidate-exclusive fixed-count baselines and separate score windows; GSTA
adds complete aligned temporal windows and gated self-training.

## Preprocessing verification

Before interpreting model accuracy, verify the preprocessing contract with a
known periodic stream. The main configuration uses a 240-sample seasonal window
and permits periods from 2 through 120 samples. A normal recurring cycle should
produce a near-zero direct univariate residual after warm-up; a phase-breaking
spike should remain in that residual. For a multivariate test, verify the
engine input values remain only scaled/smoothed while its result includes both
the raw `multivariate_score` and `decomposed_multivariate_score`. The latter,
not the raw value, is the one used by the final MAD/Z-score decision.

## MAD stability verification

After the water-hammer stream is primed, verify that result details contain
`median_absolute_deviation`, `smoothed_mad`, `stabilized_mad`,
`mad_ema_alpha`, and `epsilon`. During a flat residual interval,
`stabilized_mad` must remain at least epsilon and the score must remain finite.
After an excessive interpolation gap resets the stream, the first result after
the next full warm-up should initialize `smoothed_mad` to that window's raw
MAD rather than reuse the prior regime's state.

## Model matrix

| Scenario | Model | Inputs | Pass criterion |
| --- | --- | --- | --- |
| Pump water hammer | `water-hammer-z-score`, `water-hammer-mad` | line pressure | The 180 PSI point is anomalous on arrival in both; record whether the intentional 35 PSI collapse also alerts |
| Weekend rack drift | `rack-weekend-context-mahalanobis` | temperature, humidity, day of week | 23 C on Saturday alerts after warm-up even though weekday 24 C values are accepted |
| Cleanroom frozen moisture | `cleanroom-frozen-sequence-gsta` | moisture, cart passage | The in-range zero-variance 20-minute segment produces abnormal reconstruction scores; isolated normal points do not |
| FCCU feed spike | `fccu-feed-pressure-z-score` | discharge pressure | The single 750 PSI point alerts immediately |
| FCCU coolant restriction | `fccu-coolant-context-mahalanobis` | coolant flow, ambient temperature | 80 LPM at 39 C alerts; the same 80 LPM in the cold regime does not |
| FCCU wall erosion | `fccu-wall-erosion-gsta` | two skin thermocouples | The +0.1 C/day channel-relative trend eventually alerts while all values remain under 260 C |
| Stripper flooding | `stripper-dp-z-score` | differential pressure | The single 45 PSI point alerts immediately |
| Wrong reactor recipe | `reactor-recipe-context-mahalanobis` | agitator current, batch step | 5 A in phase 3/4 alerts; 5 A during phase 1 is accepted |
| Reactor jacket fouling | `reactor-jacket-fouling-gsta` | internal, jacket-inlet, jacket-outlet temperatures | The progressively stretched 45-to-60-minute cooling profiles alert without any value crossing 95 C |

## Point-anomaly tests

The three direct Z-score models use `ddof: 1` and deliberately high thresholds
(5–6 standard deviations). Normal pressure noise first primes the shared
event-time window. The extreme candidate then enters the direct scoring window
and should be emitted with `is_anomalous: true`. Capture detection timestamp,
source timestamp, score, and end-to-end latency. A point-test failure is either
a missed extreme sample or an alert rate on normal baseline that exceeds the
chosen acceptance budget.

Water hammer differs from the two industrial spikes because its source has a
second low-pressure impulse. Evaluate the 180 PSI point as the required point
anomaly and report the 35 PSI result separately; suppressing it is a policy
choice, not a simulator defect.

## Contextual-anomaly tests

Mahalanobis models evaluate correlation, not universal thresholds. The rack
model receives numeric day-of-week context in addition to temperature and
humidity. The coolant model receives ambient temperature with flow. The reactor
model receives batch step with motor current. For each, replay or run enough
normal paired regimes to prime the 64-vector baseline and 32-score final
window before scoring the injected context break.

Required controls are important:

- rack: verify weekday 24 C does not alert before testing Saturday 23 C;
- FCCU coolant: verify cold-regime 80 LPM does not alert before hot-regime
  80 LPM;
- reactor: verify phase-1 5 A does not alert before phase-3/4 5 A.

Because the checked-in simulator begins in Saturday, hot-day, and filling
contexts, long-running tests naturally reach their control regimes later in
the configured step cycles. A test harness may instead persist a trained model
run or accelerate both source and detector cadences consistently.

## Collective-anomaly tests

The GSTA models consume aligned channel windows and require WGPU. Their first
64 complete windows train unconditionally; afterward reconstruction errors at
or below the gate may update weights, while higher-error windows are scored but
not learned. Final alerting begins only after the post-warm-up error stream also
fills its MAD score window.

### Cleanroom

The window is 240 five-second positions, exactly 20 minutes. Moisture freezes at
10 PPM while cart-passage context continues. The event is successful only if
the sequence model recognizes the structural change; a test that depends on a
value outside 9.5–10.5 PPM is invalid.

### FCCU wall

The six-position hourly window compares two nominally synchronized daily-wave
channels. The model should accumulate evidence when one channel begins its
1.4 C/14-day micro-trend. Detection latency is expected and should be reported
in hours/days, not held to point-anomaly latency.

### Reactor jacket

The 105-position one-minute normal window spans one nominal batch. The source
then generates 15 batches with cooling duration increasing to 60 minutes while
the peak remains 85 C. Evaluate reconstruction error, alert onset by batch
number, and resistance to the normal rise/plateau/cool cycle.

## Operational procedure

1. Start Kafka and launch `iot-sim` with its checked-in YAML.
2. Start `anomaly-detect` with `config/anomaly-detect.yaml` on a host with a
   supported WGPU adapter.
3. Confirm all selected stream headers and intervals match the simulator.
4. Allow model-specific baseline and score windows to warm up before judging
   misses. GSTA has the longest multi-stage warm-up.
5. Consume `iot.anomaly.v1`, grouping results by `model_id` and source event
   time. Record true positives, normal-regime false positives, and latency.
6. Repeat with fixed simulator seeds. When accelerating, change simulator
   cadence, anomaly sample counts, detector intervals, and time-window sizes as
   one controlled test variant.

Thresholds and GSTA gate values are starting points, not production safety
limits. Calibrate them from repeated normal and injected runs, retain the
original configuration with each result set, and never interpret model output
as an automatic process-control command.
