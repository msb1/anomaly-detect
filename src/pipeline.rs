use std::collections::BTreeMap;

use thiserror::Error;

use crate::config::{AppConfig, ModelConfig, StreamConfig};
use crate::model::{Detection, ModelCoordinator, ModelError, ModelInput};
use crate::telemetry::{HeaderMap, TelemetryError, TelemetryMessage};
use crate::window::{Sample, WindowStore, WindowUpdate};

#[derive(Debug)]
pub struct Pipeline {
    streams: BTreeMap<String, StreamConfig>,
    windows: WindowStore,
    models: ModelCoordinator,
}

#[derive(Debug)]
pub struct PipelineOutcome {
    pub matched_streams: Vec<String>,
    pub late_streams: Vec<String>,
    pub reset_streams: Vec<String>,
    pub interpolated_points: BTreeMap<String, usize>,
    pub ready_models: Vec<ModelConfig>,
    pub detections: Vec<Detection>,
}

impl Pipeline {
    pub fn new(config: &AppConfig) -> Result<Self, PipelineError> {
        Ok(Self {
            streams: config.streams.clone(),
            windows: WindowStore::new(&config.streams, &config.window, &config.preprocessing),
            models: ModelCoordinator::with_preprocessing(
                config.models.clone(),
                &config.preprocessing,
            )?,
        })
    }

    /// Finds candidate streams using headers without touching the JSON payload.
    pub fn matching_streams(&self, headers: &HeaderMap) -> Vec<String> {
        self.streams
            .iter()
            .filter(|(_, stream)| {
                stream
                    .headers
                    .iter()
                    .all(|(key, expected)| headers.get(key) == Some(expected))
            })
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn ingest(
        &mut self,
        matched_streams: Vec<String>,
        headers: &HeaderMap,
        message: TelemetryMessage,
    ) -> Result<PipelineOutcome, PipelineError> {
        message.validate_headers(headers)?;
        let sample = Sample {
            timestamp_ms: message.timestamp_ms,
            value: message.value,
        };
        let mut accepted = Vec::new();
        let mut late = Vec::new();
        let mut reset = Vec::new();
        let mut interpolated = BTreeMap::new();
        for stream in matched_streams {
            match self.windows.push(&stream, sample, message.interval_ms) {
                Some(WindowUpdate::Accepted {
                    interpolated_points,
                    window_cleared,
                }) => {
                    if interpolated_points > 0 {
                        interpolated.insert(stream.clone(), interpolated_points);
                    }
                    if window_cleared {
                        reset.push(stream.clone());
                    }
                    accepted.push(stream);
                }
                Some(WindowUpdate::TooLate) => late.push(stream),
                None => return Err(PipelineError::UnknownStream(stream)),
            }
        }
        if !reset.is_empty() {
            self.models.reset_for_streams(&reset)?;
        }
        let ready_model_indices = self.models.ready_model_indices(&accepted, |input, kind| {
            self.windows
                .get(input)
                .is_some_and(|window| window.is_primed(kind))
        });
        let mut ready_models = Vec::with_capacity(ready_model_indices.len());
        let mut detections = Vec::new();
        for index in ready_model_indices {
            let config = self.models.config(index);
            let streams = config
                .inputs
                .iter()
                .map(|name| {
                    let samples = self
                        .windows
                        .get(name)
                        .and_then(|window| window.processed_samples(config.kind))
                        .ok_or_else(|| {
                            ModelError::new(format!(
                                "model '{}' input stream '{name}' is not processed and primed",
                                config.id
                            ))
                        })?;
                    Ok((name.as_str(), samples))
                })
                .collect::<Result<BTreeMap<_, _>, ModelError>>()?;
            if let Some(detection) = self.models.evaluate(index, ModelInput { streams })? {
                detections.push(detection);
            }
            ready_models.push(config.clone());
        }
        Ok(PipelineOutcome {
            matched_streams: accepted,
            late_streams: late,
            reset_streams: reset,
            interpolated_points: interpolated,
            ready_models,
            detections,
        })
    }
}

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error(transparent)]
    Telemetry(#[from] TelemetryError),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error("unknown configured stream '{0}'")]
    UnknownStream(String),
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use crate::config::*;

    use super::*;

    fn config() -> AppConfig {
        let stream = |metric: &str| StreamConfig {
            headers: BTreeMap::from([
                ("entity_id".into(), "tank-1".into()),
                ("sensor_id".into(), "sensor-1".into()),
                ("sensor_type".into(), "weather".into()),
                ("metric".into(), metric.into()),
            ]),
        };
        AppConfig {
            mode: RunMode::Streaming,
            kafka: KafkaConfig {
                brokers: "localhost:9092".into(),
                topic: "telemetry".into(),
                output_topic: "iot.anomaly.v1".into(),
                group_id: "group".into(),
                client_id: "client".into(),
                auto_offset_reset: "latest".into(),
                invalid_message_policy: InvalidMessagePolicy::Skip,
                logging: KafkaLoggingConfig::default(),
                security: KafkaSecurityConfig::default(),
            },
            dataset: DatasetConfig::default(),
            window: WindowConfig { sample_count: 4 },
            preprocessing: PreprocessingConfig::default(),
            streams: BTreeMap::from([
                ("temperature".into(), stream("temperature_c")),
                ("humidity".into(), stream("humidity_pct")),
            ]),
            models: vec![ModelConfig {
                id: "weather-context".into(),
                enabled: true,
                kind: ModelKind::Multivariate,
                algorithm: "mahalanobis".into(),
                inputs: vec!["temperature".into(), "humidity".into()],
                parameters: BTreeMap::from([
                    ("window_size".into(), Value::from(3)),
                    ("max_time_skew_ms".into(), Value::from(10)),
                    ("regularization".into(), Value::from(1e-6)),
                    ("shrinkage".into(), Value::from(0.1)),
                    ("score_detector".into(), Value::from("mad")),
                    ("score_window_size".into(), Value::from(3)),
                ]),
                thresholds: BTreeMap::from([("score".into(), Value::from(3.5))]),
            }],
        }
    }

    fn message(metric: &str, timestamp_ms: i64) -> TelemetryMessage {
        TelemetryMessage {
            entity_type: "station".into(),
            entity_id: "tank-1".into(),
            sensor_id: "sensor-1".into(),
            sensor_type: "weather".into(),
            timestamp_ms,
            interval_ms: 1,
            metric: metric.into(),
            value: 20.0,
        }
    }

    fn message_with_value(metric: &str, timestamp_ms: i64, value: f64) -> TelemetryMessage {
        TelemetryMessage {
            value,
            ..message(metric, timestamp_ms)
        }
    }

    #[test]
    fn multivariate_model_waits_for_every_input_window() {
        let mut pipeline = Pipeline::new(&config()).unwrap();
        for timestamp in [0, 10, 20, 30] {
            let headers = config().streams["temperature"].headers.clone();
            let matches = pipeline.matching_streams(&headers);
            assert!(
                pipeline
                    .ingest(matches, &headers, message("temperature_c", timestamp))
                    .unwrap()
                    .ready_models
                    .is_empty()
            );
        }
        for timestamp in [0, 10, 20, 30] {
            let headers = config().streams["humidity"].headers.clone();
            let matches = pipeline.matching_streams(&headers);
            let outcome = pipeline
                .ingest(matches, &headers, message("humidity_pct", timestamp))
                .unwrap();
            if timestamp == 30 {
                assert_eq!(outcome.ready_models[0].id, "weather-context");
            }
        }
    }

    #[test]
    fn evaluates_multiple_univariate_models_for_one_changed_stream() {
        let mut config = config();
        config.models = vec![
            ModelConfig {
                id: "temperature-z".into(),
                enabled: true,
                kind: ModelKind::Univariate,
                algorithm: "z_score".into(),
                inputs: vec!["temperature".into()],
                parameters: BTreeMap::new(),
                thresholds: BTreeMap::from([("score".into(), Value::from(1.5))]),
            },
            ModelConfig {
                id: "temperature-mad".into(),
                enabled: true,
                kind: ModelKind::Univariate,
                algorithm: "mad".into(),
                inputs: vec!["temperature".into()],
                parameters: BTreeMap::new(),
                thresholds: BTreeMap::from([("score".into(), Value::from(3.0))]),
            },
        ];
        config.validate().unwrap();
        let headers = config.streams["temperature"].headers.clone();
        let mut pipeline = Pipeline::new(&config).unwrap();

        for (timestamp, value) in [(0, 1.0), (3, 1.0), (6, 1.0)] {
            let matches = pipeline.matching_streams(&headers);
            let outcome = pipeline
                .ingest(
                    matches,
                    &headers,
                    message_with_value("temperature_c", timestamp, value),
                )
                .unwrap();
            assert!(outcome.detections.is_empty());
        }
        let matches = pipeline.matching_streams(&headers);
        let outcome = pipeline
            .ingest(
                matches,
                &headers,
                message_with_value("temperature_c", 10, 10.0),
            )
            .unwrap();

        assert_eq!(outcome.detections.len(), 2);
        assert!(outcome.detections.iter().all(|result| result.anomalous));
        assert_eq!(
            outcome
                .detections
                .iter()
                .map(|result| result.model_id.as_str())
                .collect::<Vec<_>>(),
            ["temperature-z", "temperature-mad"]
        );
    }
}
