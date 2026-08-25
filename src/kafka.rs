use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use rdkafka::Message;
use rdkafka::config::ClientConfig;
use rdkafka::consumer::{CommitMode, Consumer, StreamConsumer};
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::config::{InvalidMessagePolicy, KafkaConfig, ModelConfig, ModelKind};
use crate::model::Detection;
use crate::pipeline::Pipeline;
use crate::telemetry::{TelemetryMessage, decode_headers};

pub const ANOMALY_SCHEMA_VERSION: &str = "iot.anomaly.v1";

/// Kafka value emitted for every final univariate detection result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnomalyMessage {
    pub schema_version: String,
    pub timestamp_ms: i64,
    pub model_id: String,
    pub model_kind: ModelKind,
    pub algorithm: String,
    pub score_algorithm: String,
    pub inputs: Vec<String>,
    pub anomalous_points: usize,
    pub window_sample_count: usize,
    pub is_anomalous: bool,
    pub anomaly_score: Option<f64>,
    pub details: BTreeMap<String, Value>,
}

impl AnomalyMessage {
    fn from_detection(detection: Detection, model: &ModelConfig) -> Self {
        let score_algorithm = model
            .parameters
            .get("score_detector")
            .and_then(Value::as_str)
            .unwrap_or(&model.algorithm)
            .into();
        Self {
            schema_version: ANOMALY_SCHEMA_VERSION.into(),
            timestamp_ms: detection.timestamp_ms,
            model_id: detection.model_id,
            model_kind: model.kind,
            algorithm: model.algorithm.clone(),
            score_algorithm,
            inputs: model.inputs.clone(),
            anomalous_points: detection.anomalous_points,
            window_sample_count: detection.window_sample_count,
            is_anomalous: detection.anomalous,
            anomaly_score: detection.score,
            details: detection.details,
        }
    }
}

pub async fn consume(
    config: &KafkaConfig,
    pipeline: Arc<Mutex<Pipeline>>,
    cancellation: CancellationToken,
) -> Result<()> {
    let consumer = create_consumer(config)?;
    let producer = create_producer(config)?;
    consumer
        .subscribe(&[&config.topic])
        .context("could not subscribe to Kafka topic")?;
    tracing::info!(topic = %config.topic, brokers = %config.brokers, group_id = %config.group_id, "Kafka consumer started");

    let mut messages = consumer.stream();
    loop {
        tokio::select! {
            _ = cancellation.cancelled() => break,
            item = messages.next() => {
                let Some(item) = item else { break };
                match item {
                    Ok(message) => {
                        match process_message(&pipeline, &message).await {
                            Ok(results) => publish_results(config, &producer, results).await?,
                            Err(error) => {
                                match config.invalid_message_policy {
                                    InvalidMessagePolicy::Skip => tracing::warn!(%error, partition = message.partition(), offset = message.offset(), "skipping invalid Kafka message"),
                                    InvalidMessagePolicy::Fail => return Err(error),
                                }
                            }
                        }
                        consumer.commit_message(&message, CommitMode::Async)
                            .context("could not enqueue Kafka offset commit")?;
                    }
                    Err(error) => tracing::warn!(%error, "Kafka receive error"),
                }
            }
        }
    }
    consumer
        .commit_consumer_state(CommitMode::Sync)
        .context("could not commit Kafka offsets during shutdown")?;
    tracing::info!("Kafka consumer stopped");
    Ok(())
}

fn create_consumer(config: &KafkaConfig) -> Result<StreamConsumer> {
    let mut client = ClientConfig::new();
    client
        .set("bootstrap.servers", &config.brokers)
        .set("group.id", &config.group_id)
        .set("client.id", &config.client_id)
        .set("auto.offset.reset", &config.auto_offset_reset)
        .set("enable.auto.commit", "false")
        .set("enable.auto.offset.store", "false")
        .set("isolation.level", "read_committed");
    configure_security(&mut client, config);
    client.create().context("could not create Kafka consumer")
}

fn create_producer(config: &KafkaConfig) -> Result<FutureProducer> {
    let mut client = ClientConfig::new();
    client
        .set("bootstrap.servers", &config.brokers)
        .set("client.id", &config.client_id)
        .set("message.timeout.ms", "10000");
    configure_security(&mut client, config);
    client.create().context("could not create Kafka producer")
}

fn configure_security(client: &mut ClientConfig, config: &KafkaConfig) {
    if let Some(value) = &config.security.security_protocol {
        client.set("security.protocol", value);
    }
    if let Some(value) = &config.security.sasl_mechanism {
        client.set("sasl.mechanism", value);
    }
    if let Some(value) = &config.security.username {
        client.set("sasl.username", value);
    }
    if let Some(value) = &config.security.password {
        client.set("sasl.password", value);
    }
}

async fn process_message(
    pipeline: &Arc<Mutex<Pipeline>>,
    message: &rdkafka::message::BorrowedMessage<'_>,
) -> Result<Vec<AnomalyMessage>> {
    let headers = decode_headers(message.headers())?;
    let matched_streams = {
        let pipeline = pipeline.lock().await;
        pipeline.matching_streams(&headers)
    };
    if matched_streams.is_empty() {
        tracing::trace!(
            partition = message.partition(),
            offset = message.offset(),
            "message excluded by configured header filters"
        );
        return Ok(Vec::new());
    }

    let payload = message
        .payload()
        .context("matching Kafka message has no payload")?;
    let telemetry: TelemetryMessage = serde_json::from_slice(payload)
        .context("matching Kafka payload is not valid telemetry JSON")?;
    let outcome = {
        let mut pipeline = pipeline.lock().await;
        pipeline.ingest(matched_streams, &headers, telemetry)?
    };
    for stream in &outcome.late_streams {
        tracing::warn!(
            stream,
            partition = message.partition(),
            offset = message.offset(),
            "discarded telemetry older than the configured lateness bound"
        );
    }
    for stream in &outcome.reset_streams {
        tracing::warn!(
            stream,
            partition = message.partition(),
            offset = message.offset(),
            "interpolation gap exceeded 20% limit; cleared stream window and restarted priming"
        );
    }
    for (stream, count) in &outcome.interpolated_points {
        tracing::debug!(
            stream,
            count,
            "inserted bounded linear interpolation points"
        );
    }
    for model in &outcome.ready_models {
        tracing::trace!(model_id = %model.id, algorithm = %model.algorithm, kind = ?model.kind, "evaluated primed model inputs");
    }
    for detection in &outcome.detections {
        if detection.anomalous {
            tracing::warn!(
                model_id = %detection.model_id,
                score = ?detection.score,
                details = ?detection.details,
                "anomaly detected"
            );
        } else {
            tracing::debug!(
                model_id = %detection.model_id,
                score = ?detection.score,
                "sample is within the anomaly threshold"
            );
        }
    }
    if outcome.matched_streams.is_empty() && outcome.late_streams.is_empty() {
        bail!("message matched filters but was not routed");
    }
    outcome
        .detections
        .into_iter()
        .map(|detection| {
            let model = outcome
                .ready_models
                .iter()
                .find(|model| model.id == detection.model_id)
                .with_context(|| {
                    format!(
                        "detection references unknown ready model '{}'",
                        detection.model_id
                    )
                })?;
            Ok(AnomalyMessage::from_detection(detection, model))
        })
        .collect()
}

async fn publish_results(
    config: &KafkaConfig,
    producer: &FutureProducer,
    results: Vec<AnomalyMessage>,
) -> Result<()> {
    for result in results {
        let payload = serde_json::to_vec(&result).with_context(|| {
            format!(
                "could not encode anomaly result for model '{}'",
                result.model_id
            )
        })?;
        let record = FutureRecord::to(&config.output_topic)
            .key(&result.model_id)
            .payload(&payload);
        producer
            .send(record, Timeout::After(Duration::from_secs(5)))
            .await
            .map_err(|(error, _)| error)
            .with_context(|| {
                format!(
                    "could not publish anomaly result for model '{}' to topic '{}'",
                    result.model_id, config.output_topic
                )
            })?;
        tracing::debug!(
            topic = %config.output_topic,
            model_id = %result.model_id,
            timestamp_ms = result.timestamp_ms,
            anomalous_points = result.anomalous_points,
            anomaly_score = ?result.anomaly_score,
            "published anomaly result"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anomaly_message_carries_window_and_model_identity() {
        let model = ModelConfig {
            id: "temperature-mad".into(),
            enabled: true,
            kind: ModelKind::Univariate,
            algorithm: "mad".into(),
            inputs: vec!["outdoor_temperature".into()],
            parameters: BTreeMap::new(),
            thresholds: BTreeMap::from([("score".into(), Value::from(3.5))]),
        };
        let detection = Detection {
            model_id: model.id.clone(),
            timestamp_ms: 1_725_000_000_000,
            anomalous: true,
            score: Some(8.25),
            anomalous_points: 3,
            window_sample_count: 12,
            details: BTreeMap::from([("threshold".into(), Value::from(3.5))]),
        };

        let message = AnomalyMessage::from_detection(detection, &model);
        let json = serde_json::to_value(message).unwrap();

        assert_eq!(json["schema_version"], ANOMALY_SCHEMA_VERSION);
        assert_eq!(json["timestamp_ms"], 1_725_000_000_000_i64);
        assert_eq!(json["model_id"], "temperature-mad");
        assert_eq!(json["model_kind"], "univariate");
        assert_eq!(json["score_algorithm"], "mad");
        assert_eq!(json["inputs"][0], "outdoor_temperature");
        assert_eq!(json["anomalous_points"], 3);
        assert_eq!(json["window_sample_count"], 12);
        assert_eq!(json["anomaly_score"], 8.25);
    }

    #[test]
    fn non_finite_json_score_is_null_but_anomaly_state_is_preserved() {
        let message = AnomalyMessage {
            schema_version: ANOMALY_SCHEMA_VERSION.into(),
            timestamp_ms: 10,
            model_id: "zero-dispersion-mad".into(),
            model_kind: ModelKind::Univariate,
            algorithm: "mad".into(),
            score_algorithm: "mad".into(),
            inputs: vec!["metric".into()],
            anomalous_points: 1,
            window_sample_count: 4,
            is_anomalous: true,
            anomaly_score: Some(f64::INFINITY),
            details: BTreeMap::new(),
        };

        let json = serde_json::to_value(message).unwrap();
        assert!(json["anomaly_score"].is_null());
        assert_eq!(json["is_anomalous"], true);
        assert_eq!(json["anomalous_points"], 1);
    }

    #[test]
    fn checked_in_anomaly_schema_is_valid_json() {
        let schema = include_str!("../schemas/iot-anomaly-v1.schema.json");
        let parsed: Value = serde_json::from_str(schema).unwrap();
        assert_eq!(
            parsed["properties"]["schema_version"]["const"],
            ANOMALY_SCHEMA_VERSION
        );
    }
}
