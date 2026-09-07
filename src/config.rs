use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use config::{Config, ConfigError, Environment, File, FileFormat};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub mode: RunMode,
    #[serde(default)]
    pub kafka: KafkaConfig,
    #[serde(default)]
    pub dataset: DatasetConfig,
    pub window: WindowConfig,
    #[serde(default)]
    pub preprocessing: PreprocessingConfig,
    pub streams: BTreeMap<String, StreamConfig>,
    pub models: Vec<ModelConfig>,
}

impl AppConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigurationError> {
        let config: Self = Config::builder()
            .add_source(File::from(path.as_ref()).format(FileFormat::Yaml))
            .add_source(
                Environment::with_prefix("ANOMALY_DETECT")
                    .separator("__")
                    .try_parsing(true),
            )
            .build()?
            .try_deserialize()?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigurationError> {
        match self.mode {
            RunMode::Streaming => self.kafka.validate()?,
            RunMode::Dataset => self.dataset.validate()?,
        }
        if self.window.sample_count == 0 {
            return Err(ConfigurationError::Validation(
                "window.sample_count must be greater than zero".into(),
            ));
        }
        self.preprocessing.validate(&self.window)?;
        if self.streams.is_empty() {
            return Err(ConfigurationError::Validation(
                "at least one header-filtered stream is required".into(),
            ));
        }
        for (name, stream) in &self.streams {
            if name.trim().is_empty() || stream.headers.is_empty() {
                return Err(ConfigurationError::Validation(
                    "stream names and header filters cannot be empty".into(),
                ));
            }
            for required in ["entity_id", "sensor_id", "metric"] {
                if !stream.headers.contains_key(required) {
                    return Err(ConfigurationError::Validation(format!(
                        "stream '{name}' must filter on the '{required}' Kafka header"
                    )));
                }
            }
            if stream
                .headers
                .iter()
                .any(|(key, value)| key.trim().is_empty() || value.trim().is_empty())
            {
                return Err(ConfigurationError::Validation(format!(
                    "stream '{name}' has an empty header name or value"
                )));
            }
        }

        let mut model_ids = HashSet::new();
        for model in &self.models {
            if model.id.trim().is_empty() || !model_ids.insert(&model.id) {
                return Err(ConfigurationError::Validation(format!(
                    "model ids must be non-empty and unique; invalid id '{}'",
                    model.id
                )));
            }
            if model.algorithm.trim().is_empty() {
                return Err(ConfigurationError::Validation(format!(
                    "model '{}' must name an algorithm",
                    model.id
                )));
            }
            match model.kind {
                ModelKind::Univariate => validate_univariate_model(model, &self.window)?,
                ModelKind::Multivariate => validate_multivariate_model(model, &self.window)?,
            }
            let expected = match model.kind {
                ModelKind::Univariate => 1,
                ModelKind::Multivariate => 2,
            };
            if model.inputs.len() < expected
                || (model.kind == ModelKind::Univariate && model.inputs.len() != 1)
            {
                return Err(ConfigurationError::Validation(format!(
                    "{} model '{}' requires {} input stream{}",
                    match model.kind {
                        ModelKind::Univariate => "univariate",
                        ModelKind::Multivariate => "multivariate",
                    },
                    model.id,
                    if expected == 1 {
                        "exactly one"
                    } else {
                        "at least two"
                    },
                    if expected == 1 { "" } else { "s" }
                )));
            }
            let mut inputs = HashSet::new();
            for input in &model.inputs {
                if !self.streams.contains_key(input) {
                    return Err(ConfigurationError::Validation(format!(
                        "model '{}' references unknown stream '{input}'",
                        model.id
                    )));
                }
                if !inputs.insert(input) {
                    return Err(ConfigurationError::Validation(format!(
                        "model '{}' references stream '{input}' more than once",
                        model.id
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Only one input mode can be active for a run. Dataset mode intentionally
/// performs no Kafka consume or produce operations.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    #[default]
    Streaming,
    Dataset,
}

fn validate_score_threshold(model: &ModelConfig) -> Result<(), ConfigurationError> {
    model
        .thresholds
        .get("score")
        .and_then(Value::as_f64)
        .filter(|threshold| threshold.is_finite() && *threshold > 0.0)
        .ok_or_else(|| {
            ConfigurationError::Validation(format!(
                "model '{}' thresholds.score must be a finite positive number",
                model.id
            ))
        })?;
    Ok(())
}

fn usize_parameter(
    model: &ModelConfig,
    name: &str,
    default: Option<usize>,
) -> Result<usize, ConfigurationError> {
    match model.parameters.get(name) {
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                ConfigurationError::Validation(format!(
                    "model '{}' parameters.{name} must be a non-negative integer",
                    model.id
                ))
            }),
        None => default.ok_or_else(|| {
            ConfigurationError::Validation(format!(
                "model '{}' requires parameters.{name}",
                model.id
            ))
        }),
    }
}

fn f64_parameter(
    model: &ModelConfig,
    name: &str,
    default: Option<f64>,
) -> Result<f64, ConfigurationError> {
    match model.parameters.get(name) {
        Some(value) => value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                ConfigurationError::Validation(format!(
                    "model '{}' parameters.{name} must be a finite number",
                    model.id
                ))
            }),
        None => default.ok_or_else(|| {
            ConfigurationError::Validation(format!(
                "model '{}' requires parameters.{name}",
                model.id
            ))
        }),
    }
}

fn validate_univariate_model(
    model: &ModelConfig,
    window: &WindowConfig,
) -> Result<(), ConfigurationError> {
    if !matches!(model.algorithm.as_str(), "z_score" | "mad") {
        return Err(ConfigurationError::Validation(format!(
            "univariate model '{}' uses unsupported algorithm '{}'; expected 'z_score' or 'mad'",
            model.id, model.algorithm
        )));
    }
    validate_score_threshold(model)?;
    if model.algorithm == "z_score" {
        let ddof = usize_parameter(model, "ddof", Some(0))?;
        if ddof >= window.sample_count {
            return Err(ConfigurationError::Validation(format!(
                "Z-score model '{}' parameters.ddof must be less than window.sample_count",
                model.id
            )));
        }
    } else {
        let mad_ema_alpha = f64_parameter(model, "mad_ema_alpha", Some(0.05))?;
        if !(0.0 < mad_ema_alpha && mad_ema_alpha <= 1.0) {
            return Err(ConfigurationError::Validation(format!(
                "MAD model '{}' parameters.mad_ema_alpha must be in (0, 1]",
                model.id
            )));
        }
        let epsilon = f64_parameter(model, "epsilon", Some(1e-6))?;
        if epsilon <= 0.0 {
            return Err(ConfigurationError::Validation(format!(
                "MAD model '{}' parameters.epsilon must be positive",
                model.id
            )));
        }
    }
    Ok(())
}

fn validate_multivariate_model(
    model: &ModelConfig,
    window: &WindowConfig,
) -> Result<(), ConfigurationError> {
    if !matches!(
        model.algorithm.as_str(),
        "mahalanobis" | "pca" | "kernel_pca" | "gsta"
    ) {
        return Err(ConfigurationError::Validation(format!(
            "multivariate model '{}' uses unsupported algorithm '{}'; expected 'mahalanobis', 'pca', 'kernel_pca', or 'gsta'",
            model.id, model.algorithm
        )));
    }
    validate_score_threshold(model)?;

    // A day-of-week input is represented by its sine and cosine coordinates
    // inside multivariate engines, replacing the ordinal source coordinate.
    let dimensions = model.inputs.len();
    let window_size = usize_parameter(model, "window_size", None)?;
    if window_size < 2 {
        return Err(ConfigurationError::Validation(format!(
            "multivariate model '{}' parameters.window_size must be at least 2",
            model.id
        )));
    }
    let max_skew = usize_parameter(model, "max_time_skew_ms", None)?;
    let _ = i64::try_from(max_skew).map_err(|_| {
        ConfigurationError::Validation(format!(
            "multivariate model '{}' parameters.max_time_skew_ms is too large",
            model.id
        ))
    })?;

    let score_detector_enabled = match model.parameters.get("score_detector_enabled") {
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            return Err(ConfigurationError::Validation(format!(
                "multivariate model '{}' parameters.score_detector_enabled must be boolean",
                model.id
            )));
        }
        None => true,
    };
    if score_detector_enabled {
        let score_detector = model
            .parameters
            .get("score_detector")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ConfigurationError::Validation(format!(
                    "multivariate model '{}' requires string parameters.score_detector when parameters.score_detector_enabled is true",
                    model.id
                ))
            })?;
        if !matches!(score_detector, "z_score" | "mad") {
            return Err(ConfigurationError::Validation(format!(
                "multivariate model '{}' parameters.score_detector must be 'z_score' or 'mad'",
                model.id
            )));
        }
        let score_window_size = usize_parameter(model, "score_window_size", None)?;
        if score_window_size < 2 {
            return Err(ConfigurationError::Validation(format!(
                "multivariate model '{}' parameters.score_window_size must be at least 2",
                model.id
            )));
        }
        if score_detector == "z_score"
            && usize_parameter(model, "score_ddof", Some(0))? >= score_window_size
        {
            return Err(ConfigurationError::Validation(format!(
                "multivariate model '{}' parameters.score_ddof must be less than parameters.score_window_size",
                model.id
            )));
        }
    }

    match model.algorithm.as_str() {
        "mahalanobis" => {
            if window_size <= dimensions {
                return Err(ConfigurationError::Validation(format!(
                    "Mahalanobis model '{}' parameters.window_size must exceed its {} input dimensions",
                    model.id, dimensions
                )));
            }
            let regularization = f64_parameter(model, "regularization", Some(1e-6))?;
            if regularization <= 0.0 {
                return Err(ConfigurationError::Validation(format!(
                    "Mahalanobis model '{}' parameters.regularization must be positive",
                    model.id
                )));
            }
            let shrinkage = f64_parameter(model, "shrinkage", Some(0.0))?;
            if !(0.0..=1.0).contains(&shrinkage) {
                return Err(ConfigurationError::Validation(format!(
                    "Mahalanobis model '{}' parameters.shrinkage must be between 0 and 1",
                    model.id
                )));
            }
        }
        "pca" | "kernel_pca" => {
            let working_dim = if model.algorithm == "kernel_pca" {
                let rff_dimension = usize_parameter(model, "rff_dimension", None)?;
                if rff_dimension < 2 {
                    return Err(ConfigurationError::Validation(format!(
                        "kernel PCA model '{}' parameters.rff_dimension must be at least 2",
                        model.id
                    )));
                }
                match model.parameters.get("gamma") {
                    Some(Value::String(value)) if value == "auto" => {}
                    Some(value)
                        if value
                            .as_f64()
                            .is_some_and(|gamma| gamma.is_finite() && gamma > 0.0) => {}
                    _ => {
                        return Err(ConfigurationError::Validation(format!(
                            "kernel PCA model '{}' parameters.gamma must be a finite positive number or 'auto'",
                            model.id
                        )));
                    }
                }
                rff_dimension
            } else {
                dimensions
            };
            let retained = usize_parameter(model, "retained_components", None)?;
            if retained == 0 || retained >= working_dim.min(window_size) {
                return Err(ConfigurationError::Validation(format!(
                    "PCA model '{}' parameters.retained_components must be positive and less than min(window_size, working dimension)",
                    model.id
                )));
            }
            let _ = usize_parameter(model, "seed", Some(0))?;
        }
        "gsta" => {
            let latent_channels = usize_parameter(model, "latent_channels", Some(16))?;
            if latent_channels < 2 {
                return Err(ConfigurationError::Validation(format!(
                    "GSTA model '{}' parameters.latent_channels must be at least 2",
                    model.id
                )));
            }
            let attention_heads = usize_parameter(model, "attention_heads", Some(2))?;
            if attention_heads == 0 || !window_size.is_multiple_of(attention_heads) {
                return Err(ConfigurationError::Validation(format!(
                    "GSTA model '{}' parameters.attention_heads must be positive and divide parameters.window_size",
                    model.id
                )));
            }
            let gate_threshold = f64_parameter(model, "gate_threshold", None)?;
            if gate_threshold <= 0.0 {
                return Err(ConfigurationError::Validation(format!(
                    "GSTA model '{}' parameters.gate_threshold must be positive",
                    model.id
                )));
            }
            let learning_rate = f64_parameter(model, "learning_rate", Some(0.001))?;
            if learning_rate <= 0.0 {
                return Err(ConfigurationError::Validation(format!(
                    "GSTA model '{}' parameters.learning_rate must be positive",
                    model.id
                )));
            }
            let warmup_steps = usize_parameter(model, "warmup_steps", Some(128))?;
            if warmup_steps == 0 {
                return Err(ConfigurationError::Validation(format!(
                    "GSTA model '{}' parameters.warmup_steps must be greater than zero",
                    model.id
                )));
            }
            let _ = usize_parameter(model, "seed", Some(0))?;

            if window_size > window.sample_count {
                return Err(ConfigurationError::Validation(format!(
                    "GSTA model '{}' parameters.window_size cannot exceed window.sample_count",
                    model.id
                )));
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum ConfigurationError {
    #[error("could not load configuration: {0}")]
    Load(#[from] ConfigError),
    #[error("invalid configuration: {0}")]
    Validation(String),
}

#[derive(Debug, Clone, Deserialize)]
pub struct KafkaConfig {
    #[serde(default)]
    pub brokers: String,
    #[serde(default)]
    pub topic: String,
    #[serde(default = "default_output_topic")]
    pub output_topic: String,
    #[serde(default)]
    pub group_id: String,
    #[serde(default)]
    pub client_id: String,
    #[serde(default = "default_offset_reset")]
    pub auto_offset_reset: String,
    #[serde(default)]
    pub invalid_message_policy: InvalidMessagePolicy,
    #[serde(default)]
    pub logging: KafkaLoggingConfig,
    #[serde(default)]
    pub security: KafkaSecurityConfig,
}

impl Default for KafkaConfig {
    fn default() -> Self {
        Self {
            brokers: String::new(),
            topic: String::new(),
            output_topic: default_output_topic(),
            group_id: String::new(),
            client_id: String::new(),
            auto_offset_reset: default_offset_reset(),
            invalid_message_policy: InvalidMessagePolicy::default(),
            logging: KafkaLoggingConfig::default(),
            security: KafkaSecurityConfig::default(),
        }
    }
}

impl KafkaConfig {
    fn validate(&self) -> Result<(), ConfigurationError> {
        if self.brokers.trim().is_empty()
            || self.topic.trim().is_empty()
            || self.output_topic.trim().is_empty()
        {
            return Err(ConfigurationError::Validation("kafka.brokers, kafka.topic, and kafka.output_topic cannot be empty in streaming mode".into()));
        }
        if self.group_id.trim().is_empty() || self.client_id.trim().is_empty() {
            return Err(ConfigurationError::Validation(
                "kafka.group_id and kafka.client_id cannot be empty in streaming mode".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatasetConfig {
    /// Object key produced by iot-sim, for example dataset/iot-telemetry-....parquet.
    #[serde(default)]
    pub parquet_key: String,
    #[serde(default = "default_s3_endpoint_url")]
    pub s3_endpoint_url: String,
    #[serde(default = "default_s3_bucket_name")]
    pub s3_bucket_name: String,
    #[serde(default = "default_s3_access_key")]
    pub s3_access_key: String,
    #[serde(default = "default_s3_secret_key")]
    pub s3_secret_key: String,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            parquet_key: String::new(),
            s3_endpoint_url: default_s3_endpoint_url(),
            s3_bucket_name: default_s3_bucket_name(),
            s3_access_key: default_s3_access_key(),
            s3_secret_key: default_s3_secret_key(),
        }
    }
}

impl DatasetConfig {
    fn validate(&self) -> Result<(), ConfigurationError> {
        for (name, value) in [
            ("parquet_key", &self.parquet_key),
            ("s3_endpoint_url", &self.s3_endpoint_url),
            ("s3_bucket_name", &self.s3_bucket_name),
            ("s3_access_key", &self.s3_access_key),
            ("s3_secret_key", &self.s3_secret_key),
        ] {
            if value.trim().is_empty() {
                return Err(ConfigurationError::Validation(format!(
                    "dataset.{name} cannot be empty in dataset mode"
                )));
            }
        }
        if !self.parquet_key.ends_with(".parquet") {
            return Err(ConfigurationError::Validation(
                "dataset.parquet_key must name a .parquet object".into(),
            ));
        }
        Ok(())
    }
}

fn default_s3_endpoint_url() -> String {
    "http://192.168.1.50:9000".into()
}
fn default_s3_bucket_name() -> String {
    "iotsim".into()
}
fn default_s3_access_key() -> String {
    "access".into()
}
fn default_s3_secret_key() -> String {
    "secret".into()
}

fn default_output_topic() -> String {
    "iot.anomaly.v1".into()
}

fn default_offset_reset() -> String {
    "latest".into()
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InvalidMessagePolicy {
    #[default]
    Skip,
    Fail,
}

/// Controls structured application logs for records that pass the configured
/// Kafka header allow-list and for anomaly results sent to Kafka.
///
/// Both options default to false so a configuration upgrade does not
/// unexpectedly expose telemetry values or increase log volume.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct KafkaLoggingConfig {
    #[serde(default)]
    pub log_consumed_messages: bool,
    #[serde(default)]
    pub log_produced_results: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct KafkaSecurityConfig {
    pub security_protocol: Option<String>,
    pub sasl_mechanism: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WindowConfig {
    pub sample_count: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamConfig {
    /// Exact Kafka-header matches. Payloads are decoded only after this filter matches.
    pub headers: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PreprocessingConfig {
    #[serde(default)]
    pub interpolation: InterpolationConfig,
    #[serde(default)]
    pub scaling: ScalingConfig,
    #[serde(default)]
    pub decomposition: DecompositionConfig,
    #[serde(default)]
    pub smoothing: SmoothingConfig,
}

impl PreprocessingConfig {
    fn validate(&self, _window: &WindowConfig) -> Result<(), ConfigurationError> {
        if !(0.0..=0.20).contains(&self.interpolation.max_gap_fraction)
            || self.interpolation.max_gap_fraction == 0.0
        {
            return Err(ConfigurationError::Validation(
                "preprocessing.interpolation.max_gap_fraction must be greater than zero and no greater than 0.20".into(),
            ));
        }
        if !self.scaling.output_min.is_finite()
            || !self.scaling.output_max.is_finite()
            || self.scaling.output_min >= self.scaling.output_max
        {
            return Err(ConfigurationError::Validation(
                "preprocessing.scaling output_min must be less than output_max".into(),
            ));
        }
        if !self.scaling.epsilon.is_finite() || self.scaling.epsilon <= 0.0 {
            return Err(ConfigurationError::Validation(
                "preprocessing.scaling.epsilon must be finite and positive".into(),
            ));
        }
        if self.smoothing.window_size == 0 {
            return Err(ConfigurationError::Validation(
                "preprocessing.smoothing.window_size must be greater than zero".into(),
            ));
        }
        if !self.decomposition.trend.alpha.is_finite()
            || !(0.0..=1.0).contains(&self.decomposition.trend.alpha)
            || self.decomposition.trend.alpha == 0.0
        {
            return Err(ConfigurationError::Validation(
                "preprocessing.decomposition.trend.alpha must be finite, greater than zero, and no greater than one".into(),
            ));
        }
        if !self.decomposition.trend.beta.is_finite()
            || !(0.0..=1.0).contains(&self.decomposition.trend.beta)
            || self.decomposition.trend.beta == 0.0
        {
            return Err(ConfigurationError::Validation(
                "preprocessing.decomposition.trend.beta must be finite, greater than zero, and no greater than one".into(),
            ));
        }
        let seasonal = &self.decomposition.seasonal;
        if seasonal.window_size_samples < 4 {
            return Err(ConfigurationError::Validation(
                "preprocessing.decomposition.seasonal.window_size_samples must be at least 4"
                    .into(),
            ));
        }
        if seasonal.detection_interval_samples == 0 {
            return Err(ConfigurationError::Validation(
                "preprocessing.decomposition.seasonal.detection_interval_samples must be greater than zero".into(),
            ));
        }
        if seasonal.min_period_samples < 2
            || seasonal.max_period_samples < seasonal.min_period_samples
            || seasonal.max_period_samples > seasonal.window_size_samples
        {
            return Err(ConfigurationError::Validation(
                "preprocessing.decomposition.seasonal periods must satisfy 2 <= min_period_samples <= max_period_samples <= window_size_samples".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct InterpolationConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_max_gap_fraction")]
    pub max_gap_fraction: f64,
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_gap_fraction: default_max_gap_fraction(),
        }
    }
}

fn default_max_gap_fraction() -> f64 {
    0.20
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScalingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub method: ScalingMethod,
    #[serde(default)]
    pub output_min: f64,
    #[serde(default = "default_output_max")]
    pub output_max: f64,
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: ScalingMethod::default(),
            output_min: 0.0,
            output_max: 1.0,
            epsilon: default_epsilon(),
        }
    }
}

fn default_output_max() -> f64 {
    1.0
}

fn default_epsilon() -> f64 {
    1e-12
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScalingMethod {
    MinMax,
    #[default]
    Standard,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SmoothingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub method: SmoothingMethod,
    #[serde(default = "default_smoothing_window")]
    pub window_size: usize,
}

impl Default for SmoothingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: SmoothingMethod::default(),
            window_size: default_smoothing_window(),
        }
    }
}

fn default_smoothing_window() -> usize {
    5
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SmoothingMethod {
    #[default]
    MovingAverage,
    ExponentialMovingAverage,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecompositionConfig {
    #[serde(default)]
    pub trend: TrendDecompositionConfig,
    #[serde(default)]
    pub seasonal: SeasonalDecompositionConfig,
}

impl DecompositionConfig {
    pub fn enabled(&self) -> bool {
        self.trend.enabled || self.seasonal.enabled
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrendDecompositionConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_holt_alpha")]
    pub alpha: f64,
    #[serde(default = "default_holt_beta")]
    pub beta: f64,
}

impl Default for TrendDecompositionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            alpha: default_holt_alpha(),
            beta: default_holt_beta(),
        }
    }
}

fn default_holt_alpha() -> f64 {
    0.2
}

fn default_holt_beta() -> f64 {
    0.1
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeasonalDecompositionConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_seasonal_window_size")]
    pub window_size_samples: usize,
    #[serde(default = "default_detection_interval")]
    pub detection_interval_samples: usize,
    #[serde(default = "default_min_period")]
    pub min_period_samples: usize,
    #[serde(default = "default_max_period")]
    pub max_period_samples: usize,
}

impl Default for SeasonalDecompositionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            window_size_samples: default_seasonal_window_size(),
            detection_interval_samples: default_detection_interval(),
            min_period_samples: default_min_period(),
            max_period_samples: default_max_period(),
        }
    }
}

fn default_seasonal_window_size() -> usize {
    256
}

fn default_detection_interval() -> usize {
    1
}

fn default_min_period() -> usize {
    2
}

fn default_max_period() -> usize {
    256
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    pub kind: ModelKind,
    pub algorithm: String,
    pub inputs: Vec<String>,
    #[serde(default)]
    pub parameters: BTreeMap<String, Value>,
    #[serde(default)]
    pub thresholds: BTreeMap<String, Value>,
}

fn default_enabled() -> bool {
    true
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Univariate,
    Multivariate,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> AppConfig {
        AppConfig {
            mode: RunMode::Streaming,
            kafka: KafkaConfig {
                brokers: "localhost:9092".into(),
                topic: "telemetry".into(),
                output_topic: "iot.anomaly.v1".into(),
                group_id: "detectors".into(),
                client_id: "detector-1".into(),
                auto_offset_reset: "latest".into(),
                invalid_message_policy: InvalidMessagePolicy::Skip,
                logging: KafkaLoggingConfig::default(),
                security: KafkaSecurityConfig::default(),
            },
            dataset: DatasetConfig::default(),
            window: WindowConfig { sample_count: 12 },
            preprocessing: PreprocessingConfig::default(),
            streams: BTreeMap::from([(
                "temperature".into(),
                StreamConfig {
                    headers: BTreeMap::from([
                        ("entity_id".into(), "station-1".into()),
                        ("sensor_id".into(), "weather-1".into()),
                        ("metric".into(), "temperature_c".into()),
                    ]),
                },
            )]),
            models: vec![ModelConfig {
                id: "temperature-zscore".into(),
                enabled: true,
                kind: ModelKind::Univariate,
                algorithm: "z_score".into(),
                inputs: vec!["temperature".into()],
                parameters: BTreeMap::new(),
                thresholds: BTreeMap::from([("score".into(), Value::from(3.0))]),
            }],
        }
    }

    #[test]
    fn validates_model_arity_and_stream_references() {
        let mut config = valid_config();
        assert!(config.validate().is_ok());
        config.models[0].inputs.push("missing".into());
        assert!(config.validate().is_err());
    }

    #[test]
    fn dataset_mode_requires_a_parquet_object_key() {
        let mut config = valid_config();
        config.mode = RunMode::Dataset;
        let error = config.validate().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("dataset.parquet_key cannot be empty")
        );
    }

    #[test]
    fn allows_multiple_univariate_models_for_the_same_stream() {
        let mut config = valid_config();
        config.models.push(ModelConfig {
            id: "temperature-mad".into(),
            enabled: true,
            kind: ModelKind::Univariate,
            algorithm: "mad".into(),
            inputs: vec!["temperature".into()],
            parameters: BTreeMap::new(),
            thresholds: BTreeMap::from([("score".into(), Value::from(3.5))]),
        });
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validates_mad_stability_parameters() {
        let mut config = valid_config();
        config.models[0].algorithm = "mad".into();
        config.models[0].parameters = BTreeMap::from([
            ("mad_ema_alpha".into(), Value::from(0.05)),
            ("epsilon".into(), Value::from(0.1)),
        ]);
        assert!(config.validate().is_ok());

        config.models[0]
            .parameters
            .insert("mad_ema_alpha".into(), Value::from(0.0));
        assert!(config.validate().is_err());

        config.models[0]
            .parameters
            .insert("mad_ema_alpha".into(), Value::from(0.05));
        config.models[0]
            .parameters
            .insert("epsilon".into(), Value::from(0.0));
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_zero_window_sample_count() {
        let mut config = valid_config();
        config.window.sample_count = 0;
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("window.sample_count must be greater than zero"));
    }

    #[test]
    fn rejects_invalid_univariate_algorithm_and_threshold() {
        let mut config = valid_config();
        config.models[0].algorithm = "unknown".into();
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.models[0]
            .thresholds
            .insert("score".into(), Value::from(0.0));
        assert!(config.validate().is_err());
    }

    #[test]
    fn validates_multivariate_algorithm_parameters() {
        let mut config = valid_config();
        config.streams.insert(
            "humidity".into(),
            StreamConfig {
                headers: BTreeMap::from([
                    ("entity_id".into(), "station-1".into()),
                    ("sensor_id".into(), "weather-1".into()),
                    ("metric".into(), "humidity_pct".into()),
                ]),
            },
        );
        config.models = vec![ModelConfig {
            id: "weather-kpca".into(),
            enabled: true,
            kind: ModelKind::Multivariate,
            algorithm: "kernel_pca".into(),
            inputs: vec!["temperature".into(), "humidity".into()],
            parameters: BTreeMap::from([
                ("window_size".into(), Value::from(8)),
                ("max_time_skew_ms".into(), Value::from(5_000)),
                ("retained_components".into(), Value::from(2)),
                ("rff_dimension".into(), Value::from(16)),
                ("gamma".into(), Value::from("auto")),
                ("seed".into(), Value::from(7)),
                ("score_detector".into(), Value::from("z_score")),
                ("score_window_size".into(), Value::from(5)),
                ("score_ddof".into(), Value::from(1)),
            ]),
            thresholds: BTreeMap::from([("score".into(), Value::from(3.0))]),
        }];
        assert!(config.validate().is_ok());

        config.models[0]
            .parameters
            .insert("gamma".into(), Value::from(0.0));
        assert!(config.validate().is_err());
    }

    #[test]
    fn checked_in_configuration_loads_and_validates() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/anomaly-detect.yaml");
        let config = AppConfig::load(path).unwrap();
        assert_eq!(config.kafka.topic, "iot.telemetry.v1");
        assert_eq!(config.kafka.output_topic, "iot.anomaly.v1");
        assert!(
            config
                .models
                .iter()
                .any(|model| model.kind == ModelKind::Univariate)
        );
        assert!(
            config
                .models
                .iter()
                .any(|model| model.algorithm == "z_score")
        );
        assert!(config.kafka.logging.log_consumed_messages);
        assert!(config.kafka.logging.log_produced_results);
    }

    #[test]
    fn validates_gsta_shape_against_the_sample_window() {
        let mut config = valid_config();
        config.streams.insert(
            "humidity".into(),
            StreamConfig {
                headers: BTreeMap::from([
                    ("entity_id".into(), "station-1".into()),
                    ("sensor_id".into(), "weather-1".into()),
                    ("metric".into(), "humidity_pct".into()),
                ]),
            },
        );
        config.models = vec![ModelConfig {
            id: "weather-gsta".into(),
            enabled: true,
            kind: ModelKind::Multivariate,
            algorithm: "gsta".into(),
            inputs: vec!["temperature".into(), "humidity".into()],
            parameters: BTreeMap::from([
                ("window_size".into(), Value::from(12)),
                ("max_time_skew_ms".into(), Value::from(5_000)),
                ("latent_channels".into(), Value::from(8)),
                ("attention_heads".into(), Value::from(2)),
                ("gate_threshold".into(), Value::from(0.05)),
                ("learning_rate".into(), Value::from(0.001)),
                ("warmup_steps".into(), Value::from(128)),
                ("score_detector".into(), Value::from("mad")),
                ("score_window_size".into(), Value::from(12)),
            ]),
            thresholds: BTreeMap::from([("score".into(), Value::from(4.0))]),
        }];
        assert!(config.validate().is_ok());

        config.models[0]
            .parameters
            .insert("attention_heads".into(), Value::from(5));
        assert!(config.validate().is_err());
    }
}
