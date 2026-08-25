use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use config::{Config, ConfigError, Environment, File, FileFormat};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub kafka: KafkaConfig,
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
        if self.kafka.brokers.trim().is_empty()
            || self.kafka.topic.trim().is_empty()
            || self.kafka.output_topic.trim().is_empty()
        {
            return Err(ConfigurationError::Validation(
                "kafka.brokers, kafka.topic, and kafka.output_topic cannot be empty".into(),
            ));
        }
        if self.kafka.group_id.trim().is_empty() || self.kafka.client_id.trim().is_empty() {
            return Err(ConfigurationError::Validation(
                "kafka.group_id and kafka.client_id cannot be empty".into(),
            ));
        }
        if self.window.duration_ms <= 0 || self.window.minimum_samples == 0 {
            return Err(ConfigurationError::Validation(
                "window.duration_ms and window.minimum_samples must be greater than zero".into(),
            ));
        }
        if self.window.max_lateness_ms < 0 {
            return Err(ConfigurationError::Validation(
                "window.max_lateness_ms cannot be negative".into(),
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
            if stream.interval_ms <= 0 || stream.interval_ms > self.window.duration_ms {
                return Err(ConfigurationError::Validation(format!(
                    "stream '{name}' interval_ms must be positive and no longer than window.duration_ms"
                )));
            }
            if self.preprocessing.decomposition.enabled
                && self.preprocessing.decomposition.period_ms / stream.interval_ms < 2
            {
                return Err(ConfigurationError::Validation(format!(
                    "decomposition.period_ms must contain at least two samples for stream '{name}'"
                )));
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
                ModelKind::Multivariate => {
                    validate_multivariate_model(model, &self.streams, &self.window)?
                }
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
        if ddof >= window.minimum_samples {
            return Err(ConfigurationError::Validation(format!(
                "Z-score model '{}' parameters.ddof must be less than window.minimum_samples",
                model.id
            )));
        }
    }
    Ok(())
}

fn validate_multivariate_model(
    model: &ModelConfig,
    streams: &BTreeMap<String, StreamConfig>,
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

    let score_detector = model
        .parameters
        .get("score_detector")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ConfigurationError::Validation(format!(
                "multivariate model '{}' requires string parameters.score_detector",
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

            let first_input = model.inputs.first().ok_or_else(|| {
                ConfigurationError::Validation(format!(
                    "GSTA model '{}' requires input streams",
                    model.id
                ))
            })?;
            let interval_ms = streams
                .get(first_input)
                .ok_or_else(|| {
                    ConfigurationError::Validation(format!(
                        "model '{}' references unknown stream '{first_input}'",
                        model.id
                    ))
                })?
                .interval_ms;
            for input in &model.inputs[1..] {
                let input_interval = streams
                    .get(input)
                    .ok_or_else(|| {
                        ConfigurationError::Validation(format!(
                            "model '{}' references unknown stream '{input}'",
                            model.id
                        ))
                    })?
                    .interval_ms;
                if input_interval != interval_ms {
                    return Err(ConfigurationError::Validation(format!(
                        "GSTA model '{}' requires every input stream to have the same interval_ms",
                        model.id
                    )));
                }
            }
            let required_span = i64::try_from(window_size.saturating_sub(1))
                .ok()
                .and_then(|steps| steps.checked_mul(interval_ms))
                .ok_or_else(|| {
                    ConfigurationError::Validation(format!(
                        "GSTA model '{}' parameters.window_size is too large",
                        model.id
                    ))
                })?;
            if required_span > window.duration_ms {
                return Err(ConfigurationError::Validation(format!(
                    "GSTA model '{}' parameters.window_size does not fit within window.duration_ms at the configured input cadence",
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
    pub brokers: String,
    pub topic: String,
    #[serde(default = "default_output_topic")]
    pub output_topic: String,
    pub group_id: String,
    pub client_id: String,
    #[serde(default = "default_offset_reset")]
    pub auto_offset_reset: String,
    #[serde(default)]
    pub invalid_message_policy: InvalidMessagePolicy,
    #[serde(default)]
    pub security: KafkaSecurityConfig,
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

#[derive(Debug, Clone, Default, Deserialize)]
pub struct KafkaSecurityConfig {
    pub security_protocol: Option<String>,
    pub sasl_mechanism: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WindowConfig {
    pub duration_ms: i64,
    #[serde(default = "default_minimum_samples")]
    pub minimum_samples: usize,
    #[serde(default = "default_max_lateness_ms")]
    pub max_lateness_ms: i64,
}

fn default_minimum_samples() -> usize {
    2
}

fn default_max_lateness_ms() -> i64 {
    5_000
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamConfig {
    /// Exact Kafka-header matches. Payloads are decoded only after this filter matches.
    pub headers: BTreeMap<String, String>,
    /// Expected source cadence, used to detect and interpolate missing samples.
    pub interval_ms: i64,
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
    fn validate(&self, window: &WindowConfig) -> Result<(), ConfigurationError> {
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
        if self.decomposition.period_ms <= 0 {
            return Err(ConfigurationError::Validation(
                "preprocessing.decomposition.period_ms must be positive".into(),
            ));
        }
        if self.decomposition.enabled
            && window.duration_ms < self.decomposition.period_ms.saturating_mul(2)
        {
            return Err(ConfigurationError::Validation(
                "window.duration_ms must cover at least two decomposition periods".into(),
            ));
        }
        if self.decomposition.stl.loess_span < 3
            || self.decomposition.stl.loess_span.is_multiple_of(2)
        {
            return Err(ConfigurationError::Validation(
                "preprocessing.decomposition.stl.loess_span must be odd and at least 3".into(),
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

#[derive(Debug, Clone, Deserialize)]
pub struct DecompositionConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub method: DecompositionMethod,
    #[serde(default = "default_period_ms")]
    pub period_ms: i64,
    #[serde(default)]
    pub stl: StlConfig,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: DecompositionMethod::default(),
            period_ms: default_period_ms(),
            stl: StlConfig::default(),
        }
    }
}

fn default_period_ms() -> i64 {
    86_400_000
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecompositionMethod {
    #[default]
    Stl,
    Twitter,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StlConfig {
    #[serde(default = "default_loess_span")]
    pub loess_span: usize,
    #[serde(default = "default_robust_iterations")]
    pub robust_iterations: usize,
}

impl Default for StlConfig {
    fn default() -> Self {
        Self {
            loess_span: default_loess_span(),
            robust_iterations: default_robust_iterations(),
        }
    }
}

fn default_loess_span() -> usize {
    7
}

fn default_robust_iterations() -> usize {
    2
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
            kafka: KafkaConfig {
                brokers: "localhost:9092".into(),
                topic: "telemetry".into(),
                output_topic: "iot.anomaly.v1".into(),
                group_id: "detectors".into(),
                client_id: "detector-1".into(),
                auto_offset_reset: "latest".into(),
                invalid_message_policy: InvalidMessagePolicy::Skip,
                security: KafkaSecurityConfig::default(),
            },
            window: WindowConfig {
                duration_ms: 60_000,
                minimum_samples: 2,
                max_lateness_ms: 5_000,
            },
            preprocessing: PreprocessingConfig::default(),
            streams: BTreeMap::from([(
                "temperature".into(),
                StreamConfig {
                    interval_ms: 5_000,
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
                interval_ms: 5_000,
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
        assert!(config.models.iter().any(|model| model.algorithm == "mad"));
        assert!(
            config
                .models
                .iter()
                .any(|model| model.algorithm == "mahalanobis")
        );
        assert!(config.models.iter().any(|model| model.algorithm == "pca"));
        assert!(
            config
                .models
                .iter()
                .any(|model| model.algorithm == "kernel_pca")
        );
        assert!(config.models.iter().any(|model| model.algorithm == "gsta"));
    }

    #[test]
    fn validates_gsta_temporal_shape_and_equal_input_cadence() {
        let mut config = valid_config();
        config.streams.insert(
            "humidity".into(),
            StreamConfig {
                interval_ms: 5_000,
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

        config.streams.get_mut("humidity").unwrap().interval_ms = 4_000;
        assert!(config.validate().is_err());

        config.streams.get_mut("humidity").unwrap().interval_ms = 5_000;
        config.models[0]
            .parameters
            .insert("attention_heads".into(), Value::from(5));
        assert!(config.validate().is_err());
    }
}
