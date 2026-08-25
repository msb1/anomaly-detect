pub mod gsta;
pub mod mahalanobis;
pub mod pca;

mod score;

use std::sync::{Arc, Mutex};

use nalgebra::DVector;
use serde_json::Value;

use crate::config::{ModelConfig, ModelKind};
use crate::model::{AnomalyModel, Detection, ModelError, ModelInput};

use self::gsta::{GstaDetector, GstaUpdate, aligned_channel_window};
use self::mahalanobis::RollingWelfordMahalanobis;
use self::pca::{KernelMode, UnifiedPcaDetector};
use self::score::{ScorePipeline, float_value, usize_value};

#[derive(Debug, Clone)]
enum EngineConfig {
    Mahalanobis {
        window_size: usize,
        dimensions: usize,
        regularization: f64,
        shrinkage: f64,
    },
    Pca {
        input_dim: usize,
        retained_components: usize,
        window_size: usize,
        mode: KernelMode,
    },
    Gsta {
        channels: usize,
        window_size: usize,
        latent_channels: usize,
        attention_heads: usize,
        gate_threshold: f64,
        learning_rate: f64,
        warmup_steps: usize,
        seed: u64,
    },
}

#[derive(Debug)]
enum Engine {
    Mahalanobis(RollingWelfordMahalanobis),
    Pca(UnifiedPcaDetector),
    Gsta(Box<GstaDetector>),
}

impl Engine {
    fn new(config: &EngineConfig) -> Result<Self, ModelError> {
        match config {
            EngineConfig::Mahalanobis {
                window_size,
                dimensions,
                regularization,
                shrinkage,
            } => Ok(Self::Mahalanobis(RollingWelfordMahalanobis::new(
                *window_size,
                *dimensions,
                *regularization,
                *shrinkage,
            )?)),
            EngineConfig::Pca {
                input_dim,
                retained_components,
                window_size,
                mode,
            } => Ok(Self::Pca(UnifiedPcaDetector::new(
                *input_dim,
                *retained_components,
                *window_size,
                mode.clone(),
            )?)),
            EngineConfig::Gsta {
                channels,
                window_size,
                latent_channels,
                attention_heads,
                gate_threshold,
                learning_rate,
                warmup_steps,
                seed,
            } => Ok(Self::Gsta(Box::new(GstaDetector::new(
                *channels,
                *window_size,
                *latent_channels,
                *attention_heads,
                *gate_threshold,
                *learning_rate,
                *warmup_steps,
                *seed,
            )?))),
        }
    }

    fn update(&mut self, point: DVector<f64>) -> Result<Option<f64>, ModelError> {
        match self {
            Self::Mahalanobis(detector) => detector.update(point),
            Self::Pca(detector) => detector.update(point),
            Self::Gsta(_) => Err(ModelError::new(
                "GSTA requires a complete aligned temporal window",
            )),
        }
    }

    fn tuned_gamma(&self) -> Option<f64> {
        match self {
            Self::Pca(detector) => detector.tuned_gamma(),
            Self::Mahalanobis(_) => None,
            Self::Gsta(_) => None,
        }
    }
}

#[derive(Debug)]
struct MultivariateModel {
    id: String,
    algorithm: String,
    inputs: Vec<String>,
    max_time_skew_ms: i64,
    engine_config: EngineConfig,
    engine: Engine,
    score_pipeline: ScorePipeline,
    last_timestamps: Option<Vec<i64>>,
}

impl MultivariateModel {
    fn from_config(config: &ModelConfig) -> Result<Self, ModelError> {
        if config.kind != ModelKind::Multivariate {
            return Err(ModelError::new(format!(
                "model '{}' is not multivariate",
                config.id
            )));
        }
        let window_size = usize_value(config, "window_size", None)?;
        let max_time_skew_ms = i64::try_from(usize_value(config, "max_time_skew_ms", None)?)
            .map_err(|_| ModelError::new("max_time_skew_ms is too large"))?;
        let engine_config = match config.algorithm.as_str() {
            "mahalanobis" => EngineConfig::Mahalanobis {
                window_size,
                dimensions: config.inputs.len(),
                regularization: float_value(config, "regularization", Some(1e-6))?,
                shrinkage: float_value(config, "shrinkage", Some(0.0))?,
            },
            "pca" => EngineConfig::Pca {
                input_dim: config.inputs.len(),
                retained_components: usize_value(config, "retained_components", None)?,
                window_size,
                mode: KernelMode::Linear,
            },
            "kernel_pca" => {
                let dimension = usize_value(config, "rff_dimension", None)?;
                let seed = u64::try_from(usize_value(config, "seed", Some(0))?)
                    .map_err(|_| ModelError::new("RFF seed is too large"))?;
                let mode = match config.parameters.get("gamma") {
                    Some(Value::String(value)) if value == "auto" => {
                        KernelMode::RbfAuto { dimension, seed }
                    }
                    Some(value) => KernelMode::Rbf {
                        gamma: value.as_f64().ok_or_else(|| {
                            ModelError::new("kernel PCA gamma must be numeric or 'auto'")
                        })?,
                        dimension,
                        seed,
                    },
                    None => return Err(ModelError::new("kernel PCA requires parameters.gamma")),
                };
                EngineConfig::Pca {
                    input_dim: config.inputs.len(),
                    retained_components: usize_value(config, "retained_components", None)?,
                    window_size,
                    mode,
                }
            }
            "gsta" => EngineConfig::Gsta {
                channels: config.inputs.len(),
                window_size,
                latent_channels: usize_value(config, "latent_channels", Some(16))?,
                attention_heads: usize_value(config, "attention_heads", Some(2))?,
                gate_threshold: float_value(config, "gate_threshold", None)?,
                learning_rate: float_value(config, "learning_rate", Some(0.001))?,
                warmup_steps: usize_value(config, "warmup_steps", Some(128))?,
                seed: u64::try_from(usize_value(config, "seed", Some(0))?)
                    .map_err(|_| ModelError::new("GSTA seed is too large"))?,
            },
            algorithm => {
                return Err(ModelError::new(format!(
                    "unsupported multivariate algorithm '{algorithm}' for model '{}'",
                    config.id
                )));
            }
        };
        let engine = Engine::new(&engine_config)?;
        Ok(Self {
            id: config.id.clone(),
            algorithm: config.algorithm.clone(),
            inputs: config.inputs.clone(),
            max_time_skew_ms,
            engine_config,
            engine,
            score_pipeline: ScorePipeline::new(config)?,
            last_timestamps: None,
        })
    }

    fn evaluate(&mut self, input: ModelInput<'_>) -> Result<Option<Detection>, ModelError> {
        let mut timestamps = Vec::with_capacity(self.inputs.len());
        let mut values = Vec::with_capacity(self.inputs.len());
        for name in &self.inputs {
            let sample = input
                .streams
                .get(name.as_str())
                .and_then(|samples| samples.back())
                .ok_or_else(|| {
                    ModelError::new(format!(
                        "multivariate model '{}' did not receive a value for input '{name}'",
                        self.id
                    ))
                })?;
            timestamps.push(sample.timestamp_ms);
            values.push(sample.value);
        }
        let minimum_timestamp = *timestamps
            .iter()
            .min()
            .expect("multivariate configuration has at least two inputs");
        let maximum_timestamp = *timestamps
            .iter()
            .max()
            .expect("multivariate configuration has at least two inputs");
        if maximum_timestamp.saturating_sub(minimum_timestamp) > self.max_time_skew_ms {
            return Ok(None);
        }
        if let Some(previous) = &self.last_timestamps
            && timestamps
                .iter()
                .zip(previous)
                .any(|(current, previous)| current <= previous)
        {
            // Do not manufacture multiple multivariate rows by repeatedly
            // carrying forward coordinates whose streams have not advanced.
            return Ok(None);
        }

        let gsta_update = match &mut self.engine {
            Engine::Gsta(detector) => {
                let Some(window) = aligned_channel_window(
                    &input,
                    &self.inputs,
                    match &self.engine_config {
                        EngineConfig::Gsta { window_size, .. } => *window_size,
                        _ => unreachable!(),
                    },
                    self.max_time_skew_ms,
                )?
                else {
                    return Ok(None);
                };
                Some(detector.update(window)?)
            }
            _ => None,
        };
        let raw_score = match gsta_update {
            Some(update) => Some(update.reconstruction_error),
            None => self.engine.update(DVector::from_vec(values.clone()))?,
        };
        self.last_timestamps = Some(timestamps.clone());
        if let Some(update) = gsta_update {
            if update.warming_up {
                tracing::debug!(
                    model_id = %self.id,
                    reconstruction_error = update.reconstruction_error,
                    warmup_remaining = update.warmup_remaining,
                    "trained GSTA warmup window"
                );
            } else if !update.trained {
                tracing::debug!(
                    model_id = %self.id,
                    reconstruction_error = update.reconstruction_error,
                    "GSTA gate blocked optimizer update"
                );
            }
        }
        if gsta_update.is_some_and(|update| update.warming_up) {
            return Ok(None);
        }
        let Some(raw_score) = raw_score else {
            return Ok(None);
        };
        let Some(mut detection) = self.score_pipeline.update(maximum_timestamp, raw_score)? else {
            return Ok(None);
        };

        let score_name = match self.algorithm.as_str() {
            "mahalanobis" => "mahalanobis_distance",
            "gsta" => "reconstruction_mse",
            _ => "reconstruction_error",
        };
        detection.details.insert(
            "multivariate_algorithm".into(),
            Value::from(self.algorithm.clone()),
        );
        detection
            .details
            .insert("multivariate_score_name".into(), Value::from(score_name));
        detection
            .details
            .insert("multivariate_score".into(), Value::from(raw_score));
        detection.details.insert(
            "input_values".into(),
            Value::Object(
                self.inputs
                    .iter()
                    .cloned()
                    .zip(values)
                    .map(|(name, value)| (name, Value::from(value)))
                    .collect(),
            ),
        );
        detection.details.insert(
            "input_timestamps_ms".into(),
            Value::Object(
                self.inputs
                    .iter()
                    .cloned()
                    .zip(timestamps)
                    .map(|(name, timestamp)| (name, Value::from(timestamp)))
                    .collect(),
            ),
        );
        if let Some(gamma) = self.engine.tuned_gamma() {
            detection
                .details
                .insert("rbf_gamma".into(), Value::from(gamma));
        }
        if let Some(GstaUpdate {
            trained,
            warmup_remaining,
            ..
        }) = gsta_update
        {
            detection
                .details
                .insert("gsta_weights_updated".into(), Value::from(trained));
            detection.details.insert(
                "gsta_gate_threshold".into(),
                Value::from(match &self.engine_config {
                    EngineConfig::Gsta { gate_threshold, .. } => *gate_threshold,
                    _ => unreachable!(),
                }),
            );
            detection.details.insert(
                "gsta_warmup_remaining".into(),
                Value::from(warmup_remaining),
            );
        }
        Ok(Some(detection))
    }

    fn reset(&mut self) -> Result<(), ModelError> {
        self.engine = Engine::new(&self.engine_config)?;
        self.score_pipeline.clear();
        self.last_timestamps = None;
        Ok(())
    }
}

/// Thread-safe model handle. The inner state uses a synchronous critical
/// section and can be cloned into Tokio tasks without holding an async borrow.
#[derive(Debug, Clone)]
pub struct SharedMultivariateModel {
    id: String,
    inner: Arc<Mutex<MultivariateModel>>,
}

impl SharedMultivariateModel {
    pub fn from_config(config: &ModelConfig) -> Result<Self, ModelError> {
        Ok(Self {
            id: config.id.clone(),
            inner: Arc::new(Mutex::new(MultivariateModel::from_config(config)?)),
        })
    }

    pub fn evaluate_streams(&self, input: ModelInput<'_>) -> Result<Option<Detection>, ModelError> {
        self.inner
            .lock()
            .map_err(|_| {
                ModelError::new(format!("multivariate model '{}' lock is poisoned", self.id))
            })?
            .evaluate(input)
    }

    pub fn reset(&self) -> Result<(), ModelError> {
        self.inner
            .lock()
            .map_err(|_| {
                ModelError::new(format!("multivariate model '{}' lock is poisoned", self.id))
            })?
            .reset()
    }
}

impl AnomalyModel for SharedMultivariateModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn kind(&self) -> ModelKind {
        ModelKind::Multivariate
    }

    fn evaluate(&self, input: ModelInput<'_>) -> Result<Detection, ModelError> {
        self.evaluate_streams(input)?.ok_or_else(|| {
            ModelError::new(format!(
                "multivariate model '{}' is still warming up or awaiting aligned inputs",
                self.id
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, VecDeque};

    use super::*;
    use crate::window::Sample;

    fn config() -> ModelConfig {
        ModelConfig {
            id: "joint".into(),
            enabled: true,
            kind: ModelKind::Multivariate,
            algorithm: "mahalanobis".into(),
            inputs: vec!["left".into(), "right".into()],
            parameters: BTreeMap::from([
                ("window_size".into(), Value::from(3)),
                ("max_time_skew_ms".into(), Value::from(1)),
                ("regularization".into(), Value::from(1e-3)),
                ("shrinkage".into(), Value::from(0.2)),
                ("score_detector".into(), Value::from("mad")),
                ("score_window_size".into(), Value::from(3)),
            ]),
            thresholds: BTreeMap::from([("score".into(), Value::from(3.0))]),
        }
    }

    fn samples(timestamp_ms: i64, value: f64) -> VecDeque<Sample> {
        VecDeque::from([Sample {
            timestamp_ms,
            value,
        }])
    }

    #[test]
    fn advances_only_after_every_input_has_a_fresh_aligned_value() {
        let model = SharedMultivariateModel::from_config(&config()).unwrap();
        let left = samples(1, 0.0);
        let right = samples(1, 0.0);
        assert!(
            model
                .evaluate_streams(ModelInput {
                    streams: BTreeMap::from([("left", &left), ("right", &right)]),
                })
                .unwrap()
                .is_none()
        );

        let left = samples(2, 0.1);
        assert!(
            model
                .evaluate_streams(ModelInput {
                    streams: BTreeMap::from([("left", &left), ("right", &right)]),
                })
                .unwrap()
                .is_none()
        );
        assert_eq!(
            model.inner.lock().unwrap().last_timestamps.as_deref(),
            Some([1, 1].as_slice())
        );
    }

    #[test]
    fn composes_raw_distance_into_the_configured_univariate_detector() {
        let model = SharedMultivariateModel::from_config(&config()).unwrap();
        let mut final_detection = None;
        for (timestamp, values) in [
            (1, [0.0, 0.0]),
            (2, [0.1, 0.1]),
            (3, [-0.1, -0.1]),
            (4, [0.0, 0.0]),
            (5, [0.05, 0.05]),
            (6, [8.0, -8.0]),
        ] {
            let left = samples(timestamp, values[0]);
            let right = samples(timestamp, values[1]);
            final_detection = model
                .evaluate_streams(ModelInput {
                    streams: BTreeMap::from([("left", &left), ("right", &right)]),
                })
                .unwrap();
        }
        let detection = final_detection.expect("three raw scores warm the MAD score window");
        assert!(detection.anomalous);
        assert_eq!(detection.timestamp_ms, 6);
        assert_eq!(detection.window_sample_count, 3);
        assert!(detection.anomalous_points > 0);
        assert_eq!(
            detection.details["multivariate_score_name"],
            Value::from("mahalanobis_distance")
        );
        assert!(detection.details["multivariate_score"].as_f64().unwrap() > 10.0);
    }
}
