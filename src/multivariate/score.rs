use std::collections::{BTreeMap, VecDeque};

use serde_json::Value;

use crate::config::{
    DecompositionConfig, ModelConfig, ModelKind, SmoothingConfig, SmoothingMethod,
};
use crate::model::{AnomalyModel, Detection, ModelError, ModelInput};
use crate::preprocess::decomposition::StreamingDecomposer;
use crate::univariate::UnivariateModel;
use crate::window::Sample;

const SCORE_INPUT: &str = "multivariate_score";

#[derive(Debug)]
pub(super) struct ScorePipeline {
    detector: UnivariateModel,
    window_size: usize,
    scores: VecDeque<Sample>,
    decomposer: StreamingDecomposer,
    smoother: ScalarSmoother,
}

/// Causal smoothing state for the scalar output of a multivariate engine.
#[derive(Debug)]
struct ScalarSmoother {
    config: SmoothingConfig,
    values: VecDeque<f64>,
    sum: f64,
    ema: Option<f64>,
}

impl ScalarSmoother {
    fn new(config: &SmoothingConfig) -> Self {
        Self {
            config: config.clone(),
            values: VecDeque::with_capacity(config.window_size),
            sum: 0.0,
            ema: None,
        }
    }

    fn update(&mut self, value: f64) -> f64 {
        if !self.config.enabled {
            return value;
        }
        match self.config.method {
            SmoothingMethod::MovingAverage => {
                self.values.push_back(value);
                self.sum += value;
                if self.values.len() > self.config.window_size {
                    self.sum -= self.values.pop_front().expect("non-empty smoothing window");
                }
                self.sum / self.values.len() as f64
            }
            SmoothingMethod::ExponentialMovingAverage => {
                let alpha = 2.0 / (self.config.window_size as f64 + 1.0);
                let smoothed = self
                    .ema
                    .map_or(value, |previous| alpha * value + (1.0 - alpha) * previous);
                self.ema = Some(smoothed);
                smoothed
            }
        }
    }

    fn clear(&mut self) {
        self.values.clear();
        self.sum = 0.0;
        self.ema = None;
    }
}

impl ScorePipeline {
    pub(super) fn new(
        config: &ModelConfig,
        decomposition: &DecompositionConfig,
        smoothing: &SmoothingConfig,
    ) -> Result<Self, ModelError> {
        let detector_name = config
            .parameters
            .get("score_detector")
            .and_then(Value::as_str)
            .ok_or_else(|| ModelError::new("missing parameters.score_detector"))?;
        let window_size = usize_value(config, "score_window_size", None)?;
        let mut parameters = BTreeMap::new();
        if detector_name == "z_score" {
            parameters.insert(
                "ddof".into(),
                Value::from(usize_value(config, "score_ddof", Some(0))?),
            );
        }
        let detector_config = ModelConfig {
            id: config.id.clone(),
            enabled: config.enabled,
            kind: ModelKind::Univariate,
            algorithm: detector_name.into(),
            inputs: vec![SCORE_INPUT.into()],
            parameters,
            thresholds: config.thresholds.clone(),
        };
        Ok(Self {
            detector: UnivariateModel::from_config(&detector_config)?,
            window_size,
            scores: VecDeque::with_capacity(window_size),
            decomposer: StreamingDecomposer::new(decomposition),
            smoother: ScalarSmoother::new(smoothing),
        })
    }

    pub(super) fn update(
        &mut self,
        timestamp_ms: i64,
        score: f64,
    ) -> Result<Option<Detection>, ModelError> {
        if !score.is_finite() {
            return Err(ModelError::new("multivariate score must be finite"));
        }
        if self.scores.len() == self.window_size {
            self.scores.pop_front();
        }
        let decomposition = self.decomposer.update(score);
        let smoothed_score = self.smoother.update(decomposition.remainder);
        self.scores.push_back(Sample {
            timestamp_ms,
            value: smoothed_score,
        });
        if self.scores.len() < self.window_size {
            return Ok(None);
        }
        let mut detection = self.detector.evaluate(ModelInput {
            streams: BTreeMap::from([(SCORE_INPUT, &self.scores)]),
        })?;
        detection.details.insert(
            "decomposed_multivariate_score".into(),
            Value::from(decomposition.remainder),
        );
        detection.details.insert(
            "smoothed_multivariate_score".into(),
            Value::from(smoothed_score),
        );
        if let Some(period) = decomposition.period_samples {
            detection.details.insert(
                "detected_seasonal_period_samples".into(),
                Value::from(period),
            );
        }
        Ok(Some(detection))
    }

    pub(super) fn clear(&mut self) -> Result<(), ModelError> {
        self.scores.clear();
        self.decomposer.reset();
        self.smoother.clear();
        self.detector.reset()
    }
}

pub(super) fn usize_value(
    config: &ModelConfig,
    key: &str,
    default: Option<usize>,
) -> Result<usize, ModelError> {
    match config.parameters.get(key) {
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                ModelError::new(format!(
                    "model '{}' parameters.{key} must be a non-negative integer",
                    config.id
                ))
            }),
        None => default.ok_or_else(|| {
            ModelError::new(format!("model '{}' requires parameters.{key}", config.id))
        }),
    }
}

pub(super) fn float_value(
    config: &ModelConfig,
    key: &str,
    default: Option<f64>,
) -> Result<f64, ModelError> {
    match config.parameters.get(key) {
        Some(value) => value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                ModelError::new(format!(
                    "model '{}' parameters.{key} must be a finite number",
                    config.id
                ))
            }),
        None => default.ok_or_else(|| {
            ModelError::new(format!("model '{}' requires parameters.{key}", config.id))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrendDecompositionConfig;

    #[test]
    fn decomposes_multivariate_outputs_before_final_scoring() {
        let config = ModelConfig {
            id: "joint".into(),
            enabled: true,
            kind: ModelKind::Multivariate,
            algorithm: "mahalanobis".into(),
            inputs: vec!["left".into(), "right".into()],
            parameters: BTreeMap::from([
                ("score_detector".into(), Value::from("mad")),
                ("score_window_size".into(), Value::from(3)),
            ]),
            thresholds: BTreeMap::from([("score".into(), Value::from(3.0))]),
        };
        let decomposition = DecompositionConfig {
            trend: TrendDecompositionConfig {
                enabled: true,
                alpha: 1.0,
                beta: 1.0,
            },
            ..DecompositionConfig::default()
        };
        let mut pipeline =
            ScorePipeline::new(&config, &decomposition, &SmoothingConfig::default()).unwrap();
        assert!(pipeline.update(1, 1.0).unwrap().is_none());
        assert!(pipeline.update(2, 2.0).unwrap().is_none());
        let detection = pipeline.update(3, 3.0).unwrap().unwrap();
        assert_eq!(detection.details["decomposed_multivariate_score"], -1.0);
        assert_eq!(detection.score, Some(0.0));
    }

    #[test]
    fn smooths_decomposed_multivariate_scores_before_final_scoring() {
        let config = ModelConfig {
            id: "joint".into(),
            enabled: true,
            kind: ModelKind::Multivariate,
            algorithm: "mahalanobis".into(),
            inputs: vec!["left".into(), "right".into()],
            parameters: BTreeMap::from([
                ("score_detector".into(), Value::from("z_score")),
                ("score_window_size".into(), Value::from(3)),
            ]),
            thresholds: BTreeMap::from([("score".into(), Value::from(3.0))]),
        };
        let smoothing = SmoothingConfig {
            enabled: true,
            method: SmoothingMethod::MovingAverage,
            window_size: 2,
        };
        let mut pipeline =
            ScorePipeline::new(&config, &DecompositionConfig::default(), &smoothing).unwrap();
        assert!(pipeline.update(1, 1.0).unwrap().is_none());
        assert!(pipeline.update(2, 3.0).unwrap().is_none());
        let detection = pipeline.update(3, 5.0).unwrap().unwrap();
        assert_eq!(detection.details["decomposed_multivariate_score"], 5.0);
        assert_eq!(detection.details["smoothed_multivariate_score"], 4.0);
        assert_eq!(detection.details["value"], 4.0);
    }
}
