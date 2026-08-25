use std::collections::{BTreeMap, VecDeque};

use serde_json::Value;

use crate::config::{ModelConfig, ModelKind};
use crate::model::{AnomalyModel, Detection, ModelError, ModelInput};
use crate::univariate::UnivariateModel;
use crate::window::Sample;

const SCORE_INPUT: &str = "multivariate_score";

#[derive(Debug)]
pub(super) struct ScorePipeline {
    detector: UnivariateModel,
    window_size: usize,
    scores: VecDeque<Sample>,
}

impl ScorePipeline {
    pub(super) fn new(config: &ModelConfig) -> Result<Self, ModelError> {
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
        self.scores.push_back(Sample {
            timestamp_ms,
            value: score,
        });
        if self.scores.len() < self.window_size {
            return Ok(None);
        }
        self.detector
            .evaluate(ModelInput {
                streams: BTreeMap::from([(SCORE_INPUT, &self.scores)]),
            })
            .map(Some)
    }

    pub(super) fn clear(&mut self) {
        self.scores.clear();
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
