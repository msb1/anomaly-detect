use std::collections::BTreeMap;

use serde_json::Value;
use thiserror::Error;

use crate::config::{ModelConfig, ModelKind};
use crate::multivariate::SharedMultivariateModel;
use crate::univariate::UnivariateModel;
use crate::window::Sample;

/// Immutable, fully preprocessed and primed input presented to a detector.
pub struct ModelInput<'a> {
    pub streams: BTreeMap<&'a str, &'a std::collections::VecDeque<Sample>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub model_id: String,
    pub timestamp_ms: i64,
    pub anomalous: bool,
    pub score: Option<f64>,
    pub anomalous_points: usize,
    pub window_sample_count: usize,
    pub details: BTreeMap<String, Value>,
}

#[derive(Debug, Error)]
#[error("model evaluation failed: {message}")]
pub struct ModelError {
    pub message: String,
}

impl ModelError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Extension point for univariate and multivariate implementations.
/// The coordinator calls a detector only when every configured input is primed.
pub trait AnomalyModel: Send + Sync {
    fn id(&self) -> &str;
    fn kind(&self) -> ModelKind;
    fn evaluate(&self, input: ModelInput<'_>) -> Result<Detection, ModelError>;
}

#[derive(Debug)]
pub struct ModelCoordinator {
    models: Vec<RegisteredModel>,
}

#[derive(Debug)]
struct RegisteredModel {
    config: ModelConfig,
    univariate: Option<UnivariateModel>,
    multivariate: Option<SharedMultivariateModel>,
}

impl ModelCoordinator {
    pub fn new(models: Vec<ModelConfig>) -> Result<Self, ModelError> {
        let models = models
            .into_iter()
            .map(|config| {
                let univariate = (config.kind == ModelKind::Univariate)
                    .then(|| UnivariateModel::from_config(&config))
                    .transpose()?;
                let multivariate = (config.kind == ModelKind::Multivariate)
                    .then(|| SharedMultivariateModel::from_config(&config))
                    .transpose()?;
                Ok(RegisteredModel {
                    config,
                    univariate,
                    multivariate,
                })
            })
            .collect::<Result<_, ModelError>>()?;
        Ok(Self { models })
    }

    pub fn ready_model_indices(
        &self,
        changed_streams: &[String],
        is_primed: impl Fn(&str) -> bool,
    ) -> Vec<usize> {
        self.models
            .iter()
            .enumerate()
            .filter(|(_, model)| {
                model.config.enabled
                    && model
                        .config
                        .inputs
                        .iter()
                        .any(|input| changed_streams.contains(input))
                    && model.config.inputs.iter().all(|input| is_primed(input))
            })
            .map(|(index, _)| index)
            .collect()
    }

    pub fn config(&self, index: usize) -> &ModelConfig {
        &self.models[index].config
    }

    pub fn evaluate(
        &self,
        index: usize,
        input: ModelInput<'_>,
    ) -> Result<Option<Detection>, ModelError> {
        let model = &self.models[index];
        match (&model.univariate, &model.multivariate) {
            (Some(univariate), None) => univariate.evaluate(input).map(Some),
            (None, Some(multivariate)) => multivariate.evaluate_streams(input),
            _ => Err(ModelError::new(format!(
                "model '{}' has invalid coordinator state",
                model.config.id
            ))),
        }
    }

    pub fn reset_for_streams(&self, reset_streams: &[String]) -> Result<(), ModelError> {
        for model in &self.models {
            if model
                .config
                .inputs
                .iter()
                .any(|input| reset_streams.contains(input))
                && let Some(multivariate) = &model.multivariate
            {
                multivariate.reset()?;
            }
        }
        Ok(())
    }
}
