pub mod mad;
pub mod z_score;

use crate::config::{ModelConfig, ModelKind};
use crate::model::{AnomalyModel, Detection, ModelError, ModelInput};

use self::mad::Mad;
use self::z_score::ZScore;

/// All currently supported univariate detectors.
#[derive(Debug)]
pub enum UnivariateModel {
    ZScore(ZScore),
    Mad(Mad),
}

impl UnivariateModel {
    pub fn from_config(config: &ModelConfig) -> Result<Self, ModelError> {
        if config.kind != ModelKind::Univariate {
            return Err(ModelError::new(format!(
                "model '{}' is not univariate",
                config.id
            )));
        }
        let [input] = config.inputs.as_slice() else {
            return Err(ModelError::new(format!(
                "univariate model '{}' requires exactly one input stream",
                config.id
            )));
        };
        let threshold = config
            .thresholds
            .get("score")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| {
                ModelError::new(format!(
                    "model '{}' requires a numeric thresholds.score",
                    config.id
                ))
            })?;

        match config.algorithm.as_str() {
            "z_score" => {
                let ddof = match config.parameters.get("ddof") {
                    Some(value) => value.as_u64().ok_or_else(|| {
                        ModelError::new(format!(
                            "model '{}' parameters.ddof must be a non-negative integer",
                            config.id
                        ))
                    })?,
                    None => 0,
                };
                let ddof = usize::try_from(ddof).map_err(|_| {
                    ModelError::new(format!("model '{}' has an invalid ddof", config.id))
                })?;
                Ok(Self::ZScore(ZScore::new(
                    config.id.clone(),
                    input.clone(),
                    threshold,
                    ddof,
                )?))
            }
            "mad" => Ok(Self::Mad(Mad::new(
                config.id.clone(),
                input.clone(),
                threshold,
            )?)),
            algorithm => Err(ModelError::new(format!(
                "unsupported univariate algorithm '{algorithm}' for model '{}'",
                config.id
            ))),
        }
    }
}

impl AnomalyModel for UnivariateModel {
    fn id(&self) -> &str {
        match self {
            Self::ZScore(model) => model.id(),
            Self::Mad(model) => model.id(),
        }
    }

    fn kind(&self) -> ModelKind {
        ModelKind::Univariate
    }

    fn evaluate(&self, input: ModelInput<'_>) -> Result<Detection, ModelError> {
        match self {
            Self::ZScore(model) => model.evaluate(input),
            Self::Mad(model) => model.evaluate(input),
        }
    }
}
