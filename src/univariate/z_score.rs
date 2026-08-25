use serde_json::Value;

use crate::config::ModelKind;
use crate::model::{AnomalyModel, Detection, ModelError, ModelInput};

/// Absolute Z-score detector over one fully preprocessed sliding window.
#[derive(Debug)]
pub struct ZScore {
    id: String,
    input: String,
    threshold: f64,
    ddof: usize,
}

impl ZScore {
    pub fn new(id: String, input: String, threshold: f64, ddof: usize) -> Result<Self, ModelError> {
        if !threshold.is_finite() || threshold <= 0.0 {
            return Err(ModelError::new(format!(
                "Z-score model '{id}' threshold must be finite and positive"
            )));
        }
        Ok(Self {
            id,
            input,
            threshold,
            ddof,
        })
    }
}

impl AnomalyModel for ZScore {
    fn id(&self) -> &str {
        &self.id
    }

    fn kind(&self) -> ModelKind {
        ModelKind::Univariate
    }

    fn evaluate(&self, input: ModelInput<'_>) -> Result<Detection, ModelError> {
        let samples = input.streams.get(self.input.as_str()).ok_or_else(|| {
            ModelError::new(format!(
                "Z-score model '{}' did not receive input '{}'",
                self.id, self.input
            ))
        })?;
        if samples.len() <= self.ddof {
            return Err(ModelError::new(format!(
                "Z-score model '{}' needs more than {} samples",
                self.id, self.ddof
            )));
        }

        let count = samples.len();
        let mean = samples.iter().map(|sample| sample.value).sum::<f64>() / count as f64;
        let sum_squared_deviations = samples
            .iter()
            .map(|sample| (sample.value - mean).powi(2))
            .sum::<f64>();
        let standard_deviation = (sum_squared_deviations / (count - self.ddof) as f64).sqrt();
        let current = samples
            .back()
            .expect("a non-empty window was checked above");
        let absolute_deviation = (current.value - mean).abs();
        let score = if standard_deviation == 0.0 {
            if absolute_deviation == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            absolute_deviation / standard_deviation
        };
        let score_value = |value: f64| {
            let deviation = (value - mean).abs();
            if standard_deviation == 0.0 {
                if deviation == 0.0 { 0.0 } else { f64::INFINITY }
            } else {
                deviation / standard_deviation
            }
        };
        let anomalous_points = samples
            .iter()
            .filter(|sample| score_value(sample.value) > self.threshold)
            .count();

        Ok(Detection {
            model_id: self.id.clone(),
            timestamp_ms: current.timestamp_ms,
            anomalous: score > self.threshold,
            score: Some(score),
            anomalous_points,
            window_sample_count: count,
            details: [
                ("input".into(), Value::from(self.input.clone())),
                ("timestamp_ms".into(), Value::from(current.timestamp_ms)),
                ("value".into(), Value::from(current.value)),
                ("mean".into(), Value::from(mean)),
                ("standard_deviation".into(), Value::from(standard_deviation)),
                ("threshold".into(), Value::from(self.threshold)),
                ("sample_count".into(), Value::from(count)),
                ("ddof".into(), Value::from(self.ddof)),
            ]
            .into_iter()
            .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, VecDeque};

    use crate::window::Sample;

    use super::*;

    fn input(values: &[f64]) -> VecDeque<Sample> {
        values
            .iter()
            .enumerate()
            .map(|(timestamp_ms, value)| Sample {
                timestamp_ms: timestamp_ms as i64,
                value: *value,
            })
            .collect()
    }

    #[test]
    fn scores_latest_value_with_population_standard_deviation() {
        let values = input(&[1.0, 2.0, 3.0]);
        let model = ZScore::new("z".into(), "metric".into(), 1.2, 0).unwrap();
        let detection = model
            .evaluate(ModelInput {
                streams: BTreeMap::from([("metric", &values)]),
            })
            .unwrap();
        assert!(detection.anomalous);
        assert!((detection.score.unwrap() - 1.224_744_871_391_589).abs() < 1e-12);
        assert_eq!(detection.timestamp_ms, 2);
        assert_eq!(detection.anomalous_points, 2);
        assert_eq!(detection.window_sample_count, 3);
    }

    #[test]
    fn constant_window_has_zero_score() {
        let values = input(&[4.0, 4.0, 4.0]);
        let model = ZScore::new("z".into(), "metric".into(), 3.0, 1).unwrap();
        let detection = model
            .evaluate(ModelInput {
                streams: BTreeMap::from([("metric", &values)]),
            })
            .unwrap();
        assert_eq!(detection.score, Some(0.0));
        assert!(!detection.anomalous);
    }
}
