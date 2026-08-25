use serde_json::Value;

use crate::config::ModelKind;
use crate::model::{AnomalyModel, Detection, ModelError, ModelInput};

/// Median absolute deviation detector over one fully preprocessed sliding window.
#[derive(Debug)]
pub struct Mad {
    id: String,
    input: String,
    threshold: f64,
}

impl Mad {
    pub fn new(id: String, input: String, threshold: f64) -> Result<Self, ModelError> {
        if !threshold.is_finite() || threshold <= 0.0 {
            return Err(ModelError::new(format!(
                "MAD model '{id}' threshold must be finite and positive"
            )));
        }
        Ok(Self {
            id,
            input,
            threshold,
        })
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

impl AnomalyModel for Mad {
    fn id(&self) -> &str {
        &self.id
    }

    fn kind(&self) -> ModelKind {
        ModelKind::Univariate
    }

    fn evaluate(&self, input: ModelInput<'_>) -> Result<Detection, ModelError> {
        let samples = input.streams.get(self.input.as_str()).ok_or_else(|| {
            ModelError::new(format!(
                "MAD model '{}' did not receive input '{}'",
                self.id, self.input
            ))
        })?;
        if samples.is_empty() {
            return Err(ModelError::new(format!(
                "MAD model '{}' needs at least one sample",
                self.id
            )));
        }

        let center = median(samples.iter().map(|sample| sample.value).collect());
        let mad = median(
            samples
                .iter()
                .map(|sample| (sample.value - center).abs())
                .collect(),
        );
        let current = samples
            .back()
            .expect("a non-empty window was checked above");
        let absolute_deviation = (current.value - center).abs();
        let score = if mad == 0.0 {
            if absolute_deviation == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            absolute_deviation / mad
        };
        let score_value = |value: f64| {
            let deviation = (value - center).abs();
            if mad == 0.0 {
                if deviation == 0.0 { 0.0 } else { f64::INFINITY }
            } else {
                deviation / mad
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
            window_sample_count: samples.len(),
            details: [
                ("input".into(), Value::from(self.input.clone())),
                ("timestamp_ms".into(), Value::from(current.timestamp_ms)),
                ("value".into(), Value::from(current.value)),
                ("median".into(), Value::from(center)),
                ("median_absolute_deviation".into(), Value::from(mad)),
                ("threshold".into(), Value::from(self.threshold)),
                ("sample_count".into(), Value::from(samples.len())),
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
    fn scores_latest_value_in_mad_units() {
        let values = input(&[1.0, 1.0, 2.0, 2.0, 100.0]);
        let model = Mad::new("mad".into(), "metric".into(), 10.0).unwrap();
        let detection = model
            .evaluate(ModelInput {
                streams: BTreeMap::from([("metric", &values)]),
            })
            .unwrap();
        assert_eq!(detection.score, Some(98.0));
        assert!(detection.anomalous);
    }

    #[test]
    fn non_median_value_is_anomalous_when_mad_is_zero() {
        let values = input(&[1.0, 1.0, 1.0, 10.0]);
        let model = Mad::new("mad".into(), "metric".into(), 3.0).unwrap();
        let detection = model
            .evaluate(ModelInput {
                streams: BTreeMap::from([("metric", &values)]),
            })
            .unwrap();
        assert_eq!(detection.score, Some(f64::INFINITY));
        assert!(detection.anomalous);
        assert_eq!(detection.anomalous_points, 1);
        assert_eq!(detection.window_sample_count, 4);
    }
}
