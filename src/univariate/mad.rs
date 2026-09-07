use std::sync::Mutex;

use serde_json::Value;

use crate::config::ModelKind;
use crate::model::{AnomalyModel, Detection, ModelError, ModelInput};

pub const DEFAULT_MAD_EMA_ALPHA: f64 = 0.05;
pub const DEFAULT_MAD_EPSILON: f64 = 1e-6;
const MODIFIED_Z_SCORE_SCALE: f64 = 0.6745;

/// Median absolute deviation detector over one fully preprocessed sliding window.
#[derive(Debug)]
pub struct Mad {
    id: String,
    input: String,
    threshold: f64,
    mad_ema_alpha: f64,
    epsilon: f64,
    smoothed_mad: Mutex<Option<f64>>,
}

impl Mad {
    pub fn new(id: String, input: String, threshold: f64) -> Result<Self, ModelError> {
        Self::with_stability(
            id,
            input,
            threshold,
            DEFAULT_MAD_EMA_ALPHA,
            DEFAULT_MAD_EPSILON,
        )
    }

    pub fn with_stability(
        id: String,
        input: String,
        threshold: f64,
        mad_ema_alpha: f64,
        epsilon: f64,
    ) -> Result<Self, ModelError> {
        if !threshold.is_finite() || threshold <= 0.0 {
            return Err(ModelError::new(format!(
                "MAD model '{id}' threshold must be finite and positive"
            )));
        }
        if !mad_ema_alpha.is_finite() || !(0.0 < mad_ema_alpha && mad_ema_alpha <= 1.0) {
            return Err(ModelError::new(format!(
                "MAD model '{id}' mad_ema_alpha must be finite and in (0, 1]"
            )));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ModelError::new(format!(
                "MAD model '{id}' epsilon must be finite and positive"
            )));
        }
        Ok(Self {
            id,
            input,
            threshold,
            mad_ema_alpha,
            epsilon,
            smoothed_mad: Mutex::new(None),
        })
    }

    pub fn reset(&self) -> Result<(), ModelError> {
        *self.smoothed_mad.lock().map_err(|_| {
            ModelError::new(format!("MAD model '{}' state lock is poisoned", self.id))
        })? = None;
        Ok(())
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
        let raw_mad = median(
            samples
                .iter()
                .map(|sample| (sample.value - center).abs())
                .collect(),
        );
        let smoothed_mad = {
            let mut state = self.smoothed_mad.lock().map_err(|_| {
                ModelError::new(format!("MAD model '{}' state lock is poisoned", self.id))
            })?;
            let next = state.map_or(raw_mad, |previous| {
                self.mad_ema_alpha
                    .mul_add(raw_mad, (1.0 - self.mad_ema_alpha) * previous)
            });
            *state = Some(next);
            next
        };
        let stabilized_mad = smoothed_mad + self.epsilon;
        let current = samples
            .back()
            .expect("a non-empty window was checked above");
        let absolute_deviation = (current.value - center).abs();
        let score = MODIFIED_Z_SCORE_SCALE * absolute_deviation / stabilized_mad;
        let score_value = |value: f64| {
            let deviation = (value - center).abs();
            MODIFIED_Z_SCORE_SCALE * deviation / stabilized_mad
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
                ("median_absolute_deviation".into(), Value::from(raw_mad)),
                ("smoothed_mad".into(), Value::from(smoothed_mad)),
                ("stabilized_mad".into(), Value::from(stabilized_mad)),
                ("mad_ema_alpha".into(), Value::from(self.mad_ema_alpha)),
                ("epsilon".into(), Value::from(self.epsilon)),
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
    fn scores_latest_value_with_stabilized_modified_z_score() {
        let values = input(&[1.0, 1.0, 2.0, 2.0, 100.0]);
        let model = Mad::with_stability("mad".into(), "metric".into(), 10.0, 0.05, 0.1).unwrap();
        let detection = model
            .evaluate(ModelInput {
                streams: BTreeMap::from([("metric", &values)]),
            })
            .unwrap();
        assert_eq!(detection.score, Some(0.6745 * 98.0 / 1.1));
        assert!(detection.anomalous);
    }

    #[test]
    fn epsilon_keeps_zero_mad_score_finite() {
        let values = input(&[1.0, 1.0, 1.0, 10.0]);
        let model = Mad::with_stability("mad".into(), "metric".into(), 3.0, 0.05, 0.1).unwrap();
        let detection = model
            .evaluate(ModelInput {
                streams: BTreeMap::from([("metric", &values)]),
            })
            .unwrap();
        assert_eq!(detection.score, Some(0.6745 * 9.0 / 0.1));
        assert!(detection.score.unwrap().is_finite());
        assert!(detection.anomalous);
        assert_eq!(detection.anomalous_points, 1);
        assert_eq!(detection.window_sample_count, 4);
    }

    #[test]
    fn smooths_mad_between_windows_and_can_reset_state() {
        let model = Mad::with_stability("mad".into(), "metric".into(), 10.0, 0.25, 0.5).unwrap();
        let first = input(&[0.0, 1.0, 2.0]);
        model
            .evaluate(ModelInput {
                streams: BTreeMap::from([("metric", &first)]),
            })
            .unwrap();

        let second = input(&[0.0, 3.0, 6.0]);
        let detection = model
            .evaluate(ModelInput {
                streams: BTreeMap::from([("metric", &second)]),
            })
            .unwrap();
        assert_eq!(detection.details["median_absolute_deviation"], 3.0);
        assert_eq!(detection.details["smoothed_mad"], 1.5);
        assert_eq!(detection.details["stabilized_mad"], 2.0);

        model.reset().unwrap();
        let detection = model
            .evaluate(ModelInput {
                streams: BTreeMap::from([("metric", &second)]),
            })
            .unwrap();
        assert_eq!(detection.details["smoothed_mad"], 3.0);
    }

    #[test]
    fn rejects_invalid_stability_parameters() {
        assert!(Mad::with_stability("mad".into(), "metric".into(), 3.0, 0.0, 0.1).is_err());
        assert!(Mad::with_stability("mad".into(), "metric".into(), 3.0, 1.1, 0.1).is_err());
        assert!(Mad::with_stability("mad".into(), "metric".into(), 3.0, 0.05, 0.0).is_err());
    }
}
