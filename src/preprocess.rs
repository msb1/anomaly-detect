pub mod decomposition;
pub mod interpolation;
pub mod scaling;
pub mod smoothing;

use std::collections::VecDeque;

use crate::config::{PreprocessingConfig, ScalingMethod, SmoothingMethod, WindowConfig};
use crate::window::Sample;

use self::interpolation::{InterpolationResult, bounded_linear};
use self::scaling::{min_max, standard};
use self::smoothing::{exponential_moving_average, moving_average};

/// Applies the global preprocessing policy independently to every metric window.
#[derive(Debug, Clone)]
pub struct Preprocessor {
    config: PreprocessingConfig,
    window_sample_count: usize,
}

impl Preprocessor {
    pub fn new(config: &PreprocessingConfig, window: &WindowConfig) -> Self {
        Self {
            config: config.clone(),
            window_sample_count: window.sample_count,
        }
    }

    pub fn interpolation(
        &self,
        previous: Option<Sample>,
        current: Sample,
        interval_ms: i64,
    ) -> InterpolationResult {
        bounded_linear(
            previous,
            current,
            interval_ms,
            self.window_sample_count,
            &self.config.interpolation,
        )
    }

    /// Metric preprocessing shared by raw multivariate inputs: scaling followed
    /// by the configured moving-average filter. Decomposition is deliberately
    /// excluded from this path.
    pub fn transform_multivariate(&self, raw: &VecDeque<Sample>) -> VecDeque<Sample> {
        self.transform_values(raw, true, true)
    }

    /// The current scaled observation which feeds the stateful univariate
    /// trend/seasonal decomposition.
    pub fn latest_scaled_value(&self, raw: &VecDeque<Sample>) -> Option<f64> {
        self.scale(raw.iter().map(|sample| sample.value).collect())
            .pop()
    }

    /// Applies only smoothing to an already decomposed univariate series.
    pub fn smooth_decomposed(&self, values: &VecDeque<Sample>) -> VecDeque<Sample> {
        self.transform_values(values, false, true)
    }

    /// Preserves window-local scaling when univariate decomposition is off.
    pub fn transform_univariate_without_decomposition(
        &self,
        raw: &VecDeque<Sample>,
    ) -> VecDeque<Sample> {
        self.transform_values(raw, true, true)
    }

    fn transform_values(
        &self,
        samples: &VecDeque<Sample>,
        apply_scaling: bool,
        apply_smoothing: bool,
    ) -> VecDeque<Sample> {
        let mut values: Vec<f64> = samples.iter().map(|sample| sample.value).collect();
        if apply_scaling {
            values = self.scale(values);
        }
        if apply_smoothing && self.config.smoothing.enabled {
            values = match self.config.smoothing.method {
                SmoothingMethod::MovingAverage => {
                    moving_average(&values, self.config.smoothing.window_size)
                }
                SmoothingMethod::ExponentialMovingAverage => {
                    exponential_moving_average(&values, self.config.smoothing.window_size)
                }
            };
        }
        samples
            .iter()
            .zip(values)
            .map(|(sample, value)| Sample {
                timestamp_ms: sample.timestamp_ms,
                value,
            })
            .collect()
    }

    fn scale(&self, values: Vec<f64>) -> Vec<f64> {
        if !self.config.scaling.enabled {
            return values;
        }
        match self.config.scaling.method {
            ScalingMethod::MinMax => min_max(
                &values,
                self.config.scaling.output_min,
                self.config.scaling.output_max,
                self.config.scaling.epsilon,
            ),
            ScalingMethod::Standard => standard(&values, self.config.scaling.epsilon),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DecompositionConfig, InterpolationConfig, ScalingConfig, SmoothingConfig};

    #[test]
    fn scaling_precedes_smoothing_in_derived_window() {
        let config = PreprocessingConfig {
            interpolation: InterpolationConfig::default(),
            scaling: ScalingConfig {
                enabled: true,
                method: ScalingMethod::MinMax,
                output_min: 0.0,
                output_max: 1.0,
                epsilon: 1e-12,
            },
            decomposition: DecompositionConfig::default(),
            smoothing: SmoothingConfig {
                enabled: true,
                method: SmoothingMethod::MovingAverage,
                window_size: 2,
            },
        };
        let processor = Preprocessor::new(&config, &WindowConfig { sample_count: 3 });
        let raw = VecDeque::from([
            Sample {
                timestamp_ms: 0,
                value: 0.0,
            },
            Sample {
                timestamp_ms: 1,
                value: 10.0,
            },
            Sample {
                timestamp_ms: 2,
                value: 20.0,
            },
        ]);
        let values: Vec<f64> = processor
            .transform_multivariate(&raw)
            .iter()
            .map(|sample| sample.value)
            .collect();
        assert_eq!(values, [0.0, 0.25, 0.75]);
    }
}
