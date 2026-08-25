pub mod decomposition;
pub mod interpolation;
pub mod scaling;
pub mod smoothing;

use std::collections::VecDeque;

use crate::config::{
    DecompositionMethod, PreprocessingConfig, ScalingMethod, SmoothingMethod, WindowConfig,
};
use crate::window::Sample;

use self::decomposition::{stl, twitter};
use self::interpolation::{InterpolationResult, bounded_linear};
use self::scaling::{min_max, standard};
use self::smoothing::{exponential_moving_average, moving_average};

/// Applies the global preprocessing policy independently to every metric window.
#[derive(Debug, Clone)]
pub struct Preprocessor {
    config: PreprocessingConfig,
    window_duration_ms: i64,
}

impl Preprocessor {
    pub fn new(config: &PreprocessingConfig, window: &WindowConfig) -> Self {
        Self {
            config: config.clone(),
            window_duration_ms: window.duration_ms,
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
            self.window_duration_ms,
            &self.config.interpolation,
        )
    }

    /// Recomputes the derived window in the configured order. Recalculation is
    /// intentional because scaling and decomposition statistics move whenever
    /// a raw sample enters or leaves the sliding window.
    pub fn transform(&self, raw: &VecDeque<Sample>, interval_ms: i64) -> Option<VecDeque<Sample>> {
        let mut values: Vec<f64> = raw.iter().map(|sample| sample.value).collect();

        if self.config.scaling.enabled {
            values = match self.config.scaling.method {
                ScalingMethod::MinMax => min_max(
                    &values,
                    self.config.scaling.output_min,
                    self.config.scaling.output_max,
                    self.config.scaling.epsilon,
                ),
                ScalingMethod::Standard => standard(&values, self.config.scaling.epsilon),
            };
        }

        if self.config.decomposition.enabled {
            let period_samples = self
                .config
                .decomposition
                .period_ms
                .saturating_add(interval_ms / 2)
                / interval_ms;
            let period_samples = usize::try_from(period_samples).ok()?;
            let components = match self.config.decomposition.method {
                DecompositionMethod::Stl => {
                    stl(&values, period_samples, &self.config.decomposition.stl)
                }
                DecompositionMethod::Twitter => twitter(&values, period_samples),
            }?;
            values = components.remainder;
        }

        if self.config.smoothing.enabled {
            values = match self.config.smoothing.method {
                SmoothingMethod::MovingAverage => {
                    moving_average(&values, self.config.smoothing.window_size)
                }
                SmoothingMethod::ExponentialMovingAverage => {
                    exponential_moving_average(&values, self.config.smoothing.window_size)
                }
            };
        }

        Some(
            raw.iter()
                .zip(values)
                .map(|(sample, value)| Sample {
                    timestamp_ms: sample.timestamp_ms,
                    value,
                })
                .collect(),
        )
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
        let processor = Preprocessor::new(
            &config,
            &WindowConfig {
                duration_ms: 10,
                minimum_samples: 2,
                max_lateness_ms: 0,
            },
        );
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
            .transform(&raw, 1)
            .unwrap()
            .iter()
            .map(|sample| sample.value)
            .collect();
        assert_eq!(values, [0.0, 0.25, 0.75]);
    }
}
