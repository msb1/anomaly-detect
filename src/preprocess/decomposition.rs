use std::collections::VecDeque;
use std::f64::consts::TAU;

use rustfft::{FftPlanner, num_complex::Complex};

use crate::config::{DecompositionConfig, SeasonalDecompositionConfig};

/// The causal decomposition of one observation using estimates available when
/// that observation arrives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecompositionOutput {
    pub trend: f64,
    pub seasonal: f64,
    pub remainder: f64,
    pub period_samples: Option<usize>,
}

/// Stateful streaming decomposition: Holt's linear method is applied first,
/// then an exact sliding DFT identifies and removes repeating structure.
#[derive(Debug, Clone)]
pub struct StreamingDecomposer {
    config: DecompositionConfig,
    level: Option<f64>,
    slope: f64,
    seasonal: SlidingDft,
    detected_period: Option<usize>,
    samples_since_detection: usize,
}

impl StreamingDecomposer {
    pub fn new(config: &DecompositionConfig) -> Self {
        Self {
            config: config.clone(),
            level: None,
            slope: 0.0,
            seasonal: SlidingDft::new(config.seasonal.window_size_samples),
            detected_period: None,
            samples_since_detection: 0,
        }
    }

    pub fn update(&mut self, value: f64) -> DecompositionOutput {
        let trend = self.update_trend(value);
        let detrended = value - trend;

        let mut seasonal = 0.0;
        if self.config.seasonal.enabled && self.seasonal.is_full() {
            if self.detected_period.is_none()
                || self.samples_since_detection >= self.config.seasonal.detection_interval_samples
            {
                self.detected_period = self.seasonal.dominant_period(&self.config.seasonal);
                self.samples_since_detection = 0;
            }
            if let Some(period) = self.detected_period {
                seasonal = self.seasonal.lagged(period).unwrap_or(0.0);
            }
            self.samples_since_detection += 1;
        }
        if self.config.seasonal.enabled {
            self.seasonal.push(detrended);
        }

        DecompositionOutput {
            trend,
            seasonal,
            remainder: detrended - seasonal,
            period_samples: self.detected_period,
        }
    }

    pub fn reset(&mut self) {
        self.level = None;
        self.slope = 0.0;
        self.seasonal.clear();
        self.detected_period = None;
        self.samples_since_detection = 0;
    }

    fn update_trend(&mut self, value: f64) -> f64 {
        if !self.config.trend.enabled {
            return 0.0;
        }
        let Some(level) = self.level else {
            self.level = Some(value);
            return value;
        };
        let next_level = self.config.trend.alpha * value
            + (1.0 - self.config.trend.alpha) * (level + self.slope);
        let next_slope = self.config.trend.beta * (next_level - level)
            + (1.0 - self.config.trend.beta) * self.slope;
        self.level = Some(next_level);
        self.slope = next_slope;
        next_level + next_slope
    }
}

/// Exact sliding DFT. The first complete window is initialized with RustFFT;
/// each subsequent point updates all bins in O(N) from the outgoing and
/// incoming values instead of recalculating an O(N log N) FFT.
#[derive(Debug, Clone)]
struct SlidingDft {
    window_size: usize,
    values: VecDeque<f64>,
    spectrum: Vec<Complex<f64>>,
    rotations: Vec<Complex<f64>>,
}

impl SlidingDft {
    fn new(window_size: usize) -> Self {
        let rotations = (0..window_size)
            .map(|bin| Complex::from_polar(1.0, TAU * bin as f64 / window_size as f64))
            .collect();
        Self {
            window_size,
            values: VecDeque::with_capacity(window_size),
            spectrum: Vec::new(),
            rotations,
        }
    }

    fn is_full(&self) -> bool {
        self.values.len() == self.window_size
    }

    fn push(&mut self, value: f64) {
        if !self.is_full() {
            self.values.push_back(value);
            if self.is_full() {
                self.initialize_spectrum();
            }
            return;
        }

        let outgoing = self
            .values
            .pop_front()
            .expect("a full sliding DFT contains an outgoing value");
        self.values.push_back(value);
        for (rotation, spectrum) in self.rotations.iter().zip(self.spectrum.iter_mut()) {
            *spectrum = *rotation * (*spectrum + Complex::new(value - outgoing, 0.0));
        }
    }

    fn lagged(&self, period: usize) -> Option<f64> {
        self.values
            .get(self.values.len().checked_sub(period)?)
            .copied()
    }

    fn dominant_period(&self, config: &SeasonalDecompositionConfig) -> Option<usize> {
        if !self.is_full() {
            return None;
        }
        let minimum_bin = self.window_size.div_ceil(config.max_period_samples).max(1);
        let maximum_bin = (self.window_size / config.min_period_samples).min(self.window_size / 2);
        if minimum_bin > maximum_bin {
            return None;
        }
        let dominant_bin = (minimum_bin..=maximum_bin).max_by(|left, right| {
            self.spectrum[*left]
                .norm_sqr()
                .total_cmp(&self.spectrum[*right].norm_sqr())
        })?;
        Some(
            ((self.window_size as f64 / dominant_bin as f64).round() as usize)
                .clamp(config.min_period_samples, config.max_period_samples),
        )
    }

    fn clear(&mut self) {
        self.values.clear();
        self.spectrum.clear();
    }

    fn initialize_spectrum(&mut self) {
        let mut buffer: Vec<Complex<f64>> = self
            .values
            .iter()
            .map(|value| Complex::new(*value, 0.0))
            .collect();
        FftPlanner::new()
            .plan_fft_forward(self.window_size)
            .process(&mut buffer);
        self.spectrum = buffer;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SeasonalDecompositionConfig, TrendDecompositionConfig};

    fn seasonal_config(window_size_samples: usize) -> SeasonalDecompositionConfig {
        SeasonalDecompositionConfig {
            enabled: true,
            window_size_samples,
            detection_interval_samples: 1,
            min_period_samples: 2,
            max_period_samples: window_size_samples,
        }
    }

    #[test]
    fn sliding_update_matches_a_fresh_fft() {
        let mut sliding = SlidingDft::new(8);
        for value in 0..8 {
            sliding.push(value as f64);
        }
        sliding.push(8.0);
        let incremental = sliding.spectrum.clone();
        sliding.initialize_spectrum();
        for (actual, expected) in incremental.iter().zip(&sliding.spectrum) {
            assert!((*actual - *expected).norm() < 1e-9);
        }
    }

    #[test]
    fn detects_and_removes_a_repeating_period() {
        let config = DecompositionConfig {
            trend: TrendDecompositionConfig::default(),
            seasonal: seasonal_config(16),
        };
        let mut decomposer = StreamingDecomposer::new(&config);
        let mut output = None;
        for index in 0..33 {
            output = Some(decomposer.update(if index % 4 < 2 { 3.0 } else { -3.0 }));
        }
        let output = output.unwrap();
        assert_eq!(output.period_samples, Some(4));
        assert!(output.remainder.abs() < 1e-9);
    }

    #[test]
    fn reset_discards_holt_and_spectral_state() {
        let config = DecompositionConfig {
            trend: TrendDecompositionConfig {
                enabled: true,
                alpha: 0.2,
                beta: 0.1,
            },
            seasonal: seasonal_config(8),
        };
        let mut decomposer = StreamingDecomposer::new(&config);
        for value in 0..10 {
            decomposer.update(value as f64);
        }
        decomposer.reset();
        assert_eq!(decomposer.update(42.0).remainder, 0.0);
    }
}
