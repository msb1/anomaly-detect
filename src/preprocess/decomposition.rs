use crate::config::StlConfig;

#[derive(Debug, Clone, PartialEq)]
pub struct Components {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub remainder: Vec<f64>,
}

/// Robust STL-style additive decomposition. Trend and seasonal subseries are
/// fitted with local-linear LOESS and Tukey bisquare robustness weights.
pub fn stl(values: &[f64], period: usize, config: &StlConfig) -> Option<Components> {
    if period < 2 || values.len() < period.saturating_mul(2) {
        return None;
    }
    let mut robustness = vec![1.0; values.len()];
    let mut components = None;
    for iteration in 0..=config.robust_iterations {
        let trend_span = next_odd(period.saturating_mul(2).saturating_add(1));
        let trend = loess(values, trend_span, &robustness);
        let detrended: Vec<f64> = values
            .iter()
            .zip(&trend)
            .map(|(value, trend)| value - trend)
            .collect();
        let mut seasonal = vec![0.0; values.len()];
        for phase in 0..period {
            let indexes: Vec<usize> = (phase..values.len()).step_by(period).collect();
            let phase_values: Vec<f64> = indexes.iter().map(|index| detrended[*index]).collect();
            let phase_weights: Vec<f64> = indexes.iter().map(|index| robustness[*index]).collect();
            let fitted = loess(&phase_values, config.loess_span, &phase_weights);
            for (index, value) in indexes.into_iter().zip(fitted) {
                seasonal[index] = value;
            }
        }
        let seasonal_mean = seasonal.iter().sum::<f64>() / seasonal.len() as f64;
        seasonal
            .iter_mut()
            .for_each(|value| *value -= seasonal_mean);
        let remainder: Vec<f64> = values
            .iter()
            .zip(&trend)
            .zip(&seasonal)
            .map(|((value, trend), seasonal)| value - trend - seasonal)
            .collect();
        components = Some(Components {
            trend,
            seasonal,
            remainder,
        });
        if iteration < config.robust_iterations {
            robustness = robustness_weights(&components.as_ref()?.remainder);
        }
    }
    components
}

/// Twitter AnomalyDetection-style median decomposition: a constant median
/// trend plus a robust median seasonal profile for each phase.
pub fn twitter(values: &[f64], period: usize) -> Option<Components> {
    if period < 2 || values.len() < period.saturating_mul(2) {
        return None;
    }
    let level = median(values);
    let trend = vec![level; values.len()];
    let mut phase_pattern = Vec::with_capacity(period);
    for phase in 0..period {
        let phase_values: Vec<f64> = (phase..values.len())
            .step_by(period)
            .map(|index| values[index] - level)
            .collect();
        phase_pattern.push(median(&phase_values));
    }
    let center = median(&phase_pattern);
    phase_pattern.iter_mut().for_each(|value| *value -= center);
    let seasonal: Vec<f64> = (0..values.len())
        .map(|index| phase_pattern[index % period])
        .collect();
    let remainder = values
        .iter()
        .zip(&trend)
        .zip(&seasonal)
        .map(|((value, trend), seasonal)| value - trend - seasonal)
        .collect();
    Some(Components {
        trend,
        seasonal,
        remainder,
    })
}

fn loess(values: &[f64], requested_span: usize, robustness: &[f64]) -> Vec<f64> {
    if values.len() <= 2 {
        return values.to_vec();
    }
    let span = requested_span.clamp(3, values.len());
    let half = span / 2;
    (0..values.len())
        .map(|center| {
            let mut left = center.saturating_sub(half);
            let right = (left + span).min(values.len());
            left = right.saturating_sub(span);
            let max_distance = (center - left).max(right - 1 - center).max(1) as f64;
            let mut sum_w = 0.0;
            let mut sum_wx = 0.0;
            let mut sum_wxx = 0.0;
            let mut sum_wy = 0.0;
            let mut sum_wxy = 0.0;
            for index in left..right {
                let x = index as f64 - center as f64;
                let distance = x.abs() / max_distance;
                let tricube = (1.0 - distance.powi(3)).max(0.0).powi(3);
                let weight = tricube * robustness[index];
                sum_w += weight;
                sum_wx += weight * x;
                sum_wxx += weight * x * x;
                sum_wy += weight * values[index];
                sum_wxy += weight * x * values[index];
            }
            let denominator = sum_w * sum_wxx - sum_wx * sum_wx;
            if sum_w <= f64::EPSILON {
                values[center]
            } else if denominator.abs() <= f64::EPSILON {
                sum_wy / sum_w
            } else {
                (sum_wxx * sum_wy - sum_wx * sum_wxy) / denominator
            }
        })
        .collect()
}

fn robustness_weights(remainder: &[f64]) -> Vec<f64> {
    let absolute: Vec<f64> = remainder.iter().map(|value| value.abs()).collect();
    let scale = 6.0 * median(&absolute);
    if scale <= f64::EPSILON {
        return vec![1.0; remainder.len()];
    }
    remainder
        .iter()
        .map(|value| {
            let ratio = value.abs() / scale;
            if ratio >= 1.0 {
                0.0
            } else {
                (1.0 - ratio * ratio).powi(2)
            }
        })
        .collect()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn next_odd(value: usize) -> usize {
    if value.is_multiple_of(2) {
        value.saturating_add(1)
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twitter_removes_repeating_seasonality() {
        let values = [10.0, 12.0, 10.0, 12.0, 10.0, 12.0];
        let components = twitter(&values, 2).unwrap();
        assert!(components.remainder.iter().all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn stl_returns_finite_additive_components() {
        let values: Vec<f64> = (0..24)
            .map(|index| index as f64 * 0.1 + if index % 4 == 0 { 2.0 } else { 0.0 })
            .collect();
        let components = stl(
            &values,
            4,
            &StlConfig {
                loess_span: 5,
                robust_iterations: 1,
            },
        )
        .unwrap();
        for (index, value) in values.iter().enumerate() {
            let reconstructed =
                components.trend[index] + components.seasonal[index] + components.remainder[index];
            assert!((reconstructed - value).abs() < 1e-10);
        }
    }
}
