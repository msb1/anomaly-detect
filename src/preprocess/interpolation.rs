use crate::config::InterpolationConfig;
use crate::window::Sample;

#[derive(Debug, Clone, PartialEq)]
pub enum InterpolationResult {
    Samples(Vec<Sample>),
    GapLimitExceeded,
}

/// Returns synthetic points followed by the real current point. The gap
/// fraction measures missing points against the configured sliding-window
/// sample count.
pub fn bounded_linear(
    previous: Option<Sample>,
    current: Sample,
    interval_ms: i64,
    window_sample_count: usize,
    config: &InterpolationConfig,
) -> InterpolationResult {
    let Some(previous) = previous else {
        return InterpolationResult::Samples(vec![current]);
    };
    if !config.enabled || current.timestamp_ms <= previous.timestamp_ms {
        return InterpolationResult::Samples(vec![current]);
    }

    let delta = current.timestamp_ms.saturating_sub(previous.timestamp_ms);
    if delta <= interval_ms {
        return InterpolationResult::Samples(vec![current]);
    }
    let missing_points = (delta - 1) / interval_ms;
    let maximum_missing_points =
        (window_sample_count as f64 * config.max_gap_fraction).round() as i64;
    if missing_points > maximum_missing_points {
        return InterpolationResult::GapLimitExceeded;
    }

    let mut samples = Vec::with_capacity(usize::try_from(missing_points + 1).unwrap_or(1));
    for step in 1..=missing_points {
        let elapsed = step.saturating_mul(interval_ms);
        let timestamp_ms = previous.timestamp_ms.saturating_add(elapsed);
        let fraction = elapsed as f64 / delta as f64;
        samples.push(Sample {
            timestamp_ms,
            value: previous.value + fraction * (current.value - previous.value),
        });
    }
    samples.push(current);
    InterpolationResult::Samples(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> InterpolationConfig {
        InterpolationConfig {
            enabled: true,
            max_gap_fraction: 0.2,
        }
    }

    #[test]
    fn interpolates_only_inside_strict_gap_bound() {
        let previous = Sample {
            timestamp_ms: 0,
            value: 0.0,
        };
        let current = Sample {
            timestamp_ms: 3_000,
            value: 6.0,
        };
        let InterpolationResult::Samples(samples) =
            bounded_linear(Some(previous), current, 1_000, 10, &config())
        else {
            panic!("gap should be interpolated");
        };
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].value, 2.0);
        assert_eq!(samples[1].value, 4.0);

        let too_large = bounded_linear(
            Some(previous),
            Sample {
                timestamp_ms: 3_001,
                value: 6.0,
            },
            1_000,
            10,
            &config(),
        );
        assert_eq!(too_large, InterpolationResult::GapLimitExceeded);
    }
}
