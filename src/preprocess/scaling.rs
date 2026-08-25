pub fn min_max(values: &[f64], output_min: f64, output_max: f64, epsilon: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = maximum - minimum;
    if range <= epsilon {
        return vec![(output_min + output_max) / 2.0; values.len()];
    }
    values
        .iter()
        .map(|value| output_min + (value - minimum) / range * (output_max - output_min))
        .collect()
}

pub fn standard(values: &[f64], epsilon: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let standard_deviation = variance.sqrt();
    if standard_deviation <= epsilon {
        return vec![0.0; values.len()];
    }
    values
        .iter()
        .map(|value| (value - mean) / standard_deviation)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_min_max_and_standard() {
        assert_eq!(min_max(&[2.0, 4.0, 6.0], 0.0, 1.0, 1e-12), [0.0, 0.5, 1.0]);
        let scaled = standard(&[1.0, 2.0, 3.0], 1e-12);
        assert!(scaled.iter().sum::<f64>().abs() < 1e-12);
        assert_eq!(standard(&[7.0, 7.0], 1e-12), [0.0, 0.0]);
    }
}
