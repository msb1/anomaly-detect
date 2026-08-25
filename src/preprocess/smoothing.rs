use std::collections::VecDeque;

pub fn moving_average(values: &[f64], window_size: usize) -> Vec<f64> {
    let mut queue = VecDeque::with_capacity(window_size);
    let mut sum = 0.0;
    values
        .iter()
        .map(|value| {
            queue.push_back(*value);
            sum += value;
            if queue.len() > window_size {
                sum -= queue.pop_front().expect("queue exceeds zero-sized window");
            }
            sum / queue.len() as f64
        })
        .collect()
}

pub fn exponential_moving_average(values: &[f64], window_size: usize) -> Vec<f64> {
    let Some(first) = values.first().copied() else {
        return Vec::new();
    };
    let alpha = 2.0 / (window_size as f64 + 1.0);
    let mut current = first;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if index > 0 {
                current = alpha * value + (1.0 - alpha) * current;
            }
            current
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooths_with_trailing_mean_and_ema() {
        assert_eq!(moving_average(&[1.0, 3.0, 5.0], 2), [1.0, 2.0, 4.0]);
        assert_eq!(
            exponential_moving_average(&[1.0, 3.0, 5.0], 3),
            [1.0, 2.0, 3.5]
        );
    }
}
