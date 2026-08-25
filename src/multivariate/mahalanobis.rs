use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use nalgebra::{DMatrix, DVector};

use crate::model::ModelError;

/// A fixed-size, candidate-exclusive Mahalanobis baseline maintained with
/// forward and reverse multivariate Welford updates.
#[derive(Debug)]
pub struct RollingWelfordMahalanobis {
    window_size: usize,
    dimensions: usize,
    regularization: f64,
    shrinkage: f64,
    window: VecDeque<DVector<f64>>,
    count: usize,
    mean: DVector<f64>,
    m2: DMatrix<f64>,
}

/// Cloneable wrapper suitable for sharing a standalone processor across tasks.
pub type SharedRollingWelfordMahalanobis = Arc<Mutex<RollingWelfordMahalanobis>>;

impl RollingWelfordMahalanobis {
    pub fn new(
        window_size: usize,
        dimensions: usize,
        regularization: f64,
        shrinkage: f64,
    ) -> Result<Self, ModelError> {
        if dimensions == 0 || window_size <= dimensions {
            return Err(ModelError::new(
                "Mahalanobis window size must exceed its positive dimension count",
            ));
        }
        if !regularization.is_finite() || regularization <= 0.0 {
            return Err(ModelError::new(
                "Mahalanobis regularization must be finite and positive",
            ));
        }
        if !shrinkage.is_finite() || !(0.0..=1.0).contains(&shrinkage) {
            return Err(ModelError::new(
                "Mahalanobis shrinkage must be between zero and one",
            ));
        }
        Ok(Self {
            window_size,
            dimensions,
            regularization,
            shrinkage,
            window: VecDeque::with_capacity(window_size),
            count: 0,
            mean: DVector::zeros(dimensions),
            m2: DMatrix::zeros(dimensions, dimensions),
        })
    }

    pub fn into_shared(self) -> SharedRollingWelfordMahalanobis {
        Arc::new(Mutex::new(self))
    }

    /// Scores against the preceding full window, then rolls the candidate in.
    pub fn update(&mut self, next_point: DVector<f64>) -> Result<Option<f64>, ModelError> {
        if next_point.len() != self.dimensions {
            return Err(ModelError::new(format!(
                "Mahalanobis input has {} values; expected {}",
                next_point.len(),
                self.dimensions
            )));
        }
        if next_point.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::new("Mahalanobis input must be finite"));
        }

        let score = if self.window.len() == self.window_size {
            Some(self.calculate_distance(&next_point)?)
        } else {
            None
        };

        if self.window.len() == self.window_size {
            let oldest = self
                .window
                .pop_front()
                .expect("a full rolling window has a front value");
            self.remove_point(&oldest);
        }
        self.add_point(&next_point);
        self.window.push_back(next_point);
        Ok(score)
    }

    fn add_point(&mut self, x: &DVector<f64>) {
        self.count += 1;
        let count = self.count as f64;
        let old_mean = self.mean.clone();
        for row in 0..self.dimensions {
            self.mean[row] += (x[row] - old_mean[row]) / count;
        }
        for row in 0..self.dimensions {
            let delta_old = x[row] - old_mean[row];
            for column in 0..self.dimensions {
                self.m2[(row, column)] += delta_old * (x[column] - self.mean[column]);
            }
        }
    }

    fn remove_point(&mut self, x: &DVector<f64>) {
        if self.count <= 1 {
            self.count = 0;
            self.mean.fill(0.0);
            self.m2.fill(0.0);
            return;
        }

        let old_mean = self.mean.clone();
        self.count -= 1;
        let count = self.count as f64;
        for row in 0..self.dimensions {
            self.mean[row] -= (x[row] - old_mean[row]) / count;
        }
        for row in 0..self.dimensions {
            let delta_old = x[row] - old_mean[row];
            for column in 0..self.dimensions {
                self.m2[(row, column)] -= delta_old * (x[column] - self.mean[column]);
            }
        }
    }

    fn calculate_distance(&self, x: &DVector<f64>) -> Result<f64, ModelError> {
        let denominator = (self.count - 1) as f64;
        let mut covariance = DMatrix::zeros(self.dimensions, self.dimensions);
        let mut trace = 0.0;
        for row in 0..self.dimensions {
            for column in 0..self.dimensions {
                let symmetric_m2 = (self.m2[(row, column)] + self.m2[(column, row)]) * 0.5;
                covariance[(row, column)] = symmetric_m2 / denominator;
            }
            trace += covariance[(row, row)];
        }

        let spherical_variance = (trace / self.dimensions as f64).max(0.0);
        covariance *= 1.0 - self.shrinkage;
        for diagonal in 0..self.dimensions {
            covariance[(diagonal, diagonal)] +=
                self.shrinkage * spherical_variance + self.regularization;
        }

        let cholesky = covariance.cholesky().ok_or_else(|| {
            ModelError::new(
                "regularized Mahalanobis covariance was not positive definite; increase regularization or shrinkage",
            )
        })?;
        let difference = x - &self.mean;
        let solved = cholesky.solve(&difference);
        let distance_squared = difference.dot(&solved);
        if !distance_squared.is_finite() {
            return Err(ModelError::new("Mahalanobis distance is non-finite"));
        }
        Ok(distance_squared.max(0.0).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regularization_handles_flat_and_collinear_metrics() {
        let mut detector = RollingWelfordMahalanobis::new(4, 2, 1e-6, 0.1).unwrap();
        for _ in 0..4 {
            assert_eq!(
                detector.update(DVector::from_vec(vec![2.0, 4.0])).unwrap(),
                None
            );
        }
        assert_eq!(
            detector.update(DVector::from_vec(vec![2.0, 4.0])).unwrap(),
            Some(0.0)
        );
        assert!(
            detector
                .update(DVector::from_vec(vec![8.0, -3.0]))
                .unwrap()
                .unwrap()
                .is_finite()
        );
    }

    #[test]
    fn candidate_is_not_part_of_its_own_baseline() {
        let mut detector = RollingWelfordMahalanobis::new(4, 2, 1e-3, 0.2).unwrap();
        for point in [[-0.1, -0.1], [0.1, 0.1], [-0.1, 0.1], [0.1, -0.1]] {
            detector.update(DVector::from_row_slice(&point)).unwrap();
        }
        let distance = detector
            .update(DVector::from_vec(vec![10.0, 10.0]))
            .unwrap()
            .unwrap();
        assert!(distance > 10.0);
    }

    #[test]
    fn reverse_updates_match_a_fresh_window_recalculation() {
        let mut detector = RollingWelfordMahalanobis::new(5, 3, 1e-6, 0.05).unwrap();
        for index in 0..30 {
            detector
                .update(DVector::from_vec(vec![
                    index as f64,
                    (index as f64 * 0.3).sin(),
                    (index % 4) as f64,
                ]))
                .unwrap();
        }

        let mean = detector
            .window
            .iter()
            .fold(DVector::zeros(detector.dimensions), |sum, point| {
                sum + point
            })
            / detector.window.len() as f64;
        let recomputed_m2 = detector.window.iter().fold(
            DMatrix::zeros(detector.dimensions, detector.dimensions),
            |sum, point| {
                let difference = point - &mean;
                sum + &difference * difference.transpose()
            },
        );
        assert!((&detector.mean - mean).amax() < 1e-12);
        assert!((&detector.m2 - recomputed_m2).amax() < 1e-10);
    }
}
