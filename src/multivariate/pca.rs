use std::collections::VecDeque;

use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::model::ModelError;

#[derive(Debug, Clone)]
pub enum KernelMode {
    Linear,
    Rbf {
        gamma: f64,
        dimension: usize,
        seed: u64,
    },
    RbfAuto {
        dimension: usize,
        seed: u64,
    },
}

#[derive(Debug)]
struct RffTransform {
    omega: DMatrix<f64>,
    phase: DVector<f64>,
    gamma: f64,
}

#[derive(Debug)]
struct ActivePca {
    transform: Option<RffTransform>,
    working_dim: usize,
    window: VecDeque<DVector<f64>>,
}

#[derive(Debug)]
enum DetectorState {
    Calibrating {
        raw_points: Vec<DVector<f64>>,
        rff_dimension: usize,
        seed: u64,
    },
    Active(ActivePca),
}

/// Unified rolling linear PCA and approximate RBF kernel PCA detector.
#[derive(Debug)]
pub struct UnifiedPcaDetector {
    input_dim: usize,
    window_size: usize,
    retained_components: usize,
    state: DetectorState,
}

impl UnifiedPcaDetector {
    pub fn new(
        input_dim: usize,
        retained_components: usize,
        window_size: usize,
        mode: KernelMode,
    ) -> Result<Self, ModelError> {
        let working_dim = match mode {
            KernelMode::Linear => input_dim,
            KernelMode::Rbf { dimension, .. } | KernelMode::RbfAuto { dimension, .. } => dimension,
        };
        if input_dim == 0 || window_size < 2 || working_dim < 2 {
            return Err(ModelError::new(
                "PCA dimensions and window size must be positive (working dimension/window at least 2)",
            ));
        }
        if retained_components == 0 || retained_components >= working_dim.min(window_size) {
            return Err(ModelError::new(
                "retained PCA components must be positive and less than min(window size, working dimension)",
            ));
        }

        let state = match mode {
            KernelMode::Linear => DetectorState::Active(ActivePca {
                transform: None,
                working_dim: input_dim,
                window: VecDeque::with_capacity(window_size),
            }),
            KernelMode::Rbf {
                gamma,
                dimension,
                seed,
            } => {
                if !gamma.is_finite() || gamma <= 0.0 {
                    return Err(ModelError::new("RBF gamma must be finite and positive"));
                }
                DetectorState::Active(ActivePca {
                    transform: Some(generate_rff(input_dim, dimension, gamma, seed)),
                    working_dim: dimension,
                    window: VecDeque::with_capacity(window_size),
                })
            }
            KernelMode::RbfAuto { dimension, seed } => DetectorState::Calibrating {
                raw_points: Vec::with_capacity(window_size),
                rff_dimension: dimension,
                seed,
            },
        };
        Ok(Self {
            input_dim,
            window_size,
            retained_components,
            state,
        })
    }

    pub fn tuned_gamma(&self) -> Option<f64> {
        match &self.state {
            DetectorState::Active(active) => active.transform.as_ref().map(|rff| rff.gamma),
            DetectorState::Calibrating { .. } => None,
        }
    }

    /// Scores against the preceding full PCA window, then rolls the candidate in.
    pub fn update(&mut self, raw_point: DVector<f64>) -> Result<Option<f64>, ModelError> {
        if raw_point.len() != self.input_dim {
            return Err(ModelError::new(format!(
                "PCA input has {} values; expected {}",
                raw_point.len(),
                self.input_dim
            )));
        }
        if raw_point.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::new("PCA input must be finite"));
        }

        if let DetectorState::Calibrating {
            raw_points,
            rff_dimension,
            seed,
        } = &mut self.state
        {
            raw_points.push(raw_point);
            if raw_points.len() < self.window_size {
                return Ok(None);
            }
            let gamma = median_trick_gamma(raw_points);
            let transform = generate_rff(self.input_dim, *rff_dimension, gamma, *seed);
            let mut window = VecDeque::with_capacity(self.window_size);
            for point in raw_points.drain(..) {
                window.push_back(transform_point(&point, Some(&transform), *rff_dimension));
            }
            self.state = DetectorState::Active(ActivePca {
                transform: Some(transform),
                working_dim: *rff_dimension,
                window,
            });
            return Ok(None);
        }

        let DetectorState::Active(active) = &mut self.state else {
            unreachable!()
        };
        let point = transform_point(&raw_point, active.transform.as_ref(), active.working_dim);
        let score = if active.window.len() == self.window_size {
            Some(reconstruction_error(
                &active.window,
                &point,
                active.working_dim,
                self.retained_components,
            )?)
        } else {
            None
        };
        if active.window.len() == self.window_size {
            active.window.pop_front();
        }
        active.window.push_back(point);
        Ok(score)
    }
}

fn generate_rff(input_dim: usize, rff_dim: usize, gamma: f64, seed: u64) -> RffTransform {
    let mut random = StdRng::seed_from_u64(seed);
    let standard_deviation = (2.0 * gamma).sqrt();
    let omega = DMatrix::from_fn(rff_dim, input_dim, |_, _| {
        let u1 = random.gen_range(f64::MIN_POSITIVE..1.0_f64);
        let u2 = random.gen_range(0.0..1.0_f64);
        let standard_normal = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        standard_normal * standard_deviation
    });
    let phase = DVector::from_fn(rff_dim, |_, _| random.gen_range(0.0..std::f64::consts::TAU));
    RffTransform {
        omega,
        phase,
        gamma,
    }
}

fn transform_point(
    point: &DVector<f64>,
    transform: Option<&RffTransform>,
    working_dim: usize,
) -> DVector<f64> {
    let Some(transform) = transform else {
        return point.clone();
    };
    let projection = &transform.omega * point + &transform.phase;
    let scale = (2.0 / working_dim as f64).sqrt();
    projection.map(|value| value.cos() * scale)
}

fn median_trick_gamma(points: &[DVector<f64>]) -> f64 {
    let mut squared_distances = Vec::with_capacity(points.len() * (points.len() - 1) / 2);
    for left in 0..points.len() {
        for right in (left + 1)..points.len() {
            squared_distances.push((&points[left] - &points[right]).norm_squared());
        }
    }
    squared_distances.sort_unstable_by(f64::total_cmp);
    let middle = squared_distances.len() / 2;
    let median = if squared_distances.len().is_multiple_of(2) {
        (squared_distances[middle - 1] + squared_distances[middle]) * 0.5
    } else {
        squared_distances[middle]
    };
    if median <= 1e-12 { 1.0 } else { 1.0 / median }
}

fn reconstruction_error(
    window: &VecDeque<DVector<f64>>,
    point: &DVector<f64>,
    working_dim: usize,
    retained_components: usize,
) -> Result<f64, ModelError> {
    let mut matrix = DMatrix::zeros(window.len(), working_dim);
    for (row, value) in window.iter().enumerate() {
        matrix.row_mut(row).copy_from(&value.transpose());
    }
    let mean = DVector::from_iterator(
        working_dim,
        (0..working_dim).map(|column| {
            window.iter().map(|value| value[column]).sum::<f64>() / window.len() as f64
        }),
    );
    for mut row in matrix.row_iter_mut() {
        row -= mean.transpose();
    }
    let decomposition = matrix.svd(false, true);
    let right_vectors = decomposition
        .v_t
        .ok_or_else(|| ModelError::new("PCA SVD did not produce right singular vectors"))?
        .transpose();
    let basis = right_vectors.columns(0, retained_components);
    let centered = point - mean;
    let projected = basis.transpose() * &centered;
    let reconstructed = basis * projected;
    let error = (centered - reconstructed).norm_squared();
    if !error.is_finite() {
        return Err(ModelError::new("PCA reconstruction error is non-finite"));
    }
    Ok(error.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_pca_scores_an_off_subspace_point() {
        let mut detector = UnifiedPcaDetector::new(2, 1, 4, KernelMode::Linear).unwrap();
        for value in [-2.0, -1.0, 1.0, 2.0] {
            assert_eq!(
                detector
                    .update(DVector::from_vec(vec![value, value]))
                    .unwrap(),
                None
            );
        }
        let score = detector
            .update(DVector::from_vec(vec![0.0, 5.0]))
            .unwrap()
            .unwrap();
        assert!(score > 10.0);
    }

    #[test]
    fn auto_rff_calibrates_flat_data_safely() {
        let mut detector = UnifiedPcaDetector::new(
            2,
            1,
            4,
            KernelMode::RbfAuto {
                dimension: 8,
                seed: 42,
            },
        )
        .unwrap();
        for _ in 0..4 {
            assert_eq!(
                detector.update(DVector::from_element(2, 1.0)).unwrap(),
                None
            );
        }
        assert_eq!(detector.tuned_gamma(), Some(1.0));
        assert!(
            detector
                .update(DVector::from_element(2, 1.0))
                .unwrap()
                .unwrap()
                .is_finite()
        );
    }
}
