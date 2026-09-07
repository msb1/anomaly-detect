use std::collections::{BTreeMap, VecDeque};

use crate::config::{ModelKind, PreprocessingConfig, StreamConfig, WindowConfig};
use crate::preprocess::Preprocessor;
use crate::preprocess::decomposition::StreamingDecomposer;
use crate::preprocess::interpolation::InterpolationResult;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub timestamp_ms: i64,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowUpdate {
    Accepted {
        interpolated_points: usize,
        window_cleared: bool,
    },
    TooLate,
}

#[derive(Debug, Clone)]
pub struct TimeWindow {
    sample_count: usize,
    watermark_ms: Option<i64>,
    samples: VecDeque<Sample>,
}

impl TimeWindow {
    pub fn new(config: &WindowConfig) -> Self {
        Self {
            sample_count: config.sample_count,
            watermark_ms: None,
            samples: VecDeque::new(),
        }
    }

    pub fn push(&mut self, sample: Sample) -> WindowUpdate {
        if self
            .watermark_ms
            .is_some_and(|watermark| sample.timestamp_ms < watermark)
        {
            return WindowUpdate::TooLate;
        }

        self.watermark_ms = Some(self.watermark_ms.map_or(sample.timestamp_ms, |current| {
            current.max(sample.timestamp_ms)
        }));

        let index = self
            .samples
            .iter()
            .position(|existing| existing.timestamp_ms > sample.timestamp_ms)
            .unwrap_or(self.samples.len());
        self.samples.insert(index, sample);

        while self.samples.len() > self.sample_count {
            self.samples.pop_front();
        }
        WindowUpdate::Accepted {
            interpolated_points: 0,
            window_cleared: false,
        }
    }

    pub fn is_primed(&self) -> bool {
        self.samples.len() >= self.sample_count
    }

    pub fn samples(&self) -> &VecDeque<Sample> {
        &self.samples
    }

    fn clear(&mut self) {
        self.watermark_ms = None;
        self.samples.clear();
    }
}

#[derive(Debug)]
pub struct WindowStore {
    windows: BTreeMap<String, StreamWindow>,
    preprocessor: Preprocessor,
}

impl WindowStore {
    pub fn new(
        streams: &BTreeMap<String, StreamConfig>,
        window: &WindowConfig,
        preprocessing: &PreprocessingConfig,
    ) -> Self {
        Self {
            windows: streams
                .keys()
                .map(|name| {
                    (
                        name.clone(),
                        StreamWindow {
                            raw: TimeWindow::new(window),
                            univariate: None,
                            multivariate: None,
                            decomposed: VecDeque::with_capacity(window.sample_count),
                            decomposer: StreamingDecomposer::new(&preprocessing.decomposition),
                            decomposition_enabled: preprocessing.decomposition.enabled(),
                            sample_count: window.sample_count,
                        },
                    )
                })
                .collect(),
            preprocessor: Preprocessor::new(preprocessing, window),
        }
    }

    pub fn push(&mut self, stream: &str, sample: Sample, interval_ms: i64) -> Option<WindowUpdate> {
        let window = self.windows.get_mut(stream)?;
        let interpolation = self.preprocessor.interpolation(
            window.raw.samples().back().copied(),
            sample,
            interval_ms,
        );
        let (samples, window_cleared) = match interpolation {
            InterpolationResult::Samples(samples) => (samples, false),
            InterpolationResult::GapLimitExceeded => {
                window.raw.clear();
                window.clear_processed();
                (vec![sample], true)
            }
        };
        let interpolated_points = samples.len().saturating_sub(1);
        for next in samples {
            if matches!(window.raw.push(next), WindowUpdate::TooLate) {
                return Some(WindowUpdate::TooLate);
            }
            if window.decomposition_enabled {
                let scaled = self
                    .preprocessor
                    .latest_scaled_value(window.raw.samples())
                    .expect("a just-updated raw window contains one value");
                let remainder = window.decomposer.update(scaled).remainder;
                if window.decomposed.len() == window.sample_count {
                    window.decomposed.pop_front();
                }
                window.decomposed.push_back(Sample {
                    timestamp_ms: next.timestamp_ms,
                    value: remainder,
                });
            }
        }
        window.multivariate = Some(
            self.preprocessor
                .transform_multivariate(window.raw.samples()),
        );
        window.univariate = Some(if window.decomposition_enabled {
            self.preprocessor.smooth_decomposed(&window.decomposed)
        } else {
            self.preprocessor
                .transform_univariate_without_decomposition(window.raw.samples())
        });
        Some(WindowUpdate::Accepted {
            interpolated_points,
            window_cleared,
        })
    }

    pub fn get(&self, stream: &str) -> Option<&StreamWindow> {
        self.windows.get(stream)
    }
}

#[derive(Debug)]
pub struct StreamWindow {
    raw: TimeWindow,
    univariate: Option<VecDeque<Sample>>,
    multivariate: Option<VecDeque<Sample>>,
    decomposed: VecDeque<Sample>,
    decomposer: StreamingDecomposer,
    decomposition_enabled: bool,
    sample_count: usize,
}

impl StreamWindow {
    pub fn is_primed(&self, kind: ModelKind) -> bool {
        self.raw.is_primed() && self.processed_samples(kind).is_some()
    }

    pub fn raw_samples(&self) -> &VecDeque<Sample> {
        self.raw.samples()
    }

    pub fn processed_samples(&self, kind: ModelKind) -> Option<&VecDeque<Sample>> {
        match kind {
            ModelKind::Univariate => self.univariate.as_ref(),
            ModelKind::Multivariate => self.multivariate.as_ref(),
        }
    }

    fn clear_processed(&mut self) {
        self.univariate = None;
        self.multivariate = None;
        self.decomposed.clear();
        self.decomposer.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrendDecompositionConfig;

    fn config() -> WindowConfig {
        WindowConfig { sample_count: 3 }
    }

    fn preprocessing() -> PreprocessingConfig {
        PreprocessingConfig::default()
    }

    #[test]
    fn primes_after_the_configured_sample_count_and_evicts_oldest_samples() {
        let mut window = TimeWindow::new(&config());
        window.push(Sample {
            timestamp_ms: 0,
            value: 1.0,
        });
        window.push(Sample {
            timestamp_ms: 1,
            value: 2.0,
        });
        assert!(!window.is_primed());
        window.push(Sample {
            timestamp_ms: 2,
            value: 3.0,
        });
        assert!(window.is_primed());
        window.push(Sample {
            timestamp_ms: 3,
            value: 4.0,
        });
        assert_eq!(window.samples().front().unwrap().timestamp_ms, 1);
    }

    #[test]
    fn rejects_out_of_order_data() {
        let mut window = TimeWindow::new(&config());
        window.push(Sample {
            timestamp_ms: 5_000,
            value: 1.0,
        });
        assert_eq!(
            window.push(Sample {
                timestamp_ms: 4_500,
                value: 2.0
            }),
            WindowUpdate::TooLate
        );
        assert_eq!(window.samples().front().unwrap().timestamp_ms, 5_000);
    }

    #[test]
    fn excessive_gap_clears_stream_and_restarts_priming() {
        let streams = BTreeMap::from([(
            "metric".into(),
            StreamConfig {
                headers: BTreeMap::new(),
            },
        )]);
        let mut preprocessing = preprocessing();
        preprocessing.interpolation.enabled = true;
        let mut store = WindowStore::new(&streams, &config(), &preprocessing);
        for timestamp_ms in (0..=10_000).step_by(1_000) {
            store.push(
                "metric",
                Sample {
                    timestamp_ms,
                    value: timestamp_ms as f64,
                },
                1_000,
            );
        }
        assert!(
            store
                .get("metric")
                .unwrap()
                .is_primed(ModelKind::Univariate)
        );
        let update = store
            .push(
                "metric",
                Sample {
                    timestamp_ms: 13_001,
                    value: 3.0,
                },
                1_000,
            )
            .unwrap();
        assert_eq!(
            update,
            WindowUpdate::Accepted {
                interpolated_points: 0,
                window_cleared: true
            }
        );
        let window = store.get("metric").unwrap();
        assert_eq!(window.raw_samples().len(), 1);
        assert!(!window.is_primed(ModelKind::Univariate));
    }

    #[test]
    fn bounded_gap_adds_synthetic_points_before_reprocessing() {
        let streams = BTreeMap::from([(
            "metric".into(),
            StreamConfig {
                headers: BTreeMap::new(),
            },
        )]);
        let mut preprocessing = preprocessing();
        preprocessing.interpolation.enabled = true;
        let mut point_config = config();
        point_config.sample_count = 10;
        let mut store = WindowStore::new(&streams, &point_config, &preprocessing);
        store.push(
            "metric",
            Sample {
                timestamp_ms: 0,
                value: 0.0,
            },
            1_000,
        );
        let update = store
            .push(
                "metric",
                Sample {
                    timestamp_ms: 3_000,
                    value: 6.0,
                },
                1_000,
            )
            .unwrap();
        assert_eq!(
            update,
            WindowUpdate::Accepted {
                interpolated_points: 2,
                window_cleared: false
            }
        );
        let window = store.get("metric").unwrap();
        assert_eq!(window.raw_samples().len(), 4);
        assert_eq!(
            window.processed_samples(ModelKind::Univariate).unwrap()[1].value,
            2.0
        );
    }

    #[test]
    fn multivariate_view_bypasses_univariate_decomposition() {
        let streams = BTreeMap::from([(
            "metric".into(),
            StreamConfig {
                headers: BTreeMap::new(),
            },
        )]);
        let mut preprocessing = preprocessing();
        preprocessing.decomposition.trend = TrendDecompositionConfig {
            enabled: true,
            alpha: 1.0,
            beta: 1.0,
        };
        let mut store = WindowStore::new(&streams, &config(), &preprocessing);
        for (timestamp_ms, value) in [(0, 1.0), (1, 2.0), (2, 3.0)] {
            store.push(
                "metric",
                Sample {
                    timestamp_ms,
                    value,
                },
                1,
            );
        }
        let window = store.get("metric").unwrap();
        let multivariate: Vec<_> = window
            .processed_samples(ModelKind::Multivariate)
            .unwrap()
            .iter()
            .map(|sample| sample.value)
            .collect();
        let univariate: Vec<_> = window
            .processed_samples(ModelKind::Univariate)
            .unwrap()
            .iter()
            .map(|sample| sample.value)
            .collect();
        assert_eq!(multivariate, [1.0, 2.0, 3.0]);
        assert_eq!(univariate, [0.0, -1.0, -1.0]);
    }
}
