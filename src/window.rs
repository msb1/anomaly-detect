use std::collections::{BTreeMap, VecDeque};

use crate::config::{PreprocessingConfig, StreamConfig, WindowConfig};
use crate::preprocess::Preprocessor;
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
    duration_ms: i64,
    minimum_samples: usize,
    max_lateness_ms: i64,
    first_observed_ms: Option<i64>,
    watermark_ms: Option<i64>,
    samples: VecDeque<Sample>,
}

impl TimeWindow {
    pub fn new(config: &WindowConfig) -> Self {
        Self {
            duration_ms: config.duration_ms,
            minimum_samples: config.minimum_samples,
            max_lateness_ms: config.max_lateness_ms,
            first_observed_ms: None,
            watermark_ms: None,
            samples: VecDeque::new(),
        }
    }

    pub fn push(&mut self, sample: Sample) -> WindowUpdate {
        if let Some(watermark) = self.watermark_ms
            && sample.timestamp_ms < watermark.saturating_sub(self.max_lateness_ms)
        {
            return WindowUpdate::TooLate;
        }

        self.first_observed_ms.get_or_insert(sample.timestamp_ms);
        self.watermark_ms = Some(self.watermark_ms.map_or(sample.timestamp_ms, |current| {
            current.max(sample.timestamp_ms)
        }));

        let index = self
            .samples
            .iter()
            .position(|existing| existing.timestamp_ms > sample.timestamp_ms)
            .unwrap_or(self.samples.len());
        self.samples.insert(index, sample);

        let cutoff = self
            .watermark_ms
            .expect("watermark was just set")
            .saturating_sub(self.duration_ms);
        while self
            .samples
            .front()
            .is_some_and(|oldest| oldest.timestamp_ms < cutoff)
        {
            self.samples.pop_front();
        }
        WindowUpdate::Accepted {
            interpolated_points: 0,
            window_cleared: false,
        }
    }

    pub fn is_primed(&self) -> bool {
        match (self.first_observed_ms, self.watermark_ms) {
            (Some(first), Some(watermark)) => {
                watermark.saturating_sub(first) >= self.duration_ms
                    && self.samples.len() >= self.minimum_samples
            }
            _ => false,
        }
    }

    pub fn samples(&self) -> &VecDeque<Sample> {
        &self.samples
    }

    fn clear(&mut self) {
        self.first_observed_ms = None;
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
                .iter()
                .map(|(name, stream)| {
                    (
                        name.clone(),
                        StreamWindow {
                            interval_ms: stream.interval_ms,
                            raw: TimeWindow::new(window),
                            processed: None,
                        },
                    )
                })
                .collect(),
            preprocessor: Preprocessor::new(preprocessing, window),
        }
    }

    pub fn push(&mut self, stream: &str, sample: Sample) -> Option<WindowUpdate> {
        let window = self.windows.get_mut(stream)?;
        let interpolation = self.preprocessor.interpolation(
            window.raw.samples().back().copied(),
            sample,
            window.interval_ms,
        );
        let (samples, window_cleared) = match interpolation {
            InterpolationResult::Samples(samples) => (samples, false),
            InterpolationResult::GapLimitExceeded => {
                window.raw.clear();
                window.processed = None;
                (vec![sample], true)
            }
        };
        let interpolated_points = samples.len().saturating_sub(1);
        for next in samples {
            if matches!(window.raw.push(next), WindowUpdate::TooLate) {
                return Some(WindowUpdate::TooLate);
            }
        }
        window.processed = self
            .preprocessor
            .transform(window.raw.samples(), window.interval_ms);
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
    interval_ms: i64,
    raw: TimeWindow,
    processed: Option<VecDeque<Sample>>,
}

impl StreamWindow {
    pub fn is_primed(&self) -> bool {
        self.raw.is_primed() && self.processed.is_some()
    }

    pub fn raw_samples(&self) -> &VecDeque<Sample> {
        self.raw.samples()
    }

    pub fn processed_samples(&self) -> Option<&VecDeque<Sample>> {
        self.processed.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> WindowConfig {
        WindowConfig {
            duration_ms: 10_000,
            minimum_samples: 2,
            max_lateness_ms: 1_000,
        }
    }

    fn preprocessing() -> PreprocessingConfig {
        PreprocessingConfig::default()
    }

    #[test]
    fn primes_only_after_full_time_span_and_evicts_expired_samples() {
        let mut window = TimeWindow::new(&config());
        window.push(Sample {
            timestamp_ms: 0,
            value: 1.0,
        });
        window.push(Sample {
            timestamp_ms: 9_999,
            value: 2.0,
        });
        assert!(!window.is_primed());
        window.push(Sample {
            timestamp_ms: 10_000,
            value: 3.0,
        });
        assert!(window.is_primed());
        window.push(Sample {
            timestamp_ms: 12_000,
            value: 4.0,
        });
        assert_eq!(window.samples().front().unwrap().timestamp_ms, 9_999);
    }

    #[test]
    fn sorts_allowed_late_data_and_rejects_data_behind_lateness_bound() {
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
            WindowUpdate::Accepted {
                interpolated_points: 0,
                window_cleared: false,
            }
        );
        assert_eq!(
            window.push(Sample {
                timestamp_ms: 3_999,
                value: 3.0
            }),
            WindowUpdate::TooLate
        );
        assert_eq!(window.samples().front().unwrap().timestamp_ms, 4_500);
    }

    #[test]
    fn excessive_gap_clears_stream_and_restarts_priming() {
        let streams = BTreeMap::from([(
            "metric".into(),
            StreamConfig {
                headers: BTreeMap::new(),
                interval_ms: 1_000,
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
            );
        }
        assert!(store.get("metric").unwrap().is_primed());
        let update = store
            .push(
                "metric",
                Sample {
                    timestamp_ms: 13_001,
                    value: 3.0,
                },
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
        assert!(!window.is_primed());
    }

    #[test]
    fn bounded_gap_adds_synthetic_points_before_reprocessing() {
        let streams = BTreeMap::from([(
            "metric".into(),
            StreamConfig {
                headers: BTreeMap::new(),
                interval_ms: 1_000,
            },
        )]);
        let mut preprocessing = preprocessing();
        preprocessing.interpolation.enabled = true;
        let mut store = WindowStore::new(&streams, &config(), &preprocessing);
        store.push(
            "metric",
            Sample {
                timestamp_ms: 0,
                value: 0.0,
            },
        );
        let update = store
            .push(
                "metric",
                Sample {
                    timestamp_ms: 3_000,
                    value: 6.0,
                },
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
        assert_eq!(window.processed_samples().unwrap()[1].value, 2.0);
    }
}
