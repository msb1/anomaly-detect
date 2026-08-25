//! Gated Self-Training Autoencoder (GSTA).
//!
//! The streaming input is a channel-major tensor `[1, channels, window_size]`.
//! Two same-padded Conv1d layers extract local temporal features, a lightweight
//! multi-head graph-attention layer treats latent channels as graph nodes, and
//! two transposed convolutions reconstruct the input window. The raw mean
//! squared reconstruction error is returned to the common multivariate score
//! pipeline, where MAD or Z-score produces the final anomaly result.
//!
//! Self-training is deliberately gated. The first `warmup_steps` aligned
//! windows train unconditionally so a random network cannot deadlock behind
//! its own gate. After warmup, a window updates the weights only when its
//! pre-update reconstruction error is no greater than `gate_threshold`.
//! Suspected anomalies are still scored, but cannot teach the network to
//! reproduce the abnormal pattern.
//!
//! # Theory of operation
//!
//! A normal multichannel process is assumed to occupy a learnable manifold in
//! channel-by-time space. The encoder maps the observed window onto that
//! manifold and the decoder reconstructs it. Windows that violate learned
//! temporal or cross-channel relationships should reconstruct poorly, making
//! mean squared error a useful scalar novelty signal. GSTA does not use that
//! raw MSE as the final decision: the shared score pipeline normalizes recent
//! reconstruction errors with the configured MAD or Z-score detector.
//!
//! Conv1d is oriented along time, so each configured Kafka input is a channel
//! and no temporal flattening occurs before local feature extraction. The GAT
//! then views the latent channels as fully connected nodes whose feature vector
//! is the whole temporal footprint. Multi-head attention lets different heads
//! emphasize different inter-channel relationships. Decoder transposed
//! convolutions restore the channel window. The final decoder activation is
//! linear because the service commonly supplies signed, standard-scaled data.
//!
//! # Streaming and gating
//!
//! Inputs must share a configured cadence. Runtime alignment additionally
//! checks every temporal position against `max_time_skew_ms`; no zero padding
//! or last-value carry-forward is used. An inference pass measures the error
//! against the weights *before* any update. During warmup that window trains
//! unconditionally and produces no result. Thereafter, an error above the gate
//! skips the complete autodiff/backward/Adam path. An error at or below the gate
//! runs a separate training pass. Thus abnormal windows remain visible to the
//! final detector but cannot immediately move the learned normal manifold.
//!
//! # Why Burn and Rust
//!
//! Burn keeps tensor shape and backend selection explicit in the type system,
//! supplies WGPU portability plus an autodiff decorator, and returns gradients
//! as owned values rather than hiding them in global tensor state. Rust owns the
//! model and optimizer inside the same mutex-protected streaming state, making
//! concurrent mutation impossible without that lock. When the gate closes,
//! the non-autodiff view avoids constructing a backward graph; ownership then
//! releases temporary tensors deterministically at the end of the update.
//!
//! The weights and Adam moments are currently process-local. Restart or an
//! interpolation-gap reset creates a seeded fresh model and repeats warmup.
//! `gate_threshold`, final score threshold, window sizes, and warmup duration
//! must be calibrated on representative normal and abnormal telemetry.

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};

use burn::backend::{Autodiff, Wgpu};
use burn::module::{AutodiffModule, Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig1d};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{leaky_relu, relu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, ElementConversion, Tensor, TensorData};

use crate::model::{ModelError, ModelInput};

type InferenceBackend = Wgpu<f32, i32>;
type TrainingBackend = Autodiff<InferenceBackend>;
type GstaModel = GatedAutoencoder<TrainingBackend>;
type GstaOptimizer = OptimizerAdaptor<Adam, GstaModel, TrainingBackend>;
type GstaDevice = <TrainingBackend as Backend>::Device;

/// Builds a channel-major temporal matrix only when every channel contains a
/// complete, positionally aligned window. Configuration requires equal source
/// cadences; this runtime check protects against missing or late observations
/// when interpolation is disabled.
pub(super) fn aligned_channel_window(
    input: &ModelInput<'_>,
    inputs: &[String],
    window_size: usize,
    max_time_skew_ms: i64,
) -> Result<Option<Vec<f32>>, ModelError> {
    let mut channels = Vec::with_capacity(inputs.len());
    for name in inputs {
        let samples = input.streams.get(name.as_str()).ok_or_else(|| {
            ModelError::new(format!("GSTA did not receive input stream '{name}'"))
        })?;
        if samples.len() < window_size {
            return Ok(None);
        }
        channels.push(
            samples
                .iter()
                .skip(samples.len() - window_size)
                .copied()
                .collect::<Vec<_>>(),
        );
    }

    for position in 0..window_size {
        let mut minimum = i64::MAX;
        let mut maximum = i64::MIN;
        for channel in &channels {
            minimum = minimum.min(channel[position].timestamp_ms);
            maximum = maximum.max(channel[position].timestamp_ms);
        }
        if maximum.saturating_sub(minimum) > max_time_skew_ms {
            return Ok(None);
        }
    }

    let mut values = Vec::with_capacity(inputs.len().saturating_mul(window_size));
    for channel in channels {
        for sample in channel {
            let value = sample.value as f32;
            if !value.is_finite() {
                return Err(ModelError::new(
                    "GSTA input cannot be represented as a finite f32 value",
                ));
            }
            values.push(value);
        }
    }
    Ok(Some(values))
}

/// Multi-head attention across latent channels.
///
/// Standard GAT concatenation materializes `[B, H, N, N, 2D]`. The additive
/// source/target form below is algebraically equivalent for a linear attention
/// vector and avoids that large temporary allocation.
#[derive(Module, Debug)]
pub struct ChannelGat<B: Backend> {
    projection: Linear<B>,
    attention_source: Param<Tensor<B, 3>>,
    attention_target: Param<Tensor<B, 3>>,
    num_heads: usize,
    head_dim: usize,
}

impl<B: Backend> ChannelGat<B> {
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_heads: usize,
        device: &B::Device,
    ) -> Self {
        debug_assert!(num_heads > 0 && out_features.is_multiple_of(num_heads));
        let head_dim = out_features / num_heads;
        let deviation = 1.0 / (head_dim as f64).sqrt();
        Self {
            projection: LinearConfig::new(in_features, out_features)
                .with_bias(false)
                .init(device),
            attention_source: Param::from_tensor(Tensor::random(
                [1, num_heads, head_dim],
                Distribution::Normal(0.0, deviation),
                device,
            )),
            attention_target: Param::from_tensor(Tensor::random(
                [1, num_heads, head_dim],
                Distribution::Normal(0.0, deviation),
                device,
            )),
            num_heads,
            head_dim,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, nodes, _features] = input.dims();
        let projected =
            self.projection
                .forward(input)
                .reshape([batch, nodes, self.num_heads, self.head_dim]);
        let node_features = projected.swap_dims(1, 2); // [B, H, N, D]

        let source =
            (node_features.clone() * self.attention_source.val().unsqueeze_dim::<4>(2)).sum_dim(3); // [B, H, N, 1]
        let target = (node_features.clone() * self.attention_target.val().unsqueeze_dim::<4>(2))
            .sum_dim(3)
            .swap_dims(2, 3); // [B, H, 1, N]
        let probabilities = softmax(leaky_relu(source + target, 0.2), 3);
        probabilities
            .matmul(node_features)
            .swap_dims(1, 2)
            .reshape([batch, nodes, self.num_heads * self.head_dim])
    }
}

/// Convolutional graph-attention autoencoder used by the streaming detector.
#[derive(Module, Debug)]
pub struct GatedAutoencoder<B: Backend> {
    encoder_conv1: Conv1d<B>,
    encoder_conv2: Conv1d<B>,
    gat: ChannelGat<B>,
    decoder_conv1: ConvTranspose1d<B>,
    decoder_conv2: ConvTranspose1d<B>,
}

impl<B: Backend> GatedAutoencoder<B> {
    pub fn new(
        num_parameters: usize,
        window_size: usize,
        latent_channels: usize,
        attention_heads: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            encoder_conv1: Conv1dConfig::new(num_parameters, 64, 3)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            encoder_conv2: Conv1dConfig::new(64, latent_channels, 3)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            gat: ChannelGat::new(window_size, window_size, attention_heads, device),
            decoder_conv1: ConvTranspose1dConfig::new([latent_channels, 64], 3)
                .with_padding(1)
                .init(device),
            decoder_conv2: ConvTranspose1dConfig::new([64, num_parameters], 3)
                .with_padding(1)
                .init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let encoded = relu(self.encoder_conv1.forward(input));
        let latent = relu(self.encoder_conv2.forward(encoded));
        let attended = self.gat.forward(latent);
        let decoded = relu(self.decoder_conv1.forward(attended));
        // Keep the final layer linear: standard-scaled telemetry is signed.
        self.decoder_conv2.forward(decoded)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct GstaUpdate {
    pub reconstruction_error: f64,
    pub trained: bool,
    pub warming_up: bool,
    pub warmup_remaining: usize,
}

/// Mutable WGPU/autodiff state. The parent multivariate model keeps this value
/// behind its existing `Arc<std::sync::Mutex<_>>`, serializing inference and
/// optimizer mutation without holding a lock across an await point.
pub(super) struct GstaDetector {
    channels: usize,
    window_size: usize,
    latent_channels: usize,
    attention_heads: usize,
    gate_threshold: f64,
    learning_rate: f64,
    warmup_steps: usize,
    warmup_completed: usize,
    seed: u64,
    device: GstaDevice,
    model: Option<GstaModel>,
    optimizer: GstaOptimizer,
}

impl fmt::Debug for GstaDetector {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GstaDetector")
            .field("channels", &self.channels)
            .field("window_size", &self.window_size)
            .field("latent_channels", &self.latent_channels)
            .field("attention_heads", &self.attention_heads)
            .field("gate_threshold", &self.gate_threshold)
            .field("learning_rate", &self.learning_rate)
            .field("warmup_steps", &self.warmup_steps)
            .field("warmup_completed", &self.warmup_completed)
            .field("seed", &self.seed)
            .finish_non_exhaustive()
    }
}

impl GstaDetector {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        channels: usize,
        window_size: usize,
        latent_channels: usize,
        attention_heads: usize,
        gate_threshold: f64,
        learning_rate: f64,
        warmup_steps: usize,
        seed: u64,
    ) -> Result<Self, ModelError> {
        catch_unwind(|| {
            Self::initialize(
                channels,
                window_size,
                latent_channels,
                attention_heads,
                gate_threshold,
                learning_rate,
                warmup_steps,
                seed,
            )
        })
        .map_err(|panic| {
            ModelError::new(format!(
                "could not initialize the GSTA WGPU backend: {}",
                panic_message(panic)
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn initialize(
        channels: usize,
        window_size: usize,
        latent_channels: usize,
        attention_heads: usize,
        gate_threshold: f64,
        learning_rate: f64,
        warmup_steps: usize,
        seed: u64,
    ) -> Self {
        TrainingBackend::seed(seed);
        let device = GstaDevice::default();
        let model = GatedAutoencoder::new(
            channels,
            window_size,
            latent_channels,
            attention_heads,
            &device,
        );
        Self {
            channels,
            window_size,
            latent_channels,
            attention_heads,
            gate_threshold,
            learning_rate,
            warmup_steps,
            warmup_completed: 0,
            seed,
            device,
            model: Some(model),
            optimizer: AdamConfig::new().init(),
        }
    }

    pub(super) fn update(&mut self, values: Vec<f32>) -> Result<GstaUpdate, ModelError> {
        catch_unwind(AssertUnwindSafe(|| self.update_inner(values))).map_err(|panic| {
            ModelError::new(format!(
                "GSTA WGPU execution failed: {}",
                panic_message(panic)
            ))
        })?
    }

    fn update_inner(&mut self, values: Vec<f32>) -> Result<GstaUpdate, ModelError> {
        let expected = self.channels.saturating_mul(self.window_size);
        if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::new(format!(
                "GSTA expected {expected} finite channel-window values, received {}",
                values.len()
            )));
        }
        let data = TensorData::new(values, [1, self.channels, self.window_size]);
        let input = Tensor::<TrainingBackend, 3>::from_data(data, &self.device);
        let model = self
            .model
            .as_ref()
            .expect("GSTA model is restored after every optimizer step");

        // Evaluate with the non-autodiff view so a gated window does not build
        // or retain a backward graph.
        let inference_input = input.clone().inner();
        let reconstruction = model.valid().forward(inference_input.clone());
        let loss = (reconstruction - inference_input)
            .powf_scalar(2.0)
            .mean()
            .into_scalar()
            .elem::<f32>() as f64;
        if !loss.is_finite() {
            return Err(ModelError::new(
                "GSTA produced a non-finite reconstruction error",
            ));
        }

        let (warming_up, trained) = training_decision(
            self.warmup_completed,
            self.warmup_steps,
            loss,
            self.gate_threshold,
        );
        if trained {
            self.train(input);
        }
        if warming_up {
            self.warmup_completed += 1;
        }

        Ok(GstaUpdate {
            reconstruction_error: loss,
            trained,
            warming_up,
            warmup_remaining: self.warmup_steps.saturating_sub(self.warmup_completed),
        })
    }

    fn train(&mut self, input: Tensor<TrainingBackend, 3>) {
        let model = self
            .model
            .take()
            .expect("GSTA model is restored after every optimizer step");
        let reconstruction = model.forward(input.clone());
        let loss = (reconstruction - input).powf_scalar(2.0).mean();
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        self.model = Some(self.optimizer.step(self.learning_rate, model, gradients));
    }
}

fn training_decision(
    warmup_completed: usize,
    warmup_steps: usize,
    reconstruction_error: f64,
    gate_threshold: f64,
) -> (bool, bool) {
    let warming_up = warmup_completed < warmup_steps;
    (
        warming_up,
        warming_up || reconstruction_error <= gate_threshold,
    )
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).into()
    } else {
        "unknown WGPU failure".into()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, VecDeque};

    use crate::window::Sample;

    use super::*;

    fn samples(timestamps: &[i64], offset: f64) -> VecDeque<Sample> {
        timestamps
            .iter()
            .map(|timestamp_ms| Sample {
                timestamp_ms: *timestamp_ms,
                value: *timestamp_ms as f64 + offset,
            })
            .collect()
    }

    #[test]
    fn aligns_complete_channel_major_windows_with_bounded_skew() {
        let left = samples(&[0, 10, 20, 30], 0.0);
        let right = samples(&[1, 11, 21, 31], 100.0);
        let input = ModelInput {
            streams: BTreeMap::from([("left", &left), ("right", &right)]),
        };
        let values = aligned_channel_window(&input, &["left".into(), "right".into()], 3, 1)
            .unwrap()
            .unwrap();
        assert_eq!(values, vec![10.0, 20.0, 30.0, 111.0, 121.0, 131.0]);
    }

    #[test]
    fn rejects_incomplete_or_misaligned_channel_windows() {
        let left = samples(&[0, 10, 20], 0.0);
        let short = samples(&[0, 10], 0.0);
        let input = ModelInput {
            streams: BTreeMap::from([("left", &left), ("right", &short)]),
        };
        assert!(
            aligned_channel_window(&input, &["left".into(), "right".into()], 3, 1,)
                .unwrap()
                .is_none()
        );

        let right = samples(&[0, 10, 25], 0.0);
        let input = ModelInput {
            streams: BTreeMap::from([("left", &left), ("right", &right)]),
        };
        assert!(
            aligned_channel_window(&input, &["left".into(), "right".into()], 3, 1,)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn autoencoder_preserves_channel_window_shape_on_cpu_reference_backend() {
        type CpuBackend = burn::backend::NdArray<f32>;

        let device = Default::default();
        let model = GatedAutoencoder::<CpuBackend>::new(2, 4, 2, 2, &device);
        let input = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(vec![0.0_f32, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0, -0.1], [1, 2, 4]),
            &device,
        );
        assert_eq!(model.forward(input).dims(), [1, 2, 4]);
    }

    #[test]
    fn warmup_bypasses_gate_then_threshold_controls_training() {
        assert_eq!(training_decision(0, 128, 10.0, 0.05), (true, true));
        assert_eq!(training_decision(128, 128, 0.05, 0.05), (false, true));
        assert_eq!(training_decision(128, 128, 0.051, 0.05), (false, false));
    }

    #[test]
    #[ignore = "requires an available WGPU adapter"]
    fn wgpu_gsta_preserves_shape_and_completes_a_training_step() {
        let mut detector = GstaDetector::new(2, 4, 2, 2, 0.05, 0.001, 1, 7).unwrap();
        let update = detector
            .update(vec![0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0, -0.1])
            .unwrap();
        assert!(update.reconstruction_error.is_finite());
        assert!(update.trained);
        assert!(update.warming_up);
        assert_eq!(update.warmup_remaining, 0);
    }
}
