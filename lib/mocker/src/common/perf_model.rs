// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Performance model for timing simulations in the mocker.
//!
//! This module provides two timing models:
//! 1. Polynomial: Hardcoded polynomial formulas (default, backward compatible)
//! 2. Interpolated: Grid-based interpolation from profiler data (loaded from NPZ files)

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_interp::InterpolateError;
use ndarray_interp::interp1d::{Interp1DBuilder, Linear};
use ndarray_interp::interp2d::{Bilinear, Interp2DBuilder};
use std::path::Path;
use std::sync::Arc;

/// Inputs for one replay prefill latency prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayPrefillInput<'a> {
    pub sequence_lengths: &'a [usize],
    pub prefix_lengths: &'a [usize],
}

impl<'a> ReplayPrefillInput<'a> {
    pub fn new(sequence_lengths: &'a [usize], prefix_lengths: &'a [usize]) -> Result<Self> {
        if sequence_lengths.is_empty() {
            anyhow::bail!("replay prefill input requires at least one request");
        }
        if sequence_lengths.len() != prefix_lengths.len() {
            anyhow::bail!(
                "replay prefill input length mismatch: sequence_lengths={}, prefix_lengths={}",
                sequence_lengths.len(),
                prefix_lengths.len()
            );
        }
        if let Some((index, (prefix, sequence))) = prefix_lengths
            .iter()
            .zip(sequence_lengths)
            .enumerate()
            .find(|(_, (prefix, sequence))| prefix > sequence)
        {
            anyhow::bail!(
                "replay prefill prefix length exceeds sequence length at index {index}: prefix={prefix}, sequence={sequence}"
            );
        }
        Ok(Self {
            sequence_lengths,
            prefix_lengths,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.sequence_lengths.len()
    }

    pub fn avg_sequence_length(&self) -> usize {
        average_length(self.sequence_lengths)
    }

    pub fn avg_prefix_length(&self) -> usize {
        average_length(self.prefix_lengths)
    }

    pub fn avg_effective_input_length(&self) -> usize {
        self.avg_sequence_length()
            .saturating_sub(self.avg_prefix_length())
    }
}

/// Inputs for one replay decode latency prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayDecodeInput<'a> {
    pub sequence_lengths: &'a [usize],
    pub active_kv_tokens: usize,
    pub total_kv_tokens: usize,
    /// Sequence length passed to models that predict generation latency from
    /// an input/output pair. Replay requests one newly generated token by
    /// querying an output length of two.
    pub output_length: usize,
}

impl ReplayDecodeInput<'_> {
    pub fn batch_size(&self) -> usize {
        self.sequence_lengths.len()
    }

    pub fn avg_context_length(&self) -> usize {
        average_length(self.sequence_lengths)
    }
}

fn average_length(lengths: &[usize]) -> usize {
    if lengths.is_empty() {
        return 0;
    }
    lengths.iter().sum::<usize>() / lengths.len()
}

/// Latency model used by replay schedulers.
///
/// Implementations may call a local model, cross an FFI boundary, or use any
/// other transport. Returned values are milliseconds.
pub trait ReplayLatencyModel: Send + Sync {
    fn prefill_latency_ms(&self, input: ReplayPrefillInput<'_>) -> f64;
    fn decode_latency_ms(&self, input: ReplayDecodeInput<'_>) -> f64;
}

pub(crate) fn normalize_replay_latency_ms(
    latency_ms: f64,
    minimum_ms: f64,
    phase: &'static str,
) -> f64 {
    if latency_ms.is_finite() && latency_ms >= 0.0 {
        return latency_ms.max(minimum_ms);
    }

    tracing::warn!(
        phase,
        latency_ms,
        minimum_ms,
        "Replay latency model returned an invalid latency; using the minimum"
    );
    minimum_ms
}

/// Trait to abstract over 1D interpolation for prefill timing
pub trait PrefillInterpolator: Send + Sync {
    fn interp(&self, x: f64) -> Result<f64, InterpolateError>;
}

/// Trait to abstract over 2D interpolation for decode timing
pub trait DecodeInterpolator: Send + Sync {
    fn interp(&self, x: f64, y: f64) -> Result<f64, InterpolateError>;
}

/// Callback trait for direct AIC SDK calls.
/// Implementors call the Python AIC SDK via PyO3 GIL.
pub trait AicCallback: Send + Sync {
    /// Predict prefill latency in ms.
    /// Parameters: (batch_size, effective_isl, prefix)
    fn predict_prefill(&self, batch_size: usize, effective_isl: usize, prefix: usize) -> f64;

    /// Predict decode (generation) latency in ms.
    /// Parameters: (batch_size, isl, osl)
    fn predict_decode(&self, batch_size: usize, isl: usize, osl: usize) -> f64;
}

/// Wrapper to implement PrefillInterpolator for the concrete Interp1D type
struct PrefillInterp1D {
    inner: ndarray_interp::interp1d::Interp1D<
        ndarray::OwnedRepr<f64>,
        ndarray::OwnedRepr<f64>,
        ndarray::Ix1,
        Linear,
    >,
}

impl PrefillInterpolator for PrefillInterp1D {
    fn interp(&self, x: f64) -> Result<f64, InterpolateError> {
        self.inner.interp_scalar(x)
    }
}

/// Wrapper to implement DecodeInterpolator for the concrete Interp2D type
struct DecodeInterp2D {
    inner: ndarray_interp::interp2d::Interp2D<
        ndarray::OwnedRepr<f64>,
        ndarray::OwnedRepr<f64>,
        ndarray::OwnedRepr<f64>,
        ndarray::Ix2,
        Bilinear,
    >,
}

impl DecodeInterpolator for DecodeInterp2D {
    fn interp(&self, x: f64, y: f64) -> Result<f64, InterpolateError> {
        self.inner.interp_scalar(x, y)
    }
}

/// Performance model for predicting prefill and decode timing
#[derive(Default)]
pub enum PerfModel {
    /// Default polynomial-based model using hardcoded formulas
    #[default]
    Polynomial,
    /// Interpolation-based model using profiler data
    /// Decode axes: (active_kv_tokens, context_length)
    Interpolated {
        prefill_interp: Arc<dyn PrefillInterpolator>,
        decode_interp: Arc<dyn DecodeInterpolator>,
    },
    /// AI Configurator SDK calls via Python callback.
    /// Passes the reduced prefill inputs (batch_size, effective_isl, prefix).
    Aiconfigurator { callback: Arc<dyn AicCallback> },
}

impl Clone for PerfModel {
    fn clone(&self) -> Self {
        match self {
            PerfModel::Polynomial => PerfModel::Polynomial,
            PerfModel::Interpolated {
                prefill_interp,
                decode_interp,
            } => PerfModel::Interpolated {
                prefill_interp: Arc::clone(prefill_interp),
                decode_interp: Arc::clone(decode_interp),
            },
            PerfModel::Aiconfigurator { callback } => PerfModel::Aiconfigurator {
                callback: Arc::clone(callback),
            },
        }
    }
}

impl std::fmt::Debug for PerfModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PerfModel::Polynomial => write!(f, "PerfModel::Polynomial"),
            PerfModel::Interpolated { .. } => write!(f, "PerfModel::Interpolated {{ .. }}"),
            PerfModel::Aiconfigurator { .. } => write!(f, "PerfModel::Aiconfigurator"),
        }
    }
}

impl PerfModel {
    /// Load performance model from NPZ file
    ///
    /// Expected arrays in NPZ file:
    /// - prefill_isl: 1D array of input sequence lengths
    /// - prefill_ttft_ms: 1D array of time to first token in milliseconds
    /// - decode_active_kv_tokens: 1D array of active KV token counts
    /// - decode_context_length: 1D array of context lengths
    /// - decode_itl: 2D array of inter-token latencies in milliseconds
    pub fn from_npz(path: &Path) -> Result<Self> {
        use ndarray_npy::NpzReader;
        use std::fs::File;

        tracing::info!("Loading performance model from NPZ file: {:?}", path);

        let file =
            File::open(path).with_context(|| format!("Failed to open NPZ file: {:?}", path))?;

        let mut npz = NpzReader::new(file)
            .with_context(|| format!("Failed to create NPZ reader for: {:?}", path))?;

        // Load prefill arrays
        let prefill_isl: Array1<f64> = npz
            .by_name("prefill_isl")
            .with_context(|| "Failed to load prefill_isl from NPZ")?;
        let prefill_ttft_ms: Array1<f64> = npz
            .by_name("prefill_ttft_ms")
            .with_context(|| "Failed to load prefill_ttft_ms from NPZ")?;

        // Load decode arrays
        let decode_active_kv_tokens: Array1<f64> = npz
            .by_name("decode_active_kv_tokens")
            .with_context(|| "Failed to load decode_active_kv_tokens from NPZ")?;
        let decode_context_length: Array1<f64> = npz
            .by_name("decode_context_length")
            .with_context(|| "Failed to load decode_context_length from NPZ")?;
        let decode_itl: Array2<f64> = npz
            .by_name("decode_itl")
            .with_context(|| "Failed to load decode_itl from NPZ")?;

        // Validate dimensions
        if prefill_isl.len() != prefill_ttft_ms.len() {
            anyhow::bail!(
                "Prefill array length mismatch: isl={}, ttft={}",
                prefill_isl.len(),
                prefill_ttft_ms.len()
            );
        }

        if decode_itl.nrows() != decode_active_kv_tokens.len()
            || decode_itl.ncols() != decode_context_length.len()
        {
            anyhow::bail!(
                "Decode array dimension mismatch: itl shape=({}, {}), active_kv={}, context={}",
                decode_itl.nrows(),
                decode_itl.ncols(),
                decode_active_kv_tokens.len(),
                decode_context_length.len()
            );
        }

        tracing::info!(
            "Loaded performance model: prefill_points={}, decode_grid={}x{}",
            prefill_isl.len(),
            decode_itl.nrows(),
            decode_itl.ncols()
        );

        // Build interpolators once during loading
        let prefill_interp = Interp1DBuilder::new(prefill_ttft_ms)
            .x(prefill_isl)
            .strategy(Linear::new().extrapolate(true))
            .build()
            .with_context(|| "Failed to build prefill interpolator")?;

        let decode_interp = Interp2DBuilder::new(decode_itl)
            .x(decode_active_kv_tokens)
            .y(decode_context_length)
            .strategy(Bilinear::new().extrapolate(true))
            .build()
            .with_context(|| "Failed to build decode interpolator")?;

        Ok(PerfModel::Interpolated {
            prefill_interp: Arc::new(PrefillInterp1D {
                inner: prefill_interp,
            }),
            decode_interp: Arc::new(DecodeInterp2D {
                inner: decode_interp,
            }),
        })
    }

    /// Create an Aiconfigurator perf model from a callback.
    pub fn from_aic_callback(callback: Arc<dyn AicCallback>) -> Self {
        PerfModel::Aiconfigurator { callback }
    }

    /// Predict prefill time in milliseconds.
    pub fn predict_prefill_time(&self, batch_size: usize, isl: usize, prefix: usize) -> f64 {
        self.predict_prefill_aggregates(batch_size, isl.saturating_sub(prefix), prefix)
    }

    /// Predict decode time in milliseconds.
    pub fn predict_decode_time(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        context_length: usize,
        total_kv_tokens: usize,
    ) -> f64 {
        self.predict_decode_aggregates(
            batch_size,
            active_kv_tokens,
            context_length,
            total_kv_tokens,
            2,
        )
    }

    fn predict_prefill_aggregates(
        &self,
        batch_size: usize,
        avg_effective_input_length: usize,
        avg_prefix_length: usize,
    ) -> f64 {
        let time = match self {
            PerfModel::Polynomial => {
                let tokens = (batch_size * avg_effective_input_length) as f64;
                4.209989e-07 * tokens.powi(2) + 1.518344e-02 * tokens + 1.650142e+01
            }
            PerfModel::Interpolated { prefill_interp, .. } => {
                let tokens = (batch_size * avg_effective_input_length) as f64;
                prefill_interp.interp(tokens).unwrap_or(0.0)
            }
            PerfModel::Aiconfigurator { callback } => {
                callback.predict_prefill(batch_size, avg_effective_input_length, avg_prefix_length)
            }
        };
        time.max(0.0)
    }

    fn predict_decode_aggregates(
        &self,
        batch_size: usize,
        active_kv_tokens: usize,
        avg_context_length: usize,
        total_kv_tokens: usize,
        output_length: usize,
    ) -> f64 {
        if batch_size == 0 {
            return 0.0;
        }
        let time = match self {
            PerfModel::Polynomial => {
                let active_perc = if total_kv_tokens > 0 {
                    active_kv_tokens as f64 / total_kv_tokens as f64
                } else {
                    tracing::warn!("Total KV tokens is 0, using 1.0 as capacity");
                    1.0
                };
                -25.74 * active_perc.powi(2) + 54.01 * active_perc + 5.74
            }
            PerfModel::Interpolated { decode_interp, .. } => decode_interp
                .interp(active_kv_tokens as f64, avg_context_length as f64)
                .unwrap_or(0.0),
            PerfModel::Aiconfigurator { callback } => {
                callback.predict_decode(batch_size, avg_context_length, output_length)
            }
        };
        let result = time.max(1.0);
        tracing::trace!(
            batch_size,
            active_kv_tokens,
            avg_context_length,
            time_ms = result,
            "Decode time prediction"
        );
        result
    }
}

impl ReplayLatencyModel for PerfModel {
    fn prefill_latency_ms(&self, input: ReplayPrefillInput<'_>) -> f64 {
        self.predict_prefill_aggregates(
            input.batch_size(),
            input.avg_effective_input_length(),
            input.avg_prefix_length(),
        )
    }

    fn decode_latency_ms(&self, input: ReplayDecodeInput<'_>) -> f64 {
        self.predict_decode_aggregates(
            input.batch_size(),
            input.active_kv_tokens,
            input.avg_context_length(),
            input.total_kv_tokens,
            input.output_length,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingAicCallback {
        prefill_calls: Mutex<Vec<(usize, usize, usize)>>,
        decode_calls: Mutex<Vec<(usize, usize, usize)>>,
    }

    impl AicCallback for RecordingAicCallback {
        fn predict_prefill(&self, batch_size: usize, effective_isl: usize, prefix: usize) -> f64 {
            self.prefill_calls
                .lock()
                .unwrap()
                .push((batch_size, effective_isl, prefix));
            2.0
        }

        fn predict_decode(&self, batch_size: usize, isl: usize, osl: usize) -> f64 {
            self.decode_calls
                .lock()
                .unwrap()
                .push((batch_size, isl, osl));
            1.0
        }
    }

    #[test]
    fn replay_prefill_input_validates_request_shapes() {
        assert!(ReplayPrefillInput::new(&[], &[]).is_err());
        assert!(ReplayPrefillInput::new(&[8, 12], &[0]).is_err());
        assert!(ReplayPrefillInput::new(&[8, 12], &[0, 13]).is_err());
    }

    #[test]
    fn replay_inputs_derive_legacy_averages_from_exact_lengths() {
        let prefill = ReplayPrefillInput::new(&[8, 13], &[4, 5]).unwrap();
        assert_eq!(prefill.batch_size(), 2);
        assert_eq!(prefill.avg_sequence_length(), 10);
        assert_eq!(prefill.avg_prefix_length(), 4);
        assert_eq!(prefill.avg_effective_input_length(), 6);

        let decode = ReplayDecodeInput {
            sequence_lengths: &[9, 14],
            active_kv_tokens: 23,
            total_kv_tokens: 128,
            output_length: 2,
        };
        assert_eq!(decode.batch_size(), 2);
        assert_eq!(decode.avg_context_length(), 11);
    }

    #[test]
    fn aic_model_derives_legacy_aggregates_from_exact_lengths() {
        let callback = Arc::new(RecordingAicCallback::default());
        let model = PerfModel::from_aic_callback(callback.clone());

        model.prefill_latency_ms(ReplayPrefillInput::new(&[8, 13], &[4, 5]).unwrap());
        model.decode_latency_ms(ReplayDecodeInput {
            sequence_lengths: &[9, 14],
            active_kv_tokens: 23,
            total_kv_tokens: 128,
            output_length: 2,
        });

        assert_eq!(*callback.prefill_calls.lock().unwrap(), vec![(2, 6, 4)]);
        assert_eq!(*callback.decode_calls.lock().unwrap(), vec![(2, 11, 2)]);
    }
}
