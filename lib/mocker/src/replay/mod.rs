// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod artifacts;
mod collector;
mod entrypoints;
pub(crate) mod offline;
mod online;
mod planner_handle;
mod router_shared;
mod validate;

use std::collections::VecDeque;
use std::sync::Arc;

pub use crate::common::perf_model::{ReplayDecodeInput, ReplayLatencyModel, ReplayPrefillInput};
use crate::common::protocols::{DirectRequest, MockEngineArgs};
use dynamo_kv_router::PrefillLoadEstimator;

pub use artifacts::{
    ReplayTimedKvEvent, ReplayTimedOutputSignal, ReplayTimedRequest, ReplayWorkerArtifacts,
};
pub(crate) use collector::TraceCollector;
#[cfg(test)]
pub(crate) use collector::TraceRequestStatsSnapshot;
pub use collector::{
    PerRequestRecord, TraceDistributionStats, TraceInterTokenLatencyStats, TraceLatencyStats,
    TraceRequestCounts, TraceSimulationReport, TraceThroughputStats,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayRouterMode {
    RoundRobin,
    KvRouter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayArgsMode {
    Aggregated,
    Disagg,
}

pub type ReplayPrefillLoadEstimator = Arc<dyn PrefillLoadEstimator>;

/// Generic offline replay runner for native latency model implementations.
#[derive(Clone)]
pub struct Replay<M: ReplayLatencyModel> {
    latency_model: Arc<M>,
}

impl<M: ReplayLatencyModel> Replay<M> {
    pub fn new(latency_model: M) -> Self {
        Self {
            latency_model: Arc::new(latency_model),
        }
    }

    pub fn from_arc(latency_model: Arc<M>) -> Self {
        Self { latency_model }
    }

    pub fn latency_model(&self) -> &M {
        self.latency_model.as_ref()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn simulate_trace_requests(
        &self,
        args: MockEngineArgs,
        router_config: Option<dynamo_kv_router::config::KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        requests: Vec<DirectRequest>,
        num_workers: usize,
        arrival_speedup_ratio: f64,
        router_mode: ReplayRouterMode,
    ) -> anyhow::Result<TraceSimulationReport> {
        entrypoints::simulate_trace_requests_with_latency_model(
            Arc::clone(&self.latency_model),
            args,
            router_config,
            prefill_load_estimator,
            requests,
            num_workers,
            arrival_speedup_ratio,
            router_mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn simulate_concurrency_requests(
        &self,
        args: MockEngineArgs,
        router_config: Option<dynamo_kv_router::config::KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        requests: Vec<DirectRequest>,
        max_in_flight: usize,
        num_workers: usize,
        router_mode: ReplayRouterMode,
    ) -> anyhow::Result<TraceSimulationReport> {
        entrypoints::simulate_concurrency_requests_with_latency_model(
            Arc::clone(&self.latency_model),
            args,
            router_config,
            prefill_load_estimator,
            requests,
            max_in_flight,
            num_workers,
            router_mode,
        )
    }
}

#[derive(Clone, Debug)]
pub struct OfflineDisaggReplayConfig {
    pub prefill_args: MockEngineArgs,
    pub decode_args: MockEngineArgs,
    pub num_prefill_workers: usize,
    pub num_decode_workers: usize,
}

impl OfflineDisaggReplayConfig {
    pub fn normalized(self) -> anyhow::Result<Self> {
        Ok(Self {
            prefill_args: self.prefill_args.normalized()?,
            decode_args: self.decode_args.normalized()?,
            num_prefill_workers: self.num_prefill_workers,
            num_decode_workers: self.num_decode_workers,
        })
    }
}

pub use entrypoints::{
    generate_trace_worker_artifacts_offline, simulate_concurrency_file,
    simulate_concurrency_file_disagg_with_router_mode,
    simulate_concurrency_file_disagg_with_router_mode_and_format,
    simulate_concurrency_file_with_router_mode,
    simulate_concurrency_file_with_router_mode_and_format, simulate_concurrency_live_file,
    simulate_concurrency_live_file_with_router_mode,
    simulate_concurrency_live_file_with_router_mode_and_format, simulate_concurrency_live_requests,
    simulate_concurrency_live_requests_with_router_mode, simulate_concurrency_live_workload,
    simulate_concurrency_live_workload_with_router_mode, simulate_concurrency_requests,
    simulate_concurrency_requests_disagg_with_router_mode,
    simulate_concurrency_requests_with_router_mode, simulate_concurrency_workload,
    simulate_concurrency_workload_disagg_with_router_mode,
    simulate_concurrency_workload_with_router_mode, simulate_trace_file,
    simulate_trace_file_disagg_with_router_mode,
    simulate_trace_file_disagg_with_router_mode_and_format, simulate_trace_file_with_router_mode,
    simulate_trace_file_with_router_mode_and_format, simulate_trace_live_file,
    simulate_trace_live_file_with_router_mode,
    simulate_trace_live_file_with_router_mode_and_format, simulate_trace_live_requests,
    simulate_trace_live_requests_with_router_mode, simulate_trace_live_workload,
    simulate_trace_live_workload_with_router_mode, simulate_trace_requests,
    simulate_trace_requests_disagg_with_router_mode, simulate_trace_requests_with_router_mode,
    simulate_trace_workload, simulate_trace_workload_disagg_with_router_mode,
    simulate_trace_workload_with_router_mode,
};
pub use planner_handle::{PlannerReplayHandle, PlannerTickData};
pub use validate::validate_replay_args_mode;

pub(crate) fn normalize_trace_requests(
    mut requests: Vec<DirectRequest>,
    arrival_speedup_ratio: f64,
) -> anyhow::Result<VecDeque<DirectRequest>> {
    if !arrival_speedup_ratio.is_finite() || arrival_speedup_ratio <= 0.0 {
        anyhow::bail!(
            "arrival_speedup_ratio must be a finite positive number, got {arrival_speedup_ratio}"
        );
    }

    requests.sort_by(|left, right| {
        let left_ts = left
            .arrival_timestamp_ms
            .expect("trace replay requests must have an arrival timestamp");
        let right_ts = right
            .arrival_timestamp_ms
            .expect("trace replay requests must have an arrival timestamp");
        left_ts.total_cmp(&right_ts)
    });

    let first_arrival_ms = requests
        .first()
        .and_then(|request| request.arrival_timestamp_ms)
        .ok_or_else(|| anyhow::anyhow!("trace replay requires at least one timestamped request"))?;

    Ok(VecDeque::from(
        requests
            .into_iter()
            .map(|mut request| {
                let arrival_timestamp_ms = request
                    .arrival_timestamp_ms
                    .expect("trace replay requests must have an arrival timestamp")
                    - first_arrival_ms;
                let arrival_timestamp_ms = arrival_timestamp_ms / arrival_speedup_ratio;
                request.arrival_timestamp_ms = Some(arrival_timestamp_ms);
                request
            })
            .collect::<Vec<_>>(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use uuid::Uuid;

    #[derive(Clone, Default)]
    struct RecordingLatencyModel {
        prefill_inputs: Arc<Mutex<Vec<ReplayPrefillInput>>>,
        decode_inputs: Arc<Mutex<Vec<ReplayDecodeInput>>>,
    }

    impl ReplayLatencyModel for RecordingLatencyModel {
        fn prefill_latency_ms(&self, input: ReplayPrefillInput) -> f64 {
            self.prefill_inputs.lock().unwrap().push(input);
            2.0
        }

        fn decode_latency_ms(&self, input: ReplayDecodeInput) -> f64 {
            self.decode_inputs.lock().unwrap().push(input);
            1.0
        }
    }

    #[test]
    fn generic_replay_uses_native_latency_model() {
        let model = RecordingLatencyModel::default();
        let replay = Replay::new(model.clone());
        let args = MockEngineArgs::builder()
            .block_size(16)
            .num_gpu_blocks(128)
            .build()
            .unwrap();
        let requests = vec![
            DirectRequest {
                tokens: vec![1; 8],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(1)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
                priority: 0,
                strict_priority: 0,
            },
            DirectRequest {
                tokens: vec![2; 12],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(2)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(0.0),
                priority: 0,
                strict_priority: 0,
            },
        ];

        let report = replay
            .simulate_trace_requests(
                args,
                None,
                None,
                requests,
                2,
                1.0,
                ReplayRouterMode::RoundRobin,
            )
            .unwrap();

        assert_eq!(report.request_counts.completed_requests, 2);
        let prefill_inputs = model.prefill_inputs.lock().unwrap();
        assert!(!prefill_inputs.is_empty());
        assert!(prefill_inputs.iter().all(|input| {
            input.input_sequence_length >= input.prefix_length
                && input.effective_input_sequence_length() > 0
        }));
        let decode_inputs = model.decode_inputs.lock().unwrap();
        assert!(!decode_inputs.is_empty());
        assert!(decode_inputs.iter().all(|input| {
            input.batch_size > 0 && input.context_length > 0 && input.output_length == 2
        }));
    }

    #[test]
    fn test_replay_itl_uses_per_token_gaps() {
        let mut collector = TraceCollector::default();
        let uuid = Uuid::from_u128(11);

        collector.on_arrival(uuid, 0.0, 4, 4);
        collector.on_admit(uuid, 0.0, 0);
        collector.on_token(uuid, 10.0);
        collector.on_token(uuid, 11.0);
        collector.on_token(uuid, 12.0);
        collector.on_token(uuid, 110.0);

        let report = collector.finish();

        assert!((report.latency.tpot.mean_ms - (100.0 / 3.0)).abs() < 1e-9);
        assert!((report.latency.itl.distribution.mean_ms - (100.0 / 3.0)).abs() < 1e-9);
        assert_eq!(report.latency.itl.distribution.median_ms, 1.0);
        assert_eq!(report.latency.itl.distribution.p75_ms, 98.0);
        assert_eq!(report.latency.itl.distribution.p90_ms, 98.0);
        assert_eq!(report.latency.itl.distribution.p95_ms, 98.0);
        assert_eq!(report.latency.itl.max_ms, 98.0);
        assert_eq!(report.latency.ttst.min_ms, 1.0);
        assert_eq!(report.latency.ttst.max_ms, 1.0);
        assert_eq!(
            report.latency.output_token_throughput_per_user.min_ms,
            1000.0 / 98.0
        );
        assert_eq!(
            report.latency.output_token_throughput_per_user.max_ms,
            1000.0
        );
    }

    #[test]
    fn test_normalize_trace_requests_applies_arrival_speedup_ratio() {
        let requests = vec![
            DirectRequest {
                tokens: vec![1; 4],
                max_output_tokens: 1,
                uuid: Some(Uuid::from_u128(1)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(100.0),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![2; 4],
                max_output_tokens: 1,
                uuid: Some(Uuid::from_u128(2)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(200.0),
                ..Default::default()
            },
        ];

        let normalized = normalize_trace_requests(requests, 10.0).unwrap();
        let arrivals = normalized
            .into_iter()
            .map(|request| request.arrival_timestamp_ms.unwrap())
            .collect::<Vec<_>>();

        assert_eq!(arrivals, vec![0.0, 10.0]);
    }
}
