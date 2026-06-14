// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::protocols::WorkerId;
use uuid::Uuid;

use crate::common::perf_model::{
    PerfModel, ReplayDecodeLatencyModel, ReplayPrefillInput, ReplayPrefillLatencyModel,
    normalize_replay_latency_ms,
};
use crate::common::protocols::{DirectRequest, KvEventPublishers, MockEngineArgs, WorkerType};
use crate::common::speculative::{SpeculativeDecodeSampler, normalize_conditional_accept_rates};
use crate::kv_manager::SglangKvManager;
use crate::replay::TraceCollector;

use super::config::SglangConfig;
use super::decode::{cache_materialized_prefix, simulate_decode_step_with_sampler};
use super::policy::apply_schedule_policy;
use super::prefill::get_new_batch_prefill;
use super::request::SglangRequest;
use crate::scheduler::{
    AdmissionEvent, CapturedRouterEventBuffer, EnginePassResult, MockerMetrics,
    RouterEventVisibility, accept_length_sample, build_fpm_snapshot, capture_router_event_sink,
};

pub(crate) struct SglangCore<
    P: ReplayPrefillLatencyModel = PerfModel,
    D: ReplayDecodeLatencyModel = PerfModel,
> {
    pub(super) config: SglangConfig,
    prefill_latency_model: Arc<P>,
    decode_latency_model: Arc<D>,
    dp_rank: u32,
    pub(super) waiting: VecDeque<SglangRequest>,
    pub(super) running: Vec<SglangRequest>,
    pub(super) new_token_ratio: f64,
    pub(super) kv_manager: SglangKvManager,
    speculative_sampler: Option<SpeculativeDecodeSampler>,
    kv_event_buffer: Option<CapturedRouterEventBuffer>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl SglangCore<PerfModel, PerfModel> {
    pub(crate) fn new(args: MockEngineArgs) -> Self {
        let latency_model = Arc::clone(&args.perf_model);
        Self::new_with_latency_models(args, Arc::clone(&latency_model), latency_model)
    }

    pub(crate) fn new_with_kv_capture(args: MockEngineArgs, worker_id: WorkerId) -> Self {
        let latency_model = Arc::clone(&args.perf_model);
        Self::new_with_kv_capture_and_latency_models(
            args,
            worker_id,
            Arc::clone(&latency_model),
            latency_model,
        )
    }

    pub(super) fn new_with_sink(
        args: MockEngineArgs,
        dp_rank: u32,
        kv_event_publishers: KvEventPublishers,
    ) -> Self {
        let latency_model = Arc::clone(&args.perf_model);
        Self::new_with_sink_and_latency_models(
            args,
            dp_rank,
            kv_event_publishers,
            Arc::clone(&latency_model),
            latency_model,
        )
    }
}

impl<P: ReplayPrefillLatencyModel, D: ReplayDecodeLatencyModel> SglangCore<P, D> {
    pub(crate) fn new_with_latency_models(
        args: MockEngineArgs,
        prefill_latency_model: Arc<P>,
        decode_latency_model: Arc<D>,
    ) -> Self {
        Self::new_internal(
            args,
            prefill_latency_model,
            decode_latency_model,
            0,
            0,
            None,
            KvEventPublishers::default(),
        )
    }

    pub(crate) fn new_with_worker_id_and_latency_models(
        args: MockEngineArgs,
        worker_id: WorkerId,
        prefill_latency_model: Arc<P>,
        decode_latency_model: Arc<D>,
    ) -> Self {
        Self::new_internal(
            args,
            prefill_latency_model,
            decode_latency_model,
            0,
            worker_id,
            None,
            KvEventPublishers::default(),
        )
    }

    pub(crate) fn new_with_kv_capture_and_latency_models(
        args: MockEngineArgs,
        worker_id: WorkerId,
        prefill_latency_model: Arc<P>,
        decode_latency_model: Arc<D>,
    ) -> Self {
        let (buffer, sink) = capture_router_event_sink(worker_id);
        Self::new_internal(
            args,
            prefill_latency_model,
            decode_latency_model,
            0,
            worker_id,
            Some(buffer),
            KvEventPublishers::new(Some(sink), None),
        )
    }

    pub(super) fn new_with_sink_and_latency_models(
        args: MockEngineArgs,
        dp_rank: u32,
        kv_event_publishers: KvEventPublishers,
        prefill_latency_model: Arc<P>,
        decode_latency_model: Arc<D>,
    ) -> Self {
        Self::new_internal(
            args,
            prefill_latency_model,
            decode_latency_model,
            dp_rank,
            u64::from(dp_rank),
            None,
            kv_event_publishers,
        )
    }

    fn new_internal(
        args: MockEngineArgs,
        prefill_latency_model: Arc<P>,
        decode_latency_model: Arc<D>,
        dp_rank: u32,
        worker_id: WorkerId,
        kv_event_buffer: Option<CapturedRouterEventBuffer>,
        kv_event_publishers: KvEventPublishers,
    ) -> Self {
        let args = args.normalized().expect("invalid MockEngineArgs");
        let config = SglangConfig::from_args(&args);
        let total_tokens = args.num_gpu_blocks * args.block_size;
        let speculative_sampler = args.aic_nextn.map(|nextn| {
            let rates =
                normalize_conditional_accept_rates(nextn, args.aic_nextn_accept_rates.as_deref())
                    .expect("normalized MTP acceptance rates");
            SpeculativeDecodeSampler::new(rates, args.aic_mtp_seed.wrapping_add(worker_id))
        });

        Self {
            config,
            prefill_latency_model,
            decode_latency_model,
            dp_rank,
            waiting: VecDeque::new(),
            running: Vec::new(),
            new_token_ratio: SglangConfig::from_args(&args).init_new_token_ratio,
            kv_manager: SglangKvManager::new(
                total_tokens,
                args.block_size,
                kv_event_publishers,
                dp_rank,
            ),
            speculative_sampler,
            kv_event_buffer,
        }
    }

    pub(crate) fn receive(&mut self, request: DirectRequest) -> Uuid {
        let request = SglangRequest::from(request);
        request.debug_assert_invariants(self.config.block_size);
        let uuid = request.uuid;
        self.waiting.push_back(request);
        uuid
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    pub(crate) fn num_requests(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    pub(crate) fn execute_pass(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> EnginePassResult {
        self.execute_pass_internal(Some(collector), now_ms)
    }

    pub(crate) fn execute_hidden_pass(&mut self, now_ms: f64) -> EnginePassResult {
        self.execute_pass_internal(None, now_ms)
    }

    pub(super) fn execute_pass_internal(
        &mut self,
        mut collector: Option<&mut TraceCollector>,
        now_ms: f64,
    ) -> EnginePassResult {
        apply_schedule_policy(&mut self.waiting, &self.kv_manager, &self.config);

        let admit = get_new_batch_prefill(
            &mut self.waiting,
            &mut self.kv_manager,
            &self.config,
            self.new_token_ratio,
            &self.running,
        );

        if admit.oom {
            self.new_token_ratio = self.config.init_new_token_ratio;
        }

        let scheduled_prefills = admit.scheduled_prefills;
        for scheduled in &scheduled_prefills {
            if let Some(collector) = collector.as_deref_mut() {
                collector.on_admit(scheduled.request.uuid, now_ms, scheduled.prefix_tokens);
            }
        }

        let prefill_sequence_lengths = scheduled_prefills
            .iter()
            .map(|scheduled| scheduled.request.materialized_tokens)
            .collect::<Vec<_>>();
        let prefill_prefix_lengths = scheduled_prefills
            .iter()
            .map(|scheduled| scheduled.prefix_tokens)
            .collect::<Vec<_>>();
        let prefill_time = simulate_prefill_duration(
            &prefill_sequence_lengths,
            &prefill_prefix_lengths,
            &self.config,
            self.prefill_latency_model.as_ref(),
            true,
        );

        let prefill_fpm = scheduled_prefills
            .iter()
            .map(|scheduled| {
                (
                    scheduled.prompt_len as u64,
                    scheduled.prefix_tokens as u64,
                    scheduled.tokens_computed as u64,
                )
            })
            .collect::<Vec<_>>();
        let admissions = scheduled_prefills
            .iter()
            .map(|scheduled| AdmissionEvent {
                uuid: scheduled.request.uuid,
                reused_input_tokens: scheduled.prefix_tokens,
            })
            .collect();
        for scheduled in scheduled_prefills {
            let mut request = scheduled.request;
            if request.materialized_tokens < request.current_sequence_len() {
                cache_materialized_prefix(&mut request, &mut self.kv_manager, &self.config);
                self.waiting.push_front(request);
            } else {
                self.running.push(request);
            }
        }

        // Capture scheduled decode data before the decode step modifies running.
        let scheduled_decode_lens: Vec<u64> = self
            .running
            .iter()
            .map(|req| req.current_sequence_len() as u64)
            .collect();

        let decode_start_ms = now_ms + prefill_time.as_secs_f64() * 1000.0;
        let mut decode = simulate_decode_step_with_sampler(
            &mut self.running,
            &mut self.kv_manager,
            &self.config,
            self.decode_latency_model.as_ref(),
            self.speculative_sampler.as_mut(),
            decode_start_ms,
            true,
        );

        if let Some(collector) = collector {
            for signal in &decode.output_signals {
                collector.on_token(signal.uuid, decode.end_ms);
            }
        }

        for req in decode.requests.drain(..).rev() {
            self.waiting.push_front(req);
        }

        if decode.retracted_any {
            self.new_token_ratio = self.config.init_new_token_ratio;
        }
        self.new_token_ratio = (self.new_token_ratio - self.config.new_token_ratio_decay_step)
            .max(self.config.min_new_token_ratio);

        // Build FPM snapshot now that all state has settled.
        let sglang_cache_hit_tokens = prefill_fpm
            .iter()
            .map(|(_, prefix_tokens, _)| *prefix_tokens)
            .sum::<u64>();
        let sglang_cache_total_tokens = prefill_fpm
            .iter()
            .map(|(_, prefix_tokens, tokens_computed)| prefix_tokens + tokens_computed)
            .sum::<u64>();
        let fpm = build_fpm_snapshot(
            prefill_fpm.iter().copied(),
            scheduled_decode_lens.into_iter(),
            self.waiting
                .iter()
                .filter(|req| req.output_len() == 0)
                .map(|req| req.prompt_len() as u64),
            self.waiting
                .iter()
                .filter(|req| req.output_len() > 0)
                .map(|req| req.current_sequence_len() as u64),
            (decode.end_ms - now_ms) / 1000.0,
        );

        let (accept_length_output_tokens, accept_length_decode_forwards) =
            accept_length_sample(&decode.output_signals);
        debug_assert_sglang_scheduler_state(&self.waiting, &self.running, self.config.block_size);
        let active_decode_blocks = self.active_kv_blocks();
        EnginePassResult {
            end_ms: decode.end_ms,
            completed_requests: decode
                .output_signals
                .iter()
                .filter(|signal| signal.completed)
                .count(),
            output_signals: decode.output_signals,
            admissions,
            mocker_metrics: MockerMetrics::from_parts(
                self.dp_rank,
                active_decode_blocks,
                self.config.total_kv_tokens.div_ceil(self.config.block_size) as u64,
                self.running.len() as u64,
                self.waiting.len() as u64,
                0,
                sglang_cache_hit_tokens,
                sglang_cache_total_tokens,
            ),
            router_event_visibility: RouterEventVisibility::PassEnd,
            kv_events: self
                .kv_event_buffer
                .as_ref()
                .map(CapturedRouterEventBuffer::drain)
                .unwrap_or_default(),
            fpm: Some(fpm),
            accept_length_output_tokens,
            accept_length_decode_forwards,
        }
    }

    fn active_kv_blocks(&self) -> u64 {
        let active_reserved = self
            .waiting
            .iter()
            .map(SglangRequest::extra_reserved_tokens)
            .sum::<usize>()
            + self
                .running
                .iter()
                .map(SglangRequest::extra_reserved_tokens)
                .sum::<usize>();
        let actual_used =
            self.kv_manager.cache().total_tokens() - self.kv_manager.cache().available_tokens();
        (actual_used + active_reserved).div_ceil(self.config.block_size) as u64
    }
}

fn simulate_prefill_duration<M: ReplayPrefillLatencyModel>(
    sequence_lengths: &[usize],
    prefix_lengths: &[usize],
    config: &SglangConfig,
    latency_model: &M,
    apply_speedup: bool,
) -> Duration {
    if sequence_lengths.is_empty() || config.worker_type == WorkerType::Decode {
        return Duration::ZERO;
    }

    let prefill_time = normalize_replay_latency_ms(
        latency_model.prefill_latency_ms(
            ReplayPrefillInput::new(sequence_lengths, prefix_lengths)
                .expect("SGLang prefill batch must contain valid request shapes"),
        ),
        0.0,
        "prefill",
    );
    let total_time = Duration::from_secs_f64(prefill_time / 1000.0);

    if !apply_speedup || config.speedup_ratio <= 0.0 || total_time <= Duration::ZERO {
        return total_time;
    }

    Duration::from_secs_f64(total_time.as_secs_f64() / config.speedup_ratio)
}

fn debug_assert_sglang_scheduler_state(
    _waiting: &VecDeque<SglangRequest>,
    _running: &[SglangRequest],
    _block_size: usize,
) {
    #[cfg(debug_assertions)]
    {
        let waiting = _waiting;
        let running = _running;
        let block_size = _block_size;
        let mut seen = std::collections::HashSet::new();
        for req in waiting {
            debug_assert!(
                seen.insert(req.uuid),
                "request {} appears multiple times across waiting/running queues",
                req.uuid
            );
            req.debug_assert_invariants(block_size);
        }
        for req in running {
            debug_assert!(
                seen.insert(req.uuid),
                "request {} appears multiple times across waiting/running queues",
                req.uuid
            );
            req.debug_assert_invariants(block_size);
        }
    }
}
