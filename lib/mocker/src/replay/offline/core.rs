// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::common::perf_model::{PerfModel, ReplayLatencyModel};
use crate::common::protocols::MockEngineArgs;
use crate::replay::TraceCollector;
use crate::scheduler::{EngineCore, EnginePassResult, SglangCore, VllmCore};
use dynamo_kv_router::protocols::WorkerId;

pub(crate) struct ReplayWorkerCore<M: ReplayLatencyModel = PerfModel> {
    core: EngineCore<M>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl ReplayWorkerCore<PerfModel> {
    pub(crate) fn new(args: MockEngineArgs) -> Self {
        let latency_model = Arc::clone(&args.perf_model);
        Self::new_with_latency_model(args, latency_model)
    }

    pub(crate) fn new_with_kv_capture(args: MockEngineArgs, worker_id: WorkerId) -> Self {
        let latency_model = Arc::clone(&args.perf_model);
        Self::new_with_kv_capture_and_latency_model(args, worker_id, latency_model)
    }
}

impl<M: ReplayLatencyModel> ReplayWorkerCore<M> {
    pub(crate) fn new_with_latency_model(args: MockEngineArgs, latency_model: Arc<M>) -> Self {
        let core = match args.engine_type {
            crate::common::protocols::EngineType::Vllm
            | crate::common::protocols::EngineType::Trtllm => {
                let mut core = VllmCore::new_with_latency_model(args, latency_model);
                Self::init_offload_vllm(&mut core);
                EngineCore::Vllm(core)
            }
            crate::common::protocols::EngineType::Sglang => {
                EngineCore::Sglang(SglangCore::new_with_latency_model(args, latency_model))
            }
        };
        Self { core }
    }

    pub(crate) fn new_with_kv_capture_and_latency_model(
        args: MockEngineArgs,
        worker_id: WorkerId,
        latency_model: Arc<M>,
    ) -> Self {
        let core = match args.engine_type {
            crate::common::protocols::EngineType::Vllm
            | crate::common::protocols::EngineType::Trtllm => {
                let mut core =
                    VllmCore::new_with_kv_capture_and_latency_model(args, worker_id, latency_model);
                Self::init_offload_vllm(&mut core);
                EngineCore::Vllm(core)
            }
            crate::common::protocols::EngineType::Sglang => EngineCore::Sglang(
                SglangCore::new_with_kv_capture_and_latency_model(args, worker_id, latency_model),
            ),
        };
        Self { core }
    }

    #[cfg(feature = "kvbm-offload")]
    fn init_offload_vllm(core: &mut VllmCore<M>) {
        if let Err(e) = core.init_offload_offline() {
            tracing::error!("kvbm-offload single-worker offline init failed: {e}");
        }
    }

    #[cfg(not(feature = "kvbm-offload"))]
    fn init_offload_vllm(_core: &mut VllmCore<M>) {}

    pub(crate) fn is_empty(&self) -> bool {
        self.core.is_empty()
    }

    pub(crate) fn receive(
        &mut self,
        request: crate::common::protocols::DirectRequest,
    ) -> uuid::Uuid {
        self.core.receive(request)
    }

    pub(crate) fn num_requests(&self) -> usize {
        self.core.num_requests()
    }

    pub(crate) fn execute_pass(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> EnginePassResult {
        self.core.execute_pass(collector, now_ms)
    }
}
