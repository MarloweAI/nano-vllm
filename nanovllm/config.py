import os
from math import ceil
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    mock_backend: bool = False
    mock_mode: str = "colocated"
    mock_runner: str = "fake"
    virtual_time: bool = False
    trace_output: str = "traces/mock_trace.csv"
    prefill_base_ms: float = 1.0
    prefill_ms_per_token: float = 0.01
    decode_base_ms: float = 0.5
    decode_ms_per_token: float = 0.02
    mock_token_base: int = 1000
    mock_kv_capacity_tokens: int | None = None
    mock_block_size: int | None = None
    attention_ms_base: float = 0.4
    attention_ms_per_token: float = 0.02
    attention_ms_per_isl_token: float = 0.0001
    cs_rest_ms_base: float = 0.6
    cs_rest_ms_per_token: float = 0.03
    link_ms_one_way: float = 0.1
    num_layers: int = 32
    pipeline_mode: str = "sequential"
    microbatch_size: int = 1
    attention_replicas: int = 1
    gpu_to_cs_link_resources: int = 1
    cs_rest_resources: int = 1
    cs_to_gpu_link_resources: int = 1
    timing_backend: str = "parametric"
    analytical_profile: str = ""
    analytical_model: str = "openai/gpt-oss-120b"
    analytical_hardware: str = "helios"
    analytical_interconnect: str = ""
    analytical_collective_overhead_us: float | None = None
    analytical_send_recv_overhead_us: float | None = None
    analytical_bandwidth_efficiency: float | None = None
    analytical_overlap_comm: bool | None = None
    analytical_launch_overhead_us: float | None = None
    analytical_decode_launch_overhead_us: float | None = None
    analytical_graph_launch_overhead_us: float | None = None
    analytical_utilization: float | None = None
    analytical_hbm_utilization: float | None = None
    analytical_flop_utilization: float | None = None
    analytical_attn_hbm_utilization: float | None = None
    analytical_attn_flop_utilization: float | None = None
    analytical_prefill_attn_hbm_utilization: float | None = None
    analytical_moe_grouped_gemm_efficiency: float | None = None
    analytical_attn_proj_eager_overhead_us: float | None = None
    analytical_moe_grouped_gemm_eager_overhead_us: float | None = None
    analytical_per_layer_overhead_us: float | None = None
    analytical_prefill_per_layer_overhead_us: float | None = None
    analytical_kernel_floor_multiplier: float | None = None
    analytical_tp_sharding_beta: float | None = None
    afd_ffn_backend: str = "cs4-measured"  # cs4-measured (Cerebras submodule) | gpu
    afd_ffn_hardware: str = ""
    afd_ffn_tp: int = 1
    afd_ffn_ep: int = 1
    afd_ffn_wafers: int = 2       # cs4-measured: CS wafers holding the experts (EP unit)
    afd_ffn_cs_arch: str = "CS3"  # cs4-measured: measured-data arch (CS3 today; CS4/CS5 when landed)
    pdd_prefill_replicas: int = 1
    pdd_kv_link_gbps: float = 100.0
    pdd_kv_link_latency_ms: float = 0.1
    pdd_kv_link_lanes: int = 1
    roofline_gpu_backend: str = "roofline"
    roofline_tp_g: int = 1
    attention_groups: int = 1
    chunk_batch: int = 1
    gpu_cs_link_us: float = 12.0

    def __post_init__(self):
        if self.mock_backend:
            if self.mock_block_size is not None:
                self.kvcache_block_size = self.mock_block_size
            assert self.kvcache_block_size > 0
            assert 1 <= self.tensor_parallel_size <= 8
            assert self.mock_mode in ("colocated", "afd", "pdd")
            assert self.mock_runner in ("fake", "des")
            assert self.pipeline_mode in ("sequential", "ideal_pipeline", "discrete_pipeline")
            assert self.num_layers > 0
            assert self.microbatch_size > 0
            assert self.attention_replicas > 0
            assert self.gpu_to_cs_link_resources > 0
            assert self.cs_rest_resources > 0
            assert self.cs_to_gpu_link_resources > 0
            assert self.timing_backend in ("parametric", "analytical")
            assert self.roofline_gpu_backend in ("measured", "roofline")
            assert self.roofline_tp_g > 0
            assert self.attention_groups > 0
            assert self.chunk_batch > 0
            assert self.gpu_cs_link_us >= 0
            for value in (
                self.analytical_collective_overhead_us,
                self.analytical_send_recv_overhead_us,
                self.analytical_launch_overhead_us,
                self.analytical_decode_launch_overhead_us,
                self.analytical_graph_launch_overhead_us,
                self.analytical_utilization,
                self.analytical_hbm_utilization,
                self.analytical_flop_utilization,
                self.analytical_attn_hbm_utilization,
                self.analytical_attn_flop_utilization,
                self.analytical_prefill_attn_hbm_utilization,
                self.analytical_moe_grouped_gemm_efficiency,
                self.analytical_attn_proj_eager_overhead_us,
                self.analytical_moe_grouped_gemm_eager_overhead_us,
                self.analytical_per_layer_overhead_us,
                self.analytical_prefill_per_layer_overhead_us,
                self.analytical_kernel_floor_multiplier,
            ):
                assert value is None or value >= 0
            assert self.analytical_bandwidth_efficiency is None or self.analytical_bandwidth_efficiency > 0
            assert self.analytical_tp_sharding_beta is None or self.analytical_tp_sharding_beta > 0
            if self.mock_kv_capacity_tokens is not None:
                self.num_kvcache_blocks = max(1, ceil(self.mock_kv_capacity_tokens / self.kvcache_block_size))
            elif self.num_kvcache_blocks == -1:
                total_tokens = self.max_num_seqs * self.max_model_len
                self.num_kvcache_blocks = max(1, ceil(total_tokens / self.kvcache_block_size))
            return
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert os.path.isdir(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
