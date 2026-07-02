from __future__ import annotations

from nanovllm.mock.timing.backends import AFDStageDurations
from nanovllm.mock.timing.ac_model import cs4_offload as CS4M

from analytical_backend.comm import AnalyticalCommModel, CommSpec
from analytical_backend.gpu import GpuSpec, get_gpu_spec
from analytical_backend.pareto import (
    gpu_fmha_total as _shared_gpu_fmha_total,
    gpu_only_point as _shared_gpu_only_point,
    hybrid_point as _shared_hybrid_point,
    pareto_uplr,
    stage_times as _shared_stage_times,
    tpot_seconds as _shared_tpot_seconds,
)
from analytical_backend.interconnect import get_interconnect_spec
from analytical_backend.models import Model, load_model
from analytical_backend.serving import DTYPE_BYTES, build_roofline_backend

_MGPU = None


def _mgpu():
    global _MGPU
    if _MGPU is None:
        from nanovllm.mock.timing.ac_model.measured_gpu import MeasuredGPU

        _MGPU = MeasuredGPU()
    return _MGPU


def _measured_layer_times(arch: GpuSpec, batch_size: int, isl: int, tp_g: int) -> tuple[float, float]:
    return _mgpu().layer_times(arch, batch_size, isl, tp_g)


def _cs4_nonattn_s(batch_size: int) -> float:
    return CS4M.cs4_nonattn_us(batch_size) * 1e-6


def _cs4_comm_s(batch_size: int) -> float:
    return CS4M.comm_us(batch_size) * 1e-6


def tpot_seconds(
    arch: GpuSpec,
    m: Model,
    B: int,
    isl: int,
    P: int,
    *,
    backend: str = "measured",
    want_strategy: bool = False,
    all_full: bool = False,
) -> float | tuple[float, str]:
    return _shared_tpot_seconds(
        arch,
        m,
        B,
        isl,
        P,
        backend=backend,
        want_strategy=want_strategy,
        all_full=all_full,
        measured_layer_times=_measured_layer_times if backend == "measured" else None,
    )


def gpu_fmha_total(arch: GpuSpec, m: Model, B: int, isl: int, P: int, *, backend: str = "measured") -> float:
    return _shared_gpu_fmha_total(
        arch,
        m,
        B,
        isl,
        P,
        backend=backend,
        measured_layer_times=_measured_layer_times if backend == "measured" else None,
    )


def stage_times(arch: GpuSpec, m: Model, ck: int, isl: int, tp_g: int, *, backend: str = "measured"):
    return _shared_stage_times(
        arch,
        m,
        ck,
        isl,
        tp_g,
        backend=backend,
        cs4_nonattn_s=_cs4_nonattn_s,
        cs4_comm_s=_cs4_comm_s,
        measured_layer_times=_measured_layer_times if backend == "measured" else None,
    )


def hybrid_point(
    arch: GpuSpec,
    m: Model,
    gb: int,
    isl: int,
    tp_g: int,
    a_g: int,
    ck: int,
    *,
    backend: str = "measured",
):
    return _shared_hybrid_point(
        arch,
        m,
        gb,
        isl,
        tp_g,
        a_g,
        ck,
        backend=backend,
        cs4_nonattn_s=_cs4_nonattn_s,
        cs4_comm_s=_cs4_comm_s,
        cs4_power_kw=CS4M.P_CS4_UNIT_KW,
        measured_layer_times=_measured_layer_times if backend == "measured" else None,
    )


def gpu_only_point(arch: GpuSpec, m: Model, B: int, isl: int, tp_g: int, *, backend: str = "measured"):
    return _shared_gpu_only_point(
        arch,
        m,
        B,
        isl,
        tp_g,
        backend=backend,
        measured_layer_times=_measured_layer_times if backend == "measured" else None,
    )


class AnalyticalTimingBackend:
    """nano-vLLM timing adapter over the shared analytical backend package."""

    name = "analytical"

    def __init__(self, config):
        self.config = config
        self.model_key = getattr(config, "analytical_model", "openai/gpt-oss-120b")
        self.hardware_key = getattr(config, "analytical_hardware", "b200")
        self.interconnect_key = getattr(config, "analytical_interconnect", "")
        self.collective_overhead_us = getattr(config, "analytical_collective_overhead_us", None)
        self.send_recv_overhead_us = getattr(config, "analytical_send_recv_overhead_us", None)
        self.model = load_model(self.model_key)
        self.arch = get_gpu_spec(self.hardware_key)
        self._comm_model = None
        if self.interconnect_key:
            interconnect = get_interconnect_spec(self.interconnect_key)
            if interconnect.device_key != self.arch.key:
                raise ValueError(
                    f"interconnect {self.interconnect_key!r} targets {interconnect.device_key!r}, "
                    f"not hardware {self.arch.key!r}"
                )
            self._comm_model = AnalyticalCommModel(
                CommSpec.from_interconnect(
                    interconnect,
                    collective_overhead_us=self.collective_overhead_us,
                    send_recv_overhead_us=self.send_recv_overhead_us,
                )
            )
        self._shared_backend = build_roofline_backend(self.model, self.arch, config.roofline_tp_g)
        self._afd_stage_cache: dict[tuple[int, int, str, int, float], AFDStageDurations] = {}

    @property
    def backend(self) -> str:
        return self.config.roofline_gpu_backend

    def prefill_ms(self, batch_size: int, isl: int) -> float:
        if self.backend == "roofline":
            return self._shared_backend.prefill_step_ms(
                batch_size=batch_size,
                seq_len=isl,
            ) + self._tp_comm_ms(batch_size * isl)
        return self.config.prefill_base_ms + isl * self.config.prefill_ms_per_token * batch_size

    def colocated_decode_ms(self, batch_size: int, context_len: int) -> float:
        if self.backend == "roofline":
            return self._shared_backend.decode_step_ms(batch_size, context_len) + self._tp_comm_ms(batch_size)
        return tpot_seconds(
            self.arch,
            self.model,
            batch_size,
            context_len,
            self.config.roofline_tp_g,
            backend=self.backend,
        ) * 1e3

    def _tp_comm_ms(self, num_tokens: int) -> float:
        tp_g = self.config.roofline_tp_g
        if self._comm_model is None or tp_g <= 1 or num_tokens <= 0:
            return 0.0
        tensor_bytes = int(num_tokens * self.model.d * DTYPE_BYTES[self.model.act])
        # One attention output all-reduce and one MLP/MoE output all-reduce per layer.
        return (
            2.0
            * self.model.L
            * self._comm_model.allreduce_ms(tensor_bytes, tp_g, floor=True)
        )

    def afd_decode_stages_ms(self, microbatch_size: int, context_len: int) -> AFDStageDurations:
        cache_key = (
            microbatch_size,
            context_len,
            self.backend,
            self.config.roofline_tp_g,
            self.config.gpu_cs_link_us,
        )
        if cache_key in self._afd_stage_cache:
            return self._afd_stage_cache[cache_key]

        old_link = CS4M.CLOS_LAT_US
        CS4M.CLOS_LAT_US = self.config.gpu_cs_link_us
        try:
            if self.backend == "roofline":
                attention_s = self._shared_backend.op_time_ms(
                    "attn_decode",
                    num_tokens=microbatch_size,
                    batch_size=microbatch_size,
                    kv_cache_size=context_len,
                ) * 1e-3
                cs_rest_s = _cs4_nonattn_s(microbatch_size)
                link_s = _cs4_comm_s(microbatch_size) / 2.0
            else:
                attention_s, cs_rest_s, link_s = stage_times(
                    self.arch,
                    self.model,
                    microbatch_size,
                    context_len,
                    self.config.roofline_tp_g,
                    backend=self.backend,
                )
        finally:
            CS4M.CLOS_LAT_US = old_link

        durations = AFDStageDurations(
            attention_ms=attention_s * 1e3,
            gpu_to_cs_link_ms=link_s * 1e3,
            cs_rest_ms=cs_rest_s * 1e3,
            cs_to_gpu_link_ms=link_s * 1e3,
            notes=(
                f"timing_backend={self.config.timing_backend};"
                f"model={self.model.key};hardware={self.arch.key};"
                f"gpu_backend={self.backend};tp_g={self.config.roofline_tp_g};"
                f"collective_overhead_us={self.collective_overhead_us};"
                f"link_us={self.config.gpu_cs_link_us}"
            ),
        )
        self._afd_stage_cache[cache_key] = durations
        return durations
