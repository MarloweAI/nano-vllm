from __future__ import annotations

from nanovllm.mock.timing.backends import AFDStageDurations

from analytical_backend.calibration import calibration_from_config
from analytical_backend.disagg import (
    AFDServingModel,
    AFDSpec,
    ScaleOutLinkSpec,
    kv_transfer_ms,
)
from analytical_backend.gpu import GpuSpec
from analytical_backend.pareto import (
    gpu_fmha_total as _shared_gpu_fmha_total,
    gpu_only_point as _shared_gpu_only_point,
    pareto_uplr,
    tpot_seconds as _shared_tpot_seconds,
)
from analytical_backend.models import Model
from analytical_backend.profiles import load_profile
from analytical_backend.serving_model import AnalyticalServingModel

_MGPU = None


def _mgpu():
    global _MGPU
    if _MGPU is None:
        from analytical_backend.ac_model.measured_gpu import MeasuredGPU

        _MGPU = MeasuredGPU()
    return _MGPU


def _measured_layer_times(arch: GpuSpec, batch_size: int, isl: int, tp_g: int) -> tuple[float, float]:
    return _mgpu().layer_times(arch, batch_size, isl, tp_g)


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
        # All serving-time modeling (profile resolution, calibration, TP comm,
        # overlap composition) lives in the shared AnalyticalServingModel; this
        # adapter only maps nano config fields onto it. Explicit analytical_*
        # knobs override profile values (None = not set).
        profile_name = getattr(config, "analytical_profile", "") or ""
        explicit_calibration = calibration_from_config(config, prefix="analytical_")
        self._serving = AnalyticalServingModel(
            self.model_key,
            self.hardware_key,
            tp=config.roofline_tp_g,
            interconnect=self.interconnect_key,
            profile=profile_name or None,
            calibration=(
                calibration_from_config(
                    config,
                    prefix="analytical_",
                    base=load_profile(profile_name).calibration,
                )
                if profile_name
                else explicit_calibration
            ),
            tp_sharding_beta=getattr(config, "analytical_tp_sharding_beta", None),
            overlap_comm=getattr(config, "analytical_overlap_comm", None),
            collective_overhead_us=getattr(config, "analytical_collective_overhead_us", None),
            send_recv_overhead_us=getattr(config, "analytical_send_recv_overhead_us", None),
            bandwidth_efficiency=getattr(config, "analytical_bandwidth_efficiency", None),
        )
        self.profile = self._serving.profile
        self.model = self._serving.model
        self.arch = self._serving.arch
        self.calibration = self._serving.calibration
        self.overlap_comm = self._serving.overlap_comm
        self.tp_sharding_beta = self._serving.tp_sharding_beta
        self.collective_overhead_us = self._serving.collective_overhead_us
        self.send_recv_overhead_us = self._serving.send_recv_overhead_us
        self.bandwidth_efficiency = self._serving.bandwidth_efficiency
        self._afd_stage_cache: dict[tuple[int, int, str, int, float], AFDStageDurations] = {}

    @property
    def num_layers(self) -> int:
        return int(self.model.L)

    @property
    def backend(self) -> str:
        return self.config.roofline_gpu_backend

    def prefill_ms(self, batch_size: int, isl: int) -> float:
        if self.backend == "roofline":
            return self._serving.prefill_step_ms(batch_size=batch_size, seq_len=isl)
        return self.config.prefill_base_ms + isl * self.config.prefill_ms_per_token * batch_size

    def colocated_decode_ms(self, batch_size: int, context_len: int) -> float:
        if self.backend == "roofline":
            return self._serving.decode_step_ms(batch_size, context_len)
        return tpot_seconds(
            self.arch,
            self.model,
            batch_size,
            context_len,
            self.config.roofline_tp_g,
            backend=self.backend,
        ) * 1e3

    def kv_transfer_ms(self, context_len: int) -> float:
        """PDD KV hop, priced by the shared disagg helper (scale-out link)."""
        return kv_transfer_ms(
            self.model,
            context_len,
            link=ScaleOutLinkSpec(
                bandwidth_gbps=getattr(self.config, "pdd_kv_link_gbps", 100.0),
                latency_ms=getattr(self.config, "pdd_kv_link_latency_ms", 0.1),
            ),
        )

    def _afd_gpu_model(self) -> AFDServingModel:
        model = getattr(self, "_afd_gpu", None)
        if model is None:
            model = AFDServingModel(
                self.model,
                self.hardware_key,
                getattr(self.config, "afd_ffn_hardware", "") or self.hardware_key,
                spec=AFDSpec(
                    attn_tp=self.config.roofline_tp_g,
                    ffn_tp=getattr(self.config, "afd_ffn_tp", 1),
                    ffn_ep=getattr(self.config, "afd_ffn_ep", 1),
                ),
                interconnect=self.interconnect_key,
                profile=self.profile,
                calibration=self.calibration,
            )
            self._afd_gpu = model
        return model

    def _afd_cerebras_model(self) -> AFDServingModel:
        """AFD model with the FFN priced by the measured Cerebras submodule."""
        model = getattr(self, "_afd_cerebras", None)
        if model is None:
            from analytical_backend import CerebrasMeasuredBackend

            backend = CerebrasMeasuredBackend(
                self.model,
                wafer_count=getattr(self.config, "afd_ffn_wafers", 2),
                arch=getattr(self.config, "afd_ffn_cs_arch", "CS3"),
            )
            model = AFDServingModel(
                self.model,
                self.hardware_key,
                spec=AFDSpec(attn_tp=self.config.roofline_tp_g),
                interconnect=self.interconnect_key,
                profile=self.profile,
                calibration=self.calibration,
                ffn_backend=backend,
            )
            self._afd_cerebras = model
        return model

    def afd_decode_stages_ms(self, microbatch_size: int, context_len: int) -> AFDStageDurations:
        cache_key = (
            microbatch_size,
            context_len,
            self.backend,
            self.config.roofline_tp_g,
            self.config.gpu_cs_link_us,
            getattr(self.config, "afd_ffn_backend", "cs4-measured"),
        )
        if cache_key in self._afd_stage_cache:
            return self._afd_stage_cache[cache_key]

        if getattr(self.config, "afd_ffn_backend", "cs4-measured") == "gpu":
            # GPU-AFD: the FFN pool is GPUs priced by the shared disagg model;
            # the CS-4 path below is untouched.
            st = self._afd_gpu_model().decode_stage_times(microbatch_size, context_len)
            durations = AFDStageDurations(
                attention_ms=st.attention_ms,
                gpu_to_cs_link_ms=st.link_ms,
                cs_rest_ms=st.ffn_ms,
                cs_to_gpu_link_ms=st.link_ms,
                notes=(
                    f"timing_backend={self.config.timing_backend};afd_ffn=gpu;"
                    f"model={self.model.key};attn_hw={self.arch.key};"
                    f"ffn_hw={getattr(self.config, 'afd_ffn_hardware', '') or self.arch.key};"
                    f"attn_tp={self.config.roofline_tp_g};"
                    f"ffn_tp={getattr(self.config, 'afd_ffn_tp', 1)};"
                    f"ffn_ep={getattr(self.config, 'afd_ffn_ep', 1)}"
                ),
            )
            self._afd_stage_cache[cache_key] = durations
            return durations

        backend_kind = getattr(self.config, "afd_ffn_backend", "cs4-measured")
        if backend_kind != "cs4-measured":
            raise ValueError(
                f"afd_ffn_backend={backend_kind!r} is not supported; the Cerebras wafer "
                "path is 'cs4-measured' (Cerebras submodule) and the GPU FFN path is 'gpu'."
            )
        # CS-4 AFD with the FFN priced by the measured Cerebras submodule
        # (moe_decode_sim.perf) - the ONLY Cerebras path. The backend supplies its
        # own GPU<->CS link (its InterconnectSpec), so gpu_cs_link_us does not apply.
        st = self._afd_cerebras_model().decode_stage_times(microbatch_size, context_len)
        durations = AFDStageDurations(
            attention_ms=st.attention_ms,
            gpu_to_cs_link_ms=st.link_ms,
            cs_rest_ms=st.ffn_ms,
            cs_to_gpu_link_ms=st.link_ms,
            notes=(
                f"timing_backend={self.config.timing_backend};afd_ffn=cs4-measured;"
                f"model={self.model.key};attn_hw={self.arch.key};"
                f"wafers={getattr(self.config, 'afd_ffn_wafers', 2)};"
                f"cs_arch={getattr(self.config, 'afd_ffn_cs_arch', 'CS3')}"
            ),
        )
        self._afd_stage_cache[cache_key] = durations
        return durations
