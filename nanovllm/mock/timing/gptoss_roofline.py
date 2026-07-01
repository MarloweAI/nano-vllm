from __future__ import annotations

from pathlib import Path
import sys

from nanovllm.mock.timing.backends import AFDStageDurations
from nanovllm.mock.timing.ac_model import cs4_offload as CS4M

_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from marlowe_roofline.gptoss import (  # noqa: E402
    ARCHES,
    B200,
    DTYPE_BYTES,
    GPTOSS,
    HELIOS,
    P_MI455X_KW,
    RUBIN,
    Arch,
    Model,
    all_gather,
    all_to_all,
    arch_to_device_spec,
    attn_layer,
    experts_active,
    fmha_time,
    gemm_time,
    gpu_fmha_total as _shared_gpu_fmha_total,
    gpu_only_point as _shared_gpu_only_point,
    hybrid_point as _shared_hybrid_point,
    layer_split,
    model_to_model_spec,
    moe_layer,
    pareto_uplr,
    pipe_closed,
    roofline_backend,
    stage_times as _shared_stage_times,
    tpot_seconds as _shared_tpot_seconds,
    tp_allreduce,
)

_MGPU = None


def _mgpu():
    global _MGPU
    if _MGPU is None:
        from nanovllm.mock.timing.ac_model.measured_gpu import MeasuredGPU

        _MGPU = MeasuredGPU()
    return _MGPU


def _measured_layer_times(arch: Arch, batch_size: int, isl: int, tp_g: int) -> tuple[float, float]:
    return _mgpu().layer_times(arch, batch_size, isl, tp_g)


def _cs4_nonattn_s(batch_size: int) -> float:
    return CS4M.cs4_nonattn_us(batch_size) * 1e-6


def _cs4_comm_s(batch_size: int) -> float:
    return CS4M.comm_us(batch_size) * 1e-6


def tpot_seconds(
    arch: Arch,
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


def gpu_fmha_total(arch: Arch, m: Model, B: int, isl: int, P: int, *, backend: str = "measured") -> float:
    return _shared_gpu_fmha_total(
        arch,
        m,
        B,
        isl,
        P,
        backend=backend,
        measured_layer_times=_measured_layer_times if backend == "measured" else None,
    )


def stage_times(arch: Arch, m: Model, ck: int, isl: int, tp_g: int, *, backend: str = "measured"):
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
    arch: Arch,
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


def gpu_only_point(arch: Arch, m: Model, B: int, isl: int, tp_g: int, *, backend: str = "measured"):
    return _shared_gpu_only_point(
        arch,
        m,
        B,
        isl,
        tp_g,
        backend=backend,
        measured_layer_times=_measured_layer_times if backend == "measured" else None,
    )


class GPTOSSRooflineTimingBackend:
    """nano-vLLM timing adapter over the shared GPT-OSS analytical model."""

    name = "gptoss_roofline"

    def __init__(self, config):
        self.config = config
        self.model = GPTOSS
        self.arch = ARCHES[config.roofline_gpu_arch]
        self._shared_backend = roofline_backend(self.arch, self.model, config.roofline_tp_g)
        self._afd_stage_cache: dict[tuple[int, int, str, int, float], AFDStageDurations] = {}

    @property
    def backend(self) -> str:
        return self.config.roofline_gpu_backend

    def prefill_ms(self, batch_size: int, isl: int) -> float:
        if self.backend == "roofline":
            return self._shared_backend.prefill_step_ms(batch_size=batch_size, seq_len=isl)
        return self.config.prefill_base_ms + isl * self.config.prefill_ms_per_token * batch_size

    def colocated_decode_ms(self, batch_size: int, context_len: int) -> float:
        if self.backend == "roofline":
            return self._shared_backend.decode_step_ms(batch_size, context_len)
        return tpot_seconds(
            self.arch,
            self.model,
            batch_size,
            context_len,
            self.config.roofline_tp_g,
            backend=self.backend,
        ) * 1e3

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
                "timing_backend=gptoss_roofline;"
                f"arch={self.config.roofline_gpu_arch};"
                f"gpu_backend={self.backend};tp_g={self.config.roofline_tp_g};"
                f"link_us={self.config.gpu_cs_link_us}"
            ),
        )
        self._afd_stage_cache[cache_key] = durations
        return durations
