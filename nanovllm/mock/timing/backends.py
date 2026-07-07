from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AFDStageDurations:
    attention_ms: float
    gpu_to_cs_link_ms: float
    cs_rest_ms: float
    cs_to_gpu_link_ms: float
    notes: str = ""


class TimingBackend(Protocol):
    name: str

    @property
    def num_layers(self) -> int:
        """Transformer layer count. afd_decode_stages_ms prices ONE layer, so the
        DES scales its per-layer AFD pipeline by this to get the per-token step."""
        ...

    def prefill_ms(self, batch_size: int, isl: int) -> float:
        ...

    def colocated_decode_ms(self, batch_size: int, context_len: int) -> float:
        ...

    def afd_decode_stages_ms(self, microbatch_size: int, context_len: int) -> AFDStageDurations:
        ...

    def kv_transfer_ms(self, context_len: int) -> float:
        ...


class ParametricTimingBackend:
    """Compatibility backend for the original mock timing formulas."""

    name = "parametric"

    def __init__(self, config):
        self.config = config

    @property
    def num_layers(self) -> int:
        # The parametric AFD formula is a lumped single-stage model (not per-layer),
        # so default to 1: the DES per-layer scaling is a no-op unless a layer count
        # is explicitly configured.
        return int(getattr(self.config, "num_layers", 1) or 1)

    def prefill_ms(self, batch_size: int, isl: int) -> float:
        return self.config.prefill_base_ms + isl * self.config.prefill_ms_per_token * batch_size

    def colocated_decode_ms(self, batch_size: int, context_len: int) -> float:
        return self.config.decode_base_ms + self.config.decode_ms_per_token * batch_size

    def kv_transfer_ms(self, context_len: int) -> float:
        # PDD KV hop with a parametric per-token KV size (no model spec here);
        # 72 KiB/token is the gpt-oss-120b fp16 figure.
        kib_per_token = getattr(self.config, "pdd_kv_kib_per_token", 72.0)
        bytes_ = context_len * kib_per_token * 1024.0
        gbps = getattr(self.config, "pdd_kv_link_gbps", 100.0)
        latency = getattr(self.config, "pdd_kv_link_latency_ms", 0.1)
        return latency + bytes_ * 8.0 / (gbps * 1e6)

    def afd_decode_stages_ms(self, microbatch_size: int, context_len: int) -> AFDStageDurations:
        attention_ms = (
            self.config.attention_ms_base
            + self.config.attention_ms_per_token * microbatch_size
            + self.config.attention_ms_per_isl_token * context_len * microbatch_size
        )
        cs_rest_ms = self.config.cs_rest_ms_base + self.config.cs_rest_ms_per_token * microbatch_size
        return AFDStageDurations(
            attention_ms=attention_ms,
            gpu_to_cs_link_ms=self.config.link_ms_one_way,
            cs_rest_ms=cs_rest_ms,
            cs_to_gpu_link_ms=self.config.link_ms_one_way,
            notes="timing_backend=parametric",
        )


def build_timing_backend(config) -> TimingBackend:
    if config.timing_backend == "parametric":
        return ParametricTimingBackend(config)
    if config.timing_backend == "analytical":
        from nanovllm.mock.timing.analytical import AnalyticalTimingBackend

        return AnalyticalTimingBackend(config)
    raise ValueError(f"unknown timing backend: {config.timing_backend}")
