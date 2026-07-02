import pytest

from nanovllm.config import Config
from nanovllm.mock.timing import build_timing_backend
from nanovllm.mock.timing.ac_model import cs4_offload as CS4M
from nanovllm.mock.timing.analytical import gpu_only_point, hybrid_point
from analytical_backend.gpu import get_gpu_spec
from analytical_backend.models import load_model

HELIOS = get_gpu_spec("helios")
MODEL = load_model("openai/gpt-oss-120b")


def analytical_config(**overrides):
    kwargs = {
        "mock_backend": True,
        "mock_mode": "colocated",
        "timing_backend": "analytical",
        "analytical_model": "openai/gpt-oss-120b",
        "analytical_hardware": "b200",
        "roofline_gpu_backend": "roofline",
        "roofline_tp_g": 1,
    }
    kwargs.update(overrides)
    return Config("__mock__", **kwargs)


def test_parametric_timing_backend_preserves_existing_formulas():
    config = Config(
        "__mock__",
        mock_backend=True,
        prefill_base_ms=1.5,
        prefill_ms_per_token=0.25,
        decode_base_ms=0.7,
        decode_ms_per_token=0.05,
        attention_ms_base=0.4,
        attention_ms_per_token=0.02,
        attention_ms_per_isl_token=0.001,
        cs_rest_ms_base=0.6,
        cs_rest_ms_per_token=0.03,
        link_ms_one_way=0.2,
    )
    backend = build_timing_backend(config)

    assert backend.prefill_ms(batch_size=3, isl=10) == pytest.approx(9.0)
    assert backend.colocated_decode_ms(batch_size=4, context_len=128) == pytest.approx(0.9)

    stages = backend.afd_decode_stages_ms(microbatch_size=2, context_len=100)
    assert stages.attention_ms == pytest.approx(0.64)
    assert stages.gpu_to_cs_link_ms == pytest.approx(0.2)
    assert stages.cs_rest_ms == pytest.approx(0.66)
    assert stages.cs_to_gpu_link_ms == pytest.approx(0.2)


def test_analytical_backend_attention_is_monotonic_with_context():
    config = Config(
        "__mock__",
        mock_backend=True,
        mock_mode="afd",
        timing_backend="analytical",
        roofline_gpu_backend="measured",
        roofline_tp_g=1,
    )
    backend = build_timing_backend(config)

    short = backend.afd_decode_stages_ms(microbatch_size=4, context_len=8192)
    long = backend.afd_decode_stages_ms(microbatch_size=4, context_len=131072)

    assert long.attention_ms > short.attention_ms
    assert "timing_backend=analytical" in short.notes


def test_analytical_backend_accepts_shared_amd_hardware_specs():
    config = Config(
        "__mock__",
        mock_backend=True,
        mock_mode="colocated",
        timing_backend="analytical",
        analytical_model="openai/gpt-oss-120b",
        analytical_hardware="mi355x",
        analytical_interconnect="mi355x_ubb",
        roofline_gpu_backend="roofline",
        roofline_tp_g=1,
    )
    backend = build_timing_backend(config)

    assert backend.prefill_ms(batch_size=1, isl=1024) > 0
    assert backend.colocated_decode_ms(batch_size=4, context_len=1024) > 0


def test_analytical_backend_charges_shared_interconnect_for_tp_collectives():
    base = Config(
        "__mock__",
        mock_backend=True,
        mock_mode="colocated",
        timing_backend="analytical",
        analytical_model="openai/gpt-oss-120b",
        analytical_hardware="b200",
        roofline_gpu_backend="roofline",
        roofline_tp_g=4,
    )
    with_comm = Config(
        "__mock__",
        mock_backend=True,
        mock_mode="colocated",
        timing_backend="analytical",
        analytical_model="openai/gpt-oss-120b",
        analytical_hardware="b200",
        analytical_interconnect="b200_dgx",
        roofline_gpu_backend="roofline",
        roofline_tp_g=4,
    )
    tp1_with_comm = Config(
        "__mock__",
        mock_backend=True,
        mock_mode="colocated",
        timing_backend="analytical",
        analytical_model="openai/gpt-oss-120b",
        analytical_hardware="b200",
        analytical_interconnect="b200_dgx",
        roofline_gpu_backend="roofline",
        roofline_tp_g=1,
    )
    tp1_no_comm = Config(
        "__mock__",
        mock_backend=True,
        mock_mode="colocated",
        timing_backend="analytical",
        analytical_model="openai/gpt-oss-120b",
        analytical_hardware="b200",
        roofline_gpu_backend="roofline",
        roofline_tp_g=1,
    )

    assert build_timing_backend(with_comm).colocated_decode_ms(4, 1024) > build_timing_backend(
        base
    ).colocated_decode_ms(4, 1024)
    assert build_timing_backend(tp1_with_comm).colocated_decode_ms(4, 1024) == pytest.approx(
        build_timing_backend(tp1_no_comm).colocated_decode_ms(4, 1024)
    )


def test_analytical_backend_accepts_tuned_collective_floor_override():
    default = Config(
        "__mock__",
        mock_backend=True,
        mock_mode="colocated",
        timing_backend="analytical",
        analytical_model="openai/gpt-oss-120b",
        analytical_hardware="mi455x",
        analytical_interconnect="mi455x_helios",
        roofline_gpu_backend="roofline",
        roofline_tp_g=4,
    )
    tuned = Config(
        "__mock__",
        mock_backend=True,
        mock_mode="colocated",
        timing_backend="analytical",
        analytical_model="openai/gpt-oss-120b",
        analytical_hardware="mi455x",
        analytical_interconnect="mi455x_helios",
        analytical_collective_overhead_us=6.0,
        roofline_gpu_backend="roofline",
        roofline_tp_g=4,
    )

    assert build_timing_backend(tuned).colocated_decode_ms(4, 1024) < build_timing_backend(
        default
    ).colocated_decode_ms(4, 1024)


def test_analytical_backend_can_overlap_shared_tp_collectives():
    serial = analytical_config(
        analytical_interconnect="b200_dgx",
        roofline_tp_g=4,
    )
    overlapped = analytical_config(
        analytical_interconnect="b200_dgx",
        analytical_overlap_comm=True,
        roofline_tp_g=4,
    )

    assert build_timing_backend(overlapped).colocated_decode_ms(4, 1024) < build_timing_backend(
        serial
    ).colocated_decode_ms(4, 1024)


def test_analytical_backend_uses_shared_comm_bandwidth_efficiency_override():
    slow_comm = analytical_config(
        analytical_interconnect="b200_dgx",
        analytical_bandwidth_efficiency=0.25,
        roofline_tp_g=4,
    )
    fast_comm = analytical_config(
        analytical_interconnect="b200_dgx",
        analytical_bandwidth_efficiency=0.95,
        roofline_tp_g=4,
    )

    assert build_timing_backend(fast_comm).colocated_decode_ms(64, 1024) < build_timing_backend(
        slow_comm
    ).colocated_decode_ms(64, 1024)


def test_analytical_backend_uses_tp_sharding_beta():
    sublinear = analytical_config(analytical_tp_sharding_beta=0.6, roofline_tp_g=4)
    linear = analytical_config(analytical_tp_sharding_beta=1.0, roofline_tp_g=4)

    assert build_timing_backend(sublinear).colocated_decode_ms(16, 1024) > build_timing_backend(
        linear
    ).colocated_decode_ms(16, 1024)


def test_analytical_backend_uses_prefill_eager_calibration_knobs():
    base = analytical_config(analytical_moe_grouped_gemm_efficiency=0.12)
    extra_eager = analytical_config(
        analytical_moe_grouped_gemm_efficiency=0.12,
        analytical_moe_grouped_gemm_eager_overhead_us=100.0,
        analytical_attn_proj_eager_overhead_us=37.0,
    )

    assert build_timing_backend(extra_eager).prefill_ms(1, 1024) > build_timing_backend(
        base
    ).prefill_ms(1, 1024)


def test_analytical_backend_uses_graph_decode_launch_calibration():
    eager_like = analytical_config(analytical_launch_overhead_us=25.0)
    graph_decode = analytical_config(
        analytical_launch_overhead_us=25.0,
        analytical_decode_launch_overhead_us=0.0,
        analytical_graph_launch_overhead_us=2.0,
    )

    assert build_timing_backend(graph_decode).colocated_decode_ms(4, 1024) < build_timing_backend(
        eager_like
    ).colocated_decode_ms(4, 1024)


def test_gptoss_rawdata_regression_points_match_section5_8k():
    gpu = gpu_only_point(HELIOS, MODEL, B=256, isl=8192, tp_g=1, backend="measured")
    assert gpu["x"] == pytest_approx_pct(163.7, rel=0.005)
    assert gpu["y"] == pytest_approx_pct(41896.2, rel=0.005)

    old_link = CS4M.CLOS_LAT_US
    CS4M.CLOS_LAT_US = 12.0
    try:
        hybrid = hybrid_point(HELIOS, MODEL, gb=256, isl=8192, tp_g=1, a_g=1, ck=128, backend="measured")
    finally:
        CS4M.CLOS_LAT_US = old_link
    assert hybrid["x"] == pytest_approx_pct(327.8, rel=0.005)
    assert hybrid["y"] == pytest_approx_pct(83920.8, rel=0.005)


def test_gptoss_link_latency_changes_interactivity_not_pipeline_filled_throughput():
    old_link = CS4M.CLOS_LAT_US
    try:
        CS4M.CLOS_LAT_US = 4.0
        fast = hybrid_point(HELIOS, MODEL, gb=256, isl=8192, tp_g=1, a_g=1, ck=64, backend="measured")
        CS4M.CLOS_LAT_US = 36.0
        slow = hybrid_point(HELIOS, MODEL, gb=256, isl=8192, tp_g=1, a_g=1, ck=64, backend="measured")
    finally:
        CS4M.CLOS_LAT_US = old_link

    assert fast["x"] > slow["x"]
    assert fast["y"] == pytest_approx_pct(slow["y"], rel=0.0001)


def pytest_approx_pct(expected, rel):
    return pytest.approx(expected, rel=rel)
