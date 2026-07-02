def add_timing_backend_args(parser):
    parser.add_argument("--timing-backend", choices=["parametric", "analytical"], default="parametric")
    parser.add_argument("--analytical-model", default="openai/gpt-oss-120b")
    parser.add_argument("--analytical-hardware", default="helios")
    parser.add_argument("--analytical-interconnect", default="")
    parser.add_argument("--analytical-collective-overhead-us", type=float, default=None)
    parser.add_argument("--analytical-send-recv-overhead-us", type=float, default=None)
    parser.add_argument("--analytical-bandwidth-efficiency", type=float, default=None)
    parser.add_argument("--analytical-overlap-comm", action="store_true")
    parser.add_argument("--analytical-launch-overhead-us", type=float, default=None)
    parser.add_argument("--analytical-decode-launch-overhead-us", type=float, default=None)
    parser.add_argument("--analytical-graph-launch-overhead-us", type=float, default=None)
    parser.add_argument("--analytical-utilization", type=float, default=None)
    parser.add_argument("--analytical-hbm-utilization", type=float, default=None)
    parser.add_argument("--analytical-flop-utilization", type=float, default=None)
    parser.add_argument("--analytical-attn-hbm-utilization", type=float, default=None)
    parser.add_argument("--analytical-attn-flop-utilization", type=float, default=None)
    parser.add_argument("--analytical-prefill-attn-hbm-utilization", type=float, default=None)
    parser.add_argument("--analytical-moe-grouped-gemm-efficiency", type=float, default=None)
    parser.add_argument("--analytical-attn-proj-eager-overhead-us", type=float, default=None)
    parser.add_argument("--analytical-moe-grouped-gemm-eager-overhead-us", type=float, default=None)
    parser.add_argument("--analytical-per-layer-overhead-us", type=float, default=None)
    parser.add_argument("--analytical-prefill-per-layer-overhead-us", type=float, default=None)
    parser.add_argument("--analytical-kernel-floor-multiplier", type=float, default=None)
    parser.add_argument("--analytical-tp-sharding-beta", type=float, default=1.0)
    parser.add_argument("--roofline-gpu-backend", choices=["measured", "roofline"], default="roofline")
    parser.add_argument("--tp-g", "--roofline-tp-g", dest="roofline_tp_g", type=int, default=1)
    parser.add_argument("--attention-groups", type=int, default=1)
    parser.add_argument("--chunk-batch", type=int, default=1)
    parser.add_argument("--gpu-cs-link-us", type=float, default=12.0)


def timing_backend_kwargs(args):
    return {
        "timing_backend": args.timing_backend,
        "analytical_model": args.analytical_model,
        "analytical_hardware": args.analytical_hardware,
        "analytical_interconnect": args.analytical_interconnect,
        "analytical_collective_overhead_us": args.analytical_collective_overhead_us,
        "analytical_send_recv_overhead_us": args.analytical_send_recv_overhead_us,
        "analytical_bandwidth_efficiency": args.analytical_bandwidth_efficiency,
        "analytical_overlap_comm": args.analytical_overlap_comm,
        "analytical_launch_overhead_us": args.analytical_launch_overhead_us,
        "analytical_decode_launch_overhead_us": args.analytical_decode_launch_overhead_us,
        "analytical_graph_launch_overhead_us": args.analytical_graph_launch_overhead_us,
        "analytical_utilization": args.analytical_utilization,
        "analytical_hbm_utilization": args.analytical_hbm_utilization,
        "analytical_flop_utilization": args.analytical_flop_utilization,
        "analytical_attn_hbm_utilization": args.analytical_attn_hbm_utilization,
        "analytical_attn_flop_utilization": args.analytical_attn_flop_utilization,
        "analytical_prefill_attn_hbm_utilization": args.analytical_prefill_attn_hbm_utilization,
        "analytical_moe_grouped_gemm_efficiency": args.analytical_moe_grouped_gemm_efficiency,
        "analytical_attn_proj_eager_overhead_us": args.analytical_attn_proj_eager_overhead_us,
        "analytical_moe_grouped_gemm_eager_overhead_us": args.analytical_moe_grouped_gemm_eager_overhead_us,
        "analytical_per_layer_overhead_us": args.analytical_per_layer_overhead_us,
        "analytical_prefill_per_layer_overhead_us": args.analytical_prefill_per_layer_overhead_us,
        "analytical_kernel_floor_multiplier": args.analytical_kernel_floor_multiplier,
        "analytical_tp_sharding_beta": args.analytical_tp_sharding_beta,
        "roofline_gpu_backend": args.roofline_gpu_backend,
        "roofline_tp_g": args.roofline_tp_g,
        "attention_groups": args.attention_groups,
        "chunk_batch": args.chunk_batch,
        "gpu_cs_link_us": args.gpu_cs_link_us,
    }
