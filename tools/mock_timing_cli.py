def add_timing_backend_args(parser):
    parser.add_argument("--timing-backend", choices=["parametric", "analytical"], default="parametric")
    parser.add_argument("--analytical-model", default="openai/gpt-oss-120b")
    parser.add_argument("--analytical-hardware", default="helios")
    parser.add_argument("--analytical-interconnect", default="")
    parser.add_argument("--analytical-collective-overhead-us", type=float, default=None)
    parser.add_argument("--analytical-send-recv-overhead-us", type=float, default=None)
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
        "roofline_gpu_backend": args.roofline_gpu_backend,
        "roofline_tp_g": args.roofline_tp_g,
        "attention_groups": args.attention_groups,
        "chunk_batch": args.chunk_batch,
        "gpu_cs_link_us": args.gpu_cs_link_us,
    }
