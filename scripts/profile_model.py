"""Profile a model variant's inference efficiency (P1 protocol).

Usage:
    # Profile the fp32 model on GPU and CPU
    uv run python scripts/profile.py --name chronos2-fp32 --devices cuda cpu

    # Profile the predictor from a run config (reuses quantization transforms later)
    uv run python scripts/profile.py --config configs/evaluation/runs/chronos2_fp32_full.yaml \\
        --devices cuda

Writes results/profiles/<name>_<device>.json
"""

import argparse

import yaml

from chronosquant.evaluation.predictors import Chronos2Predictor, build_predictor
from chronosquant.evaluation.profiler import ProfilerConfig, profile_predictor, save_profile
from chronosquant.utils import ensure_truststore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Run config YAML whose predictor section to profile")
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--name", default="chronos2-fp32")
    parser.add_argument("--devices", nargs="+", default=["cuda"], choices=["cuda", "cpu"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32, 256])
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--output-dir", default="results/profiles")
    args = parser.parse_args()

    ensure_truststore()
    profiler_config = ProfilerConfig(
        context_length=args.context_length,
        horizon=args.horizon,
        batch_sizes=tuple(args.batch_sizes),
    )

    for device in args.devices:
        if args.config:
            with open(args.config) as f:
                predictor_spec = dict(yaml.safe_load(f)["predictor"])
            predictor_spec["device_map"] = device
            predictor = build_predictor(predictor_spec)
            assert isinstance(predictor, Chronos2Predictor), "profiling requires a torch pipeline"
        else:
            predictor = Chronos2Predictor(
                model_id=args.model_id, name=args.name, device_map=device
            )
        print(f"Profiling {predictor.name} on {device} (batches {args.batch_sizes}) ...")
        profile = profile_predictor(predictor, profiler_config)
        path = save_profile(profile, args.output_dir)
        for batch, stats in profile["latency"].items():
            print(
                f"  batch {batch:>4}: median {stats['median_s'] * 1e3:8.1f} ms | "
                f"p90 {stats['p90_s'] * 1e3:8.1f} ms | "
                f"{stats['throughput_series_per_s']:8.1f} series/s"
                + (
                    f" | peak GPU {stats['peak_gpu_memory_mb']:.0f} MB"
                    if "peak_gpu_memory_mb" in stats
                    else ""
                )
            )
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
