"""Downstream dev-tier evaluation of KV-activation quantization on Chronos-2.

Arms: {rtn_channel (KIVI-style baseline), inq (closed-loop DPCM)} x {2, 3, 4} bits,
applied to K (pre-RoPE) and V of all 12 TimeSelfAttention modules. Group
attention stays fp32 in all arms. Scored on the same 8 dev tasks as the repo's
weight-quantization sweeps; retention is computed against chronos2-fp32.

Run: uv run python experiments/inq/run_eval.py [--overwrite]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inq_kv import make_pipeline_transform  # noqa: E402

from chronosquant.evaluation.predictors import Chronos2Predictor
from chronosquant.evaluation.runner import RunConfig, run_evaluation
from chronosquant.utils import REPO_ROOT

BENCHMARK = "configs/evaluation/benchmarks/chronos_zeroshot.yaml"
DEV_TASKS = [
    "monash_cif_2016",
    "monash_m1_quarterly",
    "monash_tourism_quarterly",
    "monash_covid_deaths",
    "exchange_rate",
    "ercot",
    "monash_hospital",
    "monash_nn5_weekly",
]
ARMS = [(method, bits) for method in ("rtn_channel", "inq") for bits in (4, 3, 2)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for method, bits in ARMS:
        name = f"chronos2_kv{bits}_{method}_dev"
        out_dir = REPO_ROOT / "results" / "raw" / name
        if out_dir.exists() and not args.overwrite:
            print(f"skip {name} (exists)")
            continue
        predictor = Chronos2Predictor(
            name=f"chronos2-kv{bits}-{method}",
            device_map="cuda",
            batch_size=128,
            model_transform=make_pipeline_transform(method, bits),
        )
        config = RunConfig(
            name=name,
            benchmark=BENCHMARK,
            predictor={
                "type": "chronos2",
                "name": predictor.name,
                "kv_quant": {"method": method, "bits": bits, "scope": "time_attention_k_pre_rope_and_v"},
            },
            task_names=DEV_TASKS,
            overwrite=True,
            notes=f"InQ experiment: simulated {bits}-bit {method} on time-attention K/V",
        )
        run_evaluation(config, predictor=predictor)


if __name__ == "__main__":
    main()
