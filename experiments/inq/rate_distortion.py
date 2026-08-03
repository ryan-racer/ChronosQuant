"""Offline rate-distortion analysis on captured Chronos-2 K/V tensors.

Outputs (in experiments/inq/):
- rd_table.csv        : relative MSE per (dataset, tensor, method, bits), averaged
                        over layers, plus the equivalent-bits gain of InQ over the
                        KIVI-style per-channel baseline and the gain predicted by
                        Thm. 4.3 from measured lag-1 autocorrelation.
- rd_position.csv     : per-position squared error (P4: closed vs open loop).
- rd_conjugation.csv  : InQ on pre-RoPE vs post-RoPE keys (Prop. 3.3 ablation).

Run: uv run python experiments/inq/rate_distortion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from inq_kv import kv_inq, kv_open_delta, kv_rtn_channel, kv_rtn_token  # noqa: E402

HERE = Path(__file__).parent
DATASETS = ["ercot", "exchange_rate", "monash_covid_deaths", "monash_hospital", "monash_m1_quarterly"]
BITS = (2, 3, 4)

METHODS = {
    "rtn_channel": lambda x, b: kv_rtn_channel(x, b),
    "rtn_token": lambda x, b: kv_rtn_token(x, b),
    "open_delta64": lambda x, b: kv_open_delta(x, b, anchor_every=64),
    "inq": lambda x, b: kv_inq(x, b),
}


def rel_mse(x: torch.Tensor, xhat: torch.Tensor) -> float:
    return (((x - xhat) ** 2).mean() / x.var()).item()


def lag1_median(x: torch.Tensor) -> float:
    xc = x - x.mean(dim=2, keepdim=True)
    num = (xc[:, :, :-1] * xc[:, :, 1:]).sum(dim=2)
    den = (xc * xc).sum(dim=2).clamp_min(1e-12)
    return (num / den).median().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rd_rows, pos_rows, conj_rows = [], [], []

    for ds in DATASETS:
        cap = torch.load(HERE / "captures" / f"{ds}.pt", weights_only=False)
        layers = cap["layers"]
        for tensor_name in ("k_pre_rope", "v"):
            per_method: dict[tuple[str, int], list[float]] = {}
            rho_all = []
            for layer, tensors in layers.items():
                x = tensors[tensor_name].float().to(device)
                rho_all.append(lag1_median(x))
                for method, fn in METHODS.items():
                    for bits in BITS:
                        per_method.setdefault((method, bits), []).append(rel_mse(x, fn(x, bits)))
            rho = float(np.median(rho_all))
            for (method, bits), vals in per_method.items():
                rd_rows.append(
                    {
                        "dataset": ds,
                        "tensor": tensor_name,
                        "method": method,
                        "bits": bits,
                        "rel_mse": float(np.mean(vals)),
                        "rho_median": rho,
                    }
                )

        # P4: per-position error, closed vs open loop (single anchor at t=0)
        if len(layers) and next(iter(layers.values()))["k_pre_rope"].shape[2] >= 64:
            for method_name, fn in (
                ("inq_closed", lambda x: kv_inq(x, 3, anchor_every=None)),
                ("open_loop", lambda x: kv_open_delta(x, 3, anchor_every=None)),
            ):
                errs = []
                for layer, tensors in layers.items():
                    x = tensors["k_pre_rope"].float().to(device)
                    e = ((x - fn(x)) ** 2).mean(dim=(0, 1, 3)) / x.var()
                    errs.append(e.cpu())
                e = torch.stack(errs).mean(dim=0)
                for t, v in enumerate(e.tolist()):
                    pos_rows.append({"dataset": ds, "method": method_name, "position": t, "rel_sq_err": v})

        # Prop 3.3 ablation: InQ on pre- vs post-RoPE keys at 3 bits
        for frame in ("k_pre_rope", "k_post_rope"):
            vals = [
                rel_mse(t[frame].float().to(device), kv_inq(t[frame].float().to(device), 3))
                for t in layers.values()
            ]
            conj_rows.append({"dataset": ds, "frame": frame, "bits": 3, "rel_mse": float(np.mean(vals))})

    rd = pd.DataFrame(rd_rows)
    # equivalent-bits gain of inq over the per-channel baseline + theory prediction
    piv = rd.pivot_table(index=["dataset", "tensor", "bits", "rho_median"], columns="method", values="rel_mse").reset_index()
    piv["gain_bits_measured"] = 0.5 * np.log2(piv["rtn_channel"] / piv["inq"])
    piv["gain_bits_theory_lag1"] = -0.5 * np.log2(np.clip(1 - piv["rho_median"] ** 2, 1e-6, None))
    rd.to_csv(HERE / "rd_table.csv", index=False)
    piv.to_csv(HERE / "rd_gains.csv", index=False)
    pd.DataFrame(pos_rows).to_csv(HERE / "rd_position.csv", index=False)
    pd.DataFrame(conj_rows).to_csv(HERE / "rd_conjugation.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== relative MSE (mean over layers), and equivalent-bits gain of InQ vs per-channel RTN ===")
    print(piv.round(4).to_string(index=False))
    print("\n=== Prop 3.3 ablation: InQ@3b on pre- vs post-RoPE keys ===")
    print(pd.DataFrame(conj_rows).pivot_table(index="dataset", columns="frame", values="rel_mse").round(4))


if __name__ == "__main__":
    main()
