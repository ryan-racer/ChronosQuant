"""Assemble the full quantization campaign into the paper's master tables.

Reads every completed ``results/raw/*_full`` run, computes accuracy retention and
calibration metrics vs the fp32 reference (with bootstrap CIs, fev-bench convention),
merges in effective bits/weight and storage, and emits the method x bit-width matrix
(paper Tables 2-3) as a tidy CSV + a readable console table.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import yaml

from chronosquant.utils import REPO_ROOT

RAW = REPO_ROOT / "results" / "raw"
REFERENCE = "chronos2-fp32"
MIN_REL, MAX_REL = 1e-2, 1e2
N_RESAMPLES = 1000


def _gmean_ci(ratios: np.ndarray, seed: int = 123) -> tuple[float, float, float]:
    point = float(scipy.stats.gmean(ratios))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ratios), (N_RESAMPLES, len(ratios)))
    boot = scipy.stats.gmean(ratios[idx], axis=1)
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _completed_runs() -> dict[str, Path]:
    runs = {}
    for run_dir in sorted(RAW.glob("*_full")):
        meta = run_dir / "run_metadata.yaml"
        if not meta.exists():
            continue
        try:
            m = yaml.safe_load(meta.read_text()) or {}
        except Exception:
            continue
        if not m.get("num_completed"):
            continue
        df = pd.read_csv(run_dir / "summaries.csv")
        runs[df["model_name"].iloc[0]] = run_dir
    return runs


def build_campaign_table() -> pd.DataFrame:
    runs = _completed_runs()
    if REFERENCE not in runs:
        raise RuntimeError(f"Reference run {REFERENCE} not found among {sorted(runs)}")
    fp32 = pd.read_csv(runs[REFERENCE] / "summaries.csv").set_index("task_name")

    rows = []
    for model_name, run_dir in runs.items():
        if model_name in (REFERENCE, "seasonal_naive"):
            continue
        df = pd.read_csv(run_dir / "summaries.csv").set_index("task_name")
        common = fp32.index.intersection(df.index)
        wql = (df.loc[common, "WQL"] / fp32.loc[common, "WQL"]).clip(MIN_REL, MAX_REL).to_numpy()
        mase = (df.loc[common, "MASE"] / fp32.loc[common, "MASE"]).clip(MIN_REL, MAX_REL).to_numpy()
        wql_ret, wql_lo, wql_hi = _gmean_ci(wql)
        mase_ret, _, _ = _gmean_ci(mase)

        def col(name, default=None):
            return df[name].iloc[0] if name in df.columns else default

        bits = col("quant_weight_bits")
        bpw = col("quant_effective_bits_per_weight")
        if bpw is None or (isinstance(bpw, float) and np.isnan(bpw)):
            bpw = col("quant_achievable_bits_per_weight")
        rows.append(
            {
                "variant": model_name.replace("chronos2-", ""),
                "method": col("quant_method"),
                "weight_bits": bits,
                "eff_bpw": bpw,
                "size_MB": round(col("model_total_bytes", np.nan) / 1e6, 1)
                if col("model_total_bytes")
                else np.nan,
                "WQL_ret": round(wql_ret, 4),
                "WQL_lo": round(wql_lo, 4),
                "WQL_hi": round(wql_hi, 4),
                "MASE_ret": round(mase_ret, 4),
                "worst_task_WQL": round(float(wql.max()), 3),
                "win_vs_fp32": round(float((wql < 1).mean() + 0.5 * (wql == 1).mean()), 3),
                "QCR": round(float(df["QCR"].mean()), 4),
                "cov80": round(float(df["coverage[0.8]"].mean()), 4),
                "MACE": round(float(df["MACE"].mean()), 4),
                "parity_with_fp32": bool(wql_lo <= 1.0 <= wql_hi),
            }
        )

    table = pd.DataFrame(rows).sort_values(["WQL_ret"]).reset_index(drop=True)
    # append the fp32 reference row for context
    ref_row = {
        "variant": "fp32 (ref)",
        "method": "none",
        "weight_bits": 32,
        "eff_bpw": 32.0,
        "size_MB": round(fp32["model_total_bytes"].iloc[0] / 1e6, 1)
        if "model_total_bytes" in fp32.columns
        else np.nan,
        "WQL_ret": 1.0,
        "WQL_lo": 1.0,
        "WQL_hi": 1.0,
        "MASE_ret": 1.0,
        "worst_task_WQL": 1.0,
        "win_vs_fp32": np.nan,
        "QCR": round(float(fp32["QCR"].mean()), 4),
        "cov80": round(float(fp32["coverage[0.8]"].mean()), 4),
        "MACE": round(float(fp32["MACE"].mean()), 4),
        "parity_with_fp32": True,
    }
    return pd.concat([table, pd.DataFrame([ref_row])], ignore_index=True)


def main() -> None:
    table = build_campaign_table()
    out = REPO_ROOT / "results" / "tables" / "campaign_master.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(table.to_string(index=False))
    print(f"\nMaster table -> {out}")


if __name__ == "__main__":
    main()
