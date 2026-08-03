"""RQ3 module-family sensitivity study (dual-metric: accuracy vs calibration).

Defines the module-family taxonomy over the *quantized set* (all 126 nn.Linear modules
of Chronos-2 — the default in every existing run, including the input patch embedding
and the output/quantile head), builds leave-one-out sweep arms, and computes the two
cheap diagnostics that the empirical ranking is compared against:

- **Hessian sensitivity** (HAWQ-V2-style, but *exact* for the layer-reconstruction
  objective): with H = E[x x^T] collected on calibration data, the expected layer
  output MSE of a weight perturbation ΔW is exactly ``tr(ΔW H ΔW^T)`` — no Hutchinson
  sampling needed since the objective ``||ΔW x||²`` is quadratic and H is materialized
  in full by `LinearInputStats`. We report this exact value for the actual W4 RTN
  perturbation, plus the diagonal-uniform HAWQ proxy ``(tr(H)/d_in)·||ΔW||_F²``.
- **Activation-outlier statistics** per module input (channel max/mean magnitude
  ratios, kurtosis, LLM.int8-threshold channel fraction) — tests whether Chronos-2
  has LLM-style activation outliers at all.

Families partition the quantized linear set exactly (validated in tests and asserted
at analysis time against the real model). Sweep arms are plain ``rtn`` simulate-mode
configs driven entirely by ``skip_modules`` glob patterns — no changes to transforms.

CLI (module owner: sensitivity study; does not touch transforms/gptq/awq/calibration):

    uv run python -m chronosquant.quantization.sensitivity analyze
        -> results/tables/sensitivity_hessian.csv + sensitivity_outliers.csv
    uv run python -m chronosquant.quantization.sensitivity report
        -> results/tables/sensitivity_dev.csv (retention + cov80/QCR per arm)
"""

import argparse
import fnmatch
import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

#: Family name -> glob patterns over dotted module paths. Together the families
#: partition the full set of quantized Linears (all 126 in Chronos-2) exactly.
MODULE_FAMILIES: dict[str, list[str]] = {
    # residual patch-embedding input stack (48->3072->768 + 48->768 residual)
    "patch_embed": ["input_patch_embedding.*"],
    # layer.0 of every block is TimeSelfAttention (q, k, v, o)
    "time_attn": ["encoder.block.*.layer.0.self_attention.*"],
    # layer.1 of every block is GroupSelfAttention (q, k, v, o)
    "group_attn": ["encoder.block.*.layer.1.self_attention.*"],
    # layer.2 of every block is the ReLU FFN (wi, wo)
    "ffn": ["encoder.block.*.layer.2.mlp.*"],
    # output/quantile head: 768->3072->336 (16 output patches x 21 quantiles) + residual
    "head": ["output_patch_embedding.*"],
}

#: Extra positional groups (cut across families; protect-direction arms only).
EXTRA_GROUPS: dict[str, list[str]] = {
    "first_block": ["encoder.block.0.layer.*"],
    "last_block": ["encoder.block.11.layer.*"],
}

#: Expected per-family module counts for the real Chronos-2 (12 blocks).
EXPECTED_FAMILY_COUNTS = {
    "patch_embed": 3,
    "time_attn": 48,
    "group_attn": 48,
    "ffn": 24,
    "head": 3,
}


def _matches_any(name: str, patterns: list[str]) -> bool:
    """Same semantics as transforms._matches_any (glob or substring), kept local so the
    sensitivity module has no private-symbol dependency on transforms."""
    return any(fnmatch.fnmatch(name, p) or p in name for p in patterns)


def linear_module_names(model: nn.Module) -> list[str]:
    """Dotted paths of all nn.Linear modules — the quantized set of every default run."""
    return [name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]


def classify_module(name: str) -> str | None:
    """Family of a module path, or None if it matches no family."""
    matches = [fam for fam, pats in MODULE_FAMILIES.items() if _matches_any(name, pats)]
    if len(matches) > 1:
        raise ValueError(f"Module {name!r} matches multiple families: {matches}")
    return matches[0] if matches else None


def family_partition(names: list[str]) -> dict[str, list[str]]:
    """Partition module names into families; raises if any name is unclassified
    (gap) — overlap is raised by `classify_module`."""
    partition: dict[str, list[str]] = {fam: [] for fam in MODULE_FAMILIES}
    unclassified = []
    for name in names:
        fam = classify_module(name)
        if fam is None:
            unclassified.append(name)
        else:
            partition[fam].append(name)
    if unclassified:
        raise ValueError(f"Modules not covered by any family: {unclassified[:10]}")
    return partition


def skip_patterns_protect(family: str) -> list[str]:
    """PROTECT-ONE: quantize everything except `family` -> skip = family's patterns."""
    groups = {**MODULE_FAMILIES, **EXTRA_GROUPS}
    return list(groups[family])


def skip_patterns_only(family: str) -> list[str]:
    """ONLY-ONE: quantize only `family` -> skip = union of all other families."""
    if family not in MODULE_FAMILIES:
        raise ValueError(f"only-one arms are defined for families, got {family!r}")
    return [p for fam, pats in MODULE_FAMILIES.items() if fam != family for p in pats]


def build_sensitivity_variants(bits: int = 4) -> list[dict[str, Any]]:
    """The sweep arms, mirrored 1:1 by configs/evaluation/sweeps/sensitivity.yaml
    (a test asserts the YAML matches this builder)."""

    def quant(skip: list[str] | None = None) -> dict[str, Any]:
        cfg: dict[str, Any] = {"method": "rtn", "bits": bits, "simulate": True}
        if skip:
            cfg["skip_modules"] = skip
        return cfg

    def arm_name(direction: str, family: str | None = None) -> str:
        parts = ["chronos2-sens", f"w{bits}", direction]
        if family:
            parts.append(family.replace("_", "-"))
        return "-".join(parts)

    variants: list[dict[str, Any]] = [{"name": arm_name("all"), "quantization": quant()}]
    for fam in MODULE_FAMILIES:
        variants.append(
            {"name": arm_name("protect", fam), "quantization": quant(skip_patterns_protect(fam))}
        )
    for fam in MODULE_FAMILIES:
        variants.append(
            {"name": arm_name("only", fam), "quantization": quant(skip_patterns_only(fam))}
        )
    for grp in EXTRA_GROUPS:
        variants.append(
            {"name": arm_name("protect", grp), "quantization": quant(skip_patterns_protect(grp))}
        )
    return variants


def parse_arm_name(name: str) -> tuple[str, str]:
    """'chronos2-sens-w4-protect-group-attn' -> ('protect', 'group_attn')."""
    tokens = name.split("-")
    if len(tokens) < 4 or tokens[0] != "chronos2" or tokens[1] != "sens":
        raise ValueError(f"Not a sensitivity arm name: {name!r}")
    direction = tokens[3]
    family = "_".join(tokens[4:]) if len(tokens) > 4 else ""
    return direction, family


# ---------------------------------------------------------------------------
# Hessian sensitivity (exact layer-reconstruction curvature) + outlier stats
# ---------------------------------------------------------------------------


@torch.no_grad()
def _rtn_perturbation(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """ΔW = W - dequant(quant(W)) for symmetric per-channel RTN (matches transforms)."""
    q_max = 2 ** (bits - 1) - 1
    max_abs = weight.abs().amax(dim=1, keepdim=True)
    scale = (max_abs / q_max).clamp(min=torch.finfo(torch.float32).tiny)
    weight_dq = torch.round(weight / scale).clamp(-q_max, q_max) * scale
    return weight - weight_dq


@torch.no_grad()
def module_hessian_row(name: str, linear: nn.Linear, stats, bits: int = 4) -> dict[str, Any]:
    """Exact expected layer-output MSE of the W4 RTN perturbation under H = E[x x^T].

    `stats` is a `LinearInputStats` (calibration.py) holding sum(x x^T) and n_rows.
    """
    weight = linear.weight.detach().to(torch.float32)
    hessian_mean = stats.hessian.to(weight.device) / max(stats.n_rows, 1)  # E[x x^T]
    delta = _rtn_perturbation(weight, bits)
    # tr(ΔW H ΔW^T): exact expected ||ΔW x||² per token (the layer-recon objective)
    exact_mse = float(((delta @ hessian_mean) * delta).sum())
    out_energy = float(((weight @ hessian_mean) * weight).sum())  # tr(W H W^T)
    trace_h = float(torch.diagonal(hessian_mean).sum())  # E[||x||²]
    delta_fro_sq = float((delta * delta).sum())
    row = {
        "module": name,
        "family": classify_module(name),
        "in_features": linear.in_features,
        "out_features": linear.out_features,
        "n_calib_rows": int(stats.n_rows),
        "bits": bits,
        "trace_H": trace_h,  # E[||x||^2] — input energy
        "weight_fro_sq": float((weight * weight).sum()),
        "delta_fro_sq": delta_fro_sq,  # ||ΔW||_F^2 (pure weight-space RTN error)
        "exact_mse": exact_mse,  # tr(ΔW H ΔW^T) — exact expected output MSE
        "rel_recon_err": exact_mse / max(out_energy, 1e-30),
        "hawq_proxy": trace_h / linear.in_features * delta_fro_sq,
    }
    return row


@torch.no_grad()
def module_outlier_row(name: str, stats) -> dict[str, Any]:
    """Per-input-channel activation magnitude stats from calibration samples."""
    sample = stats.sample_matrix.to(torch.float32)  # [n_rows, in_features]
    if sample.numel() == 0:
        raise ValueError(f"No activation samples captured for {name}")
    abs_sample = sample.abs()
    ch_mean = abs_sample.mean(dim=0)
    ch_max = abs_sample.amax(dim=0)
    # Per-channel Fisher kurtosis over samples (the LLM-outlier-relevant axis);
    # pooled kurtosis over the flattened sample conflates ReLU zero-mass and
    # cross-channel scale spread with within-channel heavy tails, so it is kept
    # only under an explicit "pooled" name.
    ch_centered = sample - sample.mean(dim=0, keepdim=True)
    ch_var = ch_centered.pow(2).mean(dim=0)
    live = ch_var > 1e-20  # kurtosis is undefined for constant (ReLU-dead) channels
    ch_kurt = (
        ch_centered[:, live].pow(4).mean(dim=0) / ch_var[live].pow(2) - 3.0
        if bool(live.any())
        else torch.full((1,), float("nan"))
    )
    flat = sample.reshape(-1)
    centered = flat - flat.mean()
    var = centered.pow(2).mean().clamp(min=1e-30)
    kurtosis_pooled = float(centered.pow(4).mean() / var.pow(2)) - 3.0
    ch_max_median = float(ch_max.median())
    return {
        "module": name,
        "family": classify_module(name),
        "in_features": int(sample.shape[1]),
        "n_sample_rows": int(sample.shape[0]),
        # LLM outlier signatures: a few channels with magnitudes >> the rest
        "ch_absmean_max_over_mean": float(ch_mean.amax() / ch_mean.mean().clamp(min=1e-30)),
        # Raw median so a ~0 median (ReLU sparsity) is visible instead of producing
        # a clamp-artifact ratio; the ratio is only meaningful when median > 0.
        "ch_absmax_median": ch_max_median,
        "ch_absmax_max_over_median": (
            float(ch_max.amax() / ch_max.median()) if ch_max_median > 1e-12 else float("nan")
        ),
        "frac_channels_absmax_gt6": float((ch_max > 6.0).float().mean()),
        "global_absmax": float(ch_max.amax()),
        "kurtosis_ch_max": float(ch_kurt.amax()),
        "kurtosis_ch_median": float(ch_kurt.median()),
        "kurtosis_pooled": kurtosis_pooled,
    }


# ---------------------------------------------------------------------------
# CLI: analyze (hessian + outliers, one calibration pass) and report (dev table)
# ---------------------------------------------------------------------------


def _cmd_analyze(args: argparse.Namespace) -> None:
    import pandas as pd

    from chronosquant.utils import REPO_ROOT, ensure_truststore

    ensure_truststore()
    from chronos import Chronos2Pipeline

    from chronosquant.quantization.calibration import collect_linear_input_stats

    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=args.device)
    model = pipeline.model
    names = linear_module_names(model)
    partition = family_partition(names)  # raises on gaps/overlaps vs the real model
    counts = {fam: len(mods) for fam, mods in partition.items()}
    logger.info("Family partition over %d linears: %s", len(names), counts)
    if counts != EXPECTED_FAMILY_COUNTS:
        raise AssertionError(f"Unexpected family counts {counts} != {EXPECTED_FAMILY_COUNTS}")

    logger.info("Collecting calibration input stats (n=%d, ctx=%d)", args.n_series, args.context)
    stats = collect_linear_input_stats(
        pipeline, names, n_series=args.n_series, context_length=args.context, seed=args.seed
    )

    modules = dict(model.named_modules())
    hessian_rows = [
        module_hessian_row(name, modules[name], stats[name], bits=args.bits) for name in names
    ]
    outlier_rows = [module_outlier_row(name, stats[name]) for name in names]

    tables_dir = REPO_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    hessian_df = pd.DataFrame(hessian_rows).sort_values("exact_mse", ascending=False)
    outlier_df = pd.DataFrame(outlier_rows).sort_values(
        "ch_absmean_max_over_mean", ascending=False
    )
    hessian_df.to_csv(tables_dir / "sensitivity_hessian.csv", index=False)
    outlier_df.to_csv(tables_dir / "sensitivity_outliers.csv", index=False)
    pd.set_option("display.width", 200)
    # Family-level sums are CONVENTION-DEPENDENT: exact_mse is per token-row, but
    # rows-per-series differ ~17x across modules (the head sees 1 row per series,
    # encoder modules one per patch). Both conventions are printed; neither ranks
    # the head first — only the per-module view (rel_recon_err) is robust, which
    # is itself a finding: cheap local diagnostics under-flag the head family.
    hessian_df["exact_mse_rowweighted"] = hessian_df["exact_mse"] * hessian_df["n_calib_rows"]
    print("\nPer-family Hessian sensitivity (per-row sums | row-weighted sums):")
    print(
        hessian_df.groupby("family")[["exact_mse", "exact_mse_rowweighted", "hawq_proxy"]]
        .sum()
        .sort_values("exact_mse", ascending=False)
        .to_string()
    )
    print("\nTop-10 modules by exact expected output MSE:")
    print(hessian_df.head(10)[["module", "family", "exact_mse", "rel_recon_err"]].to_string())
    print("\nPer-family outlier stats (means over modules):")
    print(
        pd.DataFrame(outlier_rows)
        .groupby("family")[
            ["ch_absmean_max_over_mean", "ch_absmax_median", "kurtosis_ch_max",
             "kurtosis_pooled", "frac_channels_absmax_gt6", "global_absmax"]
        ]
        .mean()
        .to_string()
    )
    print(f"\nSaved {tables_dir / 'sensitivity_hessian.csv'}")
    print(f"Saved {tables_dir / 'sensitivity_outliers.csv'}")


def _cmd_report(args: argparse.Namespace) -> None:
    import pandas as pd

    from chronosquant.analysis import load_summaries, retention_table
    from chronosquant.utils import REPO_ROOT

    raw = REPO_ROOT / "results" / "raw"
    run_dirs = sorted(raw.glob(f"chronos2_sens_*_{args.tier}"))
    if not run_dirs:
        raise SystemExit(f"No sensitivity runs found under {raw} for tier {args.tier!r}")
    reference_dir = raw / f"chronos2_fp32_{args.tier}"
    summaries = load_summaries([str(reference_dir)] + [str(d) for d in run_dirs])

    retention = retention_table(summaries, reference_model="chronos2-fp32")
    calib = (
        summaries.groupby("model_name")[["coverage[0.8]", "QCR", "MACE"]]
        .mean()
        .rename(columns={"coverage[0.8]": "cov80"})
    )
    table = retention.join(calib, how="outer")
    # fp32 reference row: retention 1.0 by definition; keep its calibration baselines
    fp32_calib = summaries[summaries["model_name"] == "chronos2-fp32"][
        ["coverage[0.8]", "QCR", "MACE"]
    ].mean()
    table.loc["chronos2-fp32", ["cov80", "QCR", "MACE"]] = fp32_calib.to_numpy()

    directions, families = [], []
    for model_name in table.index:
        try:
            direction, family = parse_arm_name(model_name)
        except ValueError:
            direction, family = "reference", ""
        directions.append(direction)
        families.append(family)
    table.insert(0, "direction", directions)
    table.insert(1, "family", families)
    table = table.sort_values(["direction", "WQL_rel"])

    tables_dir = REPO_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / f"sensitivity_{args.tier}.csv"
    table.to_csv(out_path)
    pd.set_option("display.width", 250)
    keep = [
        "direction", "family", "WQL_rel", "WQL_rel_lower", "WQL_rel_upper",
        "MASE_rel", "cov80", "QCR", "MACE",
    ]
    print("\n" + table[[c for c in keep if c in table.columns]].round(4).to_string())
    print(f"\nSaved {out_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="Hessian traces + activation outlier stats")
    p_an.add_argument("--model-id", default="amazon/chronos-2")
    p_an.add_argument("--device", default="cuda")
    p_an.add_argument("--bits", type=int, default=4)
    p_an.add_argument("--n-series", type=int, default=128)
    p_an.add_argument("--context", type=int, default=1024)
    p_an.add_argument("--seed", type=int, default=123)
    p_an.set_defaults(func=_cmd_analyze)

    p_rp = sub.add_parser("report", help="Aggregate sensitivity sweep runs into a table")
    p_rp.add_argument("--tier", choices=["dev", "full"], default="dev")
    p_rp.set_defaults(func=_cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
