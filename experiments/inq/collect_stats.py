"""Capture Chronos-2 time-attention K/V tensors on real benchmark data and
measure the statistics the InQ proposal's predictions P1/P5 depend on.

For each dataset: run one batch of equal-length contexts (no padding, so the
patch-time axis is clean) through the fp32 pipeline with hooks on every
time-attention k/v projection, and save:

- captures/<dataset>.pt : per-layer pre-RoPE K, post-RoPE K, V ([b,h,s,d], fp16)
  plus metadata (n_ctx patches, horizon patches).
- stats.csv : per (dataset, layer, tensor) distribution of per-(series, head,
  channel) lag-1 autocorrelation (centered, context positions only) and spectral
  flatness — including the GroupSelfAttention control, whose attention axis is
  the unordered series axis (the proposal predicts structure on the time axis
  and none there).

Run: uv run python experiments/inq/collect_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from chronos.chronos2.layers import GroupSelfAttention, TimeSelfAttention

from chronosquant.evaluation.predictors import Predictor
from chronosquant.evaluation.runner import load_benchmark_tasks
from chronosquant.utils import ensure_truststore

HERE = Path(__file__).parent
BENCHMARK = "configs/evaluation/benchmarks/chronos_zeroshot.yaml"
DATASETS = ["ercot", "exchange_rate", "monash_hospital", "monash_covid_deaths", "monash_m1_quarterly"]
MAX_SERIES = 12
MAX_LEN = 2048


def get_contexts(task, max_series: int, max_len: int, patch: int):
    task.load_full_dataset(num_proc=1)
    window = next(iter(task.iter_windows(num_proc=1)))
    contexts = Predictor._extract_contexts(window, task)
    contexts = sorted(contexts, key=len, reverse=True)[:max_series]
    L = min(max_len, min(len(c) for c in contexts))
    L -= L % patch
    return [torch.as_tensor(c[-L:], dtype=torch.float32) for c in contexts], L


def channel_lag1_autocorr(x: torch.Tensor) -> torch.Tensor:
    """Centered lag-1 autocorrelation per (b, h, d) along dim=2."""
    x = x - x.mean(dim=2, keepdim=True)
    num = (x[:, :, :-1] * x[:, :, 1:]).sum(dim=2)
    den = (x * x).sum(dim=2).clamp_min(1e-12)
    return num / den


def channel_sfm(x: torch.Tensor) -> torch.Tensor:
    """Spectral flatness (GM/AM of the periodogram) per (b, h, d) along dim=2."""
    x = x - x.mean(dim=2, keepdim=True)
    spec = torch.fft.rfft(x.float(), dim=2).abs() ** 2
    spec = spec[:, :, 1:]  # drop DC (zero after centering)
    spec = spec.clamp_min(1e-12)
    gm = spec.log().mean(dim=2).exp()
    am = spec.mean(dim=2)
    return gm / am


def summarize(name_parts: dict, rho: torch.Tensor, sfm: torch.Tensor | None) -> dict:
    rho = rho.flatten().float()
    row = dict(name_parts)
    row.update(
        {
            "rho_median": rho.median().item(),
            "rho_q25": rho.quantile(0.25).item(),
            "rho_q75": rho.quantile(0.75).item(),
            "rho_frac_gt_0.8": (rho > 0.8).float().mean().item(),
        }
    )
    if sfm is not None:
        s = sfm.flatten().float()
        row.update(sfm_median=s.median().item(), bits_saved_szego=float(-0.5 * np.log2(max(s.median().item(), 1e-12))))
    return row


def main():
    ensure_truststore()
    from chronos import Chronos2Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    model = pipeline.model
    model.eval()
    patch = model.chronos_config.input_patch_size
    print(f"input_patch_size={patch}, device={device}")

    # map projection module -> (layer_idx, kind, which)
    hooks_meta = {}
    for name, module in model.named_modules():
        if isinstance(module, (TimeSelfAttention, GroupSelfAttention)):
            kind = "time" if isinstance(module, TimeSelfAttention) else "group"
            layer_idx = int(name.split(".block.")[1].split(".")[0])
            mha = module.self_attention
            for which in ("k", "v"):
                hooks_meta[getattr(mha, which)] = (layer_idx, kind, which, mha)

    captured: dict = {}
    handles = []

    def make_hook(meta):
        layer_idx, kind, which, mha = meta

        def hook(module, inputs, output):
            key = (layer_idx, kind, which)
            if key in captured:
                return
            if kind == "time":
                # [b, s, h*d] -> [b, h, s, d]; seq axis = patch time
                t = output.reshape(*output.shape[:2], mha.n_heads, mha.kv_proj_dim).permute(0, 2, 1, 3)
            else:
                # group attention input is [time, batch, d_model]; attention seq
                # axis is dim 1 (series in batch) -> [time, h, batch, d]
                t = output.reshape(*output.shape[:2], mha.n_heads, mha.kv_proj_dim).permute(0, 2, 1, 3)
            captured[key] = t.detach()

        return hook

    for module, meta in hooks_meta.items():
        handles.append(module.register_forward_hook(make_hook(meta)))

    tasks = {t.task_name: t for t in load_benchmark_tasks(BENCHMARK, DATASETS)}
    (HERE / "captures").mkdir(parents=True, exist_ok=True)
    rows = []

    for ds in DATASETS:
        task = tasks[ds]
        contexts, L = get_contexts(task, MAX_SERIES, MAX_LEN, patch)
        n_ctx = L // patch
        captured.clear()
        with torch.no_grad():
            pipeline.predict_quantiles(
                contexts, prediction_length=task.horizon, quantile_levels=[0.5], batch_size=len(contexts)
            )
        print(f"{ds}: {len(contexts)} series, context {L} steps = {n_ctx} patches")

        n_layers = model.config.num_layers
        save = {"n_ctx": n_ctx, "horizon": task.horizon, "layers": {}}
        for layer in range(n_layers):
            k_pre = captured[(layer, "time", "k")]
            v = captured[(layer, "time", "v")]
            s_total = k_pre.shape[2]
            # post-RoPE keys (as an LLM-style method would see them)
            mha = model.encoder.block[layer].layer[0].self_attention
            pos = torch.arange(s_total, device=k_pre.device).unsqueeze(0)
            cos, sin = mha.rope_embed(k_pre, pos)
            from chronos.chronos2.layers import Chronos2RotaryEmbedding

            _, k_rope = Chronos2RotaryEmbedding.apply_rotary_pos_emb(k_pre, k_pre, cos, sin)

            ctx = slice(0, n_ctx)  # context positions only for the statistics
            for tname, tensor in (("k_pre_rope", k_pre), ("k_post_rope", k_rope), ("v", v)):
                xc = tensor[:, :, ctx]
                rows.append(
                    summarize(
                        {"dataset": ds, "layer": layer, "tensor": tname, "axis": "patch_time"},
                        channel_lag1_autocorr(xc),
                        channel_sfm(xc),
                    )
                )
            # group-attention control: autocorr along the (unordered) series axis
            gk = captured[(layer, "group", "k")]
            rows.append(
                summarize(
                    {"dataset": ds, "layer": layer, "tensor": "group_k", "axis": "series"},
                    channel_lag1_autocorr(gk),
                    None,
                )
            )
            save["layers"][layer] = {
                "k_pre_rope": k_pre.half().cpu(),
                "k_post_rope": k_rope.half().cpu(),
                "v": v.half().cpu(),
            }
        torch.save(save, HERE / "captures" / f"{ds}.pt")

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "stats.csv", index=False)
    print(f"wrote {HERE / 'stats.csv'} ({len(df)} rows)")
    # headline P1 numbers
    p1 = (
        df[(df.axis == "patch_time")]
        .groupby(["dataset", "tensor"])[["rho_median", "sfm_median"]]
        .median()
        .round(3)
    )
    print(p1)


if __name__ == "__main__":
    main()
