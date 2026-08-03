"""Tests for the RQ3 module-family sensitivity study.

Validates: (1) family patterns partition the quantized linear set exactly (no overlap,
no gaps) on a structurally faithful tiny Chronos-2 clone; (2) leave-one-out configs
quantize exactly the intended module counts through the real rtn transform; (3) the
sweep YAML matches the variant builder 1:1; (4) Hessian/outlier stats behave sanely
on synthetic data (monotone in bits, detects planted outlier channels).
"""

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from chronosquant.quantization.sensitivity import (
    EXPECTED_FAMILY_COUNTS,
    EXTRA_GROUPS,
    MODULE_FAMILIES,
    build_sensitivity_variants,
    classify_module,
    family_partition,
    linear_module_names,
    module_hessian_row,
    module_outlier_row,
    parse_arm_name,
    skip_patterns_only,
    skip_patterns_protect,
)
from chronosquant.quantization.transforms import rtn_quantize_model
from chronosquant.utils import REPO_ROOT

N_BLOCKS = 12  # match the real model so first/last-block patterns are exercised


def tiny_chronos(n_blocks: int = N_BLOCKS, d: int = 8) -> nn.Module:
    """Structurally faithful miniature of Chronos-2's module tree (names match)."""

    def attention() -> nn.Module:
        sa = nn.Module()
        sa.q, sa.k, sa.v, sa.o = (nn.Linear(d, d) for _ in range(4))
        layer = nn.Module()
        layer.self_attention = sa
        return layer

    def ffn_layer() -> nn.Module:
        mlp = nn.Module()
        mlp.wi = nn.Linear(d, 2 * d)
        mlp.wo = nn.Linear(2 * d, d)
        layer = nn.Module()
        layer.mlp = mlp
        return layer

    def embed(d_in: int, d_out: int) -> nn.Module:
        e = nn.Module()
        e.hidden_layer = nn.Linear(d_in, 2 * d)
        e.output_layer = nn.Linear(2 * d, d_out)
        e.residual_layer = nn.Linear(d_in, d_out)
        return e

    model = nn.Module()
    model.input_patch_embedding = embed(4, d)
    encoder = nn.Module()
    encoder.block = nn.ModuleList()
    for _ in range(n_blocks):
        block = nn.Module()
        block.layer = nn.ModuleList([attention(), attention(), ffn_layer()])
        encoder.block.append(block)
    model.encoder = encoder
    model.output_patch_embedding = embed(d, 6)
    return model


TINY_COUNTS = {
    "patch_embed": 3,
    "time_attn": 4 * N_BLOCKS,
    "group_attn": 4 * N_BLOCKS,
    "ffn": 2 * N_BLOCKS,
    "head": 3,
}
TINY_TOTAL = sum(TINY_COUNTS.values())


class TestFamilyPartition:
    def test_partition_no_overlap_no_gaps(self):
        names = linear_module_names(tiny_chronos())
        assert len(names) == TINY_TOTAL == 126  # same count as the real model
        partition = family_partition(names)  # raises on gaps or overlaps
        counts = {fam: len(mods) for fam, mods in partition.items()}
        assert counts == TINY_COUNTS == EXPECTED_FAMILY_COUNTS
        # explicit disjointness: each name lands in exactly one family
        all_assigned = [n for mods in partition.values() for n in mods]
        assert sorted(all_assigned) == sorted(names)

    def test_classify_examples(self):
        assert classify_module("encoder.block.11.layer.0.self_attention.q") == "time_attn"
        assert classify_module("encoder.block.0.layer.1.self_attention.o") == "group_attn"
        assert classify_module("encoder.block.3.layer.2.mlp.wo") == "ffn"
        assert classify_module("input_patch_embedding.residual_layer") == "patch_embed"
        assert classify_module("output_patch_embedding.output_layer") == "head"
        assert classify_module("some.unrelated.linear") is None

    def test_gap_raises(self):
        with pytest.raises(ValueError, match="not covered"):
            family_partition(["encoder.block.0.layer.0.self_attention.q", "rogue.linear"])


class TestLeaveOneOutConfigs:
    @pytest.mark.parametrize("family", list(MODULE_FAMILIES))
    def test_protect_one_counts(self, family):
        model = tiny_chronos()
        _, info = rtn_quantize_model(
            model, bits=4, simulate=True, skip_modules=skip_patterns_protect(family)
        )
        assert info["n_modules_quantized"] == TINY_TOTAL - TINY_COUNTS[family]
        assert info["n_modules_skipped"] == TINY_COUNTS[family]

    @pytest.mark.parametrize("family", list(MODULE_FAMILIES))
    def test_only_one_counts(self, family):
        model = tiny_chronos()
        _, info = rtn_quantize_model(
            model, bits=4, simulate=True, skip_modules=skip_patterns_only(family)
        )
        assert info["n_modules_quantized"] == TINY_COUNTS[family]
        assert info["n_modules_skipped"] == TINY_TOTAL - TINY_COUNTS[family]

    def test_protect_only_modules_actually_untouched(self):
        """Protected family's weights are bit-identical; all others are perturbed."""
        model = tiny_chronos(n_blocks=2)
        before = {n: m.weight.detach().clone() for n, m in model.named_modules()
                  if isinstance(m, nn.Linear)}
        rtn_quantize_model(model, bits=4, simulate=True,
                           skip_modules=skip_patterns_protect("ffn"))
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            same = torch.equal(module.weight.detach(), before[name])
            if classify_module(name) == "ffn":
                assert same, f"protected module {name} was modified"
            else:
                assert not same, f"module {name} should have been quantized"

    @pytest.mark.parametrize("group", list(EXTRA_GROUPS))
    def test_positional_groups(self, group):
        model = tiny_chronos()
        _, info = rtn_quantize_model(
            model, bits=4, simulate=True, skip_modules=skip_patterns_protect(group)
        )
        assert info["n_modules_skipped"] == 10  # 4 + 4 attention + 2 mlp linears
        assert info["n_modules_quantized"] == TINY_TOTAL - 10

    def test_last_block_pattern_does_not_hit_block_1(self):
        """'block.11' patterns must not glob-match block.1 (and vice versa)."""
        names = linear_module_names(tiny_chronos())
        matched = [
            n for n in names
            if any(__import__("fnmatch").fnmatch(n, p) for p in EXTRA_GROUPS["last_block"])
        ]
        assert len(matched) == 10
        assert all(".block.11." in n for n in matched)


class TestSweepConfigConsistency:
    def test_yaml_matches_builder(self):
        sweep_path = REPO_ROOT / "configs" / "evaluation" / "sweeps" / "sensitivity.yaml"
        with open(sweep_path) as f:
            sweep = yaml.safe_load(f)
        assert sweep["variants"] == build_sensitivity_variants(bits=4)

    def test_arm_names_parse(self):
        for variant in build_sensitivity_variants():
            direction, family = parse_arm_name(variant["name"])
            assert direction in {"all", "protect", "only"}
            if direction == "all":
                assert family == ""
            else:
                assert family in set(MODULE_FAMILIES) | set(EXTRA_GROUPS)


class _FakeStats:
    """Minimal stand-in for calibration.LinearInputStats."""

    def __init__(self, x: torch.Tensor):
        x = x.to(torch.float32)
        self.hessian = x.T @ x
        self.n_rows = x.shape[0]
        self._x = x

    @property
    def sample_matrix(self) -> torch.Tensor:
        return self._x


class TestStats:
    def setup_method(self):
        torch.manual_seed(0)
        self.linear = nn.Linear(16, 8)
        self.x = torch.randn(256, 16)
        self.stats = _FakeStats(self.x)

    def test_hessian_row_matches_empirical_mse(self):
        """tr(ΔW H ΔW^T)/n must equal the empirical mean ||ΔW x||² on the same data."""
        row = module_hessian_row("m", self.linear, self.stats, bits=4)
        w = self.linear.weight.detach()
        from chronosquant.quantization.sensitivity import _rtn_perturbation

        delta = _rtn_perturbation(w, 4)
        empirical = (self.x @ delta.T).pow(2).sum(dim=1).mean().item()
        assert row["exact_mse"] == pytest.approx(empirical, rel=1e-4)
        assert row["rel_recon_err"] > 0
        assert row["hawq_proxy"] > 0

    def test_hessian_error_monotone_in_bits(self):
        errs = [
            module_hessian_row("m", self.linear, self.stats, bits=b)["exact_mse"]
            for b in (2, 4, 8)
        ]
        assert errs[0] > errs[1] > errs[2]

    def test_outlier_row_detects_planted_channel(self):
        x = torch.randn(512, 16)
        x[:, 3] *= 50.0  # plant one LLM-style outlier channel
        plain = module_outlier_row("m", _FakeStats(torch.randn(512, 16)))
        outlier = module_outlier_row("m", _FakeStats(x))
        assert outlier["ch_absmean_max_over_mean"] > 5 * plain["ch_absmean_max_over_mean"]
        # A scaled-up channel raises pooled kurtosis (cross-channel spread) but not
        # per-channel kurtosis (each channel is still Gaussian within itself).
        assert outlier["kurtosis_pooled"] > plain["kurtosis_pooled"]
        assert outlier["kurtosis_ch_max"] < 5.0  # within-channel tails unchanged
        for key in ("ch_absmax_max_over_median", "frac_channels_absmax_gt6", "global_absmax"):
            assert np.isfinite(plain[key]) and np.isfinite(outlier[key])
