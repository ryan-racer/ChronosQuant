# Related Work & Background

Literature review supporting the ChronosQuant project (compiled 2026-07-22 from a deep research sweep; all claims carry citations).

## 1. The target model: Chronos-2

**Paper:** *Chronos-2: From Univariate to Universal Forecasting* — [arXiv:2510.15821](https://arxiv.org/abs/2510.15821) (AWS AI Labs, Oct 2025). Lineage: *Chronos: Learning the Language of Time Series* — [arXiv:2403.07815](https://arxiv.org/abs/2403.07815).

### Architecture (facts that constrain quantization design)

| Property | Value | Source |
|---|---|---|
| Backbone | **Encoder-only** transformer, T5-inspired (NOT encoder-decoder like Chronos-Bolt; NOT tokenizing LLM like Chronos-T5) | paper, HF card |
| Size | **120M** params (`amazon/chronos-2`, fp32 safetensors 478 MB); **28M** small variant (`autogluon/chronos-2-small`) | HF file trees |
| Blocks | 12 layers × 12 heads, `d_model=768`, `d_ff=3072`, ReLU, RoPE (`rope_theta=1e4`) | `config.json` |
| Input | Non-overlapping patches (size 16), per-instance standardization + **arcsinh** transform, **continuous residual-network embeddings — no discrete tokenization** | paper §method |
| Attention | Alternating **time attention** (across patches) and **group attention** (across series/variates/covariates in a group) → native multivariate, covariates, in-context learning | paper |
| Output | Direct multi-step **quantile head**, 21 fixed levels {0.01…0.99}, up to 64 output patches × 16 = **1024-step horizon**, single forward pass (non-autoregressive) | paper, config |
| Context | Up to **8192** timesteps (pretrained 2048 → extended) | paper |
| Precision shipped | **fp32** (no official bf16/quantized checkpoint) | HF safetensors |
| Throughput | >300 series/s on 1×A10G (batch 1024, ctx 2048, h=64); CPU supported | paper |

Key implications for quantization:
- **No lm_head, no autoregressive decode, no KV cache** → LLM decode-centric arguments (KV-quant, memory-bound decode speedups from weight-only quant) do not transfer. Inference is one batched encoder forward pass → **compute-bound at batch; W8A8/FP8 buys throughput, W4A16 mainly buys footprint.**
- **Known precision sensitivity:** [PR #197](https://github.com/amazon-science/chronos-forecasting/pull/197) pins Chronos-1's input scaling to fp32 after bf16 degraded accuracy → keep Chronos-2's normalization/arcsinh path and patch-embedding residual net in high precision.
- The **group-attention layers have no prior quantization data** — their sensitivity is an open question this project can answer (RQ3).

### Software
- Package: **`chronos-forecasting>=2.0`** (PyPI; v2.3.1 as of 2026-07). Repo: [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) (Apache-2.0).
- API: `Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=...)`; `predict(...)`, `predict_quantiles(..., quantile_levels=[...])`, `predict_df(df, ...)`. Inputs: 3D tensors `(batch, variates, time)`, lists of 1D/2D tensors, dicts with covariates, or long-format DataFrames.

## 2. Prior art on compressing time series foundation models

**No published work quantizes the weights of Chronos/Chronos-Bolt/Chronos-2 as of 2026-07-22** (negative result from repeated searches — treat as "none found", not proven absence). This is the novelty gap ChronosQuant targets.

⚠️ **Terminology hazard:** in the Chronos literature, "quantization" usually means Chronos-1's *input value-binning tokenization* (4096-bin vocabulary), not weight/activation quantization. Chronos-2 has no input binning. The paper must disambiguate early.

Closest existing artifacts & adjacent work:
- [`kashif/chronos-2-onnx`](https://huggingface.co/kashif/chronos-2-onnx) — community ONNX export: fp32 456 MB → static-INT8 124.7 MB (~70% smaller), claimed <1% error, 3–8× faster than PyTorch on CPU. No sensitivity analysis, no benchmark evaluation → a baseline to beat/validate.
- TSFM structured pruning: *Less is More* — [arXiv:2505.23195](https://arxiv.org/abs/2505.23195).
- TSFM distillation: DistilTS — [arXiv:2601.12785](https://arxiv.org/abs/2601.12785).
- QAT of time-series transformers on edge hardware: [arXiv:2408.16495](https://arxiv.org/abs/2408.16495), [arXiv:2407.11041](https://arxiv.org/abs/2407.11041) (4-bit integer-only, +0.63% loss), [arXiv:2310.02654](https://arxiv.org/abs/2310.02654); INT8 edge forecasting [arXiv:2511.10680](https://arxiv.org/abs/2511.10680); FEMBA W8A8/W2A8 Mamba EEG FM on MCU [arXiv:2603.26716](https://arxiv.org/abs/2603.26716).
- Operational-viability latency study of TSFMs (flags quantization as future work): [arXiv:2605.24381](https://arxiv.org/abs/2605.24381).
- PatchTST INT8/INT4 scaling study: [MILETS 2025](https://kdd-milets.github.io/milets2025/papers/MILETS_2025_paper_17.pdf).

## 3. Quantization methods (applicability to an encoder-only, 120M, T5-style TSFM)

### Directly usable today (architecture-agnostic tooling)
| Method / tool | Bits | Calibration | Notes |
|---|---|---|---|
| RTN via **torchao** `quantize_()` | W8A16, W8A8-dyn, W4A16, FP8 | none | Recommended default; per-module-FQN configs enable cheap layer-sensitivity sweeps; composes with `torch.compile`. [arXiv:2507.16099](https://arxiv.org/abs/2507.16099) |
| **HQQ** | W8/4/3/2 | **none** | Half-quadratic solver; proven on Whisper; strong W4/W3 candidate. [blog](https://dropbox.github.io/hqq_blog/) |
| **bitsandbytes** | int8 (LLM.int8()), NF4 | none | HF-integrated; kernels can be *slower* than fp16 at 120M scale (bnb issue #1262) — always report latency. [arXiv:2208.07339](https://arxiv.org/abs/2208.07339), [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) |
| **ONNX Runtime** dynamic/static INT8 | W8A8 | static: yes | The CPU deployment story; exclude attention QKV nodes from full-INT8 (known collapse pitfall); `kashif/chronos-2-onnx` proves viability |
| **OpenVINO NNCF** | W8A8, W4-wo | ~300 samples | Has SmoothQuant + accuracy-aware mode (`max_drop`); Whisper INT8 precedent |
| **optimum-quanto** | W2/4/8, A8/F8 | optional | Works but maintenance mode |
| FP8 E4M3 (torchao / llm-compressor) | W8A8-FP8 | none (dynamic) | RTX 2000 Ada is cc 8.9 → FP8-capable; "essentially lossless" at 8 bits in LLM literature |

### Requiring adaptation (contribution-worthy)
- **GPTQ / SmoothQuant via llm-compressor** — tracing-based, validated on Whisper (>99% recovery), but needs custom wiring + a **time-series calibration set** (patched, normalized inputs). GPTQ [arXiv:2210.17323](https://arxiv.org/abs/2210.17323); SmoothQuant [arXiv:2211.10438](https://arxiv.org/abs/2211.10438).
- **AWQ-style activation-aware scaling** — concept transfers; tooling (AutoAWQ, archived) is decoder-LLM-only. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978).
- **Rotation methods (QuaRot/SpinQuant)** for W4A4 — invariance derivations assume pre-norm decoder blocks; porting to a T5-style encoder is research, not a library call. [arXiv:2404.00456](https://arxiv.org/abs/2404.00456), [arXiv:2405.16406](https://arxiv.org/abs/2405.16406).
- **QAT via torchao** (`QATConfig`, 8da4w) — feasible for 120M on one GPU using Chronos finetuning scripts, if W4 PTQ misses the accuracy bar. [arXiv:2407.11062](https://arxiv.org/abs/2407.11062) (EfficientQAT), [arXiv:2305.17888](https://arxiv.org/abs/2305.17888) (LLM-QAT).

### Architecture-specific cautions (from T5/Whisper/BERT precedent)
1. **FFN outliers:** transformer FFN down-projection input is the classic activation-outlier locus ([arXiv:2109.12948](https://arxiv.org/abs/2109.12948), [arXiv:2306.12929](https://arxiv.org/abs/2306.12929), GLU outliers [arXiv:2506.01967](https://arxiv.org/abs/2506.01967)) → prefer per-channel symmetric weights + per-token dynamic activations; consider keeping FFN down-proj higher-precision.
2. **Whisper design-choice study** ([arXiv:2511.08093](https://arxiv.org/abs/2511.08093)): dynamic > static activation quant; symmetric per-channel > asymmetric per-tensor; INT8-dynamic ≈ lossless at 57% size cut; ≤4-bit degrades on *hard* inputs → slice eval by series difficulty, not just averages.
3. **Keep in high precision:** LayerNorm/RMSNorm, patch-embedding residual net, input normalization/arcsinh, residual adds, softmax, and the **quantile output head** (tiny → cheap to keep fp32).
4. **Small-model effect:** quantization damage grows as params shrink ([arXiv:2409.11055](https://arxiv.org/abs/2409.11055), [arXiv:2405.03146](https://arxiv.org/abs/2405.03146), flips [arXiv:2407.09141](https://arxiv.org/abs/2407.09141)); at 120M expect W4 to bite harder than 70B-LLM headlines suggest; W8 should be safe.
5. **fp16 overflow** is a known T5 pathology → use bf16/fp32 baselines, never fp16.

### What rigorous quantization papers report (evaluation checklist)
From GPTQ, AWQ, SmoothQuant, LLM.int8(), QuaRot, the ACL-2024 comprehensive evaluation ([arXiv:2402.16775](https://arxiv.org/abs/2402.16775)), calibration study ([arXiv:2311.09755](https://arxiv.org/abs/2311.09755)), and *Accuracy is Not All You Need* ([arXiv:2407.09141](https://arxiv.org/abs/2407.09141)):
1. Bit-width × method sweep with **accuracy-retention %** vs the fp32/bf16 baseline;
2. Task metrics (here: WQL/SQL, MASE via standard TSFM benchmarks);
3. **Distributional fidelity** beyond aggregates (per-series ΔWQL distribution, prediction "flips");
4. Calibration ablations (size, domain, seed variance);
5. **Per-layer sensitivity** (leave-one-out high precision; find cliff layers);
6. Systems numbers on named hardware (GPU + CPU; size, peak memory, latency, throughput, backend/kernel);
7. **Pareto plots** (accuracy vs size / latency);
8. Anchor comparison: quantized-large vs fp32-small (Chronos-2 W4 vs `chronos-2-small`/`chronos-bolt-small` fp32).

*(See [EVALUATION_PLAN.md](EVALUATION_PLAN.md) for the full evaluation framework.)*
