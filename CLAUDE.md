# CLAUDE.md — agent onboarding

**What this is:** the first systematic post-training-quantization study of time-series
foundation models (Chronos-2, Chronos-Bolt, TimesFM-2.5), with a calibration-aware
evaluation harness, a mechanism finding, and repair methods. Workshop paper at
`paper/workshop.pdf` (anonymous submission — mind de-anonymization before making
anything public: artifacts, HF uploads, repo visibility).

## Read these before nontrivial work

1. `docs/RESEARCH_LOG.md` — dated narrative of every experiment, decision, infra fix,
   and finding. **The project's memory. Append an entry (newest first) for any
   substantive session.**
2. `docs/NEXT_STEPS.md` — literature-grounded roadmap (what to do next and what was
   deliberately deprioritized, with citations).
3. `docs/RELATED_WORK.md` / `docs/EVALUATION_PLAN.md` — literature base + eval design.
4. `paper/workshop.md` — markdown master of the paper (kept in byte-sync with
   `workshop.tex`; every number is machine-verified against `results/tables/*.csv` —
   never hand-transcribe numbers into the paper).

## Headline findings (full 27-task tier, as of 2026-07-31)

- 8-bit is free for every method; GPTQ holds parity to W3 (GPTAQ extends it: 1.016);
  all PTQ collapses at W2 → QAT is the sub-3-bit path.
- **Mechanism (RQ3):** calibration damage is gated *exclusively* by the 3-linear
  quantile head (3.2% of weights, a weight-space SNR problem — noise ≈ 8× signal);
  accuracy damage is distributed. No LLM-style activation outliers (that's why
  AWQ-style scaling fails here); only ReLU-sparsity channel heterogeneity.
- **Prevention:** any method + fp32 head ≈ parity with intact calibration (confirmed
  on all 4 models). **Cure:** quantile sorting is free & provably never hurts WQL;
  split-conformal (sort-before-calibrate) restores nominal coverage at any bit-width.
- Cross-architecture: Bolt (enc-dec) at parity everywhere; TimesFM (decoder-AR) W4
  parity but W3 cliff one bit higher; TimesFM fp32 natively emits crossed quantiles.

## How to run things

- **Always `uv run python ...`** — never bare python (corporate TLS proxy needs the
  truststore injection wired into scripts; uv is configured with system-certs).
- Evaluate: `uv run python scripts/evaluate.py configs/evaluation/runs/<run>.yaml`
- Sweeps: `uv run python scripts/sweep.py configs/evaluation/sweeps/<sweep>.yaml
  --tier dev|full` — resumable: completed variants are skipped, partial variants
  rerun. Dev tier = 8 tasks (fast signal, known to be noisy: dev "wins" over fp32
  routinely resolve to parity at full tier — never publish dev numbers unlabeled).
- Aggregation: `scripts/aggregate.py` (leaderboard/retention/validate),
  `python -m chronosquant.analysis.campaign` (master table),
  `scripts/repair_analysis.py` (repair arm), 
  `python -m chronosquant.quantization.sensitivity analyze|report` (RQ3 diagnostics).
- Tests: `uv run pytest` (offline) / `-m network` (reference-parity gate).

## Conventions & discipline

- **Retention** = geomean WQL(quant)/WQL(fp32-same-model) over tasks, 1000-resample
  bootstrap CI; clip [1e-2, 1e2]. Reproduction gates: seasonal-naive must match
  published per-task numbers to <1e-6; model fp32 runs to <1% where references exist.
- `results/raw/<run>/`: only `summaries.csv` + `run_metadata.yaml` are tracked;
  predictions parquets are untracked but ESSENTIAL (all offline analyses — repair,
  QCR, any new metric — recompute from them without model reruns). Don't delete them.
- Quantization plugs in via declarative `quantization:` config on a predictor
  (`model_transform` hook → `quantization/transforms.py` registry). New methods:
  register there; skip-patterns are fnmatch globs over module names.
- Multi-model: predictors registered in `evaluation/predictors.py`
  (`chronos2`, `chronos_bolt`, `timesfm`). TimesFM: `fix_quantile_crossing` must stay
  False in all arms (it is the sorting repair baked in — would erase the QCR metric).
- Review culture: substantive new modules get an adversarial code-review pass before
  their numbers are trusted (see RESEARCH_LOG 2026-07-30/31 for what reviews caught —
  e.g., conformal-guarantee attribution, alpha=0.25 GPTAQ default, pooled-vs-per-channel
  kurtosis). Paper-bound numbers get reproduced by hand from raw artifacts.

## Environment gotchas (Windows 11, RTX 2000 Ada 8 GB, Python 3.12)

- fp16 overflows T5-lineage models — use bf16/fp32. One GPU job at a time.
- The user may cap compute ("≤50%"): run GPU jobs sequentially, consider
  BelowNormal priority / half-core affinity for long sweeps.
- `datasets>=3.6,<4` pinned; project-local HF cache (`data/hf_datasets_cache`);
  `num_proc=1` on Windows+CUDA; never name scripts after stdlib modules.
- torchao int4wo unusable on Windows; bnb kernels can be slow at this scale.
- `experiments/` = independent side experiments (e.g. `experiments/inq/`, a falsified
  DPCM K/V-coding proposal — useful negative result, cite don't touch).

## Open threads (see NEXT_STEPS for full detail)

- **TiRex/TiRex-2** (recurrent xLSTM, 4th architecture family, highest novelty for a
  main-conference version). Blocker: tirex-2 pins transformers<5 — verify in a
  throwaway venv first; pure-torch sLSTM backends make Windows viable.
- ParetoQ-style short QAT for W2 (validated at ~125M scale; fits this GPU).
- GIFT-Eval / fev-bench as second benchmark + leaderboard submissions.
- Quantile-loss-guided quantization (GuidedQuant adapted to pinball loss) — the most
  novel unexplored method idea.
- HF model release: Apache-2.0 permits it (verified); wait for review decision
  (anonymity), publish only storage-real head-protected arms with calibration-documented
  cards. Competing work exists (TQS-PTQ, arXiv:2606.13300) — move fast on novelty claims.
