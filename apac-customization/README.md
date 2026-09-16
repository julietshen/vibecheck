# Customizing open models for APAC languages & cultural norms

A hands-on **customization ladder** for a Trust & Safety audience: three ways to
adapt an open-weight model to a Southeast/East-Asian language and a local norm,
in increasing order of effort and power. Everything here **runs locally on an
Apple-silicon Mac** (MLX + the repo's `.venv`), no GPU rental.

> Companion to the parent `vibecheck` eval harness. That repo *measures* models;
> this folder *customizes* them. Built for a summit talk (Singapore, 2026) on open
> models for safety workflows.

> **Start here for the talk:** [`PRESENTATION_NOTES.md`](PRESENTATION_NOTES.md) (speaker
> notes + findings) and [`results/MATRIX.md`](results/MATRIX.md) (the measured
> models×languages matrix). The published artifact is the gaps-matrix visual.
>
> **What grew beyond the ladder below:** a 5-model × 3-language matrix (SEA / Chinese /
> romanized Bengali) using real datasets (ChiFraud, TB-OLID/HASOC) plus the synthetic SEA
> set. Runners: `demo1_shieldstral/` (multilingual, fixed), `rung2_lora/` (Qwen fine-tune),
> `demo3_cope/` (steerable), `demo4_safeguard/` (steerable + reasoning). Headline: the
> failure is the *low-resource language* (Bengali), not the model type; fine-tuning helped
> the clear harm (moneylending) but **hurt** the subjective one (Bengali offensive).

## The task

One concrete, stage-safe harm: **advertising of illegal / unlicensed
moneylending** — "loan shark", "ah long", "5-6" lending, "pinjaman along",
地下黑贷. It's a real, high-priority Southeast Asian T&S harm whose markers are
language- and culture-specific, and it has a clean hard case: telling an **illegal**
loan-shark ad apart from a **legitimate licensed lender**. A Western-centric base
model gets this wrong in one of two ways depending on how you prompt it — it either
misses the local harm or nukes legitimate local businesses.

## The three rungs

| Rung | Customize by… | Effort | What it buys | What it can't do |
|---|---|---|---|---|
| **1 — Policy text** | writing the policy (in any language / encoding a local norm) | minutes, no training | cheap, transparent, works on any model; steer the model to flag **more** | a small model resists being told to **permit** something its prior dislikes — the "wall" |
| **2 — LoRA fine-tune** | training a small adapter on regional examples | ~90 s on an M4 Max | changes the **weights**: teaches nuance (licensed vs illegal) so a one-line prompt works | needs labeled regional data; adapter is task-specific |
| **3 — Purpose-built guard** | using a model already trained for the region | none (download) | broadest multilingual coverage out of the box (Qwen3Guard; **SEA-Guard** for SEA norms) | **fixed taxonomy** — no carve-outs, no per-platform policy |

The rungs are complementary, not competing — a real stack uses all three (cheap
policy-conditioned model up front, a fine-tuned adapter for the priority local
harms, a purpose-built guard as a multilingual backstop).

## Results (real, reproducible)

Base model everywhere: **Qwen3-1.7B** (Apache-2.0), the same family AI Singapore's
SEA-Guard is built on. Positive class = VIOLATION.

**Rung 2, real Chinese data (ChiFraud, temporal-shift 2023 test split):**

| | Precision | Recall | F1 | False positives on *normal* finance text |
|---|---|---|---|---|
| Base (generic policy) | 0.69 | 0.88 | 0.769 | **16/40 (40%)** |
| **+ LoRA (90 s, 300 ex.)** | **0.94** | 0.82 | **0.880** | **2/40 (5%)** |

**Rung 2, synthetic SEA-language set (id/ms/th/tl/vi):**

| | Precision | Recall | F1 | False positives on *licensed lenders* |
|---|---|---|---|---|
| Base (generic policy) | 0.75 | 1.00 | 0.857 | 6/8 |
| Base (expert detailed policy = rung 1) | 0.75 | 1.00 | 0.857 | 5/8 |
| **+ LoRA** | **1.00** | 1.00 | **1.00** | **0/8** |

The through-line: the base model **over-flags legitimate lenders**, and no prompt
fixes it on a small model (rung 1's wall). ~90 seconds of LoRA on regional
examples teaches the distinction — false positives on legitimate businesses
collapse. On real, harder data the gain is honest-sized (F1 +0.11, FP 40%→5%);
on the small synthetic set it saturates (flagged as such — see `data/NOTICE.md`).

**Rung 3:** Qwen3Guard-Gen-8B, off the shelf, classified an 18-post multilingual
sample perfectly (loan-shark ads → "Non-violent Illegal Acts", legit/benign →
Safe) — with zero training and no policy, but a taxonomy you can't reconfigure.

## Run it

From this folder, using the parent repo's venv (`../.venv`, torch + MPS). One-time:
`../.venv/bin/pip install "mlx-lm>=0.28"`.

```bash
# data (synthetic SEA set — regenerates deterministically)
../.venv/bin/python data/build_dataset.py

# Rung 1 — policy-only: language works, the permit "wall" shows
../.venv/bin/python rung1_prompt/run_prompt.py

# Rung 2 — LoRA fine-tune + before/after (synthetic SEA set)
../.venv/bin/python -m mlx_lm lora --config rung2_lora/lora_config.yaml
../.venv/bin/python run_eval.py --model mlx-community/Qwen3-1.7B-bf16 --policy minimal --label base_minimal
../.venv/bin/python run_eval.py --model mlx-community/Qwen3-1.7B-bf16 --adapter rung2_lora/adapters --policy minimal --label tuned_minimal

# Rung 2 on REAL Chinese data (ChiFraud) — clone the source first (CC BY-NC 4.0)
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/xuemingxxx/ChiFraud.git /tmp/ChiFraud
../.venv/bin/python rung2_lora/prepare_chifraud.py --source /tmp/ChiFraud
../.venv/bin/python -m mlx_lm lora --config rung2_lora/lora_config_chifraud.yaml
../.venv/bin/python run_eval.py --adapter rung2_lora/adapters_chifraud --test data/chifraud/chifraud_sample.csv --label zh_tuned --model mlx-community/Qwen3-1.7B-bf16

# Rung 3 — purpose-built multilingual guard (already cached)
../.venv/bin/python rung3_regional/run_guard.py --limit 18
```

## Which base model? (Qwen / Kimi / MiniMax / SEA-LION)

Two tiers, two customization stories:

- **Frontier open MoE — Kimi K2/K3, MiniMax M2, Qwen3-235B** (200 B – 2.8 T params):
  too big to fine-tune or run on a laptop. You customize them by **policy/prompt
  only** (rung 1). Watch licenses (MiniMax M2.7 restricts commercial use).
- **Small dense — Qwen3 0.6/1.7/4B, Gemma-3-4B** (Apache-2.0): small enough to
  **LoRA on a Mac** (rung 2). This is the customization vehicle, and **Qwen is the
  base AI Singapore's SEA-Guard chose** — so rung 2 here mirrors the region's own
  recipe at small scale.
- **Purpose-built regional — SEA-Guard** (`aisingapore/Qwen-SEA-Guard-8B-2602`,
  from SEA-LION v4; id/ms/my/ta/th/tl/vi/en) and **Qwen3Guard** (119 langs): rung 3.

## Datasets

Full per-dataset attribution and licenses are in [`DATA_CREDITS.md`](DATA_CREDITS.md)
(ChiFraud CC BY-NC 4.0 · TB-OLID/HASOC AGPL-3.0 · SWSR CC BY 4.0 · Uli CC BY 4.0). Raw
data is not vendored — regenerate the samples from source with the `build_*` scripts.


- **Synthetic SEA set** (`data/`) — hand-authored, multilingual, **must be
  validated by a native speaker**; demonstrates the mechanism, not real accuracy.
  See `data/NOTICE.md`.
- **ChiFraud** (Tang et al., COLING 2025) — real Chinese fraud benchmark, CC BY-NC
  4.0; class 9 (地下黑贷, underground loans) vs class 0 (normal). Used in rung 2.
- **Further tracks not yet built** (user-suggested, strong regional fit):
  - **HASOC 2024** — hate/offensive in **English + Bangla** → a "what counts as
    offensive is culturally defined" track (cultural-norm axis rather than language).
  - **Tattle** (tattle.co.in) — Indian-language misinfo / fear-speech (Zenodo, GitHub).

## Provenance

Results and code generated by Claude (Claude Code, Opus 4.8) with Juliet Shen,
2026-09-16. Numbers reproduce from the scripts above; artefacts in `results/`.
Small-sample, single-labeler caveats from the parent `README.md` apply.
