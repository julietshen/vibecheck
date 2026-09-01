# vibecheck

A harness for evaluating policy-conditioned safety classifiers — you supply a written policy plus labelled content, and it measures how well a model applies that policy. One model-agnostic core plus a small adapter per model, so adding a model is a ~30-line file. See [GUIDE.md](GUIDE.md#adapting-this-whole-setup-for-a-different-model) for the harness design.

Runs to date evaluate `zentropi-ai/cope-b-a4b`, `zentropi-ai/cope-a-9b`, `openai/gpt-oss-safeguard-20b`, and `mistralai/Shieldstral-1.0-3B` across two harm domains — self-harm and sexually explicit content — under policy prompts of varying detail. Run in support of ROOST Model Community (RMC) inclusion review.

> Previously named `cope-evaluation`, when it was a single-model eval of cope; renamed as it grew into a general harness.

**Round 2 (August 2026):** `mistralai/Shieldstral-1.0-3B` run on the same test sets and policies, locally (no GPU rental). Best F1 0.898 (self-harm) / 0.803 (sexual content) — just behind cope-b at 1/16th the size. See the round-2 section of RESULTS.md. This round also added a **steerability** axis — how much each model tracks the policy text vs. the content's topic (see [STEERABILITY.md](STEERABILITY.md)).

## What's in here

- **[RESULTS.md](RESULTS.md)** — the findings. Per-policy precision/recall/F1, head-to-head against gpt-oss-safeguard, comparison against Zentropi's published benchmark, RMC inclusion recommendation. **Start here.**
- **[STEERABILITY.md](STEERABILITY.md)** — a separate axis: does a model apply the *policy text* it's given, or a fixed notion of the *topic*? Three probes — inverted policy (verdict flip rate), off-topic policy (false-flag rate), and within-domain carve-out (selective release). cope-a/cope-b/safeguard are policy-steerable (90–100% flip under an inverted policy); Shieldstral is a fixed-prior topic detector (weak, asymmetric flip — 27% self-harm, 0% sexual). Includes a defense-in-depth model-stacking argument from the complementary failure modes.
- **[GUIDE.md](GUIDE.md)** — the general playbook this evaluation followed. Serving any open-weight policy classifier on Modal, building test sets, running the eval harness, adapting for a different model.
- `serve_cope.py` — Modal deployment recipe.
- `eval/` — the harness (`eval.py` + `models/` adapters), policies, test sets, and per-run prediction/summary CSVs.

## Running an eval

The generalized harness (`eval/eval.py`) drives every model through one interface; each model is an adapter under `eval/models/`. Pick a model with `--model`. See [GUIDE.md](GUIDE.md#adapting-this-whole-setup-for-a-different-model) for adapter internals and how to add a new model.

**Local model (no server, e.g. Shieldstral)** — a venv with `torch` and `transformers>=5` on an Apple-silicon Mac or any GPU box:

```bash
cd eval
python eval.py --model shieldstral --label sh \
  --policies minimal simple medium full very_long zentropi_official \
  --probes probes/selfharm.csv
python eval.py --model shieldstral --test-set sexual_content_eval/test_set.csv --label sex \
  --policies sexual_content_minimal sexual_content_simple sexual_content_medium \
  sexual_content_zentropi_long sexual_content_oai sexual_content_oai_adapted sexual_content_very_long
```

**Served model (Modal + vLLM, e.g. cope-b, cope-a, safeguard)** — needs a Modal account, an HF token with access to the model, and Python 3.11+:

```bash
# 1. Stand up the endpoint (Part 1 of GUIDE.md)
uv tool install modal
modal token new
modal secret create cope-secrets HF_TOKEN=hf_... VLLM_API_KEY=sk-pick-something
modal run serve_cope.py::download_model    # one-time, ~$0.01
modal deploy serve_cope.py

# 2. Run the eval (Part 2 of GUIDE.md)
export VLLM_API_KEY=sk-pick-something      # same value as above; pip install requests
cd eval
python eval.py --model cope_b --policies minimal simple medium full zentropi_official --concurrency 16
python eval.py --model cope_b --test-set sexual_content_eval/test_set.csv --label sex \
  --policies sexual_content_minimal sexual_content_simple sexual_content_medium sexual_content_zentropi_long sexual_content_oai sexual_content_oai_adapted --concurrency 16
```

Expect ~$2 in Modal GPU time for the served evals end-to-end; the local Shieldstral runs are free.

<details><summary>Original single-model scripts (superseded, kept as worked examples)</summary>

```bash
cd eval
python eval_shieldstral.py --label sh --policies minimal simple medium full very_long zentropi_official
python eval_shieldstral.py --test-set sexual_content_eval/test_set.csv --label sex \
  --policies sexual_content_minimal sexual_content_simple sexual_content_medium \
  sexual_content_zentropi_long sexual_content_oai sexual_content_oai_adapted sexual_content_very_long
```

The original `eval_cope.py` (Modal/cope) works the same way with `--endpoint`/`--model`/`--max-tokens` flags. Both are superseded by `eval.py --model ...`.

</details>

## Provenance and sanitization

- The **self-harm test set** (`eval/test_set.csv`) was provided to ROOST by an external partner for evaluation use. All labelling-org identifiers in the original column names have been replaced with neutral descriptors (`ground_truth`). Post content is preserved as-is.
- The **sexually-explicit Bluesky candidates** (`eval/sexual_content_eval/candidates_to_label.csv`) are sampled from the public `withalim/bluesky-posts` Hugging Face dataset. Labels in that file are this author's editorial judgment (labeller: Juliet Shen, ROOST, 2026-05-20), not ground truth from a benchmark partner.
- The **red-team set** (`eval/sexual_content_eval/redteam_set.csv`) is synthetic content hand-crafted to stress-test policy clauses. **NSFW**: contains explicit text by design.
- The **policy files** under `eval/policies/` are either (a) hand-written for this evaluation, (b) pulled from Zentropi's public labelers API, or (c) sourced from OpenAI's `teen-safety-policy-pack` repository.

## Limitations and bias risks

These are small, single-labeler studies and should be read as directional, not definitive.

- **Single labeler, no inter-annotator agreement.** Every test set is labeled by one source: the sexual-content and scam/spam sets by the author (Juliet Shen), and the self-harm set by a single external partner. No set has a second annotator or an agreement measure, so label subjectivity — especially on judgment-heavy boundaries (scam vs spam, self-disclosure vs glorification, suggestive vs explicit) — is not quantified.
- **Selection bias from keyword/similarity sampling.** The sexual and scam positives were surfaced by keyword or signal ranking, so the sets over-represent *findable* violations and under-represent the long tail (novel phrasing, image-only, non-English, subtle social engineering). Reported **recall is an upper bound** — performance on items the sampling could surface, not on the harm in general.
- **Small samples.** n = 100 (self-harm), 129 (sexual), 100 (scam). Confidence intervals are wide; treat cross-model differences under ~0.1 F1 as ties rather than rankings.
- **Single platform and time window.** The sexual and scam sets are drawn from Bluesky snapshots (2024–2025); tactics and norms drift, and other platforms differ.
- **Label-philosophy dependence.** Scores depend on how each set was labeled and how each policy was authored; a published policy calibrated to a different labeling philosophy can score low here without being deficient (see the self-harm framing-disagreement finding in RESULTS.md).
- **Overfitting / benchmark-gaming risk.** Because the sets are small, iterating policy wording to raise F1 on them would overfit. Policies here are frozen against the test items and not tuned on their scores; exact items are kept out of published numbers so they cannot be trained against.

A differently-sampled held-out validation set (fresh time window, disjoint sampling method, and ideally a second labeler) would bound the selection and labeler bias directly. This applies to all three domains, not only scam.

## Status

The initial evaluation was conducted as part of an RMC inclusion review for cope-b-a4b in May 2026; the Shieldstral round followed in August 2026. Findings will be shared with Zentropi for response and with the wider RMC community. The harness (`eval/eval.py`) is intended for reuse on future models.
