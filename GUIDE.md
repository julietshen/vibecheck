# Model Testing Guide

A walkthrough of how we serve and evaluate open-weight safety models, written for someone who doesn't write code day-to-day. No jargon without explanation.

The recipe here is general — it works for any policy-conditioned safety classifier, whether it runs on a rented GPU or a laptop. The `vibecheck` harness (`eval/eval.py`) evaluates every model through one interface; each model is a small **adapter** under `eval/models/`, and adding a new model means writing one ~30-line adapter, not touching the harness. We use several models as running examples — `zentropi-ai/cope-b-a4b`, `cope-a-9b`, `openai/gpt-oss-safeguard-20b`, and `mistralai/Shieldstral-1.0-3B` — because they span the three shapes you'll meet: a served MoE model, a served reasoning model, and a laptop-runnable local model.

This guide has two halves:

- **Part 1 — Serving**: getting an open-weight model deployed on a rented GPU and reachable from the laptop. **Skip this for local models** (e.g. Shieldstral), which run in-process with no server.
- **Part 2 — Evaluating**: feeding test content through a model under different "policy" prompts and measuring how well it labels things. This is where the harness lives — start here if your model is already reachable (or runs locally).

### What to have ready before a demo

Prepare these ahead of time so the demo isn't spent on installs and downloads:

- **A Python 3.11+ virtualenv** with dependencies installed: `pip install -r requirements.txt`. That covers both paths — `requests` for served models, and `torch` / `transformers>=5` / `pillow` / `torchvision` for the local (Shieldstral) and multimodal runs. Apple-silicon Macs run Shieldstral on the MPS GPU; any CUDA box works too. (`modal`, for serving, is installed separately as a CLI tool — see Part 1.)
- **A Hugging Face account + token, with the model's terms accepted.** `mistralai/Shieldstral-1.0-3B` is **gated** — click "Agree and access repository" on its HF page while logged in, then `huggingface-cli login` with a read token *before* the first run, or the weight download 401s. This applies to the local models too, not just the served ones.
- **Weights pre-downloaded.** The first `python eval.py --model shieldstral …` pulls ~6 GB from HF (multimodal pulls a bit more). Run one throwaway `--limit 2` sweep beforehand so the download is already cached when your audience is watching.
- **For image evals: the image files present on disk.** The scam image set (`eval/scam_eval/scam_images/`) is tracked in the repo, so a clone has it. The sexual-content image set (`eval/sexual_content_eval/sexual_content_images/`) is **intentionally not distributed** (real, potentially NSFW Bluesky content — gitignored), so you must regenerate/re-fetch it locally before running that eval (see [Multimodal (image) evals](#multimodal-image-evals)).
- **For served models only:** a Modal account (`modal token new` done), the Modal secret created, and ideally the endpoint already deployed and warmed (the first request pays a 2–3 min cold start). See Part 1.

---

## Part 1: Serving

### What we're trying to do

We want a deployed AI model — running on a remote GPU — that our laptop can send test inputs to. "Running" an AI model means loading its weights (a large file full of numbers that define what the model "knows") into the memory of a powerful computer with a graphics card, and then having a program take a user's input, do the math, and produce a response.

Our worked example: **cope-b-a4b** (made by a company called Zentropi AI). It's pre-release — not available through any hosted API like OpenAI or Anthropic — so we have to run it ourselves. Its weights are ~50 GB. Most useful open-weight safety models are in the 8–80 GB range; the steps below scale up or down accordingly.

### Why we can't just run it on the laptop

Three problems with running it on the MacBook:

1. **Memory.** A typical safety model needs 10–100 GB of memory just to load. The MacBook has 36 GB. Larger models physically can't fit.
2. **GPU.** AI models run thousands of times faster on NVIDIA graphics cards than on a regular CPU. The MacBook has Apple's own GPU, which most AI inference tools don't support yet.
3. **The serving software.** The standard tool for serving large AI models, called **vLLM**, doesn't run on Macs at all. It's built for Linux servers with NVIDIA GPUs.

So we need a remote computer with the right hardware.

### Why we picked Modal

We considered several options for renting GPU computers:

| Option | Best for | Why we didn't pick it |
|---|---|---|
| **RunPod** | Long testing sessions (3+ hrs at a stretch) | You pay even when the computer is idle. Easy to forget to turn off. |
| **Lambda Labs** | Reliable, well-documented usage | Slightly more expensive; no auto-scaling |
| **AWS / GCP / Azure** | Big enterprise deployments | Slow to set up, most expensive, paperwork |
| **Modal** ✓ | **Intermittent testing** | Pay only when you're actively using it. Scales to zero. |

Modal is what's called a **serverless GPU platform**. The idea: you write a Python script describing what you want to run. Modal handles the rest — provisioning a computer with a GPU when a request comes in, charging you per second of actual use, and turning everything off when you're done. You never SSH into anything or babysit a running server.

**Cost expectation:** an H100 GPU on Modal is ~$4/hour while actively serving a request. For light testing (a few requests, then walking away for hours), you'd spend maybe $1–5/day. For a heavy day of nonstop testing, more like $20–40.

### The pieces involved

A few proper nouns it helps to know:

- **Hugging Face (HF).** A website/registry where AI researchers publish their models. Like GitHub but for AI weights. Cope lives at `huggingface.co/zentropi-ai/cope-b-a4b`; other safety models you might evaluate include `openai/gpt-oss-safeguard-20b`, `meta-llama/Llama-Guard-3-8B`, `google/shieldgemma-9b`.
- **HF token.** A password-like string that proves you have permission to download a specific model. Some models (including cope) are "gated" — you have to click "I agree to the terms" on the model's page first, then your token unlocks downloads.
- **vLLM.** The software that loads the model into GPU memory and exposes a familiar OpenAI-style API (`POST /v1/chat/completions` and `POST /v1/completions`) so we can talk to it. It does a lot of optimization tricks under the hood to make inference fast. Works for most open-weight transformer models.
- **Modal Secret.** A safe place to store sensitive strings (like the HF token). The code references the secret by name; the actual values never appear in files we commit.
- **Modal Volume.** A persistent disk that survives between runs. We use it to cache model weights so we don't have to re-download them every time we start the server.

### The workflow, step by step

The steps below use cope as the example. For a different model, change three things: the Hugging Face model name, the Modal app name, and the secret name. Everything else stays the same.

#### Step 1: Accept the model's terms (if gated)

On Hugging Face, visit the model page while logged in and click "Agree and access repository." This is a one-time thing per model. After this, your account is on the allowlist. Non-gated models skip this step.

#### Step 2: Create an HF token

Go to Hugging Face settings → tokens, and make a new "Fine-grained" token scoped to just this one model with read access. Copy the long `hf_...` string. Treat it like a password — anyone with it can download things as you.

#### Step 3: Install Modal locally

We need a small tool on the laptop that talks to Modal's servers:

```bash
uv tool install modal     # one-time install
modal token new           # one-time login, opens browser
```

#### Step 4: Tell Modal our secrets

We store the HF token (so Modal's computer can download the model) and a custom API key (so random people on the internet can't use our endpoint):

```bash
modal secret create cope-secrets HF_TOKEN=hf_yourtoken VLLM_API_KEY=sk-pick-something
```

🔧 **Adapt for a different model**: change `cope-secrets` to `<your-model>-secrets`, and reference that name in the Python file (Step 5).

Modal saves these on their servers. The Python script reads them at runtime through environment variables (`os.environ["HF_TOKEN"]`, etc.).

#### Step 5: The Python serving recipe (`serve_cope.py`)

This file is the recipe Modal follows. The important parts in plain English:

- "Build a Linux machine with NVIDIA's CUDA toolkit and Python 3.12 installed, then install vLLM on it."
- "Mount a persistent disk at a specific path so cached files survive between runs."
- "When a web request comes in, give the machine an H100 GPU and run `vllm serve <MODEL_NAME>` to start the model."
- "After 10 minutes of no requests, shut the GPU machine down to save money."

The cope-specific bits are at the top of the file:

```python
MODEL_NAME = "zentropi-ai/cope-b-a4b"
GPU_CONFIG = "H100:1"        # 80GB GPU; smaller models fit on A100:1 or L40S:1
MAX_MODEL_LEN = 8192          # max prompt+response length, in tokens
```

We pass a few flags to vLLM:

- `--trust-remote-code`: many open-weight models ship their own custom Python loader. We're saying "yes, run it." Required for cope.
- `--enforce-eager`: skip a slow but optional GPU-kernel compilation step. Cope's pre-release weights had a bug in that compilation step; safe to drop this flag once a model is well-shaken.
- `--max-model-len 8192`: how long an input the model accepts. 8192 "tokens" is roughly 6,000 words. Set this to the largest prompt you actually need — higher values use more GPU memory.
- `--api-key`: require requests to include our custom API key.

🔧 **Adapt for a different model**: change `MODEL_NAME`, possibly `GPU_CONFIG` (rule of thumb: model weights in GB × 1.3 ≤ GPU memory), and check the model card for any required vLLM flags.

#### Step 6: Pre-download the weights (one-time, optional)

```bash
modal run serve_cope.py::download_model
```

This runs a tiny cheap machine (no GPU, ~$0.01) that downloads the model weights and stores them on the persistent volume. For cope's 50 GB it takes 10–20 minutes. After this, every future startup is fast because the weights are already there. **Optional**, but saves ~$1.50 per future cold start.

#### Step 7: Deploy the server

```bash
modal deploy serve_cope.py
```

This publishes the recipe to Modal. Modal prints a public URL like `https://juliet--cope-b-a4b-serve.modal.run`. The server isn't *running* yet — Modal will spin up a machine only when a request arrives. From this point on the laptop can be closed; the endpoint stays available.

(`modal deploy` is the production-style command. There's also `modal serve` which is for interactive development — that one tears down when you close the terminal.)

#### Step 8: Send a test request

The cope prompt format is unusual — see Part 2 for the full template. For a quick smoke test against most vLLM-served models, the chat-completions endpoint works:

```bash
curl https://juliet--cope-b-a4b-serve.modal.run/v1/chat/completions \
  -H "Authorization: Bearer sk-your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zentropi-ai/cope-b-a4b",
    "messages": [{"role":"user","content":"say hi"}],
    "max_tokens": 64
  }'
```

The very first request after a long idle period is a **cold start** — Modal provisions a GPU machine, loads the weights into GPU memory, and starts serving. Takes ~2–3 minutes for a 50 GB model. Subsequent requests are fast (a few seconds). If 10 minutes go by with no requests, the machine spins down, and the next request pays the cold-start cost again.

#### Step 9: Shut it down when fully done

```bash
modal app stop cope-b-a4b
```

This guarantees the GPU machine can't spin up. The volume (with cached weights) sticks around for a few cents per month. To delete the volume too, use `modal volume delete hf-cache`.

### Errors we hit along the way, decoded

These are common when standing up a new model. Some background on each:

1. **"Deprecated on 2025-02-24: 'container_idle_timeout' renamed to 'scaledown_window'."**
   Modal renamed a parameter in their library. Our Python file used the old name. One-line fix.

2. **"Each item should be of the form `<KEY>=VALUE`."**
   When we tried to type the secret-creation command across multiple lines using backslash-continuation, blank lines between the lines broke the continuation. The shell ran a command without any actual values. Fix: write it on one line.

3. **"modal-http: invalid function call."**
   We tried the wrong URL — guessed at it instead of looking it up on the dashboard.

4. **"modal-http: app for invoked web endpoint is stopped."**
   We had stopped the app earlier and never re-deployed it.

5. **"RuntimeError: Engine core initialization failed."**
   The big one. vLLM was trying to JIT-compile a small piece of GPU code on-the-fly the first time someone asked for a response. The Linux machine we'd configured had CUDA's runtime libraries but not the compiler (`nvcc`). Fix: switch to a beefier base image that includes the compiler. Costs about 5 minutes of one-time image-rebuild.

### What "good" looks like

When everything is working, the test curl above returns JSON in the shape:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [{
    "message": {"role": "assistant", "content": "Hi there!"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
}
```

That's the OpenAI Chat Completions API shape — meaning anything that already speaks "OpenAI" (the OpenAI SDK, LangChain, evaluation harnesses, etc.) can be pointed at our endpoint by changing the `base_url` and `api_key` and will work without other code changes.

### Quick reference

```bash
# Start of session
modal deploy serve_cope.py
# … test things, send requests …
# End of session (optional, saves the cents)
modal app stop cope-b-a4b

# Check what's happening on the GPU machine
modal app logs cope-b-a4b

# Update the code, redeploy
modal deploy serve_cope.py     # same command, picks up changes

# Replace a secret value
modal secret create --force cope-secrets HF_TOKEN=hf_new VLLM_API_KEY=sk-new
```

### Costs in one place

| What | When billed | Rough cost |
|---|---|---|
| Modal GPU (H100, 1×) | Per second, only while serving a request | ~$4/hour active |
| Modal CPU container (`download_model`) | Per second | ~$0.01 for the 15-min download |
| Persistent volume (50 GB) | Continuously | ~$1/month |
| Hugging Face | Free | $0 |
| Our laptop | N/A | $0 — laptop just sends curl requests |

For a typical week of intermittent testing: expect **$5–15/week**, mostly GPU time. Smaller models on smaller GPUs (A100, L40S) cost roughly half.

---

## Part 2: Evaluating

This is where Part 1 starts paying off. With a working endpoint, we can run the same model against a labelled test set under different "policy" prompts and see how its decisions change.

### What kind of models this section is about

We're focused on **policy-conditioned safety classifiers**: open-weight models that take both a *policy* (a written description of what counts as a violation) and *content* (a post, message, transcript) and output a label, typically binary. Cope, gpt-oss-safeguard, Llama-Guard, and ShieldGemma all fit this pattern, though they differ in their exact prompt formats and output shapes.

The appeal of these models compared to a traditional classifier is that **the same weights can do many different jobs** — you swap the policy text, and now you have a different classifier. That's the whole point. It also raises the question this section is built around: **how does performance change as the policy text gets more or less detailed?**

### Why varying-detail policies?

A policy can be a one-liner ("flag self-harm content") or a multi-page Includes/Excludes document. Different teams will have different appetites for writing detail — and different operational tolerances for false positives vs false negatives. By writing the same policy at multiple levels of detail and running the same test set against each, we can see:

- How much recall (catching the bad stuff) you gain from added detail.
- Whether precision (not flagging the good stuff) holds up as detail grows.
- Whether one well-written paragraph is enough, or whether you really do need the full structure.
- Whether the *official* policy from a model creator agrees with how a downstream platform would label content.

The last point turns out to matter a lot — see the self-harm case study below.

### How models receive a prompt — and why this varies

Different policy-conditioned classifiers expect different prompt shapes and produce different answer shapes. Some examples:

- **Cope** uses a text template (`INSTRUCTIONS / POLICY / CONTENT / ANSWER` blocks, model emits a single `0` or `1` after the ANSWER header). cope-b is chat-template-aware (`/v1/chat/completions`); cope-a is a Gemma-2 base with no chat template (`/v1/completions`). `max_tokens=1`, `temperature=0`.
- **gpt-oss-safeguard** reasons (chain-of-thought) then emits a final `0` or `1`; needs a large `max_tokens` (~2048) so it has room to finish reasoning before answering, and the answer is the last `0`/`1` in the response.
- **Shieldstral** doesn't generate text at all — it scores in one forward pass, and we read the yes/no token logits directly and renormalize to a 0–1 score. It has a **multimodal variant** (`models/shieldstral_mm.py`) that judges an *image* the same way — same logit-scoring, but the Document is a picture (optionally paired with the post's caption). See [Multimodal (image) evals](#multimodal-image-evals).
- **Llama-Guard** uses a fixed taxonomy embedded in the prompt and emits `safe` / `unsafe` plus a category code.

The harness isolates exactly these differences into a small **adapter** per model, so the eval loop, metrics, and output format never change. The three things an adapter encapsulates:

1. **Transport** — how you reach the model (OpenAI-compatible HTTP endpoint, local in-process inference, a vendor API).
2. **Prompt construction** — how policy + content become the model's expected input.
3. **Answer extraction** — first token, last `0`/`1` after reasoning, yes/no logits → score, or a label string.

You do **not** edit the harness to add a model — you add an adapter file. See ["Adapting this whole setup for a different model"](#adapting-this-whole-setup-for-a-different-model) below for the adapter contract and a copy-paste starting point.

### Directory layout (the `eval/` folder)

```
eval/
├── eval.py                       # the generalized harness — pick a model with --model
├── models/                       # one adapter per model (the only per-model code)
│   ├── _openai_http.py           # shared helper for vLLM/OpenAI-style HTTP endpoints
│   ├── cope_b.py                 # zentropi-ai/cope-b-a4b   (HTTP, chat)
│   ├── cope_a.py                 # zentropi-ai/cope-a-9b    (HTTP, raw completions)
│   ├── safeguard.py              # openai/gpt-oss-safeguard-20b (HTTP, CoT)
│   ├── shieldstral.py            # mistralai/Shieldstral-1.0-3B (local, text)
│   └── shieldstral_mm.py         # mistralai/Shieldstral-1.0-3B (local, image / multimodal)
├── compare_predictions.py        # A/B a run vs another (format-sensitivity check)
├── steerability.py               # readout: flip rate under an INVERTED policy (see Robustness probes)
├── offtopic.py                   # readout: flag rate under an OFF-TOPIC (null) policy
├── carveout.py                   # readout: selective release under a CARVE-OUT policy
├── rescore.py                    # re-score existing predictions against an alternate label set
├── probes/
│   └── selfharm.csv              # calibration probes run before a sweep with --probes
├── test_set.csv                  # default: 100-row self-harm test set
├── policies/                     # one .md file per policy variant
│   ├── minimal / simple / medium / full / very_long .md   # self-harm, increasing detail
│   ├── zentropi_official.md      # official Zentropi self-harm policy (from their API)
│   ├── selfharm_inverted.md      # INVERTED mirror — steerability probe
│   ├── sexual_content_*.md       # sexual-content domain (same spread + kink carve-outs, inverted)
│   ├── scam_*.md                 # scam & spam domain
│   └── violent_extremism_offtopic.md   # OFF-TOPIC null policy — off-topic probe
├── sexual_content_eval/                     # data-prep workspace for the sexual-content eval
│   ├── build_redteam.py              # generates synthetic red-team set
│   ├── merge_test_set.py             # combines labelled + red-team into the final test set
│   ├── test_set.csv                  # text test set (produced by merge_test_set.py)
│   └── test_set_sex_images.csv       # image test set (content = relative image path)
├── scam_eval/                    # data-prep workspace for the scam/spam eval
│   ├── build_image_set.py            # downloads + assembles the image test set
│   ├── test_set_scam.csv             # scam text set
│   ├── test_set_spam.csv             # spam-inclusive re-labeling of the same content
│   ├── test_set_nonenglish.csv       # multilingual text set
│   ├── test_set_images.csv           # image set (content = relative image path)
│   └── test_set_images_caption.csv   # image + caption set (content = JSON {image, text})
├── eval_cope.py, eval_shieldstral.py # original single-model scripts (superseded)
└── results/                          # CSVs of predictions and per-policy metrics
```

### Running an evaluation, step by step

The command is the same for every model — you only change `--model`. Local models (like Shieldstral) need no server; served models (cope, safeguard) need their Modal endpoint alive first.

#### Step 1 (served models only): make sure the endpoint is alive

Local adapters skip this entirely. For an HTTP-served model:

```bash
export VLLM_API_KEY="<the key you put in the Modal secret>"   # same value as Part 1 Step 4
modal app list        # find your app; if stopped, redeploy:
modal deploy serve_cope.py
```

The first request after a deploy pays the cold-start cost (~2–3 minutes); the harness's first call absorbs it.

#### Step 2: Run the harness

Self-harm eval, all policies. Add `--probes` to sanity-check the model's orientation before the sweep (see Step 4):

```bash
cd eval
python eval.py --model shieldstral --label sh \
  --policies minimal simple medium full very_long zentropi_official \
  --probes probes/selfharm.csv
```

Same test set, a served model instead — only `--model` and `--concurrency` change:

```bash
python eval.py --model cope_b --label sh \
  --policies minimal simple medium full zentropi_official \
  --concurrency 16
```

Sexual-content eval (any model):

```bash
python eval.py --model shieldstral --test-set sexual_content_eval/test_set.csv --label sex \
  --policies sexual_content_minimal sexual_content_simple sexual_content_medium \
  sexual_content_zentropi_long sexual_content_oai sexual_content_oai_adapted sexual_content_very_long
```

Useful flags:

- `--model NAME` — adapter module under `models/` (`shieldstral`, `cope_b`, `cope_a`, `safeguard`).
- `--test-set PATH` — any CSV with columns `id, content, ground_truth` (default: `test_set.csv`).
- `--label TAG` — tag in the output filenames so runs don't collide (e.g. `sh`, `sex`).
- `--policies A B C` — policy files (without `.md`) from `policies/`; each becomes a column.
- `--limit N` — first N rows only (smoke tests).
- `--concurrency N` — parallel requests. Applies to HTTP adapters; local adapters run sequentially regardless (one model instance is the bottleneck).
- `--probes PATH` — a small calibration CSV run before the sweep (Step 4).
- `--model-arg KEY=VALUE` — override any adapter default without editing code; repeatable. Examples: `--model-arg endpoint=https://...` (point at a redeployed URL), `--model-arg max_tokens=4096`, `--model-arg threshold=0.7` (Shieldstral's score cutoff), `--model-arg prompt_style=chat` (safeguard's native format).

The harness **checkpoints per policy** to `results/ckpt_<model>_<tag>_<policy>.csv` and resumes if interrupted, and **retries once** on an empty response (bumping the output budget) — so a truncated reasoning answer gets a second chance before being recorded as an error.

#### Step 3: Read the outputs

Each run writes two files under `eval/results/`:

- **`predictions_<model>_<tag>_<timestamp>.csv`** — one row per test sample, with a `<policy>_pred` and `<policy>_raw` column per policy. The `_raw` column holds the model's raw output — or `score=<0-1>` for score-based models like Shieldstral, which is what lets you re-threshold later without re-running. Open this to see exactly which samples each policy got wrong.
- **`summary_<model>_<tag>_<timestamp>.csv`** — one row per policy with TP/FP/FN/TN, an `errors` count (responses that weren't a usable `0`/`1`), and precision/recall/F1/accuracy. Errors are excluded from the rate denominators, not silently counted as misses.

The harness also prints the summary table to the terminal at the end of each run.

#### Step 4: Trust the numbers before you report them

Two cheap checks the harness makes easy — do both for any model you haven't run before:

- **Calibration probes (`--probes probes/selfharm.csv`)** run a handful of unambiguous items (an explicit violation, neutral text, a safe near-miss like a recovery post) *before* the sweep and print each result. If the obvious violation scores "safe" or everything comes out identical, your prompt template or score orientation is wrong — the run aborts if every probe is inverted. This is what caught Shieldstral's setup early; it costs seconds and saves a wasted sweep.
- **Format-sensitivity A/B (`compare_predictions.py`)** — run the same set/policy two ways (e.g. a different prompt phrasing) and compare, to confirm your numbers aren't an artifact of one arbitrary formatting choice. We used this twice — raw-vs-harmony for safeguard, criterion-in-Query for Shieldstral — before believing either model's sweep:

  ```bash
  python compare_predictions.py results/predictions_A.csv results/predictions_B.csv --policy medium
  # prints per-run P/R/F1 and the number of prediction flips
  ```

  A large flip count means the model is sensitive to how you phrased the prompt, and you need to decide which phrasing is the fair one (and say so in the writeup) rather than quietly picking the higher score.

### Building a test set when one doesn't exist

The self-harm eval used a pre-labelled CSV from a partner. For sexual content (and most new domains you'll evaluate), you'll need to build a labelled test set from scratch. We use a two-part approach:

#### Part A — Stratified sample from a public dataset for manual labelling

> **Note:** the original `sexual_content_eval/sample_bsky_for_sexual_content_eval.py` sampler is no longer checked in — its labelled output (`sexual_content_eval/candidates_to_label.csv`) and the merged `test_set.csv` remain. The pattern below is still the methodology we follow; `scam_eval/build_image_set.py` is the closest live worked example (it assembles the image test set with the same tiering + hard-drop approach). Treat the commands here as the recipe to re-implement, not a script to run as-is.

The sampler pattern downloads one shard of a public Bluesky post dataset (~130 MB, ~390k posts) and filters it into three "tiers":

- **Tier A** — strong-signal keywords (likely-violating) — drives recall measurement
- **Tier B** — borderline / suggestive keywords — judgment-call zone
- **Tier C** — no signal keywords — drives precision measurement

The sampler hard-drops:

- posts with images (we only label text)
- posts shorter than 12 chars or longer than 400 chars
- mostly-URL or mostly-non-Latin posts
- **anything where a sexual term co-occurs with a minor-related term** (CSAM-adjacent; dropped entirely rather than risked)

It pulls roughly 35 / 30 / 15 = **80 posts** to a `candidates_to_label.csv` with an empty `ground_truth` column. Open the CSV in Numbers/Excel and fill it in: `1` = violating, `0` = not, blank to skip.

🔧 **Adapt for a different domain**: swap the keyword regex lists. The hard-drop rules around image posts, length, and minor-related terms should stay.

#### Part B — Synthetic red-team set

A real dataset alone leaves blind spots: the exclusions and edge cases that policy writers worry about (recovery narratives, educational content, fictional framings) appear too rarely in random samples to drive metrics. So we also write a 50-row hand-crafted set covering categories chosen to stress-test specific policy clauses.

`sexual_content_eval/build_redteam.py` is a worked example. Each of its 10 categories targets a clause in the cope sexual-content policy:

- clear explicit sex acts (5, expected 1)
- explicit invitations / participation offers (5, expected 1)
- body / anatomy with erotic framing (5, expected 1)
- coded language / euphemisms (5, expected 1)
- sexual humor / hyperbole — judgment calls (3 expected 1, 2 expected 0)
- educational / clinical content (5, expected 0)
- recovery / addiction narratives (5, expected 0)
- fictional creative writing — graphic vs non-graphic (3 expected 1, 2 expected 0)
- sexually degrading speech vs critique of such speech (3 expected 1, 2 expected 0)
- factual body-part mentions without sexual framing (5, expected 0)

🔧 **Adapt for a different domain**: write your own category list from the policy's Includes/Excludes, then ~5 examples per category. The goal is one example per policy clause, plus one or two "almost but not quite" examples per exclusion clause.

#### Part C — Merge and run

Once labels are filled in:

```bash
python merge_test_set.py
# combines labelled + redteam into sexual_content_eval/test_set.csv
cd ..
python eval.py --model shieldstral \
  --test-set sexual_content_eval/test_set.csv \
  --label sex \
  --policies sexual_content_minimal sexual_content_simple sexual_content_medium sexual_content_zentropi_long sexual_content_oai
```

### Worked examples

We've run this end-to-end on two harm domains using cope-b-a4b: **self-harm** (pre-labelled 100-row test set) and **sexually explicit content** (test set built from a stratified Bluesky sample + a synthetic red-team set). See [RESULTS.md](RESULTS.md) for the test-set details, the per-policy precision/recall/F1 numbers, the head-to-head against gpt-oss-safeguard, and the findings around Zentropi's published policies and policy-format alignment.

The general-purpose takeaway from those runs, that future evals should bake in:

**Measuring "accuracy" only makes sense after you've checked that the test set's labelling framework matches the policy's framework.** Both cope evaluations turned up substantial disagreements between Zentropi's published policies and the labelled test sets — not because the model was wrong, but because the policies and the labels were optimising for different things. When the two disagree, the numbers report the *disagreement*, not the model. Always look at the false positives and false negatives by hand before drawing model-quality conclusions from F1.

### Multimodal (image) evals

The same harness scores **images** — no core changes. The trick: the test-set schema is unchanged (`id, content, ground_truth`), but for an image set the `content` column holds an **image file path** instead of text, and the multimodal adapter (`models/shieldstral_mm.py`) opens that path and judges the picture. Everything else — policies, checkpointing, metrics, output format — is identical.

```bash
cd eval
python eval.py --model shieldstral_mm --test-set scam_eval/test_set_images.csv --label images \
  --policies scam_minimal scam_simple scam_full scam_spam_inclusive
python eval.py --model shieldstral_mm --test-set sexual_content_eval/test_set_sex_images.csv --label seximg \
  --policies sexual_content_minimal sexual_content_simple sexual_content_medium sexual_content_very_long
```

Two content shapes the multimodal adapter accepts in the `content` column:

- **A plain image path** — judge the image alone (`scam_eval/test_set_images.csv`).
- **A JSON object `{"image": <path>, "text": <caption>}`** — judge the image *together with* the post's caption text (`scam_eval/test_set_images_caption.csv`). This is how you test whether a caption changes the model's read of the picture.

**Image paths are stored relative to the `eval/` directory** (e.g. `scam_eval/scam_images/scamimg001.jpg`) so the test sets are portable across machines. The adapter resolves a relative path against `eval/` regardless of where you launch from, or against `--model-arg image_base=/some/dir` if your images live elsewhere. The **scam** image binaries are tracked in the repo. The **sexual-content** image binaries (`sexual_content_eval/sexual_content_images/`) are gitignored and not distributed — regenerate them locally before running that eval. `scam_eval/build_image_set.py` is the worked example of downloading and assembling an image set.

### Scam, spam, and multilingual domains

Beyond self-harm and sexual content, the repo carries a **scam/spam** domain (`scam_eval/`, `policies/scam_*.md`) and **multilingual** text sets (`*_nonenglish_*.csv`). They run through the identical command — only `--test-set` and `--policies` change:

```bash
python eval.py --model shieldstral --test-set scam_eval/test_set_scam.csv --label scam \
  --policies scam_minimal scam_simple scam_full
python eval.py --model shieldstral --test-set scam_eval/test_set_nonenglish.csv --label scam_ml \
  --policies scam_simple scam_full
```

`scam` vs `spam` is the same content under two labelings (scam-only, or scam+spam). Rather than re-run inference, run once and re-score against the alternate labels with `rescore.py` (below).

### Robustness probes — reading beyond F1

A set of small **readout scripts** turn a normal sweep into a robustness probe. Each pairs a normal policy with a *twisted* one in the **same** eval.py sweep, then reads the predictions file with a purpose-built metric — because for these the `summary_*.csv` F1 is **meaningless** (the ground_truth no longer matches what the twisted policy asks). Always read these with the dedicated script, not the summary CSV.

- **`steerability.py`** — pair a policy with its **inverted** mirror (`selfharm_inverted.md`, `sexual_content_inverted.md`). On truly-violating rows, measures the *flip rate* 1→0 when the policy is inverted. Flip≈0 means the model held its safety prior (ignored the policy); flip≈1 means it followed the policy. This is the steerability number to report — **not** the summary F1.

  ```bash
  python eval.py --model shieldstral --label steer \
    --policies simple selfharm_inverted --test-set test_set.csv
  python steerability.py 'results/predictions_shieldstral_*_steer_*.csv' \
    --baseline simple --inverted selfharm_inverted
  ```

- **`offtopic.py`** — pair with an **off-topic** policy about a harm the set doesn't contain (`violent_extremism_offtopic.md`). Correct answer is 0 on every row, so every flag is a false flag; a policy-literal model flags ~0%. Reads flag rate.
- **`carveout.py`** — pair a broad policy with a **carve-out** variant that reclassifies one subcategory (e.g. kink) from violating to permitted. Measures whether the model releases *only* the carved-out subcategory while still flagging the rest. Needs a per-row `carveout` label in the test set (passed via `--labels`).
- **`rescore.py`** — not a probe: re-scores an existing predictions CSV against a different label set (joined on `id`), so one inference pass covers multiple scopes (e.g. scam-only vs scam+spam).

### Reproducing or extending

To add a new policy variant: drop a markdown file into `eval/policies/` (any structure — the harness reads it as a single text blob), then pass its name (without `.md`) to `--policies`.

To add a new harm domain: build a test CSV with columns `id, content, ground_truth`, write 3–5 policy variants under `eval/policies/`, and run:

```bash
python eval.py --model <model> --test-set path/to/your_test_set.csv --label your_domain --policies policy_a policy_b ...
```

The harness is policy-agnostic — it hands whatever you put in `policies/<name>.md` to the chosen model's adapter unchanged.

### Adapting this whole setup for a different model

The `eval_cope.py` / `eval_shieldstral.py` split proved the pattern, then got refactored into a single generalized harness (`eval/eval.py`) so adding a model is writing one small adapter file rather than copying a script. **This is the recommended path for any new model.**

#### The generalized harness (`eval/eval.py` + `eval/models/`)

The core (`eval.py`) owns everything model-agnostic: the test-set schema (`id, content, ground_truth`), policy loading, per-policy checkpointing (resume after an interrupt), one automatic retry on empty responses, optional calibration probes, metrics with explicit error accounting, and the shared `predictions_*.csv` / `summary_*.csv` output format. It never knows how any specific model works.

Each model is an **adapter** under `eval/models/` — a module exposing:

```python
NAME = "my-model-1.0"
SUPPORTS_CONCURRENCY = True    # True for HTTP endpoints, False for local inference

def make_classifier(opts: dict):
    # opts comes from --model-arg KEY=VALUE flags, merged over the adapter's DEFAULTS
    def classify(policy_text, content, attempt=0):
        # ... call the model ...
        # return (pred, raw): pred in {"1", "0", ""};  raw is the model's output,
        # or "score=<0-1>" for score-based models.  attempt>0 means the harness is
        # retrying an empty answer, so you may raise the output budget here.
        return pred, raw
    return classify
```

The four shipped adapters cover the three transport shapes you'll encounter:

- `models/cope_b.py`, `models/cope_a.py`, `models/safeguard.py` — HTTP against an OpenAI-compatible vLLM endpoint, via the shared `models/_openai_http.py` helper (handles chat-vs-completions detection and 0/1 parsing). These differ only in a prompt template and a few defaults.
- `models/shieldstral.py` — **local inference**, no server at all: loads the model with PyTorch/transformers and reads the yes/no logits in-process. This is the template for any laptop-runnable model.
- `models/shieldstral_mm.py` — the **multimodal** sibling: same local logit-scoring, but the content is an image path (or `{image, text}` JSON) and it judges the picture. The template for any image-capable local model.

Run any of them the same way:

```bash
python eval.py --model shieldstral --label sh \
  --policies minimal simple medium full very_long zentropi_official \
  --probes probes/selfharm.csv
python eval.py --model cope_b --test-set sexual_content_eval/test_set.csv --label sex \
  --policies sexual_content_minimal ... --concurrency 16
# override any adapter default without editing code:
python eval.py --model safeguard ... --model-arg max_tokens=4096 --model-arg prompt_style=chat
```

**To add a model:** copy the closest adapter (HTTP → start from `cope_b.py`; local → start from `shieldstral.py`), change the prompt template, the transport defaults, and the answer parsing. Nothing else moves. If it's served, you still stand up a `serve_<model>.py` on Modal as in Part 1 and point the adapter's `endpoint` default at it.

#### Two habits baked into the core — use them

- **Calibration probes (`--probes`)** run a tiny labelled set (one explicit violation, one neutral, one safe near-miss like a recovery post) *before* the sweep and abort if the result comes out fully inverted. This is what caught Shieldstral's score orientation early; it's free insurance against a broken template burning a full run. A starter set is at `eval/probes/selfharm.csv`.
- **Format-sensitivity checks (`compare_predictions.py`)** re-run one set × one policy under an alternative prompt phrasing and report the P/R/F1 delta and prediction flips. We used this twice to earn trust in the numbers (raw-vs-harmony for safeguard, criterion-in-Query for Shieldstral). Do it for any new model before believing its sweep:

  ```bash
  python compare_predictions.py results/predictions_A.csv results/predictions_B.csv --policy medium
  ```

Everything else — the stratified sampler, the red-team construction approach, the merge script, the metrics — generalises without changes. The old `eval_cope.py` and `eval_shieldstral.py` remain in the repo as the worked examples this refactor came from; new work should use `eval.py`.

#### Worked example: the two adapter shapes

The shipped adapters show both transports end to end:

- **Served (`models/cope_b.py`)** — a prompt template plus a couple of defaults, handed to the shared `models/_openai_http.py` helper. `cope_a.py` and `safeguard.py` are near-identical: same helper, different template/endpoint/budget. Start here for anything behind a vLLM endpoint.
- **Local (`models/shieldstral.py`)** — loads the model with PyTorch/transformers, runs one forward pass, reads the yes/no logits, renormalizes to a 0–1 score, thresholds. No server, no HTTP. Start here for anything small enough to run in-process. It also honors `--model-arg threshold=...` so you can re-threshold without re-running, and `--model-arg device=...` to force cpu/cuda/mps.

Because every adapter returns the same `(pred, raw)` and the harness owns the output format, results from any model drop straight into the same comparison tables — which is what makes cross-model rows in RESULTS.md possible.

### Files cheat sheet

| File | Purpose |
|---|---|
| `eval/eval.py` | **The harness** — pick a model with `--model`, writes predictions + summary |
| `eval/models/*.py` | Per-model adapters (the only per-model code); `_openai_http.py` is a shared helper; `shieldstral_mm.py` is the image adapter |
| `eval/compare_predictions.py` | A/B two runs on one policy — the format-sensitivity check |
| `eval/steerability.py` | Flip rate under an inverted policy (read this, not summary F1) |
| `eval/offtopic.py` | Flag rate under an off-topic (null) policy |
| `eval/carveout.py` | Selective-release rate under a carve-out policy |
| `eval/rescore.py` | Re-score existing predictions against an alternate label set |
| `eval/probes/selfharm.csv` | Calibration probes, passed via `--probes` |
| `serve_cope.py` | Modal deployment recipe for served models — Part 1 |
| `eval/test_set.csv` | Default test set (self-harm, 100 rows; column identifiers sanitized) |
| `eval/policies/*.md` | Policy variants — one per file, name passed to `--policies` |
| `eval/results/` | Output CSVs (predictions + summary) timestamped per run |
| `eval/sexual_content_eval/`, `eval/scam_eval/` | Data-prep workspaces (samplers, red-team, image sets) per domain |
| `eval/sexual_content_eval/build_redteam.py` | Generates 50 synthetic red-team cases |
| `eval/sexual_content_eval/merge_test_set.py` | Combines labelled + red-team into final test set |
| `eval/*/test_set_images*.csv` | Image test sets — `content` column is a relative image path |
| `eval/eval_cope.py`, `eval/eval_shieldstral.py` | Original single-model scripts, superseded by `eval.py` |
