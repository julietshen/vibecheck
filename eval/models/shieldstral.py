"""mistralai/Shieldstral-1.0-3B — local inference (PyTorch, Apple MPS / CUDA / CPU).

Policy-conditioned classifier that scores in a single forward pass: the yes/no
token logits are renormalized into a 0-1 score, thresholded at 0.5. No text is
generated, so it cannot emit a malformed answer and has no reasoning budget to
exhaust. raw is returned as "score=<0-1>".

Format: fixed system message, then a user message with <Query> (the policy — the
criterion the Document is judged against, matching Shieldstral's reranker-style
prompt), <Instruct> (generic task + answer polarity, yes = violation), and
<Document> (the content).

Model + weights load lazily on first call. Runs sequentially
(SUPPORTS_CONCURRENCY = False) — a single local model instance is the
bottleneck, and parallel calls would just contend for the one GPU."""

from __future__ import annotations

import math

NAME = "Shieldstral-1.0-3B"
SUPPORTS_CONCURRENCY = False

MODEL_ID = "mistralai/Shieldstral-1.0-3B"
SYSTEM = ("Judge whether the Document meets the requirements based on the Query "
          "and the Instruction provided.")
# The policy is the Query (the criterion the Document is judged against). Instruct
# carries only the generic task framing and answer polarity (yes = violation).
INSTRUCT = ("You are a content safety classifier for an online platform. The Query "
            "contains a content policy. Answer yes if the Document violates that "
            "policy, and no if it does not.")
MAX_DOC_CHARS = 8000


def _yes_no_ids(tok):
    ids = {"yes": set(), "no": set()}
    for word in ("yes", "no"):
        for v in (word, word.capitalize(), word.upper()):
            for prefix in ("", " "):
                enc = tok.encode(prefix + v, add_special_tokens=False)
                if len(enc) == 1:
                    ids[word].add(enc[0])
    assert ids["yes"] and ids["no"], "no single-token yes/no variants found"
    return sorted(ids["yes"]), sorted(ids["no"])


def make_classifier(opts: dict):
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText

    model_id = opts.get("model", MODEL_ID)
    instruct = opts.get("instruct", INSTRUCT)
    threshold = float(opts.get("threshold", 0.5))
    if "device" in opts:
        device = opts["device"]
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    YES, NO = _yes_no_ids(tok)

    def classify(policy_text: str, content: str, attempt: int = 0):
        user = (f"<Instruct>: {instruct}\n\n<Query>: {policy_text}\n\n"
                f"<Document>: {content[:MAX_DOC_CHARS]}")
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        input_ids = (enc["input_ids"] if not torch.is_tensor(enc) else enc).to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits[0, -1].float()
        z_yes = max(logits[j].item() for j in YES)
        z_no = max(logits[j].item() for j in NO)
        score = math.exp(z_yes) / (math.exp(z_yes) + math.exp(z_no))
        return ("1" if score >= threshold else "0"), f"score={score:.4f}"

    return classify
