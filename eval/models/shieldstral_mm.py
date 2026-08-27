"""mistralai/Shieldstral-1.0-3B — multimodal (image) variant of the shieldstral
adapter. The `content` passed by the harness is an **image file path** (or a
local path to a downloaded Bluesky blob); the policy is text in <Instruct> and
the Document being judged is the image.

Same single-forward-pass scoring as the text adapter: yes/no token logits at the
final position are renormalized into a 0-1 score, thresholded at 0.5. Needs
torchvision (Pixtral image processor). Runs sequentially.
"""
from __future__ import annotations
import math
from pathlib import Path

NAME = "Shieldstral-1.0-3B-mm"
SUPPORTS_CONCURRENCY = False

# Test-set CSVs store image paths relative to the eval/ directory (e.g.
# "scam_eval/scam_images/scamimg001.jpg") so they are portable across machines.
# EVAL_DIR is eval/ (this file lives at eval/models/shieldstral_mm.py).
EVAL_DIR = Path(__file__).resolve().parents[1]


def _resolve_image_path(path: str, base: Path) -> str:
    """Return an existing path for `path`. Absolute paths are used as-is; a
    relative path is tried against the process cwd first, then `base` (the eval/
    dir, or an --model-arg image_base override), so evals run from anywhere."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return str(p)
    candidate = base / path
    return str(candidate if candidate.exists() else p)

MODEL_ID = "mistralai/Shieldstral-1.0-3B"
SYSTEM = ("Judge whether the Document meets the requirements based on the Query "
          "and the Instruction provided.")
QUERY = "Does the image Document violate the policy?"
INSTRUCT_PREAMBLE = ("You are a content safety classifier for an online platform. "
                     "Judge the image Document strictly according to the policy below.\n\n")


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
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_id = opts.get("model", MODEL_ID)
    query = opts.get("query", QUERY)
    threshold = float(opts.get("threshold", 0.5))
    image_base = Path(opts["image_base"]) if opts.get("image_base") else EVAL_DIR
    if "device" in opts:
        device = opts["device"]
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    proc = AutoProcessor.from_pretrained(model_id)
    tok = proc.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    YES, NO = _yes_no_ids(tok)

    def classify(policy_text: str, content: str, attempt: int = 0):
        import json
        instruct = INSTRUCT_PREAMBLE + policy_text
        # content is either a plain image path, or a JSON object
        # {"image": <path>, "text": <caption>} to judge image + post caption together.
        caption = ""
        path = content.strip()
        if path.startswith("{"):
            try:
                obj = json.loads(path)
                path = (obj.get("image") or "").strip()
                caption = (obj.get("text") or "").strip()
            except Exception:
                pass
        path = _resolve_image_path(path, image_base)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            return "", f"ERROR loading image {path!r}: {e}"
        doc = f"<Document>:\n{caption}" if caption else "<Document>:"
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "text", "text": f"<Instruct>: {instruct}\n\n<Query>: {query}\n\n{doc}"},
                {"type": "image"},
            ]},
        ]
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=prompt, images=[img], return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1].float()
        z_yes = max(logits[j].item() for j in YES)
        z_no = max(logits[j].item() for j in NO)
        score = math.exp(z_yes) / (math.exp(z_yes) + math.exp(z_no))
        return ("1" if score >= threshold else "0"), f"score={score:.4f}"

    return classify
