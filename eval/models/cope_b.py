"""zentropi-ai/cope-b-a4b — standalone MoE classifier (4B active / 27B total).
Served via vLLM on Modal; requires /v1/chat/completions (chat-template-aware).
Emits a single 0/1 token."""

from ._openai_http import make_http_classifier

NAME = "cope-b-a4b"
SUPPORTS_CONCURRENCY = True

PROMPT_TEMPLATE = """INSTRUCTIONS
============

Examine the given POLICY and determine if the given CONTENT meets the criteria for ANY of the LABELS. Answer "1" if yes, and "0" if no.


POLICY
======

{policy}


CONTENT
=======

{content}


ANSWER
======

"""

DEFAULTS = {
    "endpoint": "https://juliet--cope-b-a4b-serve.modal.run/v1/chat/completions",
    "model": "zentropi-ai/cope-b-a4b",
    "max_tokens": "1",
}


def make_classifier(opts: dict):
    o = {**DEFAULTS, **opts}
    return make_http_classifier(
        lambda policy, content: PROMPT_TEMPLATE.format(policy=policy, content=content), o)
