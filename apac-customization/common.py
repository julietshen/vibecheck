"""Shared prompt formatting + labels for the APAC customization ladder.

Every rung (prompt-only, LoRA, regional model) uses the SAME classifier prompt
so before/after numbers are comparable. The task is a binary Trust & Safety
verdict on a single post.

Domain: illegal / unlicensed moneylending advertising ("loan shark" / "ah long"
lending). This is a high-priority Trust & Safety harm across Southeast Asia whose
linguistic and cultural markers are regional — a Western-centric base model tends
to read these posts as ordinary, legal short-term loan offers and lets them pass.
"""

VIOLATION = "VIOLATION"
OK = "OK"

# English base policy — a concise, honest description of the harm and the
# distinction that matters (illegal/unlicensed vs. a legitimate licensed lender).
POLICY_EN = """You are a Trust & Safety classifier for a social platform used across Southeast Asia.

Decide whether the POST is ADVERTISING ILLEGAL OR UNLICENSED MONEYLENDING — i.e. a
"loan shark" / "ah long" / "5-6" / "pinjaman along" / unlicensed private-lending offer.

Signals of illegal moneylending (VIOLATION):
- Loans offered with NO licence number and NO regulated lender named.
- "Instant" / "guaranteed" approval with no documents, no credit check, "blacklist / CTOS / CMND OK".
- Contact only via personal WhatsApp / Telegram / DM; no registered business.
- Asking to hold the borrower's IC / ATM card / bank login as collateral.
- Very high or unstated interest, daily interest, or threats about repayment.

NOT a violation (OK):
- A licensed bank, finance company, or cooperative naming its regulator / licence.
- General talk about money, debt, or budgeting; news; personal stories.
- Ordinary commerce, food, and daily-life posts.

Reply with EXACTLY ONE WORD: VIOLATION or OK.""".strip()


# Minimal policy — what a team WITHOUT Southeast-Asian regional expertise would
# realistically write. No loan-shark markers, no local slang, no SEA context. A
# general model reads region-specific loan-shark ads as ordinary loan offers and
# passes them. Fine-tuning (rung 2) bakes the regional knowledge into the weights
# so this one-liner is enough; an expert in-language policy (rung 1) is the other
# way to close the same gap.
POLICY_MINIMAL = """You are a Trust & Safety classifier. Decide whether the POST is a scam or a
harmful financial offer that violates platform policy. Reply with EXACTLY ONE
WORD: VIOLATION or OK.""".strip()

POLICIES = {"minimal": POLICY_MINIMAL, "detailed": POLICY_EN}

# --- Second harm/language: offensive language (for the Bengali TB-OLID set) ---
# A generic policy a non-regional team would write, and a fuller one. "Offensive"
# follows the OLID definition: targeted or untargeted insults, profanity, or slurs
# against a person or group. Criticism of ideas / non-targeted venting is not.
POLICY_OFFENSIVE_MINIMAL = """You are a Trust & Safety classifier. Decide whether the POST is offensive,
abusive, or harassing and violates platform policy. Reply with EXACTLY ONE WORD:
VIOLATION or OK.""".strip()

POLICY_OFFENSIVE = """You are a Trust & Safety classifier for a social platform used across Asia.
Decide whether the POST is OFFENSIVE — a targeted or untargeted insult, profanity,
slur, threat, or harassment aimed at a person or group.

VIOLATION: insults or slurs aimed at someone; abusive name-calling; demeaning a
person or group; threats; sexual harassment; profanity used to attack.
OK: criticism of ideas, brands, or policies; non-targeted venting; ordinary
strong language not aimed at anyone; neutral or supportive posts.

The POST may be in romanized / code-mixed Bangla. Reply with EXACTLY ONE WORD:
VIOLATION or OK.""".strip()

POLICIES.update({"offensive_minimal": POLICY_OFFENSIVE_MINIMAL,
                 "offensive": POLICY_OFFENSIVE})

# --- Third harm/language: gendered abuse (Chinese SWSR + Indic Uli) ---
POLICY_GENDERED_MINIMAL = """You are a Trust & Safety classifier. Decide whether the POST is sexist or
gendered abuse that violates platform policy. Reply with EXACTLY ONE WORD:
VIOLATION or OK.""".strip()

POLICY_GENDERED = """You are a Trust & Safety classifier for a social platform used across Asia.
Decide whether the POST is SEXIST or GENDERED ABUSE — content that demeans,
insults, threatens, objectifies, or expresses hatred toward a person or group
based on gender, or that endorses gender-based violence or misogyny/misandry.

VIOLATION: misogynistic or misandrist insults; slut-shaming; gendered slurs;
threats or jokes about gender-based violence; demeaning stereotypes used as an
attack; harassment of someone for their gender or sexual orientation.
OK: neutral discussion of gender or feminism; personal experience; criticism of
ideas; non-gendered content.

The POST may be in Chinese, Hindi, Tamil, or Indian English. Reply with EXACTLY
ONE WORD: VIOLATION or OK.""".strip()

POLICIES.update({"gendered_minimal": POLICY_GENDERED_MINIMAL, "gendered": POLICY_GENDERED})

# Which policy each evaluation dataset should be scored under.
DATASET_DOMAIN = {"moneylending": "minimal", "offensive": "offensive_minimal",
                  "gendered": "gendered_minimal"}


def build_prompt(post: str, policy: str = POLICY_EN) -> str:
    """The single user-turn shown to the model. Kept identical across all rungs."""
    return f"{policy}\n\nPOST:\n{post}\n\nVERDICT:"


def chat_prompt(tok, post: str, policy: str = POLICY_EN):
    """Apply the model's chat template (Qwen3 thinking disabled) so base and
    fine-tuned models are driven identically. mlx_lm's LoRA loader templates the
    training data the same way, so the adapter expects this format."""
    msgs = [{"role": "user", "content": build_prompt(post, policy)}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True)


def label_word(label: int) -> str:
    return VIOLATION if int(label) == 1 else OK


def parse_verdict(text: str) -> int:
    """Map a model's free-text reply to {1,0,-1}. -1 = unparseable/abstain."""
    t = (text or "").strip().upper()
    # strip any leaked reasoning / punctuation, look at the first ~40 chars
    head = t[:40]
    has_v = "VIOLATION" in head or head.startswith("VIOLAT")
    has_ok = "OK" in head or "NOT A VIOLATION" in t.upper()
    if "NOT A VIOLATION" in t.upper() or "NOT ILLEGAL" in t.upper():
        return 0
    if has_v and not has_ok:
        return 1
    if has_ok and not has_v:
        return 0
    if has_v:
        return 1
    return -1
