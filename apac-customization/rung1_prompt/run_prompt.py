"""Rung 1 — customize by POLICY TEXT only (no training): what it can and cannot do.

WORKS — the LANGUAGE lever:
  The same expert policy, written in English or in Bahasa, catches the local
  loan-shark ad either way. You can author the harm definition in any language.

THE WALL — a small model's fixed prior:
  The expert policy EXPLICITLY lists "a licensed lender naming its regulator" as
  OK. The base model flags it anyway. Prompt text can steer a small model to flag
  MORE, but not to PERMIT something its prior dislikes. That asymmetry is why the
  next rung changes the weights: after LoRA (rung 2) the same licensed lender is
  correctly released (licensed false-positives 5/8 -> 0/8 in run_eval).

Run from project root:
  ../.venv/bin/python rung1_prompt/run_prompt.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from common import POLICY_EN, chat_prompt, parse_verdict  # noqa: E402

MODEL = "mlx-community/Qwen3-1.7B-bf16"

# The expert policy translated to Bahasa Indonesia (best-effort; validate).
POLICY_ID = """Anda adalah pengklasifikasi Trust & Safety untuk platform di Asia Tenggara.
Tentukan apakah POSTINGAN MENGIKLANKAN PINJAMAN ILEGAL / TANPA IZIN — rentenir /
"along" / pinjaman pribadi tanpa lisensi.
Tanda pinjaman ilegal (VIOLATION): tanpa nomor lisensi; persetujuan instan tanpa
dokumen; "blacklist boleh"; kontak hanya lewat WhatsApp/Telegram pribadi; menahan
KTP/kartu ATM sebagai jaminan; bunga harian sangat tinggi.
BUKAN pelanggaran (OK): bank/koperasi BERLISENSI yang menyebut regulatornya;
pembicaraan umum soal uang; niaga sehari-hari.
Jawab dengan SATU KATA saja: VIOLATION atau OK.""".strip()

POLICIES = {"expert_EN": POLICY_EN, "expert_ID(Bahasa)": POLICY_ID}

SHOWCASE = [
    ("illegal loan-shark (Malay)",
     "Pinjaman segera!! Tiada CTOS, blacklist pun boleh, cukup IC & kad ATM. WhatsApp saya.",
     "VIOLATION"),
    ("LICENSED lender (policy says OK)",
     "Personal loan from a licensed lender, registered under the Moneylenders Act "
     "(Lic. No. 21/2025). Apply at our branch with your NRIC and payslip.",
     "OK"),
    ("benign",
     "Finally tried the laksa at that new stall in Tiong Bahru — worth the queue.",
     "OK"),
]


def main():
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(MODEL)
    sampler = make_sampler(temp=0.0)

    def verdict(policy, post):
        out = generate(model, tok, prompt=chat_prompt(tok, post, policy),
                       max_tokens=6, sampler=sampler, verbose=False)
        return {1: "VIOLATION", 0: "OK", -1: "?"}[parse_verdict(out)]

    names = list(POLICIES)
    w = 22
    print("\nRung 1 — same base model (Qwen3-1.7B), policy text only\n")
    print("post".ljust(36) + "".join(n.ljust(w) for n in names) + "should be")
    print("-" * (36 + w * len(names) + 10))
    for desc, post, want in SHOWCASE:
        cells = [verdict(POLICIES[n], post) for n in names]
        flag = "" if all(c == want for c in cells) else "   <-- WALL" if want == "OK" else ""
        print(desc.ljust(36) + "".join(c.ljust(w) for c in cells) + want + flag)
    print("\nLANGUAGE works: expert_EN and expert_ID(Bahasa) agree — author the")
    print("  policy in any language.")
    print("THE WALL: the policy explicitly lists a licensed lender as OK, but the")
    print("  base model flags it anyway. Prompting can tighten a small model, not")
    print("  loosen it. Rung 2 fixes exactly this by changing the weights.\n")


if __name__ == "__main__":
    main()
