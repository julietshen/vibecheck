"""Build the APAC illegal-moneylending demo dataset.

Produces a small, stage-safe, multilingual Trust & Safety set for the
customization-ladder demo. The classification target is:

    Is this post ADVERTISING ILLEGAL / UNLICENSED MONEYLENDING
    ("loan shark" / "ah long" / "5-6" / "pinjaman along") ?  ->  VIOLATION / OK

Why this domain: it is a real, high-priority harm across Southeast Asia whose
markers are language- and culture-specific. A Western-centric base model reads
these as ordinary legal short-term loans and lets them through (false negatives),
and a naive keyword filter over-flags legitimate licensed lenders (false
positives). Learning the *distinction* is exactly the regional customization the
talk is about.

Classes per language:
  - illegal_ml   (label 1): unlicensed loan-shark advertising, regional slang
  - licensed     (label 0): HARD negative — a real licensed lender / bank naming
                            its regulator; keyword-similar but legitimate
  - benign       (label 0): EASY negative — unrelated everyday post

Languages: en (English/Singlish), ms (Malay/Indonesian), tl (Tagalog/Filipino),
th (Thai), vi (Vietnamese).

============================  PROVENANCE / HONESTY  ============================
Every row here is SYNTHETIC, hand-authored for illustration by Claude (Opus 4.8)
with Juliet Shen. It is NOT scraped from any platform and NOT a validated
benchmark. The non-English phrasing is best-effort and MUST be reviewed by a
native speaker before it is shown or quoted publicly. Treat all numbers this set
produces as a DEMO of the customization mechanics, not evidence of real-world
accuracy. See NOTICE.md.
===============================================================================
"""
import csv
import json
import os
import random

random.seed(20260916)  # deterministic; APIs that forbid time-seeding are unaffected
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Template banks.  {n} = a phone/handle slot, {amt}/{rate} numeric slots.
# Positives deliberately carry the *illegal* markers: no licence, "blacklist ok",
# hold IC/ATM, personal DM only, daily interest.
# ---------------------------------------------------------------------------

ILLEGAL = {
    "en": [
        "Fast cash loan!! No CTOS no income proof, blacklist also can. WhatsApp {n}, money in 15 min 💰",
        "Ah long here bro. ${amt} today, keep your IC + ATM card as guarantee. Low low interest, DM only.",
        "Need urgent $$$? No documents no credit check. Instant approval, {rate}% daily. Telegram @{n}",
        "Loan {amt} approved on the spot, no bank no paperwork. Just give me your ATM pin as security ok?",
        "Blacklisted? Bankrupt? Still can loan! Cash same day, settle old debt first. Text {n} now.",
    ],
    "ms": [
        "Pinjaman segera!! Tiada CTOS, tiada dokumen, blacklist pun boleh. Duit masuk 10 minit. WhatsApp {n}",
        "Along boss di sini. Loan RM{amt} cepat lulus, cukup IC dan kad ATM sebagai cagaran. Faedah rendah.",
        "Duit kecemasan tanpa slip gaji, tanpa semak kredit. Lulus segera, faedah {rate}% sehari. Telegram {n}",
        "Loan mudah RM{amt}, tak payah bank tak payah borang. Serah kad ATM & pin sahaja. DM sekarang.",
        "CTOS teruk pun boleh pinjam! Cash hari ini, settle hutang lama dulu. Taip {n} sekarang.",
    ],
    "tl": [
        "5-6 lending po! Mabilis na cash, walang requirements, kahit blacklisted ok. Text {n}, 15 min lang.",
        "Instant loan ₱{amt}, sangla ATM card at IC mo bilang collateral. Mababang interest, DM lang po.",
        "Need pera agad? Walang credit check walang papeles. Approved agad, {rate}% per araw. Telegram @{n}",
        "Loan ₱{amt} approved on the spot, walang bank walang requirements. Ibigay pin ng ATM as security.",
        "Blacklisted? Bankrupt? Pwede pa rin umutang! Cash today, bayaran muna luma. Text {n} ngayon.",
    ],
    "th": [
        "เงินกู้ด่วน!! ไม่เช็คเครดิต ไม่ต้องมีเอกสาร ติดแบล็คลิสต์ก็กู้ได้ เงินเข้าใน 10 นาที ทัก {n}",
        "เงินกู้นอกระบบ อนุมัติไว {amt} บาท ใช้บัตร ATM กับบัตรประชาชนค้ำ ดอกเบี้ยถูก แชทมาเลย",
        "ต้องการเงินด่วน? ไม่ต้องมีสลิปเงินเดือน ไม่เช็คเครดิต อนุมัติทันที ดอก {rate}% ต่อวัน ไลน์ {n}",
        "กู้ {amt} บาทอนุมัติทันที ไม่ต้องผ่านธนาคาร แค่ให้รหัส ATM ไว้เป็นประกัน ทักแชทเลย",
        "ติดแบล็คลิสต์ก็กู้ได้! ได้เงินวันนี้ ปิดหนี้เก่าก่อน พิมพ์ {n} มาเลย",
    ],
    "vi": [
        "Vay tiền nhanh!! Không cần thế chấp, không cần chứng minh thu nhập, nợ xấu vẫn vay được. Nhắn {n}",
        "Cho vay nóng {amt}, chỉ cần CMND và giữ thẻ ATM làm tin. Lãi thấp, nhắn tin riêng thôi nhé.",
        "Cần tiền gấp? Không kiểm tra tín dụng, giải ngân trong 15 phút, lãi {rate}%/ngày. Zalo {n}",
        "Vay {amt} duyệt ngay, không cần ngân hàng không giấy tờ. Đưa mã PIN ATM giữ làm bảo đảm nhé.",
        "Nợ xấu, nợ chú ý vẫn vay! Tiền trong ngày, trả nợ cũ trước. Nhắn {n} ngay.",
    ],
}

# HARD negatives: real, LICENSED lenders / banks. Keyword-similar (loan, fast,
# approval, interest) but legitimate — a naive filter over-flags these.
LICENSED = {
    "en": [
        "Personal loan from a licensed lender. Reg. under the Moneylenders Act, Lic. No. {amt}/2025. Apply at our branch with NRIC + payslip.",
        "Maybank personal financing — apply in the official app. Rates from {rate}% p.a., subject to approval and credit assessment.",
        "Credit counselling & debt restructuring by a registered non-profit. Free consultation. Terms apply.",
    ],
    "ms": [
        "Pembiayaan peribadi dari pemberi pinjaman berlesen. Berdaftar di bawah Akta Pemberi Pinjam Wang, Lesen No. {amt}/2025. Mohon di cawangan dengan MyKad + slip gaji.",
        "Pinjaman peribadi Bank Rakyat — mohon melalui aplikasi rasmi. Kadar dari {rate}% setahun, tertakluk kelulusan.",
        "Khidmat nasihat kredit oleh AKPK (agensi berdaftar). Rundingan percuma. Terma dikenakan.",
    ],
    "tl": [
        "Personal loan mula sa lisensyadong lender. Rehistrado sa SEC, Certificate No. {amt}. Mag-apply sa branch dala ang valid ID at payslip.",
        "BPI personal loan — mag-apply sa opisyal na app. Rates mula {rate}% per annum, subject sa credit assessment.",
        "Libreng financial literacy webinar ng isang rehistradong cooperative. Register sa opisyal na page.",
    ],
    "th": [
        "สินเชื่อส่วนบุคคลจากผู้ให้บริการที่ได้รับอนุญาต กำกับโดยธนาคารแห่งประเทศไทย เลขที่ใบอนุญาต {amt}/2568 สมัครที่สาขาพร้อมบัตรประชาชนและสลิปเงินเดือน",
        "สินเชื่อธนาคารกสิกรไทย สมัครผ่านแอปทางการ ดอกเบี้ยตั้งแต่ {rate}% ต่อปี ขึ้นอยู่กับการพิจารณา",
        "บริการให้คำปรึกษาหนี้โดยหน่วยงานที่จดทะเบียน ปรึกษาฟรี มีเงื่อนไข",
    ],
    "vi": [
        "Vay tiêu dùng từ tổ chức được cấp phép, quản lý bởi Ngân hàng Nhà nước, giấy phép số {amt}/2025. Đăng ký tại chi nhánh với CCCD và sao kê lương.",
        "Vay tiêu dùng Vietcombank — đăng ký qua ứng dụng chính thức. Lãi suất từ {rate}%/năm, tùy theo thẩm định.",
        "Hội thảo miễn phí về quản lý tài chính do một hợp tác xã đã đăng ký tổ chức. Đăng ký tại trang chính thức.",
    ],
}

# EASY negatives: unrelated everyday posts.
BENIGN = {
    "en": [
        "Finally tried the laksa at that new stall in Tiong Bahru — worth the queue 🍜",
        "Reminder: MRT maintenance this weekend on the East-West line, plan extra time.",
        "Anyone got recommendations for a good badminton court near Jurong?",
    ],
    "ms": [
        "Akhirnya cuba nasi lemak kaw kaw kat kedai baru tu — memang sedap gila 🍚",
        "Peringatan: jalan tol PLUS akan ada kerja penyelenggaraan hujung minggu ini.",
        "Ada sesiapa cadang gelanggang badminton area Shah Alam?",
    ],
    "tl": [
        "Sa wakas natikman ko yung sisig sa bagong karinderya sa Maginhawa — sulit ang pila! 🍛",
        "Paalala: may maintenance ang MRT ngayong weekend, mag-allot ng dagdag na oras.",
        "May marerecommend ba kayong magandang badminton court malapit sa QC?",
    ],
    "th": [
        "ในที่สุดก็ได้ลองข้าวมันไก่ร้านใหม่แถวอารีย์ อร่อยมากกกก 🍗",
        "แจ้งเตือน: รถไฟฟ้าสายสีเขียวปิดซ่อมบำรุงสุดสัปดาห์นี้ เผื่อเวลาเดินทางด้วยนะ",
        "มีใครแนะนำสนามแบดมินตันแถวลาดพร้าวไหมครับ",
    ],
    "vi": [
        "Cuối cùng cũng thử bún bò ở quán mới ngoài quận 1 — ngon xứng đáng xếp hàng 🍜",
        "Nhắc nhẹ: tuyến metro bảo trì cuối tuần này, mọi người tính thêm thời gian nhé.",
        "Có ai gợi ý sân cầu lông gần Thủ Đức không ạ?",
    ],
}

LANG_NAME = {"en": "English/Singlish", "ms": "Malay/Indonesian",
             "tl": "Tagalog", "th": "Thai", "vi": "Vietnamese"}


def fill(t: str) -> str:
    return (t.replace("{n}", str(random.randint(80000000, 99999999)))
             .replace("{amt}", f"{random.choice([500, 800, 1000, 2000, 3000, 5000]):,}")
             .replace("{rate}", str(random.choice([10, 15, 20, 20, 30]))))


def expand(bank, label, category, per_template):
    rows = []
    for lang, templates in bank.items():
        for t in templates:
            for _ in range(per_template):
                rows.append({"text": fill(t), "label": label,
                             "lang": lang, "category": category})
    return rows


def main():
    rows = []
    # illegal: 5 templates x 5 langs x 3 = 75 positives
    rows += expand(ILLEGAL, 1, "illegal_ml", 3)
    # licensed hard-neg: 3 x 5 x 3 = 45
    rows += expand(LICENSED, 0, "licensed", 3)
    # benign easy-neg: 3 x 5 x 2 = 30
    rows += expand(BENIGN, 0, "benign", 2)

    # de-duplicate identical filled strings, then shuffle deterministically
    seen, uniq = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"]); uniq.append(r)
    random.shuffle(uniq)

    # stratified-ish split by (category, lang): ~25% test, ~12% valid, rest train
    for i, r in enumerate(uniq):
        r["split"] = "test" if i % 4 == 0 else ("valid" if i % 8 == 1 else "train")

    # write human-readable CSV
    csv_path = os.path.join(HERE, "apac_moneylending.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "lang", "category", "split"])
        w.writeheader(); w.writerows(uniq)

    # write MLX prompt/completion jsonl (no chat template -> clean verdict tokens)
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    from common import build_prompt, label_word, POLICY_MINIMAL
    mlx_dir = os.path.join(HERE, "mlx")
    os.makedirs(mlx_dir, exist_ok=True)
    counts = {"train": 0, "valid": 0, "test": 0}
    handles = {s: open(os.path.join(mlx_dir, f"{s}.jsonl"), "w") for s in counts}
    for r in uniq:
        # train on the MINIMAL policy so the model internalises the regional norm
        # in its weights and needs only a one-line prompt at inference time.
        rec = {"prompt": build_prompt(r["text"], POLICY_MINIMAL),
               "completion": " " + label_word(r["label"])}
        handles[r["split"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[r["split"]] += 1
    for h in handles.values():
        h.close()

    # summary
    from collections import Counter
    by_split_label = Counter((r["split"], r["label"]) for r in uniq)
    by_lang = Counter(r["lang"] for r in uniq)
    print(f"Total rows: {len(uniq)}")
    print("Splits:", counts)
    print("By split/label (label 1 = VIOLATION):",
          {f"{s}/{l}": by_split_label[(s, l)] for s in counts for l in (1, 0)})
    print("By language:", {LANG_NAME[k]: v for k, v in by_lang.items()})
    print(f"CSV  -> {csv_path}")
    print(f"MLX  -> {mlx_dir}/(train|valid|test).jsonl")


if __name__ == "__main__":
    main()
