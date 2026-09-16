# Dataset credits & attribution

This project **references** (does not redistribute) several third-party datasets. The raw
data and any prediction files that embed their text are git-ignored; only build scripts and
aggregate metrics are committed. Each dataset is credited below using the **citation its
authors request**. If you use these datasets, cite them as below and follow each upstream
license — these summaries do not replace it.

---

### ChiFraud — Chinese fraud detection
- **Source:** https://github.com/xuemingxxx/ChiFraud · **License:** CC BY-NC 4.0 (attribution, **non-commercial**)
- **Used here:** sampled class 9 (地下黑贷, underground loans) vs class 0 (normal) for evaluation and a LoRA fine-tune; text truncated. Non-commercial use only.
- **Cite (as requested in the repo):**
```bibtex
@inproceedings{tang2025chifraud,
  title={{ChiFraud}: A Long-Term Web Text Benchmark for {Chinese} Fraud Detection},
  author={Tang, Min and Zou, Lixin and Liang, Shiuan Ni and Jin, Zhe and Wang, Weiqing and Cui, Shujie},
  booktitle={International Conference on Computational Linguistics (COLING 2025)},
  pages={5962--5974},
  year={2025},
  publisher={Association for Computational Linguistics}
}
```

### SWSR — Sina Weibo Sexism Review (Chinese)
- **Source:** https://github.com/aggiejiang/SWSR · https://doi.org/10.5281/zenodo.4443063 · **License:** CC BY 4.0 (attribution)
- **Used here:** sampled `SexComment.csv` by the `label` field (sexist / not) for evaluation; text truncated.
- **Cite (as requested by the authors):**
```
Aiqi Jiang, Xiaohan Yang, Yang Liu, Arkaitz Zubiaga,
SWSR: A Chinese dataset and lexicon for online sexism detection,
Online Social Networks and Media, Volume 27, 2022, 100182, ISSN 2468-6964,
https://doi.org/10.1016/j.osnem.2021.100182.
```

### TB-OLID — Transliterated Bangla offensive language (HASOC 2024 Bangla source)
- **Source:** https://github.com/LanguageTechnologyLab/TB-OLID (also https://github.com/mraihan-gmu/TB-OLID) · **License:** AGPL-3.0 (copyleft)
- **Used here:** sampled a balanced subset by `offensive_gold` (O/N) for evaluation and a LoRA fine-tune; text truncated.
- **Cite:**
```bibtex
@inproceedings{raihan-etal-2023-offensive,
  title = "Offensive Language Identification in Transliterated and Code-Mixed {B}angla",
  author = "Raihan, Md Nishat and Tanmoy, Umma and Islam, Anika Binte and North, Kai and Ranasinghe, Tharindu and Anastasopoulos, Antonios and Zampieri, Marcos",
  booktitle = "Proceedings of the First Workshop on Bangla Language Processing (BLP-2023)",
  month = dec, year = "2023", address = "Singapore",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2023.banglalp-1.1/"
}
```

### Uli — gendered abuse in Indic languages (Tattle Civic Technologies)
- **Source:** https://github.com/tattle-made/uli_dataset · **License:** CC BY 4.0 (attribution)
- **Used here:** used the gendered-abuse label (question 1) for en/hi/ta with a majority vote over annotators; sampled a balanced subset for evaluation; text truncated.
- **Cite:**
```bibtex
@misc{arora2023uli,
  title  = {The {Uli} Dataset: An Exercise in Experience Led Annotation of oGBV},
  author = {Arora, Arnav and Jinadoss, Maha and Arora, Cheshta and George, Denny and
            Brindaalakshmi and Khan, Haseena Dawood and Rawat, Kirti and Div and Ritash and
            Mathur, Seema and Yadav, Shivani and Shora, Shehla Rashid and Raut, Rie and
            Pawar, Sumit and Paithane, Apurva and Sonia and Vivek and Priscilla, Dharini and
            Khairunnisha and Banu, Grace and Tandon, Ambika and Thakker, Rishav and
            Korra, Rahul Dev and Vaidya, Aatman and Prabhakar, Tarunima},
  year = {2023}, eprint = {2311.09086}, archivePrefix = {arXiv}, primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2311.09086}
}
```

---

### Also referenced (not used as training/eval data)
- **ADHAR** — multi-dialectal Arabic hate speech (Charfi, Besghaier, Akasheh, Atalla, Zaghouani; *Frontiers in AI*, 2024) — cited as corroboration of the cross-dialect / cultural-norm effect.
- **LatticeFlow** political-bias framework (`bias.latticeflow.ai`) and *"We're Cooked!"* (arXiv:2609.07568) — cited for the imported-bias discussion.
- **Indic safety resources** — MuRIL, AI4Bharat IndicLLMSuite (IndicAlign-Toxic), L3Cube-MahaHate — mentioned as the regional "guardrail" layer.

The **SEA moneylending set** (`data/apac_moneylending.csv`) is synthetic and authored for this
project (see `data/NOTICE.md`) — no third-party attribution required, but it needs
native-speaker validation before external use.
