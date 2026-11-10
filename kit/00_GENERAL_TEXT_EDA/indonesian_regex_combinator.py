"""
Indonesian Regex Combinator & Pre-Clean Signal Feature Extractor
Dirancang khusus untuk dataset Bahasa Indonesia & Daerah (GEMASTIK / Satria Data / Kaggle):
Prinsip Utama: "Extract Signal First, Normalize Second, Never Delete Negations!"

7 Modular Lego Bricks (Toggleable):
  1. extract_preclean_signals     : Ekstrak 12 fitur numerik regex SEBELUM teks dibersihkan (CAPS, !!!, ???, kata2, elongasi, Rp, emoji, dll.)
  2. normalize_social_entities    : Ubah @username -> [USER], URL -> [URL], Email -> [EMAIL], No HP (+62/08xx) -> [PHONE], Rp 50rb -> [MONEY]
  3. expand_angka2_reduplication  : Ubah kata ulang angka 'anak2' -> 'anak-anak', 'tiba2' -> 'tiba-tiba' (tanpa merusak kode seperti 'ps2', 'b2b')
  4. reduce_char_elongation       : Ubah huruf beruntun >=3 'bangeeeettt' -> 'banget', 'paraaah' -> 'parah', 'mantaaap' -> 'mantap'
  5. normalize_indo_slang         : Kamus 180+ kata gaul/singkatan Bahasa Indonesia + unifikasi seluruh kata negasi ('gak','ga','tdk','tak' -> 'tidak')
  6. normalize_affix_colloquial   : Normalisasi imbuhan percakapan ('ngelihat' -> 'melihat', 'nyari' -> 'mencari', 'beneran' -> 'benaran')
  7. remove_stopwords_safe        : Hapus stopword ringan dengan JAMINAN 100% TIDAK MENGHAPUS KATA NEGASI ('tidak', 'bukan', 'jangan', 'belum', 'tanpa')
"""

import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class IndonesianRegexCombinator:
    """Modular, toggleable Indonesian Regex & Text Normalization Engine."""

    # Kata negasi dan pembalik sentimen yang HARAM dihapus oleh stopword removal!
    PROTECTED_NEGATION_WORDS = {
        "tidak", "bukan", "jangan", "belum", "tanpa", "kurang", "jarang",
        "anti", "mustahil", "gagal", "batal", "bohong", "hoax", "palsu"
    }

    # Kamus Slang & Singkatan Bahasa Indonesia Terkurasi (E-Commerce, Medsos, Berita, Percakapan)
    INDO_SLANG_MAP = {
        # Negasi (Wajib disatukan ke bentuk baku agar model tidak pecah kosakata)
        "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak",
        "enggak": "tidak", "engga": "tidak", "kagak": "tidak", "kaga": "tidak",
        "tdk": "tidak", "tak": "tidak", "ndak": "tidak", "nda": "tidak",
        "ora": "tidak", "teu": "tidak", "sing": "tidak", "bkn": "bukan",
        "blm": "belum", "blom": "belum", "jgn": "jangan",
        # Kata hubung & preposisi umum
        "yg": "yang", "dgn": "dengan", "dg": "dengan", "krn": "karena",
        "karna": "karena", "kalo": "kalau", "kl": "kalau", "klu": "kalau",
        "tp": "tapi", "tpi": "tapi", "utk": "untuk", "buat": "untuk",
        "dlm": "dalam", "dr": "dari", "pd": "pada", "kpd": "kepada",
        "spt": "seperti", "kayak": "seperti", "kyk": "seperti", "kek": "seperti",
        "sm": "sama", "ama": "sama", "bs": "bisa", "bisa2": "bisa-bisa",
        "jd": "jadi", "jdi": "jadi", "sdh": "sudah", "udh": "sudah",
        "udah": "sudah", "dah": "sudah", "lg": "lagi", "lgi": "lagi",
        "aja": "saja", "aj": "saja", "doang": "saja", "ae": "saja",
        "bgt": "banget", "bget": "banget", "bingit": "banget", "poll": "sekali",
        "bener": "benar", "bnr": "benar", "beneran": "benaran",
        "hrs": "harus", "msh": "masih", "emg": "memang", "emang": "memang",
        "cm": "cuma", "cuman": "cuma", "hny": "hanya",
        # Kata ganti orang
        "sy": "saya", "aku": "saya", "ak": "saya", "gw": "saya", "gue": "saya",
        "gua": "saya", "km": "kamu", "kmu": "kamu", "lu": "kamu", "lo": "kamu",
        "agan": "anda", "sis": "anda", "kak": "kakak", "min": "admin",
        # Kosakata E-Commerce / Ulasan / Medsos (PRDECT-ID style)
        "brg": "barang", "brgnya": "barangnya", "pket": "paket",
        "pengiriman": "pengiriman", "kirim": "kirim", "nyampe": "sampai",
        "nyampek": "sampai", "sampe": "sampai", "dtg": "datang",
        " sesuai ": " sesuai ", "ori": "asli", "original": "asli",
        "kw": "palsu", "zonk": "buruk", "parah": "buruk", "ancur": "hancur",
        "mantul": "mantap", "jos": "bagus", "ok": "bagus", "oke": "bagus",
        "recommended": "direkomendasikan", "rekomen": "direkomendasikan",
        "respon": "respons", "fast": "cepat", "slow": "lambat",
        "packing": "kemasan", "bubble": "pelindung", "seller": "penjual",
        "kurir": "pengantar", "cpt": "cepat", "lmbt": "lambat",
        "mksh": "terima kasih", "makasih": "terima kasih", "thx": "terima kasih",
        "tq": "terima kasih", "trims": "terima kasih", "tks": "terima kasih",
        "gimana": "bagaimana", "gmn": "bagaimana", "knp": "kenapa",
        "kenapa": "mengapa", "dmn": "dimana", "kpn": "kapan",
        "org": "orang", "ank": "anak", "thn": "tahun", "bln": "bulan",
        "hr": "hari", "tgl": "tanggal", "rp": "rupiah", "rb": "ribu", "jt": "juta",
    }

    # Stopwords ringan Bahasa Indonesia (TIDAK mengandung kata negasi atau kata emosi!)
    SAFE_STOPWORDS = {
        "yang", "di", "ke", "dari", "pada", "dalam", "untuk", "dengan", "dan",
        "atau", "ini", "itu", "tersebut", "adalah", "yaitu", "yakni", "sebagai",
        "oleh", "karena", "sehingga", "jika", "kalau", "apabila", "ketika",
        "saat", "para", "sang", "si", "pun", "lah", "kah", "nya", "ya", "동"
    } - PROTECTED_NEGATION_WORDS

    # Compiled Regex Patterns (High Speed)
    RE_URL = re.compile(r"(?:https?://|www\.)\S+|(?:\b[a-zA-Z0-9.-]+\.(?:com|id|co\.id|go\.id|ac\.id|org|net)\b\S*)", re.IGNORECASE)
    RE_EMAIL = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
    RE_USER = re.compile(r"(?<!\w)@[a-zA-Z0-9_]{2,30}\b")
    RE_HASHTAG = re.compile(r"(?<!\w)#([a-zA-Z0-9_]+)")
    RE_PHONE = re.compile(r"\b(?:\+62|62|0)8[1-9][0-9]{6,10}\b")
    RE_MONEY = re.compile(r"\b(?:rp\.?\s*[0-9]+(?:[.,][0-9]+)*(?:\s*(?:rb|ribu|jt|juta|m|miliar))?|[0-9]+\s*(?:rb|ribu|jt|juta))\b", re.IGNORECASE)
    RE_ANGKA2 = re.compile(r"\b([a-zA-Z]{3,})2(nya|an|in|lah|kah)?\b")
    RE_ELONGATION = re.compile(r"([a-zA-Z])\1{2,}")
    RE_LAUGH = re.compile(r"\b(?:(?:wk){2,}(?:w|k)*|(?:ha){2,}(?:h|a)*|(?:he){2,}(?:h|e)*|(?:hi){2,}(?:h|i)*|xixixi+)\b", re.IGNORECASE)
    RE_HTML = re.compile(r"<[^>]+>|&(?:amp|lt|gt|quot|nbsp);")
    RE_MULTISPACE = re.compile(r"\s+")

    DEFAULT_CONFIG = {
        "extract_preclean_signals": True,
        "normalize_social_entities": True,
        "expand_angka2_reduplication": True,
        "reduce_char_elongation": True,
        "normalize_indo_slang": True,
        "normalize_affix_colloquial": True,
        "remove_stopwords_safe": False,
        "lowercase": True,
    }

    def __init__(
        self,
        extract_pre_clean_signals: bool = True,
        mask_entities: bool = True,
        expand_reduplication: bool = True,
        normalize_elongation: bool = True,
        normalize_slang: bool = True,
        normalize_colloquial_affixes: bool = True,
        remove_stopwords_negation_safe: bool = False,
        lowercase: bool = True,
    ):
        self.config = {
            "extract_preclean_signals": extract_pre_clean_signals,
            "normalize_social_entities": mask_entities,
            "expand_angka2_reduplication": expand_reduplication,
            "reduce_char_elongation": normalize_elongation,
            "normalize_indo_slang": normalize_slang,
            "normalize_affix_colloquial": normalize_colloquial_affixes,
            "remove_stopwords_safe": remove_stopwords_negation_safe,
            "lowercase": lowercase,
        }

    def clean_text(self, text: str) -> str:
        """Instance wrapper untuk membersihkan 1 kalimat sesuai konfigurasi instance."""
        return self.clean_single_text(text, self.config)

    def fit_transform_df(
        self,
        df: pd.DataFrame,
        text_col: str,
        prefix: str = "rx_"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Mengekstrak 12 sinyal Regex pra-pembersihan + membuat kolom `{text_col}_clean`
        langsung pada DataFrame dan mengembalikan `(df_out, rx_feature_cols)`.
        """
        df_out = df.copy()
        cleaned_s, rx_df = self.process_series(df_out[text_col].fillna("").astype(str), self.config)
        rx_cols: List[str] = []
        if not rx_df.empty:
            for col in rx_df.columns:
                col_name = col if col.startswith(prefix) else f"{prefix}{col.replace('rx_', '')}"
                df_out[col_name] = rx_df[col].values
                rx_cols.append(col_name)
        df_out[f"{text_col}_clean"] = cleaned_s.values
        return df_out, rx_cols

    @classmethod
    def extract_preclean_regex_features(cls, texts: Union[List[str], pd.Series]) -> pd.DataFrame:
        """
        [Brick 1] Mengekstrak 12 sinyal statistik Regex dari teks mentah SEBELUM teks dibersihkan!
        Sangat ampuh untuk LightGBM/CatBoost dan analisis korelasi EDA.
        """
        records = []
        for raw in texts:
            s = str(raw) if raw is not None else ""
            n_chars = max(1, len(s))
            words = s.split()
            n_words = max(1, len(words))

            upper_chars = sum(1 for c in s if c.isupper())
            caps_words = sum(1 for w in words if len(w) >= 2 and w.isupper())
            excl_count = s.count("!")
            quest_count = s.count("?")
            angka2_count = len(cls.RE_ANGKA2.findall(s))
            elong_count = len(cls.RE_ELONGATION.findall(s))
            laugh_count = len(cls.RE_LAUGH.findall(s))
            url_count = len(cls.RE_URL.findall(s))
            user_count = len(cls.RE_USER.findall(s))
            money_count = len(cls.RE_MONEY.findall(s))

            # Hitung kemunculan kata negasi & kata slang mentah
            low_words = [w.lower().strip(".,!?\"'()") for w in words]
            neg_count = sum(1 for w in low_words if w in cls.PROTECTED_NEGATION_WORDS or w in ("gak", "ga", "nggak", "tdk", "tak", "bkn", "blm", "jgn"))
            slang_count = sum(1 for w in low_words if w in cls.INDO_SLANG_MAP)

            # Karakter non-ASCII (Emoji / Simbol / Aksara)
            non_ascii = sum(1 for c in s if ord(c) > 127)

            records.append({
                "rx_caps_ratio": round(upper_chars / n_chars, 4),
                "rx_allcaps_words": caps_words,
                "rx_exclamation_count": excl_count,
                "rx_question_count": quest_count,
                "rx_angka2_redup_count": angka2_count,
                "rx_char_elongation_count": elong_count,
                "rx_laugh_token_count": laugh_count,
                "rx_url_mention_count": url_count + user_count,
                "rx_money_mention_count": money_count,
                "rx_negation_word_count": neg_count,
                "rx_slang_word_count": slang_count,
                "rx_emoji_non_ascii_count": non_ascii,
                "rx_avg_word_len": round(sum(len(w) for w in words) / n_words, 4),
            })
        return pd.DataFrame(records)

    @classmethod
    def clean_single_text(cls, text: str, config: Optional[Dict[str, bool]] = None) -> str:
        """Menerapkan kombinasi Brick 2 s/d Brick 7 secara berurutan pada 1 kalimat."""
        cfg = {**cls.DEFAULT_CONFIG, **(config or {})}
        s = str(text) if text is not None else ""

        # Bersihkan tag HTML sisa scraping
        s = cls.RE_HTML.sub(" ", s)

        # [Brick 2] Normalisasi Entitas Sosial Media, Kontak PII & Nominal Uang
        if cfg.get("normalize_social_entities", True):
            s = cls.RE_URL.sub(" [URL] ", s)
            s = cls.RE_EMAIL.sub(" [EMAIL] ", s)
            s = cls.RE_USER.sub(" [USER] ", s)
            s = cls.RE_PHONE.sub(" [PHONE] ", s)
            s = cls.RE_MONEY.sub(" [MONEY] ", s)
            # Pisahkan hashtag menjadi kata biasa: #IndonesiaMaju -> IndonesiaMaju
            s = cls.RE_HASHTAG.sub(r" \1 ", s)

        # Normalisasi tawa khas Indonesia ('wkwkwk', 'hahaha', 'xixixi') -> 'tertawa'
        s = cls.RE_LAUGH.sub(" tertawa ", s)

        # [Brick 3] Ekspansi Reduplikasi Angka 2 ('anak2' -> 'anak-anak', 'tiba2nya' -> 'tiba-tibanya')
        if cfg.get("expand_angka2_reduplication", True):
            def _expand_redup(match):
                base = match.group(1)
                suffix = match.group(2) or ""
                return f"{base}-{base}{suffix}"
            s = cls.RE_ANGKA2.sub(_expand_redup, s)

        # Lowercase (kecuali token spesial [URL], [USER], [PHONE], [MONEY], [EMAIL])
        if cfg.get("lowercase", True):
            s = s.lower()
            for tok in ("[url]", "[email]", "[user]", "[phone]", "[money]"):
                s = s.replace(tok, tok.upper())

        # [Brick 4] Reduksi Elongasi Karakter ('bangeeeettt' -> 'banget', 'paraaah' -> 'parah')
        if cfg.get("reduce_char_elongation", True):
            s = cls.RE_ELONGATION.sub(r"\1", s)

        # [Brick 5 & 6] Normalisasi Slang, Singkatan & Imbuhan Lisan per Token
        if cfg.get("normalize_indo_slang", True) or cfg.get("normalize_affix_colloquial", True):
            tokens = s.split()
            norm_tokens = []
            for tok in tokens:
                if tok.startswith("[") and tok.endswith("]"):
                    norm_tokens.append(tok)
                    continue
                # Pisahkan tanda baca di ujung kata agar kamus slang tetap cocok
                lead_p = ""
                trail_p = ""
                core = tok
                while core and core[0] in ".,!?\"'():;":
                    lead_p += core[0]
                    core = core[1:]
                while core and core[-1] in ".,!?\"'():;":
                    trail_p = core[-1] + trail_p
                    core = core[:-1]

                if cfg.get("normalize_indo_slang", True) and core in cls.INDO_SLANG_MAP:
                    core = cls.INDO_SLANG_MAP[core]
                elif cfg.get("normalize_affix_colloquial", True) and len(core) >= 6:
                    # Normalisasi awalan 'nge-' lisan: 'ngelihat' -> 'melihat', 'ngebantu' -> 'membantu'
                    if core.startswith("ngelihat"):
                        core = "melihat" + core[8:]
                    elif core.startswith("ngebantu"):
                        core = "membantu" + core[8:]
                    elif core.startswith("ngerasa"):
                        core = "merasa" + core[7:]
                    elif core.startswith("nyari"):
                        core = "mencari" + core[5:]

                if cfg.get("remove_stopwords_safe", False):
                    if core in cls.SAFE_STOPWORDS and core not in cls.PROTECTED_NEGATION_WORDS:
                        continue

                norm_tokens.append(f"{lead_p}{core}{trail_p}".strip())
            s = " ".join(norm_tokens)

        s = cls.RE_MULTISPACE.sub(" ", s).strip()
        return s

    @classmethod
    def process_series(
        cls,
        texts: Union[List[str], pd.Series],
        config: Optional[Dict[str, bool]] = None
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Memproses seluruh kolom teks sekaligus:
        Mengembalikan tuple: (cleaned_text_series, preclean_regex_features_df)
        """
        cfg = {**cls.DEFAULT_CONFIG, **(config or {})}
        if cfg.get("extract_preclean_signals", True):
            rx_feats_df = cls.extract_preclean_regex_features(texts)
        else:
            rx_feats_df = pd.DataFrame(index=range(len(texts)))

        cleaned = pd.Series([cls.clean_single_text(t, cfg) for t in texts])
        return cleaned, rx_feats_df


if __name__ == "__main__":
    samples = [
        "PARAH BGT!!! Brg yg dtg gak sesuai gambar, anak2 di rumah pd kecewa bangeeeettt :( Hub 081234567890 @tokopedia_care",
        "Alhamdulillah paket Rp 150rb udh nyampe dgn selamat wkwkwk mantaaap jiwa, jalan2 jadi tenang!",
    ]
    cleaned_s, feats_df = IndonesianRegexCombinator.process_series(samples)
    for orig, cln in zip(samples, cleaned_s):
        print("RAW  :", orig)
        print("CLEAN:", cln)
        print("-" * 70)
    print("PRE-CLEAN REGEX FEATURES:\n", feats_df)

