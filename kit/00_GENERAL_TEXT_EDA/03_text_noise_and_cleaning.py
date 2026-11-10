# %% [markdown]
# # 📊 General Text EDA: Text Noise & Artifacts Detection
# ### Universal Diagnostic for All Text Competitions (NLU & NLG)
# Tujuan: Mengaudit artefak kotoran pada teks mentah (tag HTML, entity escaping, emoji liar, mojibake unicode, URL, email).

# %%
# 1. Master Config
DATA_PATH = "demo_data/train.csv"
TEXT_COL = "full_text"

# %%
# 2. Imports & Noise Scanners
import pandas as pd
import re
import html

df = pd.read_csv(DATA_PATH)

scanners = {
    'html_tags': re.compile(r'<[^>]+>'),
    'html_entities': re.compile(r'&(?:[a-z\d]+|#\d+|#x[a-f\d]+);', re.I),
    'urls': re.compile(r'https?://\S+|www\.\S+'),
    'user_mentions': re.compile(r'@\w+'),
    'excessive_punctuation': re.compile(r'([!?.]){3,}'),
    'excessive_whitespaces': re.compile(r'\s{2,}')
}

noise_counts = {}
for name, pattern in scanners.items():
    matches = df[TEXT_COL].dropna().apply(lambda t: len(pattern.findall(str(t))))
    noise_counts[name] = {
        'total_occurrences': int(matches.sum()),
        'rows_affected': int((matches > 0).sum()),
        'pct_rows': round((matches > 0).mean() * 100, 2)
    }

noise_df = pd.DataFrame(noise_counts).T
print("=== TEXT NOISE ARTIFACT AUDIT ===")
print(noise_df)

# %%
# 3. Robust Text Cleaning Pipeline
def clean_text_pipeline(text: str, preserve_for_transformer: bool = True) -> str:
    text = str(text)
    # 1. Unescape HTML entities (&amp; -> &)
    text = html.unescape(text)
    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # 3. Normalize URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    # 4. Normalize excessive whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

sample_noisy = "<p>Selamat pagi &amp; selamat datang! Cek info di https://example.com/info   ... Mantap!</p>"
print("Original Noisy Sample:\n", sample_noisy)
print("\nCleaned Output:\n", clean_text_pipeline(sample_noisy))

# %% [markdown]
# ### 📋 Master Decision Matrix (Text Cleaning):
# 1. **HTML Entities & Tags:** Wajib dibersihkan (`html.unescape` dan strip `<tag>`) di semua model karena tag HTML bukan bagian dari semantik bahasa.
# 2. **Punctuation & Case (Huruf Kapital):**
#    * Untuk model **GBDT (TF-IDF)**: Boleh dibersihkan secara agresif (hapus tanda baca, lowercase).
#    * Untuk model **Transformer (DeBERTa/IndoBERT)**: **JANGAN hapus tanda baca atau huruf kapital!** Tanda baca krusial untuk attention mask memahami batas klausa dan tone emosional.
