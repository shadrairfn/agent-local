import pandas as pd
import numpy as np

np.random.seed(42)

prompts = [
    ("P101", "Describe the impact of renewable energy on global climate change."),
    ("P102", "Jelaskan pengaruh kecerdasan buatan terhadap masa depan dunia kerja di Indonesia."),
    ("P103", "Analyze the advantages and disadvantages of remote work culture."),
    ("P104", "Bagaimana peran pemuda dalam memajukan inovasi teknologi dan riset nasional?"),
    ("P105", "Discuss the ethical implications of genetic engineering in healthcare.")
]

categories = ["Technology", "Environment", "Society", "Education", "Science"]

train_rows = []
for i in range(1, 61):
    pid, prompt = prompts[i % len(prompts)]
    cat = categories[i % len(categories)]
    
    # Mix Indonesian and English essays with slight noise
    if i % 2 == 0:
        base_text = (
            f"Kecerdasan buatan dan inovasi teknologi membawa transformasi besar bagi masyarakat. "
            f"Dalam konteks {cat.lower()}, peran pemuda sangat penting untuk mendorong produktivitas nasional. "
            f"Namun demikian, tantangan adaptasi regulasi dan kesiapan tenaga kerja harus diatasi dgn bijak. "
            f"Secara keseluruhan, integrasi sistem digital memberikan potensi kemajuan yg sangat signifikan. "
            f"Evaluasi berkelanjutan diperlukan agar manfaat teknologi dirasakan secara merata."
        )
    else:
        base_text = (
            f"The rapid advancement of {cat.lower()} has fundamentally altered human civilization. "
            f"Regarding the topic of {prompt.lower()}, researchers emphasize both structural efficiency and ethical constraints. "
            f"Empirical evidence demonstrates that continuous education and balanced regulatory frameworks provide optimal long-term outcomes. "
            f"Consequently, multi-stakeholder collaboration remains vital to ensure sustainable progress and equity."
        )
        
    word_count = len(base_text.split()) + np.random.randint(-5, 15)
    score_cont = np.round(2.5 + 0.8 * np.sin(i) + 0.02 * word_count, 3)
    score_cont = np.clip(score_cont, 1.0, 5.0)
    grade_ord = int(np.clip(np.round(score_cont + 0.5), 1, 6))
    toxicity = np.round(np.clip(0.05 + 0.15 * np.random.rand(), 0.0, 1.0), 4)
    likes = int(np.clip(np.random.poisson(lam=12), 0, 100))
    
    # 3 multi-output traits
    cohesion = np.round(np.clip(score_cont + np.random.normal(0, 0.2), 1.0, 5.0), 2)
    syntax = np.round(np.clip(score_cont + np.random.normal(0, 0.2), 1.0, 5.0), 2)
    vocab = np.round(np.clip(score_cont + np.random.normal(0, 0.2), 1.0, 5.0), 2)
    
    train_rows.append({
        "id": f"TRAIN_{i:04d}",
        "full_text": base_text,
        "prompt_text": prompt,
        "prompt_id": pid,
        "category": cat,
        "word_count": word_count,
        "score": score_cont,
        "grade": grade_ord,
        "toxicity": toxicity,
        "likes_count": likes,
        "cohesion": cohesion,
        "syntax": syntax,
        "vocabulary": vocab
    })

train_df = pd.DataFrame(train_rows)
train_df.to_csv("c:/Users/yasse/OneDrive/Desktop/Mine All Ds/kitGemastikk/NLP-FULL KIT/text_regression/level_a_teks_utuh/demo_data/train.csv", index=False)

test_rows = []
for i in range(1, 21):
    pid, prompt = prompts[(i + 2) % len(prompts)]
    cat = categories[(i + 1) % len(categories)]
    if i % 2 == 0:
        base_text = (
            f"Perkembangan teknologi modern di era industri 4.0 memerlukan kesiapan literasi digital. "
            f"Terkait {cat.lower()}, kolaborasi lintas sektor menjadi kunci keberhasilan implementasi solusi cerdas di Indonesia."
        )
    else:
        base_text = (
            f"Technological innovations continue to shape future workforce dynamics. "
            f"Addressing challenges in {cat.lower()} necessitates comprehensive strategic planning and agile execution."
        )
    word_count = len(base_text.split()) + np.random.randint(-3, 10)
    test_rows.append({
        "id": f"TEST_{i:04d}",
        "full_text": base_text,
        "prompt_text": prompt,
        "prompt_id": pid,
        "category": cat,
        "word_count": word_count
    })

test_df = pd.DataFrame(test_rows)
test_df.to_csv("c:/Users/yasse/OneDrive/Desktop/Mine All Ds/kitGemastikk/NLP-FULL KIT/text_regression/level_a_teks_utuh/demo_data/test.csv", index=False)

# Sample submission for single regression ('score')
sample_sub = test_df[["id"]].copy()
sample_sub["score"] = 3.0
sample_sub.to_csv("c:/Users/yasse/OneDrive/Desktop/Mine All Ds/kitGemastikk/NLP-FULL KIT/text_regression/level_a_teks_utuh/demo_data/sample_submission.csv", index=False)

print("Demo data successfully created:")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Sample submission shape: {sample_sub.shape}")

