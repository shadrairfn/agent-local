# GEMASTIK NLP: opencode + Ollama lokal

Workspace agen untuk lomba offline: opencode (utama) dan aider (cadangan) diarahkan ke Ollama lokal.

## Isi
- `AGENTS.md` aturan agen (masuk ke system prompt, sengaja pendek)
- `opencode.json` provider Ollama (`http://localhost:11434/v1`), model 4B/8B, tool web dimatikan, `.ipynb` diblokir untuk dibaca
- `.aider.conf.yml`, `.env` konfigurasi aider dan env offline
- `kit/` template NLP (tiap notebook punya salinan `.py`; bobot `03_OFFLINE_WEIGHTS` tidak ikut repo)
- `tools/check_submission.py` validator submission
- `ollama/` Modelfile varian ctx besar + `setup_ollama.ps1`

## Setup di PC baru
1. Pasang Ollama dan opencode (`npm i -g opencode-ai`), lalu jalankan `ollama/setup_ollama.ps1`.
2. Salin bobot pretrained ke `kit/03_OFFLINE_WEIGHTS/weights/` (tidak ada di repo).
3. Taruh data lomba (csv) di folder kerja, lalu `opencode` dari root repo ini.

Di GPU 8 GB, `num_ctx` bisa dinaikkan (mis. 32768) di Modelfile dan `opencode.json` (`limit.context`).
