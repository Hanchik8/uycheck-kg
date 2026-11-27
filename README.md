# UyCheck KG

Bishkek apartment valuation, listing text check, and Kyrgyz Republic legal & GIK mortgage assistant.

**Team:** Nurlanov Azizkhan · Zhanybekov Nurdos · Abdilazizov Nurmukhamed  
**Course:** Generative AI (Fall 2026), Ala-Too International University  
**Live Demo:** https://uycheck-kg.streamlit.app/  
**Repository:** https://github.com/Hanchik8/uycheck-kg  

## Architecture

1. **Price Valuation (ML):** LightGBM + CatBoost ensemble trained on 8,821 real Bishkek apartments (`raimbekovm/bishkek-real-estate`, CC BY 4.0) with SHAP feature breakdown and input validation.
2. **Listing Analysis (NLP):** Multilingual MiniLM + TF-IDF classifier trained on 7,023 Russian/Kyrgyz descriptions across 5 renovation levels, deal tag extraction, and low-confidence guardrails.
3. **Legal & Mortgage Q&A (RAG on LM Studio):** Hybrid vector + BM25 retrieval over 17 official Kyrgyz Republic documents (GIK «Менин үйүм», Law No. 153, Housing Code No. 117, GU «Кадастр», DDU rules, KR Tax Code). Both **answer generation** and **document/query embeddings** run locally through **LM Studio** (`http://localhost:1234/v1`).

## Run Locally with LM Studio

1. Install **LM Studio** from https://lmstudio.ai/ (regular desktop app).
2. Download two models in LM Studio:
   - **LLM:** `google/gemma-3-4b` (Q4_K_M) — or `qwen3-4b` / `llama-3.2-3b`.
   - **Embeddings:** `bge-m3` (recommended for Russian/Kyrgyz/English) or `nomic-embed-text-v1.5`.
3. Open the **Developer** tab in LM Studio, load both models, and start the local server (`http://localhost:1234/v1`).
4. Install Python dependencies and launch the app:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

By default, `.env.example` connects both LLM completions and embeddings directly to `http://localhost:1234/v1` and auto-detects the loaded models.

## Verification

```bash
python tests/run_all_tests.py
python tests/redteam_10_missions.py
```
