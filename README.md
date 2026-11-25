# UyCheck KG — Bishkek Real Estate AI Assistant

**Team:** Nurlanov Azizkhan · Zhanybekov Nurdos · Abdilazizov Nurmukhamed  
**Course:** Generative AI (Fall 2026) · Ala-Too International University  

**UyCheck KG** is an integrated Streamlit web application for Bishkek apartment buyers combining:
1. **ML Price Valuation:** LightGBM + CatBoost ensemble on 8,821 real Bishkek apartments (`raimbekovm/bishkek-real-estate`, CC BY 4.0) with SHAP feature explanations and strict input validation.
2. **NLP Listing Analyzer:** Multilingual MiniLM + TF-IDF classifier trained on 7,023 Russian/Kyrgyz apartment descriptions (5 renovation classes + deal tags + low-confidence/sarcasm guard).
3. **KR Legal & GIK RAG Assistant:** Hybrid dense + BM25 search over 17 official Kyrgyz Republic legal & housing documents (GIK KR «Менин үйүм», KR Law No. 153, Housing Code No. 117, GU «Кадастр», DDU risks, KR Tax Code) with mandatory source citations, strict *"I don't know"* fallback, and prompt-injection defense.

## Quick Local Run (For Every Team Member)

```bash
# 1. Create virtual environment and install dependencies
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure API key (NEVER commit .env!)
cp .env.example .env
# Edit .env and paste your OPENAI_API_BASE, OPENAI_API_KEY, and LLM_MODEL

# 3. Generate RAG documents and train models (fixed seed = 42)
python src/build_rag_docs.py
python src/ml_model.py
python src/nlp_model.py

# 4. Run 20 RAG tests + all 10 Client Day mission checks
python tests/run_all_tests.py

# 5. Launch Streamlit web app
streamlit run app.py
```

## Deploy Free Public Link (Streamlit Community Cloud)
1. Push this repository to GitHub (`.env` is gitignored — never push API keys!).
2. Go to https://share.streamlit.io → **New app** → select your repo and `app.py`.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   OPENAI_API_BASE = "https://api.reformboss.com/v1"
   OPENAI_API_KEY = "YOUR_KEY_HERE"
   LLM_MODEL = "deepseek-v4-flash"
   ```
4. Click **Deploy** — share the live URL and `PRODUCT_DESCRIPTION.md` on Client Day!
