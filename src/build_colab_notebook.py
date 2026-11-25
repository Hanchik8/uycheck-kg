import os
import nbformat as nbf

ROOT = r"C:\Users\user\.gemini\antigravity\scratch\uycheck-kg"
OUT_NB = os.path.join(ROOT, "notebooks", "UyCheck_KG_Pipeline.ipynb")
os.makedirs(os.path.dirname(OUT_NB), exist_ok=True)

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""# UyCheck KG — End-to-End Generative AI Pipeline (Colab Notebook)

**Team:** Nurlanov Azizkhan · Zhanybekov Nurdos · Abdilazizov Nurmukhamed  
**Course:** Generative AI (Fall 2026) · Ala-Too International University  
**Fixed Random Seed:** `42` (fully reproducible top-to-bottom on Free Colab or Colab Pro/Ultra)

### What This Notebook Runs Top-to-Bottom:
1. Installs dependencies & downloads the public **Bishkek Real Estate Dataset** (`raimbekovm/bishkek-real-estate`, **8,821 rows**, **CC BY 4.0**).
2. Cleans PII fields and runs exploratory checks (price distribution, districts, renovation levels).
3. Trains and benchmarks **ML Valuation Models** (Global Median & Ridge baselines vs. **LightGBM + CatBoost ensemble**) + **SHAP** per-listing explanations + Mission 5/6 input validation.
4. Trains and evaluates the **Multilingual NLP Listing Analyzer** (MiniLM + TF-IDF on **7,023 RU/KG descriptions**) + Mission 10 low-confidence/sarcasm test.
5. Builds the **17-Document Kyrgyz Republic Legal & GIK RAG Engine** (hybrid dense + BM25, mandatory citations, *"I don't know"* cutoff, prompt-injection guard).
6. Runs the **20-Question RAG Evaluation Benchmark** and all **10 Client Day Missions** + Integrated ML+NLP+RAG consultation demo.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Install dependencies (uncomment when running in Google Colab)
# !pip -q install pandas numpy scikit-learn lightgbm catboost shap sentence-transformers rank-bm25 openai huggingface-hub matplotlib joblib python-dotenv

import json, math, os, re, urllib.request, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
print("Fixed seed set to:", SEED)
"""))

cells.append(nbf.v4.new_code_cell("""# 2. Load API key safely (Colab Secrets or Environment Variables — NEVER hardcode!)
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.reformboss.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")

try:
    from google.colab import userdata
    OPENAI_API_KEY = userdata.get("OPENAI_API_KEY") or OPENAI_API_KEY
    OPENAI_API_BASE = userdata.get("OPENAI_API_BASE") or OPENAI_API_BASE
except Exception:
    pass

print("API base:", OPENAI_API_BASE, "| Key configured:", bool(OPENAI_API_KEY))
"""))

cells.append(nbf.v4.new_code_cell("""# 3. Download Bishkek Real Estate CSV directly from Hugging Face (CC BY 4.0, 12.1 MB)
os.makedirs("data", exist_ok=True)
CSV_PATH = "data/bishkek_apartments.csv"
if not os.path.exists(CSV_PATH):
    url = "https://huggingface.co/datasets/raimbekovm/bishkek-real-estate/resolve/main/data/bishkek_apartments.csv"
    print("Downloading Bishkek Real Estate CSV...")
    urllib.request.urlretrieve(url, CSV_PATH)

raw_df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(raw_df):,} rows x {raw_df.shape[1]} columns")
print("Median price_usd:", f"${raw_df['price_usd'].median():,.0f}")
print("Top districts:\\n", raw_df["district"].value_counts().head(6))
print("\\nCondition counts:\\n", raw_df["condition"].value_counts().head(6))
"""))

cells.append(nbf.v4.new_markdown_cell("""## Part 1 — ML Tabular Valuation Model (LightGBM + CatBoost + SHAP)
We drop all PII/ID columns, remove extreme scrape outliers, engineer Bishkek geospatial features (distance to Ala-Too Square & South Magistral), train with `seed=42`, and prove we beat Median & Ridge baselines."""))

cells.append(nbf.v4.new_code_cell("""import sys
# If running from the repo folder, reuse src/ directly; in standalone Colab, clone repo first:
# !git clone <YOUR_GITHUB_REPO_URL> .
if os.path.isdir("src") and "src" not in sys.path:
    sys.path.insert(0, "src")
elif os.path.isdir("../src") and "../src" not in sys.path:
    sys.path.insert(0, "../src")
    CSV_PATH = "../data/bishkek_apartments.csv"

from ml_model import train_and_save_ml, BishkekPricePredictor

MODELS_DIR = "../models" if os.path.isdir("../src") else "models"
ml_metrics = train_and_save_ml(CSV_PATH, MODELS_DIR)

summary_df = pd.DataFrame([
    {"Model": "Baseline 1: Global Median", **ml_metrics["baselines"]["global_median"]},
    {"Model": "Baseline 2: Ridge Regression", **ml_metrics["baselines"]["ridge_regression"]},
    {"Model": "LightGBM Regressor", **ml_metrics["models"]["lightgbm"]},
    {"Model": "CatBoost Regressor", **ml_metrics["models"]["catboost"]},
    {"Model": "Final Ensemble (LGB + CB)", **ml_metrics["models"]["ensemble_lgb_cb"]},
])
summary_df
"""))

cells.append(nbf.v4.new_code_cell("""# Test ML Prediction + SHAP Explainability (Mission 1 & 2) + Impossible Values (Mission 5)
ml_pred = BishkekPricePredictor(os.path.join(MODELS_DIR, "ml_valuation_bundle.joblib"))

dzhal_sample = {
    "district": "джал м-н", "area": 65, "rooms": 2, "floor": 6, "total_floors": 12,
    "year_built": 2024, "house_type": "монолит", "condition": "под самоотделку (псо)",
    "asking_price_usd": 115000,
}
res_ok = ml_pred.predict(dzhal_sample)
print(f"Fair Price: ${res_ok['fair_price_usd']:,.0f} (${res_ok['fair_price_per_m2']:,.0f}/m²) | {res_ok['price_verdict']}")
print("Top SHAP drivers:")
for d in res_ok["top_drivers"][:5]:
    print(f"  - {d['label']} ({d['value']}): {d['usd_impact']:+,.0f}$")

res_bad = ml_pred.predict({"district": "джал м-н", "area": -20, "rooms": 2, "floor": 50, "total_floors": 5})
print("\\nMission 5 (Impossible input) safe errors:", res_bad["errors"])
"""))

cells.append(nbf.v4.new_markdown_cell("""## Part 2 — Multilingual NLP Listing Analyzer (7,023 RU/KG Descriptions)
Combines `paraphrase-multilingual-MiniLM-L12-v2` embeddings with Russian/Kyrgyz TF-IDF n-grams, extracts legal deal signals (`ДДУ`, `ПСО`, `Красная книга`, `срочно`), and detects sarcastic/tricky inputs (Mission 10)."""))

cells.append(nbf.v4.new_code_cell("""from sentence_transformers import SentenceTransformer
from nlp_model import train_and_save_nlp, ListingNLPAnalyzer, EMBED_MODEL_NAME

nlp_metrics = train_and_save_nlp(CSV_PATH, MODELS_DIR)
print(f"NLP Labeled Texts: {nlp_metrics['total_labeled_texts']:,}")
print(f"Majority Baseline Acc: {nlp_metrics['baseline_majority_accuracy']:.1%}")
print(f"NLP Test Accuracy:     {nlp_metrics['test_accuracy']:.1%}")
print(f"NLP Weighted F1:       {nlp_metrics['test_weighted_f1']:.3f} | Macro F1: {nlp_metrics['test_macro_f1']:.3f}")

embedder = SentenceTransformer(EMBED_MODEL_NAME)
nlp = ListingNLPAnalyzer(os.path.join(MODELS_DIR, "nlp_listing_bundle.joblib"), embedder=embedder)

real_ad = "Срочно продаётся 2-комн квартира в Джале, ПСО, документы ДДУ, сдача дома в конце года."
tricky_ad = "Элитный дворец в центре за 5 сомов, стены из воздуха, 1000% евроремонт на Луне, ахаха!"

print("\\nReal ad NLP:", nlp.analyze(real_ad))
print("\\nTricky/Sarcastic ad NLP (Mission 10):", nlp.analyze(tricky_ad))
"""))

cells.append(nbf.v4.new_markdown_cell("""## Part 3 — Official Kyrgyz Republic Legal & GIK RAG Assistant (17 Docs)
Hybrid retrieval (multilingual MiniLM + BM25) over 17 curated Kyrgyz Republic documents with exact source citations, strict *"I don't know"* fallback, and prompt-injection defense."""))

cells.append(nbf.v4.new_code_cell("""from rag_engine import KyrgyzRealEstateRAG

RAG_DIR = "../data/rag_docs" if os.path.isdir("../data/rag_docs") else "data/rag_docs"
rag = KyrgyzRealEstateRAG(
    docs_dir=RAG_DIR,
    embedder=embedder,
    api_base=OPENAI_API_BASE,
    api_key=OPENAI_API_KEY,
    llm_model=LLM_MODEL,
)
print(f"Indexed {rag.num_docs} official KR documents ({len(rag.chunks)} sections).")

# Integration Rule Demo (ML + NLP + RAG in ONE answer!)
nlp_real = nlp.analyze(real_ad)
integrated = rag.answer_integrated(
    ml_result=res_ok,
    nlp_result=nlp_real,
    user_question="Адекватная ли цена $115,000 и можно ли купить эту квартиру через ГИК «Менин үйүм»?",
    apartment_inputs=dzhal_sample,
)
print("\\n=== INTEGRATED ANSWER (ML + NLP + RAG) ===")
print(integrated["answer"])
"""))

cells.append(nbf.v4.new_markdown_cell("""## Part 4 — 20-Question RAG Evaluation & All 10 Client Day Missions
Runs our full automated verification suite covering Russian, Kyrgyz, English, out-of-scope refusals, prompt injections, impossible inputs, and tricky NLP inputs."""))

cells.append(nbf.v4.new_code_cell("""TESTS_DIR = "../tests" if os.path.isdir("../tests") else "tests"
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from run_all_tests import run_20_rag_eval, run_10_client_day_missions

rag_summary = run_20_rag_eval(rag)
mission_results = run_10_client_day_missions(ml_pred, nlp, rag)
"""))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open(OUT_NB, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Saved valid Colab notebook:", OUT_NB)
