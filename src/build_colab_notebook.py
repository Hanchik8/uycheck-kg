import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_NB = os.path.join(ROOT, "notebooks", "UyCheck_KG_Pipeline.ipynb")
os.makedirs(os.path.dirname(OUT_NB), exist_ok=True)

nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell("""# UyCheck KG — Reproducible Pipeline

Team: Nurlanov Azizkhan, Zhanybekov Nurdos, Abdilazizov Nurmukhamed  
Course: Generative AI (Fall 2026), Ala-Too International University  
Seed: `42`

Covers data cleaning, LightGBM + CatBoost price regression with SHAP, multilingual NLP listing classification, and local LM Studio RAG (`http://localhost:1234/v1` for both LLM completions and embeddings).
"""),
    nbf.v4.new_code_cell("""import json, os, sys, urllib.request, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
LLM_MODEL = os.getenv("LLM_MODEL", "auto")
EMBED_MODEL = os.getenv("EMBED_MODEL", "auto")

try:
    from google.colab import userdata
    OPENAI_API_KEY = userdata.get("OPENAI_API_KEY") or OPENAI_API_KEY
    OPENAI_API_BASE = userdata.get("OPENAI_API_BASE") or OPENAI_API_BASE
except Exception:
    pass

print("Endpoint:", OPENAI_API_BASE, "| LLM:", LLM_MODEL, "| Embeddings:", EMBED_MODEL)
"""),
    nbf.v4.new_code_cell("""os.makedirs("data", exist_ok=True)
CSV_PATH = "data/bishkek_apartments.csv"
if not os.path.exists(CSV_PATH):
    urllib.request.urlretrieve(
        "https://huggingface.co/datasets/raimbekovm/bishkek-real-estate/resolve/main/data/bishkek_apartments.csv",
        CSV_PATH,
    )

raw_df = pd.read_csv(CSV_PATH)
print(f"Rows: {len(raw_df):,} | Median USD: ${raw_df['price_usd'].median():,.0f}")
"""),
    nbf.v4.new_markdown_cell("## 1. Price Regression (LightGBM + CatBoost + SHAP)"),
    nbf.v4.new_code_cell("""if os.path.isdir("src") and "src" not in sys.path:
    sys.path.insert(0, "src")
elif os.path.isdir("../src") and "../src" not in sys.path:
    sys.path.insert(0, "../src")
    CSV_PATH = "../data/bishkek_apartments.csv"

from ml_model import BishkekPricePredictor, train_and_save_ml

MODELS_DIR = "../models" if os.path.isdir("../src") else "models"
ml_metrics = train_and_save_ml(CSV_PATH, MODELS_DIR)

pd.DataFrame([
    {"Model": "Baseline: Global Median", **ml_metrics["baselines"]["global_median"]},
    {"Model": "Baseline: Ridge", **ml_metrics["baselines"]["ridge_regression"]},
    {"Model": "LightGBM", **ml_metrics["models"]["lightgbm"]},
    {"Model": "CatBoost", **ml_metrics["models"]["catboost"]},
    {"Model": "Ensemble (50/50)", **ml_metrics["models"]["ensemble_lgb_cb"]},
])
"""),
    nbf.v4.new_code_cell("""ml_pred = BishkekPricePredictor(os.path.join(MODELS_DIR, "ml_valuation_bundle.joblib"))
sample = {
    "district": "джал м-н", "area": 65, "rooms": 2, "floor": 6, "total_floors": 12,
    "year_built": 2024, "house_type": "монолит", "condition": "под самоотделку (псо)",
    "asking_price_usd": 115000,
}
res_ok = ml_pred.predict(sample)
print(f"Fair: ${res_ok['fair_price_usd']:,.0f} (${res_ok['fair_price_per_m2']:,.0f}/m²) | {res_ok['price_verdict']}")
for d in res_ok["top_drivers"][:5]:
    print(f"  {d['label']} ({d['value']}): {d['usd_impact']:+,.0f}$")

res_bad = ml_pred.predict({"district": "джал м-н", "area": -20, "rooms": 2, "floor": 50, "total_floors": 5})
print("Invalid input check:", res_bad["errors"])
"""),
    nbf.v4.new_markdown_cell("## 2. Multilingual Listing Text Classifier (NLP)"),
    nbf.v4.new_code_cell("""from nlp_model import ListingNLPAnalyzer, train_and_save_nlp

nlp_metrics = train_and_save_nlp(CSV_PATH, MODELS_DIR)
print(f"Labeled: {nlp_metrics['total_labeled_texts']:,} | Majority: {nlp_metrics['baseline_majority_accuracy']:.1%} | Test Acc: {nlp_metrics['test_accuracy']:.1%} | Weighted F1: {nlp_metrics['test_weighted_f1']:.3f}")

nlp = ListingNLPAnalyzer(os.path.join(MODELS_DIR, "nlp_listing_bundle.joblib"))
real_ad = "Срочно продаётся 2-комн квартира в Джале, ПСО, документы ДДУ, сдача дома в конце года."
tricky_ad = "Элитный дворец в центре за 5 сомов, стены из воздуха, 1000% евроремонт на Луне, ахаха!"
print("Real ad:", nlp.analyze(real_ad))
print("Tricky ad:", nlp.analyze(tricky_ad))
"""),
    nbf.v4.new_markdown_cell("## 3. KR Legal & GIK RAG on LM Studio (LLM + Embeddings)"),
    nbf.v4.new_code_cell("""from rag_engine import KyrgyzRealEstateRAG

RAG_DIR = "../data/rag_docs" if os.path.isdir("../data/rag_docs") else "data/rag_docs"
rag = KyrgyzRealEstateRAG(
    docs_dir=RAG_DIR,
    embedder=nlp.embedder,
    api_base=OPENAI_API_BASE,
    api_key=OPENAI_API_KEY,
    llm_model=LLM_MODEL,
    embed_model=EMBED_MODEL,
)
print(f"Backend: {rag.backend_mode} | LLM: {rag.llm_model} | Embeddings: {rag.embed_backend} ({rag.embed_model})")

integrated = rag.answer_integrated(
    ml_result=res_ok,
    nlp_result=nlp.analyze(real_ad),
    user_question="Адекватная ли цена $115,000 и можно ли купить эту квартиру через ГИК «Менин үйүм»?",
    apartment_inputs=sample,
    ad_text=real_ad,
)
print(integrated["answer"])
"""),
    nbf.v4.new_markdown_cell("## 4. 20-Question RAG Benchmark & 10 Client Day Missions"),
    nbf.v4.new_code_cell("""TESTS_DIR = "../tests" if os.path.isdir("../tests") else "tests"
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from run_all_tests import run_10_client_day_missions, run_20_rag_eval

rag_summary = run_20_rag_eval(rag)
mission_results = run_10_client_day_missions(ml_pred, nlp, rag)
"""),
]

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
with open(OUT_NB, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Saved notebook:", OUT_NB)
