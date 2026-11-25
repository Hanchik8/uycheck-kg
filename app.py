import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ml_model import BishkekPricePredictor, CONDITION_LABELS_EN, HOUSE_TYPE_LABELS
from nlp_model import EMBED_MODEL_NAME, ListingNLPAnalyzer
from rag_engine import KyrgyzRealEstateRAG

load_dotenv(os.path.join(ROOT_DIR, ".env"))

st.set_page_config(
    page_title="UyCheck KG — Bishkek Real Estate AI Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Clean Monochrome Architectural CSS (Gray & White — matches proposal slides!)
st.markdown(
    """
    <style>
    .stApp { background-color: #F6F7F8; color: #1F2226; }
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1180px; }
    h1, h2, h3 { color: #1F2226 !important; font-family: 'Segoe UI', sans-serif; }
    .hero-bar {
        background-color: #1F2226;
        color: #FFFFFF;
        padding: 1.25rem 1.6rem;
        border-left: 6px solid #6B7280;
        margin-bottom: 1.2rem;
    }
    .hero-title { font-size: 1.85rem; font-weight: 700; margin: 0; color: #FFFFFF; }
    .hero-sub { font-size: 0.96rem; color: #D1D5DB; margin-top: 0.25rem; }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #D8DADE;
        border-top: 4px solid #1F2226;
        padding: 0.95rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    .metric-label { font-size: 0.80rem; color: #666B73; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-val { font-size: 1.50rem; font-weight: 700; color: #1F2226; margin-top: 0.15rem; }
    .metric-note { font-size: 0.82rem; color: #4B5057; margin-top: 0.2rem; }
    .answer-box {
        background: #FFFFFF;
        border: 1px solid #D8DADE;
        border-left: 5px solid #1F2226;
        padding: 1.15rem 1.35rem;
        margin: 0.8rem 0;
        line-height: 1.55;
    }
    .warn-box {
        background: #EFEFF1;
        border: 1px solid #52575E;
        border-left: 5px solid #1F2226;
        padding: 0.9rem 1.1rem;
        margin: 0.6rem 0;
    }
    .tag-pill {
        display: inline-block;
        background: #ECEEF0;
        border: 1px solid #C8CCD2;
        color: #1F2226;
        padding: 0.18rem 0.55rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .disclaimer-bar {
        font-size: 0.82rem;
        color: #52575E;
        border-top: 1px solid #D8DADE;
        padding-top: 0.6rem;
        margin-top: 1.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


@st.cache_resource(show_spinner="Loading UyCheck KG models (ML + NLP + KR Legal RAG)...")
def load_all_components():
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    ml_predictor = BishkekPricePredictor(os.path.join(ROOT_DIR, "models", "ml_valuation_bundle.joblib"))
    nlp_analyzer = ListingNLPAnalyzer(
        os.path.join(ROOT_DIR, "models", "nlp_listing_bundle.joblib"),
        embedder=embedder,
    )
    rag_engine = KyrgyzRealEstateRAG(
        docs_dir=os.path.join(ROOT_DIR, "data", "rag_docs"),
        embedder=embedder,
        api_base=_get_secret("OPENAI_API_BASE", "https://api.reformboss.com/v1"),
        api_key=_get_secret("OPENAI_API_KEY", ""),
        llm_model=_get_secret("LLM_MODEL", "deepseek-v4-flash"),
    )
    return ml_predictor, nlp_analyzer, rag_engine


def plot_shap_bars(drivers):
    labels = [f"{d['label']}\n({d['value']})" for d in drivers][::-1]
    vals = [d["usd_impact"] for d in drivers][::-1]
    colors = ["#1F2226" if v >= 0 else "#7B8088" for v in vals]

    fig, ax = plt.subplots(figsize=(6.8, 3.4), dpi=180, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=colors, height=0.56)
    ax.axvline(0, color="#1F2226", linewidth=0.9)

    max_abs = max([abs(v) for v in vals] + [2000])
    ax.set_xlim(-max_abs * 1.62, max_abs * 1.55)

    for bar, v in zip(bars, vals):
        offset = max_abs * 0.05 if v >= 0 else -max_abs * 0.05
        ha = "left" if v >= 0 else "right"
        ax.text(
            v + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{v:+,.0f}$",
            va="center",
            ha=ha,
            fontsize=8.3,
            fontweight="bold",
            color="#1F2226",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.0, color="#1F2226")
    ax.xaxis.set_visible(False)
    for sp in ["top", "right", "bottom", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_title("Top Feature Impacts on Price Estimate (SHAP, USD)", loc="left", fontsize=9.8, fontweight="bold", color="#1F2226", pad=8)
    plt.subplots_adjust(left=0.35, right=0.97, top=0.88, bottom=0.05)
    return fig


PRESETS = {
    "dzhal_pso": {
        "district": "джал м-н",
        "area": 65.0,
        "rooms": 2,
        "floor": 6,
        "total_floors": 12,
        "year_built": 2024,
        "house_type": "монолит",
        "condition": "Auto-detect from ad text (NLP)",
        "heating": "газовое",
        "documents": "договор долевого участия",
        "asking_price": 115000.0,
        "ad_text": "Срочно продаётся 2-комн квартира в Джале, ПСО, документы ДДУ, сдача дома в конце года, газ подведён.",
        "question": "Адекватная ли цена $115,000, можно ли купить через ГИК «Менин үйүм» и какие риски у ДДУ?",
    },
    "center_euro": {
        "district": "московская - уметалиева",
        "area": 92.0,
        "rooms": 3,
        "floor": 7,
        "total_floors": 14,
        "year_built": 2021,
        "house_type": "кирпич",
        "condition": "Auto-detect from ad text (NLP)",
        "heating": "центральное",
        "documents": "договор купли-продажи, технический паспорт",
        "asking_price": 172000.0,
        "ad_text": "Продаётся просторная 3-комнатная квартира с дизайнерским евроремонтом, вся мебель и техника остаются, техпаспорт на руках, чистые документы.",
        "question": "Насколько справедлива цена и какой ежегодный налог на имущество придётся платить за 92 м² в Бишкеке?",
    },
    "penthouse_extreme": {
        "district": "магистраль",
        "area": 295.0,
        "rooms": 5,
        "floor": 12,
        "total_floors": 12,
        "year_built": 2023,
        "house_type": "монолит",
        "condition": "евроремонт",
        "heating": "центральное",
        "documents": "договор купли-продажи, технический паспорт, красная книга",
        "asking_price": 490000.0,
        "ad_text": "Продаётся двухуровневый пентхаус на Магистрали, 295 м², премиальный евроремонт, техпаспорт и договор купли-продажи.",
        "question": "How fair is this luxury price and what should I verify in GU Cadastre before buying?",
    },
    "tricky_sarcasm": {
        "district": "12 м-н",
        "area": 54.0,
        "rooms": 2,
        "floor": 4,
        "total_floors": 9,
        "year_built": 2018,
        "house_type": "панель",
        "condition": "Auto-detect from ad text (NLP)",
        "heating": "центральное",
        "documents": "технический паспорт",
        "asking_price": 82000.0,
        "ad_text": "Элитный дворец в центре за 5 сомов, стены из воздуха, 1000% евроремонт на Луне, ахаха!",
        "question": "Проверь это объявление и объясни, выдаётся ли Красная книга на квартиру?",
    },
}


def init_form_state():
    if "form" not in st.session_state:
        st.session_state.form = dict(PRESETS["dzhal_pso"])
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def apply_preset(key: str):
    st.session_state.form = dict(PRESETS[key])


def main():
    init_form_state()
    ml_predictor, nlp_analyzer, rag_engine = load_all_components()

    # Header
    st.markdown(
        """
        <div class="hero-bar">
            <div class="hero-title">UyCheck KG</div>
            <div class="hero-sub">Bishkek Real Estate AI Assistant — Fair Price Valuation (ML) · RU/KG Listing Analysis (NLP) · Official Kyrgyz Republic Legal & GIK Guide (RAG)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs([
        "1. Integrated Apartment Check (ML + NLP + RAG)",
        "2. KR Legal & GIK Mortgage Q&A (RAG Chat)",
        "3. Model Quality & Data Card",
    ])

    # ==========================================================
    # TAB 1: INTEGRATED APARTMENT CHECK (ML + NLP + RAG)
    # ==========================================================
    with tab1:
        st.caption("Quick presets to test the integrated pipeline (or enter any custom values below):")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        if pcol1.button("Dzhal 2-room ПСО ($115k)", use_container_width=True):
            apply_preset("dzhal_pso")
            st.rerun()
        if pcol2.button("Center 3-room Euro ($172k)", use_container_width=True):
            apply_preset("center_euro")
            st.rerun()
        if pcol3.button("Extreme Valid: 295 m² Penthouse", use_container_width=True):
            apply_preset("penthouse_extreme")
            st.rerun()
        if pcol4.button("Tricky NLP: Sarcastic Ad", use_container_width=True):
            apply_preset("tricky_sarcasm")
            st.rerun()

        f = st.session_state.form
        districts = ml_predictor.meta["ui_districts"]
        if f["district"] not in districts:
            districts = [f["district"]] + districts

        cond_options = ["Auto-detect from ad text (NLP)"] + list(CONDITION_LABELS_EN.keys())[:-1]
        house_options = list(HOUSE_TYPE_LABELS.keys())[:-1]
        heat_options = ml_predictor.meta["ui_heating"]
        if f["heating"] not in heat_options:
            heat_options = [f["heating"]] + heat_options

        with st.form("integrated_form"):
            left_col, right_col = st.columns([1.05, 1.0], gap="large")

            with left_col:
                st.subheader("Apartment Parameters (ML Input)")
                c1, c2 = st.columns(2)
                district = c1.selectbox("Bishkek District", districts, index=districts.index(f["district"]))
                house_type = c2.selectbox(
                    "Building Material",
                    house_options,
                    index=house_options.index(f["house_type"]) if f["house_type"] in house_options else 0,
                    format_func=lambda x: HOUSE_TYPE_LABELS.get(x, x),
                )

                c3, c4, c5 = st.columns(3)
                # Note: intentionally allow negative/extreme numbers so our Python validator catches Mission 5 cleanly!
                area = c3.number_input("Area (m²)", value=float(f["area"]), step=1.0)
                rooms = c4.number_input("Rooms", value=int(f["rooms"]), step=1)
                year_built = c5.number_input("Year Built", value=int(f["year_built"]), step=1)

                c6, c7, c8 = st.columns(3)
                floor = c6.number_input("Floor", value=int(f["floor"]), step=1)
                total_floors = c7.number_input("Total Floors", value=int(f["total_floors"]), step=1)
                asking_price = c8.number_input("Asking Price (USD)", value=float(f["asking_price"]), step=1000.0)

                c9, c10 = st.columns(2)
                condition_choice = c9.selectbox(
                    "Renovation Condition",
                    cond_options,
                    index=cond_options.index(f["condition"]) if f["condition"] in cond_options else 0,
                    format_func=lambda x: CONDITION_LABELS_EN.get(x, x),
                )
                heating = c10.selectbox("Heating", heat_options, index=heat_options.index(f["heating"]))
                documents = st.text_input("Listed Documents (e.g., ДДУ, техпаспорт, Красная книга)", value=f["documents"])

            with right_col:
                st.subheader("Listing Description (NLP) & Buyer Question (RAG)")
                ad_text = st.text_area(
                    "Paste apartment ad text (Russian, Kyrgyz, or English)",
                    value=f["ad_text"],
                    height=135,
                    placeholder="Например: Продаётся 2-комн квартира в Джале, ПСО, ДДУ...",
                )
                question = st.text_area(
                    "Your question (price fairness, GIK mortgage, Cadastre, DDU, taxes — RU / KG / EN)",
                    value=f["question"],
                    height=105,
                    placeholder="Можно ли оформить через ГИК «Менин үйүм» и какие документы проверить?",
                )

            submitted = st.form_submit_button("Analyze Apartment & Answer (ML + NLP + RAG)", type="primary", use_container_width=True)

        if submitted:
            # 1. Run NLP analysis first (if ad text is provided)
            nlp_res = nlp_analyzer.analyze(ad_text) if ad_text.strip() else None

            # Use NLP predicted condition if Auto-detect selected and confidence is reliable
            effective_condition = condition_choice
            used_nlp_condition = False
            if condition_choice == "Auto-detect from ad text (NLP)":
                if nlp_res and not nlp_res["low_confidence"] and nlp_res["predicted_condition"] != "unknown":
                    effective_condition = nlp_res["predicted_condition"]
                    used_nlp_condition = True
                else:
                    effective_condition = "unknown"

            # Combine user-entered documents field + NLP extracted signals
            combined_docs = documents
            if nlp_res and nlp_res.get("signals"):
                combined_docs = f"{documents} {' '.join(nlp_res['signals'])}".strip()

            payload = {
                "district": district,
                "area": area,
                "rooms": rooms,
                "floor": floor,
                "total_floors": total_floors,
                "year_built": year_built,
                "house_type": house_type,
                "condition": effective_condition,
                "heating": heating,
                "documents": combined_docs,
                "asking_price_usd": asking_price if asking_price != 0 else None,
            }

            # 2. Run ML valuation + strict Mission 5/6 validation
            ml_res = ml_predictor.predict(payload)

            if not ml_res["ok"]:
                st.error("**Invalid apartment input detected — prediction stopped safely (no crash):**")
                for err in ml_res["errors"]:
                    st.markdown(f"- {err}")
                if nlp_res:
                    st.info(f"**NLP Note:** {nlp_res['display_condition']} (confidence {nlp_res['confidence']:.2f}) — {nlp_res['reason']}")
            else:
                if ml_res["warnings"]:
                    for w in ml_res["warnings"]:
                        st.warning(w)

                # Top 4 KPI summary cards
                k1, k2, k3, k4 = st.columns(4)
                k1.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Fair Market Value (ML)</div>"
                    f"<div class='metric-val'>${ml_res['fair_price_usd']:,.0f}</div>"
                    f"<div class='metric-note'>${ml_res['fair_price_per_m2']:,.0f} / m²</div></div>",
                    unsafe_allow_html=True,
                )

                if ml_res["overprice_pct"] is not None:
                    diff_lbl = f"{ml_res['overprice_pct']:+.1f}%"
                    diff_note = ml_res["price_verdict"]
                else:
                    diff_lbl = "N/A"
                    diff_note = "No asking price entered"
                k2.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Asking vs. Market</div>"
                    f"<div class='metric-val'>{diff_lbl}</div>"
                    f"<div class='metric-note'>{diff_note}</div></div>",
                    unsafe_allow_html=True,
                )

                if nlp_res:
                    nlp_title = "LOW CONFIDENCE" if nlp_res["low_confidence"] else f"Confidence {nlp_res['confidence']:.0%}"
                    nlp_sub = nlp_res["display_condition"]
                else:
                    nlp_title = "No text"
                    nlp_sub = CONDITION_LABELS_EN.get(effective_condition, effective_condition)
                k3.markdown(
                    f"<div class='metric-card'><div class='metric-label'>NLP Listing Check</div>"
                    f"<div class='metric-val' style='font-size:1.18rem;'>{nlp_title}</div>"
                    f"<div class='metric-note'>{nlp_sub}</div></div>",
                    unsafe_allow_html=True,
                )

                k4.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Condition Used in ML</div>"
                    f"<div class='metric-val' style='font-size:1.08rem;'>{CONDITION_LABELS_EN.get(effective_condition, effective_condition)}</div>"
                    f"<div class='metric-note'>{'From NLP text' if used_nlp_condition else 'Manual / fallback'}</div></div>",
                    unsafe_allow_html=True,
                )

                # Prominent Low-Confidence warning banner for Mission 10!
                if nlp_res and nlp_res["low_confidence"]:
                    st.warning(
                        f"**NLP Low Confidence Warning ({nlp_res['confidence']:.2f}):** {nlp_res['reason']} "
                        "Because the text is unreliable, ML did not trust it for renovation classification."
                    )

                # 3. Unified Integrated Response (ML + NLP + RAG together!)
                with st.spinner("Combining ML valuation, NLP signals, and official KR legal documents..."):
                    rag_res = rag_engine.answer_integrated(ml_res, nlp_res, question, payload)

                st.subheader("Unified Assessment (ML + NLP + KR Legal RAG)")
                st.markdown(f"<div class='answer-box'>{rag_res['answer']}</div>", unsafe_allow_html=True)

                # Detailed breakdown: SHAP (Mission 2) + NLP details
                dcol1, dcol2 = st.columns([1.25, 0.95], gap="large")
                with dcol1:
                    st.subheader("Why This Price? (SHAP Explanation)")
                    fig = plot_shap_bars(ml_res["top_drivers"])
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                with dcol2:
                    st.subheader("NLP Extracted Signals & Probabilities")
                    if nlp_res:
                        if nlp_res["signals"]:
                            pills = "".join(f"<span class='tag-pill'>{s}</span>" for s in nlp_res["signals"])
                            st.markdown(f"**Detected deal/legal tags:**<br>{pills}", unsafe_allow_html=True)
                        else:
                            st.markdown("**Detected deal/legal tags:** None explicitly mentioned.")
                        st.markdown(f"**Domain similarity to Bishkek ads:** `{nlp_res['domain_similarity']:.3f}`")
                        if nlp_res.get("top_probs"):
                            st.markdown("**Top class probabilities:**")
                            for cls_name, prob in nlp_res["top_probs"].items():
                                st.progress(min(max(prob, 0.0), 1.0), text=f"{cls_name}: {prob:.1%}")
                    else:
                        st.info("Paste an apartment description above to activate NLP renovation & deal-tag extraction.")

                if rag_res.get("retrieved"):
                    with st.expander("Inspect Retrieved Official Kyrgyz Republic Legal Sources"):
                        for ch in rag_res["retrieved"]:
                            st.markdown(
                                f"**{ch['doc_id']}** — *{ch['section']}* "
                                f"(dense: `{ch['dense_score']:.3f}`, hybrid: `{ch['hybrid_score']:.3f}`)"
                            )
                            if ch.get("source_url"):
                                st.caption(f"Official source: {ch['source_url']}")
                            st.markdown(ch["text"])
                            st.divider()

    # ==========================================================
    # TAB 2: STANDALONE KR LEGAL & GIK RAG CHAT
    # ==========================================================
    with tab2:
        st.subheader("Ask About Kyrgyz Republic Real Estate Rules (RU / KG / EN)")
        st.caption("Grounded strictly in 17 verified KR documents (GIK «Менин үйүм», Law No. 153, Housing Code No. 117, GU «Кадастр», KR Tax Code).")

        qc1, qc2, qc3, qc4 = st.columns(4)
        sample_q = None
        if qc1.button("Ставки ГИК «Менин үйүм»?", use_container_width=True):
            sample_q = "Какие ставки и условия по ипотеке ГИК «Менин үйүм» для бюджетников и частного сектора?"
        if qc2.button("Бишкекте батирге Кызыл китеп берилеби?", use_container_width=True):
            sample_q = "Бишкекте батирге Кызыл китеп берилеби? Кандай документтер керек?"
        if qc3.button("Tax on selling a flat in KR?", use_container_width=True):
            sample_q = "Do I pay income tax when selling an apartment in Kyrgyzstan within 1 year vs after 1 year?"
        if qc4.button("Out-of-scope test (Kazakhstan)", use_container_width=True):
            sample_q = "What is the 7-20-25 mortgage rate in Almaty, Kazakhstan?"

        for turn in st.session_state.chat_history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn.get("sources"):
                    st.caption("Verified sources: " + " · ".join(turn["sources"][:3]))
                if turn.get("retrieved"):
                    with st.expander("View exact retrieved excerpts"):
                        for ch in turn["retrieved"]:
                            st.markdown(f"**{ch['doc_id']}** ({ch['section']}) — score `{ch['hybrid_score']:.3f}`")
                            st.markdown(ch["text"])
                            st.divider()

        user_msg = st.chat_input("Ask any question in Russian, Kyrgyz, or English...")
        active_q = sample_q or user_msg

        if active_q:
            st.session_state.chat_history.append({"role": "user", "content": active_q})
            with st.spinner("Searching verified KR documents..."):
                ans = rag_engine.answer_question(active_q)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ans["answer"],
                "sources": ans.get("sources", []),
                "retrieved": ans.get("retrieved", [])[:3] if ans["status"] == "answered" else [],
            })
            st.rerun()

        if st.session_state.chat_history:
            if st.button("Clear chat history"):
                st.session_state.chat_history = []
                st.rerun()

    # ==========================================================
    # TAB 3: MODEL QUALITY & TRANSPARENCY
    # ==========================================================
    with tab3:
        st.subheader("Verified Benchmark Metrics (Fixed Seed = 42)")
        m_ml = ml_predictor.metrics
        m_nlp = nlp_analyzer.metrics

        st.markdown("**ML Price Regression — Bishkek Real Estate (8,821 raw / " + f"{m_ml['clean_rows']:,} clean rows)**")
        rows_tbl = [
            {"Model": "Baseline 1: Global Median", "MAE ($)": m_ml["baselines"]["global_median"]["mae_usd"], "RMSE ($)": m_ml["baselines"]["global_median"]["rmse_usd"], "MAPE (%)": m_ml["baselines"]["global_median"]["mape_pct"], "R²": m_ml["baselines"]["global_median"]["r2"]},
            {"Model": "Baseline 2: Ridge Regression", "MAE ($)": m_ml["baselines"]["ridge_regression"]["mae_usd"], "RMSE ($)": m_ml["baselines"]["ridge_regression"]["rmse_usd"], "MAPE (%)": m_ml["baselines"]["ridge_regression"]["mape_pct"], "R²": m_ml["baselines"]["ridge_regression"]["r2"]},
            {"Model": "LightGBM Regressor", "MAE ($)": m_ml["models"]["lightgbm"]["mae_usd"], "RMSE ($)": m_ml["models"]["lightgbm"]["rmse_usd"], "MAPE (%)": m_ml["models"]["lightgbm"]["mape_pct"], "R²": m_ml["models"]["lightgbm"]["r2"]},
            {"Model": "CatBoost Regressor", "MAE ($)": m_ml["models"]["catboost"]["mae_usd"], "RMSE ($)": m_ml["models"]["catboost"]["rmse_usd"], "MAPE (%)": m_ml["models"]["catboost"]["mape_pct"], "R²": m_ml["models"]["catboost"]["r2"]},
            {"Model": "Final Ensemble (50% LGB + 50% CB)", "MAE ($)": m_ml["models"]["ensemble_lgb_cb"]["mae_usd"], "RMSE ($)": m_ml["models"]["ensemble_lgb_cb"]["rmse_usd"], "MAPE (%)": m_ml["models"]["ensemble_lgb_cb"]["mape_pct"], "R²": m_ml["models"]["ensemble_lgb_cb"]["r2"]},
        ]
        st.dataframe(rows_tbl, use_container_width=True, hide_index=True)

        st.markdown(f"**NLP Multilingual Listing Classifier ({m_nlp['total_labeled_texts']:,} RU/KG descriptions) & RAG Status**")
        st.markdown(
            f"- **Encoder & Features:** `{m_nlp['features']}`\n"
            f"- **Test Accuracy:** `{m_nlp['test_accuracy']:.1%}` (vs. `{m_nlp['baseline_majority_accuracy']:.1%}` majority baseline)\n"
            f"- **Test Weighted F1:** `{m_nlp['test_weighted_f1']:.3f}` · **Macro F1:** `{m_nlp['test_macro_f1']:.3f}`\n"
            f"- **Low-Confidence Threshold:** `{m_nlp['confidence_threshold']}` · **OOD Similarity Cutoff:** `{m_nlp['ood_similarity_threshold']}`\n"
            f"- **RAG Corpus:** `{rag_engine.num_docs}` official KR documents (`{len(rag_engine.chunks)}` indexed sections) · **20/20** eval benchmark\n"
            f"- **LLM Status:** `{'Connected (' + rag_engine.llm_model + ')' if rag_engine.client else 'Offline Grounded Fallback'}`"
        )

        st.markdown("**Data Source & License:** `raimbekovm/bishkek-real-estate` (Hugging Face, **CC BY 4.0**). Zero PII — listing IDs, URLs, and contact fields removed before training.")

    st.markdown(
        "<div class='disclaimer-bar'><strong>Disclaimer:</strong> UyCheck KG is an educational market-valuation and legal-orientation assistant built for the Generative AI course (Ala-Too International University, Team: Nurlanov Azizkhan, Zhanybekov Nurdos, Abdilazizov Nurmukhamed). It does not replace a licensed notary, official GU «Кадастр» title extract, or certified property appraiser.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
