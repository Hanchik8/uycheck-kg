import os
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ml_model import (
    BishkekPricePredictor,
    CONDITION_LABELS_EN,
    CONDITION_LABELS_RU,
    HEATING_LABELS_EN,
    HEATING_LABELS_RU,
    HOUSE_TYPE_LABELS,
    HOUSE_TYPE_LABELS_RU,
    format_district,
)
from nlp_model import ListingNLPAnalyzer
from rag_engine import DEFAULT_LMSTUDIO_URL, KyrgyzRealEstateRAG, _lmstudio_alive, detect_language

load_dotenv(os.path.join(ROOT_DIR, ".env"))

st.set_page_config(
    page_title="UyCheck KG — Оценка квартир в Бишкеке",
    page_icon="U",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Manrope:wght@400;500;600;700&display=swap');

:root {
    --bg: #F7F5F0;
    --surface: #FFFFFF;
    --surface-soft: #F2EFE8;
    --border: #E5E0D6;
    --border-strong: #CFC8BC;
    --ink: #181A1D;
    --muted: #62656C;
    --pine: #1B4332;
    --pine-soft: #EBF3EE;
    --amber: #9A3412;
    --amber-soft: #FCF0E8;
}

html, body, [class*="css"], .stApp {
    font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--bg);
    color: var(--ink);
}

#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 2.2rem; }

.block-container {
    max-width: 1120px;
    padding-top: 1.1rem;
    padding-bottom: 3rem;
}

.masthead {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    padding-bottom: 1.05rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.1rem;
    gap: 1rem;
}
.brand-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.brand-name {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: var(--ink);
    line-height: 1;
}
.brand-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--pine);
    background: var(--pine-soft);
    padding: 0.16rem 0.45rem;
    border-radius: 4px;
    letter-spacing: 0.04em;
}
.brand-sub {
    font-size: 0.90rem;
    color: var(--muted);
    margin-top: 0.35rem;
}
.engine-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: var(--muted);
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 0.28rem 0.65rem;
    border-radius: 999px;
    white-space: nowrap;
}
.engine-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-right: 0.4rem;
    vertical-align: middle;
}
.dot-local { background: var(--pine); }
.dot-remote { background: #52606D; }
.dot-off { background: var(--amber); }

.stTabs [data-baseweb="tab-list"] {
    gap: 1.6rem;
    border-bottom: 1px solid var(--border);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.65rem 0.1rem;
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--muted);
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: var(--ink) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--pine) !important;
    height: 2px !important;
}

[data-testid="stForm"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.35rem 1.45rem 1.25rem 1.45rem;
}

.sec-eyebrow {
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--muted);
    margin-bottom: 0.55rem;
}

.stButton > button[kind="secondary"] {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--ink);
    border-radius: 8px;
    font-size: 0.83rem;
    font-weight: 500;
    padding: 0.42rem 0.75rem;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--pine);
    background: var(--pine-soft);
    color: var(--pine);
}
.stFormSubmitButton > button {
    background: var(--pine) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.62rem 1.2rem !important;
}
.stFormSubmitButton > button:hover {
    background: #143326 !important;
}

.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.95rem 1.1rem;
    height: 100%;
}
.kpi-label {
    font-size: 0.73rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.42rem;
    font-weight: 600;
    color: var(--ink);
    margin-top: 0.28rem;
    line-height: 1.2;
}
.kpi-value-text {
    font-family: 'Manrope', sans-serif;
    font-size: 1.04rem;
    font-weight: 700;
    color: var(--ink);
    margin-top: 0.32rem;
    line-height: 1.3;
}
.kpi-sub {
    font-size: 0.80rem;
    color: var(--muted);
    margin-top: 0.30rem;
}
.tone-good {
    display: inline-block;
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--pine);
    background: var(--pine-soft);
    padding: 0.16rem 0.5rem;
    border-radius: 6px;
    margin-top: 0.35rem;
}
.tone-warn {
    display: inline-block;
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--amber);
    background: var(--amber-soft);
    padding: 0.16rem 0.5rem;
    border-radius: 6px;
    margin-top: 0.35rem;
}
.tone-neutral {
    display: inline-block;
    font-size: 0.76rem;
    font-weight: 500;
    color: var(--muted);
    background: var(--surface-soft);
    padding: 0.16rem 0.5rem;
    border-radius: 6px;
    margin-top: 0.35rem;
}

.panel-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.35rem 1.55rem;
    margin-top: 0.85rem;
    line-height: 1.66;
    font-size: 0.94rem;
}
.panel-card h3, .panel-card h4 {
    margin-top: 0.8rem;
    margin-bottom: 0.35rem;
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--ink);
}
.panel-card p { margin-bottom: 0.65rem; }

.tag-chip {
    display: inline-block;
    background: var(--surface-soft);
    border: 1px solid var(--border);
    color: var(--ink);
    padding: 0.22rem 0.6rem;
    margin: 0.18rem 0.3rem 0.18rem 0;
    border-radius: 6px;
    font-size: 0.80rem;
    font-weight: 500;
}

.prob-wrap { margin-top: 0.65rem; }
.prob-row { margin-bottom: 0.55rem; }
.prob-head {
    display: flex;
    justify-content: space-between;
    font-size: 0.81rem;
    margin-bottom: 0.22rem;
}
.prob-name { color: var(--ink); font-weight: 500; }
.prob-pct { font-family: 'JetBrains Mono', monospace; color: var(--muted); font-size: 0.78rem; }
.prob-track {
    height: 6px;
    background: var(--surface-soft);
    border-radius: 999px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    background: var(--pine);
    border-radius: 999px;
}

.runtime-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.15rem;
}
.runtime-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.84rem;
    color: var(--ink);
    margin-top: 0.28rem;
    word-break: break-all;
}

.foot-note {
    font-size: 0.78rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    padding-top: 0.85rem;
    margin-top: 2rem;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

COPY = {
    "ru": {
        "subtitle": "Рыночная оценка квартир Бишкека и правовая проверка сделок по документам КР",
        "tabs": ["Проверка квартиры", "Вопросы по законам и ГИК", "О системе"],
        "examples_hint": "Примеры для быстрого заполнения",
        "presets": [
            ("dzhal_pso", "Джал · 2-комн · ПСО"),
            ("center_euro", "Центр · 3-комн · Евро"),
            ("penthouse_extreme", "Магистраль · 295 м²"),
            ("tricky_sarcasm", "Сомнительный текст"),
        ],
        "sec_params": "Параметры объекта",
        "sec_text": "Описание и вопрос",
        "district": "Район",
        "house_type": "Материал дома",
        "area": "Площадь, м²",
        "rooms": "Комнат",
        "year_built": "Год постройки",
        "floor": "Этаж",
        "total_floors": "Этажность",
        "asking_price": "Цена продавца, $",
        "condition": "Состояние ремонта",
        "auto_cond": "Определить по тексту объявления",
        "heating": "Отопление",
        "documents": "Документы в объявлении",
        "ad_label": "Текст объявления",
        "ad_placeholder": "Вставьте описание квартиры на русском, кыргызском или английском...",
        "q_label": "Ваш вопрос",
        "q_placeholder": "Например: адекватна ли цена, можно ли оформить через ГИК «Менин үйүм»?",
        "submit": "Оценить и проверить",
        "spinner_eval": "Считаем рыночную цену и сверяем документы КР...",
        "err_title": "Проверьте введённые параметры — расчёт остановлен:",
        "kpi_fair": "Рыночная оценка",
        "kpi_asking": "Цена продавца",
        "kpi_no_asking": "Не указана",
        "kpi_text": "Анализ текста",
        "kpi_cond": "Ремонт в расчёте",
        "from_text": "Определён по тексту",
        "from_manual": "Указан вручную / база",
        "low_conf": "Низкая уверенность",
        "conf_fmt": "Уверенность {:.0%}",
        "no_text": "Текст не задан",
        "low_conf_banner": "Ненадёжное описание ({:.0%}): {} Модель оценки цены не стала опираться на этот текст.",
        "sec_brief": "Итог проверки",
        "sec_shap": "Из чего сложилась цена",
        "sec_nlp": "Сигналы из текста объявления",
        "tags_none": "Явных юридических или ценовых меток в тексте не найдено.",
        "sim_label": "Сходство с реальными объявлениями Бишкека",
        "probs_label": "Вероятность состояния ремонта",
        "paste_hint": "Добавьте текст объявления, чтобы определить ремонт и извлечь ключевые метки.",
        "sources_exp": "Использованные фрагменты документов КР",
        "official_src": "Официальный источник",
        "chat_head": "Законы КР, ГУ «Кадастр» и ипотека ГИК",
        "chat_sub": "Ответы строятся строго по 17 официальным документам Кыргызстана. Если факта нет в базе — ассистент скажет об этом прямо.",
        "chat_samples": [
            ("Ставки ГИК «Менин үйүм»", "Какие ставки и условия по ипотеке ГИК «Менин үйүм» для бюджетников и частного сектора?"),
            ("Батирге Кызыл китеп берилеби?", "Бишкекте батирге Кызыл китеп берилеби? Кандай документтер керек?"),
            ("Налог при продаже до 1 года", "Какой подоходный налог нужно платить при продаже квартиры в Кыргызстане до 1 года и после 1 года владения?"),
            ("Риски покупки по ДДУ", "На что обязательно проверить застройщика и договор ДДУ в Бишкеке до оплаты?"),
        ],
        "chat_placeholder": "Задайте вопрос на русском, кыргызском или английском...",
        "chat_spinner": "Ищем ответ в документах КР...",
        "sources_cap": "Источники",
        "excerpts_exp": "Показать фрагменты документов",
        "clear_chat": "Очистить диалог",
        "sys_runtime": "Среда выполнения RAG",
        "sys_llm": "Модель генерации (LLM)",
        "sys_emb": "Модель эмбеддингов",
        "sys_bench": "Качество моделей (фиксированный seed = 42)",
        "sys_ml_head": "Оценка стоимости квартир Бишкека — {clean:,} проверенных объявлений из {raw:,}",
        "sys_nlp_head": "Анализ текстов объявлений ({n:,} описаний) и база документов",
        "footer": "UyCheck KG — учебный аналитический проект (Ала-Тоо, курс Generative AI). Расчёты и справки носят информационный характер и не заменяют нотариуса или официальную выписку ГУ «Кадастр».",
    },
    "en": {
        "subtitle": "Bishkek apartment valuation & Kyrgyz Republic legal verification",
        "tabs": ["Apartment Check", "Laws & GIK Q&A", "System & Data"],
        "examples_hint": "Sample listings",
        "presets": [
            ("dzhal_pso", "Dzhal · 2-room · Shell"),
            ("center_euro", "Center · 3-room · Euro"),
            ("penthouse_extreme", "Magistral · 295 m²"),
            ("tricky_sarcasm", "Suspicious listing"),
        ],
        "sec_params": "Property details",
        "sec_text": "Listing text & question",
        "district": "District",
        "house_type": "Building material",
        "area": "Area, m²",
        "rooms": "Rooms",
        "year_built": "Year built",
        "floor": "Floor",
        "total_floors": "Total floors",
        "asking_price": "Asking price, $",
        "condition": "Renovation",
        "auto_cond": "Detect from listing text",
        "heating": "Heating",
        "documents": "Listed documents",
        "ad_label": "Listing description",
        "ad_placeholder": "Paste the apartment ad text in Russian, Kyrgyz, or English...",
        "q_label": "Your question",
        "q_placeholder": "e.g., Is the asking price fair, and is this eligible for GIK «Menin Uyum»?",
        "submit": "Evaluate Apartment",
        "spinner_eval": "Estimating market value and checking KR legal docs...",
        "err_title": "Please check the entered parameters:",
        "kpi_fair": "Market estimate",
        "kpi_asking": "Asking price",
        "kpi_no_asking": "Not entered",
        "kpi_text": "Listing text",
        "kpi_cond": "Renovation used",
        "from_text": "Extracted from text",
        "from_manual": "Manual / fallback",
        "low_conf": "Low confidence",
        "conf_fmt": "Confidence {:.0%}",
        "no_text": "No text",
        "low_conf_banner": "Unreliable description ({:.0%}): {} Valuation did not rely on this text.",
        "sec_brief": "Assessment",
        "sec_shap": "Price breakdown (SHAP)",
        "sec_nlp": "Listing signals",
        "tags_none": "No explicit deal or document tags detected.",
        "sim_label": "Similarity to real Bishkek listings",
        "probs_label": "Renovation probabilities",
        "paste_hint": "Paste a listing description to classify renovation and extract tags.",
        "sources_exp": "Retrieved Kyrgyz Republic legal excerpts",
        "official_src": "Official source",
        "chat_head": "Kyrgyz Republic Real Estate Law & GIK Q&A",
        "chat_sub": "Answers are grounded strictly in 17 verified KR documents. Out-of-scope questions are refused directly.",
        "chat_samples": [
            ("GIK «Menin Uyum» rates", "What are the GIK Menin Uyum mortgage rates and down payment requirements for public and private sector buyers?"),
            ("Батирге Кызыл китеп берилеби?", "Бишкекте батирге Кызыл китеп берилеби? Кандай документтер керек?"),
            ("1-year property sale tax", "Do I pay income tax when selling an apartment in Kyrgyzstan within 1 year vs after 1 year?"),
            ("DDU developer checks", "What must a buyer verify before signing a DDU shared construction contract in Bishkek?"),
        ],
        "chat_placeholder": "Ask in English, Russian, or Kyrgyz...",
        "chat_spinner": "Searching verified KR documents...",
        "sources_cap": "Sources",
        "excerpts_exp": "View retrieved excerpts",
        "clear_chat": "Clear conversation",
        "sys_runtime": "RAG Runtime",
        "sys_llm": "Chat model (LLM)",
        "sys_emb": "Embedding model",
        "sys_bench": "Benchmark Metrics (fixed seed = 42)",
        "sys_ml_head": "Bishkek Apartment Price Model — {clean:,} verified listings from {raw:,}",
        "sys_nlp_head": "Listing Text Classifier ({n:,} descriptions) & Document Corpus",
        "footer": "UyCheck KG — Generative AI course project (Ala-Too International University). Informational tool only; does not replace a licensed notary or official GU Cadastre extract.",
    },
}

PRESETS = {
    "dzhal_pso": {
        "district": "джал м-н",
        "area": 65.0,
        "rooms": 2,
        "floor": 6,
        "total_floors": 12,
        "year_built": 2024,
        "house_type": "монолит",
        "condition": "__auto__",
        "heating": "газовое",
        "documents": "договор долевого участия",
        "asking_price": 115000.0,
        "ad_text": "Срочно продаётся 2-комн квартира в Джале, ПСО, документы ДДУ, сдача дома в конце года, газ подведён.",
        "question_ru": "Адекватная ли цена $115,000, можно ли купить через ГИК «Менин үйүм» и какие риски у ДДУ?",
        "question_en": "Is $115,000 a fair price, can this be financed via GIK «Menin Uyum», and what are the DDU risks?",
    },
    "center_euro": {
        "district": "московская - уметалиева",
        "area": 92.0,
        "rooms": 3,
        "floor": 7,
        "total_floors": 14,
        "year_built": 2021,
        "house_type": "кирпич",
        "condition": "__auto__",
        "heating": "центральное",
        "documents": "договор купли-продажи, технический паспорт",
        "asking_price": 172000.0,
        "ad_text": "Продаётся просторная 3-комнатная квартира с дизайнерским евроремонтом, вся мебель и техника остаются, техпаспорт на руках, чистые документы.",
        "question_ru": "Насколько справедлива цена и какой ежегодный налог на имущество придётся платить за 92 м² в Бишкеке?",
        "question_en": "How fair is $172,000 and what annual property tax applies to a 92 m² apartment in Bishkek?",
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
        "question_ru": "Насколько обоснована цена пентхауса и что обязательно проверить в ГУ «Кадастр» перед сделкой?",
        "question_en": "How reasonable is this penthouse price and what should be verified in GU Cadastre before buying?",
    },
    "tricky_sarcasm": {
        "district": "12 м-н",
        "area": 54.0,
        "rooms": 2,
        "floor": 4,
        "total_floors": 9,
        "year_built": 2018,
        "house_type": "панель",
        "condition": "__auto__",
        "heating": "центральное",
        "documents": "технический паспорт",
        "asking_price": 82000.0,
        "ad_text": "Элитный дворец в центре за 5 сомов, стены из воздуха, 1000% евроремонт на Луне, ахаха!",
        "question_ru": "Проверь это объявление и объясни, выдаётся ли Красная книга на квартиру?",
        "question_en": "Check this listing text and explain whether a Red Book is issued for an apartment.",
    },
}


def _get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


@st.cache_resource(show_spinner=False)
def load_components():
    ml = BishkekPricePredictor(os.path.join(ROOT_DIR, "models", "ml_valuation_bundle.joblib"))
    nlp = ListingNLPAnalyzer(os.path.join(ROOT_DIR, "models", "nlp_listing_bundle.joblib"))
    local_url = _get_secret("LMSTUDIO_BASE_URL", DEFAULT_LMSTUDIO_URL)
    if _lmstudio_alive(local_url):
        rag = KyrgyzRealEstateRAG(
            docs_dir=os.path.join(ROOT_DIR, "data", "rag_docs"),
            embedder=nlp.embedder,
            api_base=local_url,
            api_key="lm-studio",
            llm_model="auto",
            embed_model="auto",
        )
    else:
        rag = KyrgyzRealEstateRAG(
            docs_dir=os.path.join(ROOT_DIR, "data", "rag_docs"),
            embedder=nlp.embedder,
            api_base=_get_secret("OPENAI_API_BASE", local_url),
            api_key=_get_secret("OPENAI_API_KEY", "lm-studio"),
            llm_model=_get_secret("LLM_MODEL", "auto"),
            embed_model=_get_secret("EMBED_MODEL", "auto"),
        )
    return ml, nlp, rag


def render_shap_chart(drivers, lang: str):
    key_lbl = "label_ru" if lang == "ru" else "label"
    key_val = "value_ru" if lang == "ru" else "value_en"
    labels = [f"{d.get(key_lbl, d['label'])}\n{d.get(key_val, d['value'])}" for d in drivers][::-1]
    vals = [d["usd_impact"] for d in drivers][::-1]
    colors = ["#1B4332" if v >= 0 else "#B45309" for v in vals]

    fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=180, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=colors, height=0.52)
    ax.axvline(0, color="#D6D1C7", linewidth=1.0, zorder=1)

    max_abs = max([abs(v) for v in vals] + [2500])
    ax.set_xlim(-max_abs * 1.55, max_abs * 1.50)

    for bar, v in zip(bars, vals):
        offset = max_abs * 0.05 if v >= 0 else -max_abs * 0.05
        ha = "left" if v >= 0 else "right"
        ax.text(
            v + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{v:+,.0f}$",
            va="center",
            ha=ha,
            fontsize=8.2,
            fontweight="600",
            color="#181A1D",
            family="monospace",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.0, color="#373A40")
    ax.xaxis.set_visible(False)
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    plt.subplots_adjust(left=0.36, right=0.96, top=0.96, bottom=0.04)
    return fig


def init_state():
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "ru"
    if "preset_key" not in st.session_state:
        st.session_state.preset_key = "dzhal_pso"
    if "form" not in st.session_state:
        st.session_state.form = dict(PRESETS["dzhal_pso"])
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def apply_preset(key: str):
    st.session_state.preset_key = key
    st.session_state.form = dict(PRESETS[key])


def is_why_price_query(q: str) -> bool:
    low = (q or "").lower()
    if re.search(
        r"(гик|ипотек|налог|салык|кадастр|нотариус|закон|кфв|kfw|дду|красн\w*\s+книг|кызыл\s+китеп|"
        r"алмат|казахстан|ош\b|штраф|перепланиров|выписк|түндүк|mortgage|tax|cadastre|notary|law)",
        low,
    ):
        return False
    return bool(re.search(
        r"((why|explain|how\s+did|what\s+drove|reason|breakdown).*(result|price|valuat|predict|estimat|number|output|cost|fair))"
        r"|((result|price|valuat|predict|estimat).*(why|explain|reason|breakdown))"
        r"|((почему|объясни|откуда|как\s+получил|как\s+рассчита|из\s+чего|что\s+повлиял|за\s+сч[её]т\s+чего).*(такая|такой|эта|этот|оценк|цен|стоимост|результат|прогноз|расч[её]т|модел|сумм))"
        r"|((оценк|цен|стоимост|результат|прогноз).*(почему|объясни|откуда|из\s+чего))"
        r"|((эмне\s+үчүн|кантип\s+чык|эмнеден\s+чык|түшүндүр).*(баа|жыйынтык|эсеп|модел|мындай))",
        low,
    ))


def format_why_price_reply(lang: str) -> dict:
    lr = st.session_state.get("last_ml_result")
    if not lr:
        hints = {
            "ru": "Сначала выполните расчёт квартиры на вкладке «Проверка квартиры», и я подробно разберу, какие факторы повлияли на цену.",
            "ky": "Адегенде биринчи өтмөктө батирди баалап алыңыз, андан кийин баага таасир эткен факторлорду так көрсөтүп берем.",
            "en": "Run an apartment check on the first tab first, and I will break down the exact factors behind that estimate.",
        }
        return {"status": "answered", "answer": hints.get(lang, hints["en"]), "sources": [], "retrieved": []}

    li = st.session_state.get("last_ml_inputs", {})
    ln = st.session_state.get("last_nlp_result")
    key_lbl = "label_ru" if lang in ("ru", "ky") else "label"
    key_val = "value_ru" if lang in ("ru", "ky") else "value_en"
    drivers_md = "\n".join(
        f"- **{d.get(key_lbl, d['label'])}** ({d.get(key_val, d['value'])}): **{d['usd_impact']:+,.0f}$**"
        for d in lr.get("top_drivers", [])[:6]
    )
    dist = format_district(li.get("district", ""))

    if lang == "ru":
        verdict = f" Цена продавца — **${lr['asking_price_usd']:,.0f}** ({lr['price_verdict_ru'].lower()})." if lr.get("asking_price_usd") and lr.get("price_verdict_ru") else ""
        nlp_line = f"\n\nИз текста объявления учтено состояние: **{ln['display_condition']}** (уверенность {ln['confidence']:.0%})." if ln else ""
        text = (
            f"Для квартиры (**{dist}**, {li.get('area', '')} м², {li.get('rooms', '')} комн.) "
            f"рыночная оценка составила **${lr['fair_price_usd']:,.0f}** (**${lr['fair_price_per_m2']:,.0f}/м²**).{verdict}\n\n"
            f"Отправная точка — средний уровень по Бишкеку (**${lr['base_market_usd']:,.0f}**). "
            f"Главный вклад в итоговую сумму внесли:\n{drivers_md}{nlp_line}"
        )
        src = "Модель оценки недвижимости Бишкека (SHAP)"
    elif lang == "ky":
        text = (
            f"Акыркы текшерилген батир (**{dist}**, {li.get('area', '')} м², {li.get('rooms', '')} бөлмө) "
            f"үчүн рыноктук баа **${lr['fair_price_usd']:,.0f}** (**${lr['fair_price_per_m2']:,.0f}/м²**) болду.\n\n"
            f"Бишкектин орточо деңгээлинен (**${lr['base_market_usd']:,.0f}**) төмөнкү факторлор таасир этти:\n{drivers_md}"
        )
        src = "Бишкек кыймылсыз мүлк баалоо модели (SHAP)"
    else:
        verdict = f" Asking price is **${lr['asking_price_usd']:,.0f}** ({lr['price_verdict'].lower()})." if lr.get("asking_price_usd") and lr.get("price_verdict") else ""
        text = (
            f"The estimate for **{dist}** ({li.get('area', '')} m², {li.get('rooms', '')} rooms) is "
            f"**${lr['fair_price_usd']:,.0f}** (**${lr['fair_price_per_m2']:,.0f}/m²**).{verdict}\n\n"
            f"Starting from the Bishkek baseline (**${lr['base_market_usd']:,.0f}**), key drivers were:\n{drivers_md}"
        )
        src = "Bishkek Price Model (SHAP)"

    return {"status": "answered", "answer": text, "sources": [src], "retrieved": []}


def main():
    init_state()
    ml, nlp, rag = load_components()

    top_left, top_right = st.columns([3.4, 1.3], vertical_alignment="bottom")
    with top_right:
        c_lang, c_eng = st.columns([1.05, 1.15], vertical_alignment="center")
        with c_lang:
            picked = st.segmented_control(
                "lang", ["RU", "EN"], default=st.session_state.ui_lang.upper(), label_visibility="collapsed"
            )
            if picked and picked.lower() != st.session_state.ui_lang:
                st.session_state.ui_lang = picked.lower()
                st.rerun()
        with c_eng:
            if rag.backend_mode == "lmstudio":
                dot_cls, eng_txt = "dot-local", "LM Studio"
            elif rag.backend_mode == "remote_api":
                dot_cls, eng_txt = "dot-remote", "Cloud API"
            else:
                dot_cls, eng_txt = "dot-off", "Offline"
            st.markdown(
                f"<div style='text-align:right;'><span class='engine-pill'>"
                f"<span class='engine-dot {dot_cls}'></span>{eng_txt}</span></div>",
                unsafe_allow_html=True,
            )

    lang = st.session_state.ui_lang
    t = COPY[lang]

    with top_left:
        st.markdown(
            f"<div class='brand-row'><span class='brand-name'>UyCheck</span><span class='brand-tag'>KG</span></div>"
            f"<div class='brand-sub'>{t['subtitle']}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='border-bottom:1px solid #E5E0D6; margin:0.65rem 0 0.8rem 0;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(t["tabs"])

    with tab1:
        st.markdown(f"<div class='sec-eyebrow'>{t['examples_hint']}</div>", unsafe_allow_html=True)
        pcols = st.columns(4)
        for col, (pkey, plabel) in zip(pcols, t["presets"]):
            if col.button(plabel, key=f"preset_{pkey}", use_container_width=True):
                apply_preset(pkey)
                st.rerun()

        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

        f = st.session_state.form
        districts = ml.meta["ui_districts"]
        if f["district"] not in districts:
            districts = [f["district"]] + districts

        cond_map = CONDITION_LABELS_RU if lang == "ru" else CONDITION_LABELS_EN
        house_map = HOUSE_TYPE_LABELS_RU if lang == "ru" else HOUSE_TYPE_LABELS
        heat_map = HEATING_LABELS_RU if lang == "ru" else HEATING_LABELS_EN

        cond_options = ["__auto__"] + [k for k in cond_map.keys() if k != "unknown"]
        house_options = [k for k in house_map.keys() if k != "unknown"]
        heat_options = ml.meta["ui_heating"]
        if f["heating"] not in heat_options:
            heat_options = [f["heating"]] + heat_options

        default_q = f.get(f"question_{lang}") or f.get("question_ru", "")

        with st.form("check_form"):
            left_col, right_col = st.columns([1.08, 1.0], gap="large")

            with left_col:
                st.markdown(f"<div class='sec-eyebrow'>{t['sec_params']}</div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                district = c1.selectbox(
                    t["district"], districts, index=districts.index(f["district"]),
                    format_func=format_district,
                )
                house_type = c2.selectbox(
                    t["house_type"], house_options,
                    index=house_options.index(f["house_type"]) if f["house_type"] in house_options else 0,
                    format_func=lambda x: house_map.get(x, x),
                )

                c3, c4, c5 = st.columns(3)
                area = c3.number_input(t["area"], value=float(f["area"]), step=1.0)
                rooms = c4.number_input(t["rooms"], value=int(f["rooms"]), step=1)
                year_built = c5.number_input(t["year_built"], value=int(f["year_built"]), step=1)

                c6, c7, c8 = st.columns(3)
                floor = c6.number_input(t["floor"], value=int(f["floor"]), step=1)
                total_floors = c7.number_input(t["total_floors"], value=int(f["total_floors"]), step=1)
                asking_price = c8.number_input(t["asking_price"], value=float(f["asking_price"]), step=1000.0)

                c9, c10 = st.columns(2)
                condition_choice = c9.selectbox(
                    t["condition"], cond_options,
                    index=cond_options.index(f["condition"]) if f["condition"] in cond_options else 0,
                    format_func=lambda x: t["auto_cond"] if x == "__auto__" else cond_map.get(x, x),
                )
                heating = c10.selectbox(
                    t["heating"], heat_options,
                    index=heat_options.index(f["heating"]),
                    format_func=lambda x: heat_map.get(x, x.capitalize()),
                )
                documents = st.text_input(t["documents"], value=f["documents"])

            with right_col:
                st.markdown(f"<div class='sec-eyebrow'>{t['sec_text']}</div>", unsafe_allow_html=True)
                ad_text = st.text_area(t["ad_label"], value=f["ad_text"], height=138, placeholder=t["ad_placeholder"])
                question = st.text_area(t["q_label"], value=default_q, height=118, placeholder=t["q_placeholder"])
                submitted = st.form_submit_button(t["submit"], use_container_width=True)

        if submitted:
            nlp_res = nlp.analyze(ad_text, lang=lang) if ad_text.strip() else None

            effective_condition = condition_choice
            used_nlp_condition = False
            if condition_choice == "__auto__":
                if nlp_res and not nlp_res["low_confidence"] and nlp_res["predicted_condition"] != "unknown":
                    effective_condition = nlp_res["predicted_condition"]
                    used_nlp_condition = True
                else:
                    effective_condition = "unknown"

            combined_docs = f"{documents} {' '.join(nlp_res['signals'])}".strip() if (nlp_res and nlp_res.get("signals")) else documents

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

            ml_res = ml.predict(payload, lang=lang)

            if not ml_res["ok"]:
                st.error(t["err_title"] + "\n" + "\n".join(f"- {e}" for e in ml_res["errors"]))
                if nlp_res:
                    st.info(f"{nlp_res['display_condition']} ({nlp_res['confidence']:.0%}) — {nlp_res['reason']}")
            else:
                for w in ml_res["warnings"]:
                    st.warning(w)

                st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)
                k1, k2, k3, k4 = st.columns(4)

                k1.markdown(
                    f"<div class='kpi-card'><div class='kpi-label'>{t['kpi_fair']}</div>"
                    f"<div class='kpi-value'>${ml_res['fair_price_usd']:,.0f}</div>"
                    f"<div class='kpi-sub'>${ml_res['fair_price_per_m2']:,.0f} / м²</div></div>",
                    unsafe_allow_html=True,
                )

                if ml_res["overprice_pct"] is not None:
                    diff_val = f"${ml_res['asking_price_usd']:,.0f}"
                    verdict_txt = ml_res["price_verdict_ru"] if lang == "ru" else ml_res["price_verdict"]
                    tone_cls = f"tone-{ml_res['price_tone']}"
                else:
                    diff_val = "—"
                    verdict_txt = t["kpi_no_asking"]
                    tone_cls = "tone-neutral"
                k2.markdown(
                    f"<div class='kpi-card'><div class='kpi-label'>{t['kpi_asking']}</div>"
                    f"<div class='kpi-value'>{diff_val}</div>"
                    f"<span class='{tone_cls}'>{verdict_txt}</span></div>",
                    unsafe_allow_html=True,
                )

                if nlp_res:
                    nlp_badge = t["low_conf"] if nlp_res["low_confidence"] else t["conf_fmt"].format(nlp_res["confidence"])
                    nlp_tone = "tone-warn" if nlp_res["low_confidence"] else "tone-good"
                    nlp_main = nlp_res["display_condition"]
                else:
                    nlp_badge = t["no_text"]
                    nlp_tone = "tone-neutral"
                    nlp_main = "—"
                k3.markdown(
                    f"<div class='kpi-card'><div class='kpi-label'>{t['kpi_text']}</div>"
                    f"<div class='kpi-value-text'>{nlp_main}</div>"
                    f"<span class='{nlp_tone}'>{nlp_badge}</span></div>",
                    unsafe_allow_html=True,
                )

                k4.markdown(
                    f"<div class='kpi-card'><div class='kpi-label'>{t['kpi_cond']}</div>"
                    f"<div class='kpi-value-text'>{cond_map.get(effective_condition, effective_condition)}</div>"
                    f"<div class='kpi-sub'>{t['from_text'] if used_nlp_condition else t['from_manual']}</div></div>",
                    unsafe_allow_html=True,
                )

                if nlp_res and nlp_res["low_confidence"]:
                    st.warning(t["low_conf_banner"].format(nlp_res["confidence"], nlp_res["reason"]))

                st.session_state.last_ml_result = ml_res
                st.session_state.last_ml_inputs = payload
                st.session_state.last_nlp_result = nlp_res

                with st.spinner(t["spinner_eval"]):
                    rag_res = rag.answer_integrated(ml_res, nlp_res, question, payload, ad_text=ad_text)

                st.markdown(f"<div style='height:0.9rem;'></div><div class='sec-eyebrow'>{t['sec_brief']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='panel-card'>{rag_res['answer']}</div>", unsafe_allow_html=True)

                st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
                dcol1, dcol2 = st.columns([1.22, 0.95], gap="large")

                with dcol1:
                    st.markdown(f"<div class='sec-eyebrow'>{t['sec_shap']}</div>", unsafe_allow_html=True)
                    with st.container(border=True):
                        fig = render_shap_chart(ml_res["top_drivers"], lang=lang)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                with dcol2:
                    st.markdown(f"<div class='sec-eyebrow'>{t['sec_nlp']}</div>", unsafe_allow_html=True)
                    with st.container(border=True):
                        if nlp_res:
                            if nlp_res["signals"]:
                                chips = "".join(f"<span class='tag-chip'>{s}</span>" for s in nlp_res["signals"])
                                st.markdown(chips, unsafe_allow_html=True)
                            else:
                                st.caption(t["tags_none"])
                            st.markdown(
                                f"<div class='kpi-sub' style='margin-top:0.75rem;'>{t['sim_label']}: "
                                f"<span style='font-family:JetBrains Mono,monospace;color:#181A1D;font-weight:600;'>"
                                f"{nlp_res['domain_similarity']:.2f}</span></div>",
                                unsafe_allow_html=True,
                            )
                            if nlp_res.get("top_probs"):
                                rows_html = [f"<div class='prob-wrap'><div class='kpi-sub' style='margin-bottom:0.45rem;'>{t['probs_label']}</div>"]
                                for cls_name, prob in nlp_res["top_probs"].items():
                                    pct = min(max(prob * 100.0, 0.0), 100.0)
                                    rows_html.append(
                                        f"<div class='prob-row'>"
                                        f"<div class='prob-head'><span class='prob-name'>{cls_name}</span>"
                                        f"<span class='prob-pct'>{pct:.1f}%</span></div>"
                                        f"<div class='prob-track'><div class='prob-fill' style='width:{pct:.1f}%;'></div></div>"
                                        f"</div>"
                                    )
                                rows_html.append("</div>")
                                st.markdown("".join(rows_html), unsafe_allow_html=True)
                        else:
                            st.caption(t["paste_hint"])

                if rag_res.get("retrieved"):
                    with st.expander(t["sources_exp"]):
                        for ch in rag_res["retrieved"]:
                            st.markdown(f"**{ch['doc_title']}** · {ch['section']}")
                            meta_line = f"`{ch['doc_id']}`"
                            if ch.get("source_url"):
                                meta_line += f" · [{t['official_src']}]({ch['source_url']})"
                            st.caption(meta_line)
                            st.markdown(ch["text"])
                            st.divider()

    with tab2:
        st.markdown(f"<div style='font-size:1.08rem;font-weight:700;margin-bottom:0.2rem;'>{t['chat_head']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='brand-sub' style='margin-bottom:0.9rem;'>{t['chat_sub']}</div>", unsafe_allow_html=True)

        qcols = st.columns(4)
        sample_q = None
        for idx, (col, (btn_lbl, full_q)) in enumerate(zip(qcols, t["chat_samples"])):
            if col.button(btn_lbl, key=f"qbtn_{idx}", use_container_width=True):
                sample_q = full_q

        for turn in st.session_state.chat_history:
            avatar = ":material/person:" if turn["role"] == "user" else ":material/home_work:"
            with st.chat_message(turn["role"], avatar=avatar):
                st.markdown(turn["content"])
                if turn.get("sources"):
                    st.caption(f"{t['sources_cap']}: " + " · ".join(turn["sources"][:3]))
                if turn.get("retrieved"):
                    with st.expander(t["excerpts_exp"]):
                        for ch in turn["retrieved"]:
                            st.markdown(f"**{ch['doc_title']}** ({ch['section']}) · `{ch['doc_id']}`")
                            st.markdown(ch["text"])
                            st.divider()

        user_msg = st.chat_input(t["chat_placeholder"])
        active_q = sample_q or user_msg

        if active_q:
            st.session_state.chat_history.append({"role": "user", "content": active_q})
            q_lang = detect_language(active_q)
            if is_why_price_query(active_q):
                ans = format_why_price_reply(q_lang)
            else:
                with st.spinner(t["chat_spinner"]):
                    ans = rag.answer_question(active_q)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ans["answer"],
                "sources": ans.get("sources", []),
                "retrieved": ans.get("retrieved", [])[:3] if ans["status"] == "answered" else [],
            })
            st.rerun()

        if st.session_state.chat_history:
            if st.button(t["clear_chat"], key="clear_chat_btn"):
                st.session_state.chat_history = []
                st.rerun()

    with tab3:
        st.markdown(f"<div class='sec-eyebrow'>{t['sys_runtime']}</div>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        llm_src = "LM Studio (localhost:1234)" if rag.backend_mode == "lmstudio" else ("OpenAI-compatible API" if rag.backend_mode == "remote_api" else "Offline fallback")
        emb_src = "LM Studio (/v1/embeddings)" if rag.embed_backend == "lmstudio" else ("API /v1/embeddings" if rag.embed_backend == "api_embeddings" else "Local SentenceTransformer")
        r1.markdown(
            f"<div class='runtime-card'><div class='kpi-label'>{t['sys_llm']}</div>"
            f"<div class='runtime-val'>{rag.llm_model or '—'}</div>"
            f"<div class='kpi-sub'>{llm_src} · {rag.api_base}</div></div>",
            unsafe_allow_html=True,
        )
        r2.markdown(
            f"<div class='runtime-card'><div class='kpi-label'>{t['sys_emb']}</div>"
            f"<div class='runtime-val'>{rag.embed_model or '—'}</div>"
            f"<div class='kpi-sub'>{emb_src} · {rag.num_docs} docs / {len(rag.chunks)} sections</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(f"<div style='height:1rem;'></div><div class='sec-eyebrow'>{t['sys_bench']}</div>", unsafe_allow_html=True)
        m_ml = ml.metrics
        m_nlp = nlp.metrics

        st.markdown(f"**{t['sys_ml_head'].format(clean=m_ml['clean_rows'], raw=m_ml['total_raw_rows'])}**")
        rows_tbl = [
            {"Model": "Baseline: Global Median", "MAE ($)": m_ml["baselines"]["global_median"]["mae_usd"], "RMSE ($)": m_ml["baselines"]["global_median"]["rmse_usd"], "MAPE (%)": m_ml["baselines"]["global_median"]["mape_pct"], "R²": m_ml["baselines"]["global_median"]["r2"]},
            {"Model": "Baseline: Ridge Regression", "MAE ($)": m_ml["baselines"]["ridge_regression"]["mae_usd"], "RMSE ($)": m_ml["baselines"]["ridge_regression"]["rmse_usd"], "MAPE (%)": m_ml["baselines"]["ridge_regression"]["mape_pct"], "R²": m_ml["baselines"]["ridge_regression"]["r2"]},
            {"Model": "LightGBM", "MAE ($)": m_ml["models"]["lightgbm"]["mae_usd"], "RMSE ($)": m_ml["models"]["lightgbm"]["rmse_usd"], "MAPE (%)": m_ml["models"]["lightgbm"]["mape_pct"], "R²": m_ml["models"]["lightgbm"]["r2"]},
            {"Model": "CatBoost", "MAE ($)": m_ml["models"]["catboost"]["mae_usd"], "RMSE ($)": m_ml["models"]["catboost"]["rmse_usd"], "MAPE (%)": m_ml["models"]["catboost"]["mape_pct"], "R²": m_ml["models"]["catboost"]["r2"]},
            {"Model": "Ensemble (50% LGB + 50% CB)", "MAE ($)": m_ml["models"]["ensemble_lgb_cb"]["mae_usd"], "RMSE ($)": m_ml["models"]["ensemble_lgb_cb"]["rmse_usd"], "MAPE (%)": m_ml["models"]["ensemble_lgb_cb"]["mape_pct"], "R²": m_ml["models"]["ensemble_lgb_cb"]["r2"]},
        ]
        st.dataframe(rows_tbl, use_container_width=True, hide_index=True)

        st.markdown(f"**{t['sys_nlp_head'].format(n=m_nlp['total_labeled_texts'])}**")
        n1, n2, n3, n4 = st.columns(4)
        n1.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>NLP Accuracy</div>"
            f"<div class='kpi-value'>{m_nlp['test_accuracy']:.1%}</div>"
            f"<div class='kpi-sub'>Baseline: {m_nlp['baseline_majority_accuracy']:.1%}</div></div>",
            unsafe_allow_html=True,
        )
        n2.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Weighted / Macro F1</div>"
            f"<div class='kpi-value'>{m_nlp['test_weighted_f1']:.3f}</div>"
            f"<div class='kpi-sub'>Macro F1: {m_nlp['test_macro_f1']:.3f}</div></div>",
            unsafe_allow_html=True,
        )
        n3.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>RAG Corpus</div>"
            f"<div class='kpi-value'>{rag.num_docs}</div>"
            f"<div class='kpi-sub'>{len(rag.chunks)} sections · RU / KG / EN</div></div>",
            unsafe_allow_html=True,
        )
        n4.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Dataset License</div>"
            f"<div class='kpi-value-text'>CC BY 4.0</div>"
            f"<div class='kpi-sub'>raimbekovm/bishkek-real-estate</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"<div class='foot-note'>{t['footer']}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
