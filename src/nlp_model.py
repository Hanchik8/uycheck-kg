import json
import os
import re
import warnings
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEED = 42
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CONFIDENCE_THRESHOLD = 0.55

CONDITION_DISPLAY = {
    "евроремонт": "Euro-renovation",
    "под самоотделку (псо)": "Shell / Unfinished (PSO)",
    "хорошее": "Good condition",
    "среднее": "Average condition",
    "не достроено": "Under construction",
}

CONDITION_DISPLAY_RU = {
    "евроремонт": "Евроремонт",
    "под самоотделку (псо)": "Под самоотделку (ПСО)",
    "хорошее": "Хорошее состояние",
    "среднее": "Среднее состояние",
    "не достроено": "Не достроено",
}

SIGNAL_PATTERNS = {
    "ddu": (r"\bдду\b|долев\w*\s+участ|үлуштүк|shared construction", "ДДУ / долевое", "DDU contract"),
    "pso": (r"\bпсо\b|под\s+самоотделк|чернов\w*\s+отделк|shell", "ПСО / самоотделка", "Shell / PSO"),
    "redbook": (r"красн\w*\s+книг|кызыл\s+китеп|red\s+book", "Красная книга", "Red Book"),
    "techpass": (r"техпаспорт|тех\.?\s*паспорт|техническ\w*\s+паспорт", "Техпаспорт", "Tech Passport"),
    "dkp": (r"\bдкп\b|договор\s+купли|купля[- ]продаж", "Договор купли-продажи", "Purchase contract"),
    "urgent": (r"срочн\w*|шашылыш|urgent", "Срочная продажа", "Urgent sale"),
    "mortgage": (r"ипотек\w*|\bгик\b|mortgage|менин\s+үйүм", "Ипотека / ГИК", "Mortgage / GIK"),
    "negotiable": (r"\bторг\b|уступк|negotiable|соодалашуу", "Торг уместен", "Negotiable"),
    "installment": (r"рассрочк|бөлүп\s+төлөө|installment", "Рассрочка", "Installment"),
}

SARCASM_OR_ABSURD_PATTERNS = [
    r"стен\w*\s+из\s+воздух|без\s+стен|голый\s+бетон.*люкс|люкс.*без\s+стен",
    r"walls\s+(made\s+)?of\s+air|flying\s+balcon|golden\s+toilet|trust\s+me\s+bro",
    r"за\s+[0-9]\s*(сом|доллар|\$)|for\s+\$?[0-9]\b",
    r"на\s+луне|на\s+марсе|on\s+mars|on\s+the\s+moon",
    r"замок\s+дракон|дворец\s+.*\s+5\s+сом|palace\s+for\s+5",
    r"крыша\s+теч[её]т|водопад|на\s+честном\s+слове|зато\s+душ\s+бесплатн|тарак\w*\s+в\s+подарок",
    r"\b1000%\b|бессмертн|immortal|сарказм|sarcasm|шутка|just\s+kidding|ахах|хаха|lol\b|лол\b",
    r"totally\s+(real|a\s+five[- ]star)|obviously\s+not|yeah\s+right|конечно\s+же\s+нет|oh\s+yeah",
    r"продам\s+гараж|пицц|погод|анекдот|котик|bitcoin|крипт",
]

REAL_ESTATE_ANCHOR_PATTERN = (
    r"квартир|батир|комн|бөлмө|этаж|кабат|м2|м²|кв\.?\s*м|ремонт|псо|дду|жк|район|"
    r"дом|үй|серия|монолит|кирпич|панел|отоплен|санузел|лоджи|балкон|застройщик|"
    r"сдач|техпаспорт|красн|apartment|flat|room|floor|renovat|bishkek|бишкек|джал|магистрал"
)


def extract_deal_signals(text: str, lang: str = "ru") -> List[str]:
    low = text.lower()
    idx = 1 if lang == "ru" else 2
    return [spec[idx] for spec in SIGNAL_PATTERNS.values() if re.search(spec[0], low, flags=re.IGNORECASE)]


def train_and_save_nlp(csv_path: str, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    raw = pd.read_csv(csv_path)
    df = raw[["description", "condition"]].dropna().copy()
    df["description"] = df["description"].astype(str).str.strip()
    df["condition"] = df["condition"].astype(str).str.strip().str.lower()
    valid_classes = list(CONDITION_DISPLAY.keys())
    df = df[df["condition"].isin(valid_classes) & (df["description"].str.len() > 20)].reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["condition"])

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    X_train_emb = embedder.encode(train_df["description"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    X_test_emb = embedder.encode(test_df["description"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), max_features=8000, min_df=2, sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Zа-яА-ЯёЁөӨүҮңҢ0-9]{2,}\b",
    )
    X_train_tfidf = tfidf.fit_transform(train_df["description"].tolist())
    X_test_tfidf = tfidf.transform(test_df["description"].tolist())

    X_train_comb = hstack([csr_matrix(X_train_emb * 1.4), X_train_tfidf * 1.8]).tocsr()
    X_test_comb = hstack([csr_matrix(X_test_emb * 1.4), X_test_tfidf * 1.8]).tocsr()

    clf = LogisticRegression(C=2.0, max_iter=1000, random_state=SEED)
    clf.fit(X_train_comb, train_df["condition"].values)

    preds = clf.predict(X_test_comb)
    acc = float(accuracy_score(test_df["condition"].values, preds))
    macro_f1 = float(f1_score(test_df["condition"].values, preds, average="macro"))
    weighted_f1 = float(f1_score(test_df["condition"].values, preds, average="weighted"))
    maj_class = train_df["condition"].value_counts().index[0]
    maj_acc = float(accuracy_score(test_df["condition"].values, [maj_class] * len(test_df)))

    centroid = np.mean(X_train_emb, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    ood_threshold = float(np.percentile(X_train_emb @ centroid, 2.0))

    metrics = {
        "seed": SEED,
        "embed_model": EMBED_MODEL_NAME,
        "features": "Multilingual MiniLM (384-d) + TF-IDF (8,000 word 1-2 grams)",
        "total_labeled_texts": int(len(df)),
        "train_texts": int(len(train_df)),
        "test_texts": int(len(test_df)),
        "classes": valid_classes,
        "class_counts": {k: int(v) for k, v in df["condition"].value_counts().items()},
        "baseline_majority_accuracy": round(maj_acc, 4),
        "test_accuracy": round(acc, 4),
        "test_macro_f1": round(macro_f1, 4),
        "test_weighted_f1": round(weighted_f1, 4),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "ood_similarity_threshold": round(ood_threshold, 4),
        "classification_report": classification_report(test_df["condition"].values, preds, output_dict=True),
    }

    joblib.dump(
        {
            "clf": clf, "tfidf": tfidf, "embed_model_name": EMBED_MODEL_NAME,
            "centroid": centroid, "ood_threshold": ood_threshold,
            "confidence_threshold": CONFIDENCE_THRESHOLD, "metrics": metrics,
        },
        os.path.join(out_dir, "nlp_listing_bundle.joblib"),
    )
    with open(os.path.join(out_dir, "nlp_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


class ListingNLPAnalyzer:
    def __init__(self, bundle_path: str, embedder: SentenceTransformer = None):
        bundle = joblib.load(bundle_path)
        self.clf = bundle["clf"]
        self.tfidf = bundle["tfidf"]
        self.embed_model_name = bundle["embed_model_name"]
        self.centroid = bundle["centroid"]
        self.ood_threshold = float(bundle["ood_threshold"])
        self.conf_threshold = float(bundle["confidence_threshold"])
        self.metrics = bundle["metrics"]
        self.embedder = embedder if embedder is not None else SentenceTransformer(self.embed_model_name)

    def analyze(self, text: str, lang: str = "ru") -> Dict[str, Any]:
        clean = (text or "").strip()
        ru = lang == "ru"
        signals = extract_deal_signals(clean, lang=lang)
        disp_map = CONDITION_DISPLAY_RU if ru else CONDITION_DISPLAY

        words = re.findall(r"[A-Za-zА-Яа-яЁёӨөҮүҢң0-9]+", clean)
        if len(clean) < 15 or len(words) < 3:
            return {
                "predicted_condition": "unknown",
                "display_condition": "Слишком короткий текст" if ru else "Too short to classify",
                "confidence": 0.15,
                "low_confidence": True,
                "reason": "Добавьте более подробное описание квартиры, чтобы определить ремонт." if ru else "Paste a fuller apartment description to classify renovation.",
                "signals": signals,
                "domain_similarity": 0.0,
                "top_probs": {},
            }

        low = clean.lower()
        alpha_chars = re.findall(r"[A-Za-zА-Яа-яЁёӨөҮүҢң]", clean)
        vowels = re.findall(r"[AEIOUYaeiouyАЕЁИОУЫЭЮЯаеёиоуыэюяӨөҮү]", clean)
        vowel_ratio = (len(vowels) / len(alpha_chars)) if alpha_chars else 0.0
        is_gibberish = (len(alpha_chars) >= 8 and (vowel_ratio < 0.16 or vowel_ratio > 0.85)) or bool(
            re.search(r"asdf|qwerty|zxcv|фыва|йцук", low)
        )
        if is_gibberish:
            return {
                "predicted_condition": "unknown",
                "display_condition": "Неразборчивый ввод" if ru else "Unclear input",
                "confidence": 0.12,
                "low_confidence": True,
                "reason": "Текст похож на случайный набор символов, а не на реальное объявление." if ru else "Input appears to be random characters rather than a listing.",
                "signals": signals,
                "domain_similarity": 0.0,
                "top_probs": {},
            }

        sarcasm_hit = any(re.search(pat, low, flags=re.IGNORECASE) for pat in SARCASM_OR_ABSURD_PATTERNS)
        has_euro = bool(re.search(r"евроремонт|дизайнерск|люкс|euro[- ]?renovat", low))
        has_shell = bool(re.search(r"\bпсо\b|под\s+самоотделк|не\s+достроен|без\s+стен|руин|unfinished|no\s+walls", low))
        contradiction_hit = has_euro and has_shell

        emb = self.embedder.encode([clean], normalize_embeddings=True)[0]
        domain_sim = float(emb @ self.centroid)
        tf_vec = self.tfidf.transform([clean])
        comb = hstack([csr_matrix(emb.reshape(1, -1) * 1.4), tf_vec * 1.8]).tocsr()

        probs = self.clf.predict_proba(comb)[0]
        classes = list(self.clf.classes_)
        top_idx = int(np.argmax(probs))
        top_class = classes[top_idx]
        top_prob = float(probs[top_idx])

        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))) / np.log(len(probs)))
        has_anchor = bool(re.search(REAL_ESTATE_ANCHOR_PATTERN, low))

        reasons = []
        low_conf = False
        effective_conf = top_prob

        if sarcasm_hit:
            low_conf = True
            effective_conf = min(effective_conf, 0.28)
            reasons.append("В тексте заметен сарказм или нереалистичные утверждения." if ru else "Detected sarcastic or exaggerated wording.")
        if contradiction_hit:
            low_conf = True
            effective_conf = min(effective_conf, 0.34)
            reasons.append("Противоречивое описание: одновременно указаны евроремонт и черновая отделка." if ru else "Contradictory claims (both finished euro-renovation and unfinished shell).")
        if not has_anchor and domain_sim < max(self.ood_threshold, 0.62):
            low_conf = True
            effective_conf = min(effective_conf, 0.25)
            reasons.append("Текст не похож на объявление о продаже квартиры в Бишкеке." if ru else "Text does not resemble a Bishkek apartment listing.")
        elif domain_sim < 0.36:
            low_conf = True
            effective_conf = min(effective_conf, 0.30)
            reasons.append("Низкое сходство с реальными объявлениями Бишкека." if ru else "Low semantic similarity to Bishkek apartment listings.")
        if top_prob < self.conf_threshold or entropy > 0.78:
            low_conf = True
            reasons.append("Описание неоднозначно — модель не уверена в состоянии ремонта." if ru else "Description is ambiguous — below confidence threshold.")

        top_probs = {
            disp_map.get(c, c): round(float(p), 3)
            for c, p in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]
        }

        if low_conf:
            display = f"Неоднозначно (возможно: {disp_map.get(top_class, top_class)})" if ru else f"Uncertain (possibly {disp_map.get(top_class, top_class)})"
        else:
            display = disp_map.get(top_class, top_class)

        return {
            "predicted_condition": "unknown" if low_conf else top_class,
            "raw_top_condition": top_class,
            "display_condition": display,
            "confidence": round(effective_conf, 2),
            "low_confidence": low_conf,
            "reason": " ".join(reasons) if reasons else ("Текст соответствует реальным объявлениям Бишкека." if ru else "Matches typical Bishkek listing phrasing."),
            "signals": signals,
            "domain_similarity": round(domain_sim, 3),
            "top_probs": top_probs,
        }


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = train_and_save_nlp(os.path.join(base, "data", "bishkek_apartments.csv"), os.path.join(base, "models"))
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2, ensure_ascii=False))
