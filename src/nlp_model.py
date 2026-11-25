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
    "евроремонт": "Euro-renovation (Евроремонт)",
    "под самоотделку (псо)": "Shell / Unfinished (ПСО)",
    "хорошее": "Good condition (Хорошее)",
    "среднее": "Average condition (Среднее)",
    "не достроено": "Under construction (Не достроено)",
}

SIGNAL_PATTERNS = {
    "DDU / Shared Construction (ДДУ)": r"\bдду\b|долев\w*\s+участ|үлуштүк|shared construction",
    "Shell / Unfinished (ПСО)": r"\bпсо\b|под\s+самоотделк|чернов\w*\s+отделк|shell",
    "Red Book on Land (Красная книга)": r"красн\w*\s+книг|кызыл\s+китеп|red\s+book",
    "Technical Passport (Техпаспорт)": r"техпаспорт|тех\.?\s*паспорт|техническ\w*\s+паспорт",
    "Purchase Contract (ДКП)": r"\bдкп\b|договор\s+купли|купля[- ]продаж",
    "Urgent Sale (Срочно)": r"срочн\w*|шашылыш|urgent",
    "Mortgage / GIK Mentioned (Ипотека/ГИК)": r"ипотек\w*|\bгик\b|mortgage|менин\s+үйүм",
    "Negotiable (Торг)": r"\bторг\b|уступк|negotiable|соодалашуу",
    "Installment (Рассрочка)": r"рассрочк|бөлүп\s+төлөө|installment",
}

SARCASM_OR_ABSURD_PATTERNS = [
    r"стен\w*\s+из\s+воздух",
    r"walls\s+(made\s+)?of\s+air",
    r"за\s+[0-9]\s*(сом|доллар|\$)",
    r"for\s+\$?[0-9]\b",
    r"на\s+луне|на\s+марсе|on\s+mars|on\s+the\s+moon",
    r"замок\s+дракон|дворец\s+.*\s+5\s+сом|palace\s+for\s+5",
    r"\b1000%\b|бессмертн|immortal|сарказм|sarcasm|шутка|just\s+kidding|ахах|хаха|lol\b|лол\b",
    r"totally\s+real|obviously\s+not|yeah\s+right|конечно\s+же\s+нет",
    r"продам\s+гараж|пицц|погод|анекдот|котик|bitcoin|крипт",
]

REAL_ESTATE_ANCHOR_PATTERN = (
    r"квартир|батир|комн|бөлмө|этаж|кабат|м2|м²|кв\.?\s*м|ремонт|псо|дду|жк|район|"
    r"дом|үй|серия|монолит|кирпич|панел|отоплен|санузел|лоджи|балкон|застройщик|"
    r"сдач|техпаспорт|красн|apartment|flat|room|floor|renovat|bishkek|бишкек|джал|магистрал"
)


def extract_deal_signals(text: str) -> List[str]:
    low = text.lower()
    found = []
    for label, pat in SIGNAL_PATTERNS.items():
        if re.search(pat, low, flags=re.IGNORECASE):
            found.append(label)
    return found


def train_and_save_nlp(csv_path: str, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    raw = pd.read_csv(csv_path)
    df = raw[["description", "condition"]].dropna().copy()
    df["description"] = df["description"].astype(str).str.strip()
    df["condition"] = df["condition"].astype(str).str.strip().str.lower()
    valid_classes = list(CONDITION_DISPLAY.keys())
    df = df[df["condition"].isin(valid_classes) & (df["description"].str.len() > 20)].reset_index(drop=True)

    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df["condition"]
    )

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    X_train_emb = embedder.encode(
        train_df["description"].tolist(),
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    X_test_emb = embedder.encode(
        test_df["description"].tolist(),
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    # Combine multilingual MiniLM dense vectors with Russian/Kyrgyz TF-IDF n-grams
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Zа-яА-ЯёЁөӨүҮңҢ0-9]{2,}\b",
    )
    X_train_tfidf = tfidf.fit_transform(train_df["description"].tolist())
    X_test_tfidf = tfidf.transform(test_df["description"].tolist())

    # Weight embeddings & TF-IDF cleanly so both RU/KG exact cues and cross-lingual semantics shine
    X_train_comb = hstack([csr_matrix(X_train_emb * 1.4), X_train_tfidf * 1.8]).tocsr()
    X_test_comb = hstack([csr_matrix(X_test_emb * 1.4), X_test_tfidf * 1.8]).tocsr()

    clf = LogisticRegression(
        C=2.0,
        max_iter=1000,
        random_state=SEED,
    )
    clf.fit(X_train_comb, train_df["condition"].values)

    preds = clf.predict(X_test_comb)
    acc = float(accuracy_score(test_df["condition"].values, preds))
    macro_f1 = float(f1_score(test_df["condition"].values, preds, average="macro"))
    weighted_f1 = float(f1_score(test_df["condition"].values, preds, average="weighted"))

    # Also train a majority-class baseline for honest comparison
    maj_class = train_df["condition"].value_counts().index[0]
    maj_acc = float(accuracy_score(test_df["condition"].values, [maj_class] * len(test_df)))

    centroid = np.mean(X_train_emb, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    train_sims = X_train_emb @ centroid
    ood_threshold = float(np.percentile(train_sims, 2.0))

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

    artifact = {
        "clf": clf,
        "tfidf": tfidf,
        "embed_model_name": EMBED_MODEL_NAME,
        "centroid": centroid,
        "ood_threshold": ood_threshold,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "metrics": metrics,
    }
    joblib.dump(artifact, os.path.join(out_dir, "nlp_listing_bundle.joblib"))
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

    def analyze(self, text: str) -> Dict[str, Any]:
        clean = (text or "").strip()
        signals = extract_deal_signals(clean)

        words = re.findall(r"[A-Za-zА-Яа-яЁёӨөҮүҢң0-9]+", clean)
        if len(clean) < 15 or len(words) < 3:
            return {
                "predicted_condition": "unknown",
                "display_condition": "Unclear / Too short",
                "confidence": 0.15,
                "low_confidence": True,
                "reason": "Input text is too short to classify renovation condition reliably. Please paste a fuller apartment description.",
                "signals": signals,
                "domain_similarity": 0.0,
                "top_probs": {},
            }

        low = clean.lower()

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
            reasons.append("Detected sarcastic, exaggerated, or non-serious wording.")
        if contradiction_hit:
            low_conf = True
            effective_conf = min(effective_conf, 0.34)
            reasons.append("Listing contains contradictory condition claims (e.g., both finished euro-renovation and unfinished/ПСО).")
        if domain_sim < max(self.ood_threshold, 0.34) and not has_anchor:
            low_conf = True
            effective_conf = min(effective_conf, 0.25)
            reasons.append("Text does not resemble a Bishkek apartment listing (out-of-domain input).")
        if top_prob < self.conf_threshold or entropy > 0.78:
            low_conf = True
            reasons.append("Model confidence is below the reliability threshold (0.55) — description is ambiguous.")

        top_probs = {
            CONDITION_DISPLAY.get(c, c): round(float(p), 3)
            for c, p in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]
        }

        return {
            "predicted_condition": "unknown" if low_conf else top_class,
            "raw_top_condition": top_class,
            "display_condition": (
                f"Low Confidence — possible {CONDITION_DISPLAY.get(top_class, top_class)}"
                if low_conf else CONDITION_DISPLAY.get(top_class, top_class)
            ),
            "confidence": round(effective_conf, 2),
            "low_confidence": low_conf,
            "reason": " ".join(reasons) if reasons else "Clear real-estate description matched Bishkek listing patterns.",
            "signals": signals,
            "domain_similarity": round(domain_sim, 3),
            "top_probs": top_probs,
        }


if __name__ == "__main__":
    base = r"C:\Users\user\.gemini\antigravity\scratch\uycheck-kg"
    metrics = train_and_save_nlp(
        os.path.join(base, "data", "bishkek_apartments.csv"),
        os.path.join(base, "models"),
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2, ensure_ascii=False))
