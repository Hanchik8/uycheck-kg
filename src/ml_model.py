import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
ALATOO_LAT, ALATOO_LON = 42.8765, 74.6037
MAGISTRAL_LAT, MAGISTRAL_LON = 42.8285, 74.6020

PII_AND_DROP_COLS = [
    "id", "listing_id", "url", "has_phone", "address", "parsed_at", "updated_at",
    "source", "residential_complex_id", "is_active", "photos_downloaded",
    "price_local", "price_per_m2", "description", "raw_площадь_участка",
    "raw_период_аренды", "raw_питьевая_вода", "offer_type",
]

CONDITION_LABELS_EN = {
    "евроремонт": "Euro-renovation (Евроремонт)",
    "хорошее": "Good condition (Хорошее)",
    "среднее": "Average condition (Среднее)",
    "под самоотделку (псо)": "Shell / Unfinished (ПСО)",
    "не достроено": "Under construction (Не достроено)",
    "unknown": "Unknown",
}

HOUSE_TYPE_LABELS = {
    "монолит": "Monolith (Монолит)",
    "кирпич": "Brick (Кирпич)",
    "панель": "Panel (Панель)",
    "unknown": "Other / Unknown",
}

FEATURE_READABLE = {
    "area": "Apartment area (m²)",
    "rooms": "Number of rooms",
    "floor": "Floor number",
    "total_floors": "Total building floors",
    "floor_ratio": "Relative floor height",
    "is_first_floor": "First floor indicator",
    "is_top_floor": "Top floor indicator",
    "area_per_room": "Area per room (m²)",
    "year_built": "Year built",
    "building_age": "Building age (years)",
    "latitude": "North–South location (lat)",
    "longitude": "East–West location (lon)",
    "dist_alatoo_km": "Distance to Center / Ala-Too (km)",
    "dist_magistral_km": "Distance to South Magistral (km)",
    "documents_has_ddu": "DDU contract flag",
    "documents_has_redbook": "Land Red Book flag",
    "documents_has_techpassport": "Technical Passport flag",
    "mortgage_flag": "Mortgage-eligible flag",
    "district": "Bishkek district",
    "house_type": "Building material",
    "building_series": "Building series",
    "condition": "Renovation condition",
    "heating": "Heating type",
    "jk_class": "Residential complex class",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def validate_apartment_input(payload: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Strict schema validation for Mission 5 (impossible values) & Mission 6 (valid extremes)."""
    errors: List[str] = []
    warnings_list: List[str] = []

    def _num(key: str) -> Optional[float]:
        v = payload.get(key)
        if v is None or v == "":
            return None
        try:
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                errors.append(f"'{key}' must be a valid finite number.")
                return None
            return fv
        except (ValueError, TypeError):
            errors.append(f"'{key}' must be numeric (received: {v}).")
            return None

    area = _num("area")
    rooms = _num("rooms")
    floor = _num("floor")
    total_floors = _num("total_floors")
    year_built = _num("year_built")
    asking_price = _num("asking_price_usd")

    if area is None:
        errors.append("Apartment area (m²) is required.")
    elif area <= 0:
        errors.append(f"Impossible apartment area ({area} m²). Area must be greater than 0 m².")
    elif area < 10:
        errors.append(f"Apartment area ({area} m²) is below minimum livable studio size (10 m²).")
    elif area > 600:
        errors.append(f"Apartment area ({area} m²) exceeds valid residential apartment range (max 600 m²).")
    elif area >= 220 or area <= 18:
        warnings_list.append(
            f"Extreme but valid apartment size ({area:.0f} m²): estimate reflects tail market segment."
        )

    if rooms is None:
        errors.append("Number of rooms is required.")
    elif rooms <= 0 or rooms != int(rooms):
        errors.append(f"Impossible room count ({rooms}). Must be a positive integer (1–12).")
    elif rooms > 12:
        errors.append(f"Room count ({int(rooms)}) exceeds realistic residential apartment range (max 12).")

    if area is not None and rooms is not None and area > 0 and rooms > 0:
        if area / rooms < 4.5:
            errors.append(
                f"Impossible layout: {area:.1f} m² cannot realistically hold {int(rooms)} rooms (< 4.5 m²/room)."
            )

    if floor is None or total_floors is None:
        errors.append("Both floor and total building floors are required.")
    else:
        if total_floors < 1 or total_floors > 35 or total_floors != int(total_floors):
            errors.append(f"Impossible total floors ({total_floors}). Valid range in Bishkek: 1–35.")
        if floor < 1 or floor > 35 or floor != int(floor):
            errors.append(f"Impossible floor number ({floor}). Valid range: 1–35.")
        if floor >= 1 and total_floors >= 1 and floor > total_floors:
            errors.append(
                f"Impossible floor: apartment floor ({int(floor)}) cannot exceed total building floors ({int(total_floors)})."
            )

    if year_built is not None:
        if year_built < 1945 or year_built > 2029:
            errors.append(f"Impossible construction year ({int(year_built)}). Valid range: 1945–2029.")

    if asking_price is not None:
        if asking_price <= 0:
            errors.append(f"Impossible asking price (${asking_price:,.0f}). Price must be positive.")
        elif asking_price < 8000 or asking_price > 3000000:
            errors.append(
                f"Asking price (${asking_price:,.0f}) is outside realistic Bishkek apartment range ($8,000–$3,000,000)."
            )

    return len(errors) == 0, errors, warnings_list


def _clean_cat(series: pd.Series, top_values: Optional[List[str]] = None) -> Tuple[pd.Series, List[str]]:
    s = series.fillna("unknown").astype(str).str.strip().str.lower()
    s = s.replace({"": "unknown", "nan": "unknown", "none": "unknown"})
    if top_values is None:
        top_values = s.value_counts().head(45).index.tolist()
    s = s.where(s.isin(top_values), "other")
    return s, top_values


def engineer_features(
    df: pd.DataFrame,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build clean numeric + categorical feature frame; learns vocab/medians if meta is None."""
    d = df.copy()
    is_train = meta is None
    if is_train:
        meta = {}

    # Numeric fields
    for col in ["rooms", "area", "floor", "total_floors", "year_built", "latitude", "longitude"]:
        if col not in d.columns:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # Clean out-of-Bishkek GPS coordinates
    bad_gps = (
        (d["latitude"] < 42.78) | (d["latitude"] > 42.96) |
        (d["longitude"] < 74.48) | (d["longitude"] > 74.72)
    )
    d.loc[bad_gps, ["latitude", "longitude"]] = np.nan

    # Clean impossible year_built / floor rows inside raw scrape
    d.loc[(d["year_built"] < 1945) | (d["year_built"] > 2029), "year_built"] = np.nan
    d.loc[(d["floor"] < 1) | (d["floor"] > 35), "floor"] = np.nan
    d.loc[(d["total_floors"] < 1) | (d["total_floors"] > 35), "total_floors"] = np.nan

    # Categorical cleaning
    cat_limits = {
        "district": 45,
        "house_type": 6,
        "building_series": 18,
        "condition": 8,
        "heating": 8,
        "jk_class": 8,
    }
    if is_train:
        meta["cat_vocabs"] = {}
    for col, lim in cat_limits.items():
        if col not in d.columns:
            d[col] = "unknown"
        s = d[col].fillna("unknown").astype(str).str.strip().str.lower().replace({"": "unknown", "nan": "unknown"})
        if is_train:
            vocab = s.value_counts().head(lim).index.tolist()
            if "unknown" not in vocab:
                vocab.append("unknown")
            meta["cat_vocabs"][col] = vocab
        else:
            vocab = meta["cat_vocabs"][col]
        d[col] = s.where(s.isin(vocab), "other")

    # Learn district median GPS coordinates so UI users only pick district!
    if is_train:
        dist_coords = {}
        for dist_name, grp in d.groupby("district"):
            if grp["latitude"].notna().sum() >= 3:
                dist_coords[dist_name] = (
                    float(grp["latitude"].median()),
                    float(grp["longitude"].median()),
                )
        meta["district_coords"] = dist_coords
        meta["global_lat"] = float(d["latitude"].median())
        meta["global_lon"] = float(d["longitude"].median())

    # Fill missing lat/lon from district median, then global Bishkek median
    dist_coords = meta["district_coords"]
    fill_lat = d["district"].map(lambda x: dist_coords.get(x, (meta["global_lat"], meta["global_lon"]))[0])
    fill_lon = d["district"].map(lambda x: dist_coords.get(x, (meta["global_lat"], meta["global_lon"]))[1])
    d["latitude"] = d["latitude"].fillna(fill_lat)
    d["longitude"] = d["longitude"].fillna(fill_lon)

    # Numeric medians
    if is_train:
        meta["num_medians"] = {
            c: float(d[c].median()) for c in ["rooms", "area", "floor", "total_floors", "year_built"]
        }
    for c, med in meta["num_medians"].items():
        d[c] = d[c].fillna(med)

    # Fix any scraped row where floor > total_floors
    d["total_floors"] = np.maximum(d["total_floors"], d["floor"])

    # Document / mortgage flags
    docs_str = d["documents"].fillna("").astype(str).str.lower() if "documents" in d.columns else pd.Series("", index=d.index)
    d["documents_has_ddu"] = docs_str.str.contains("долев|дду", regex=True).astype(int)
    d["documents_has_redbook"] = docs_str.str.contains("красн", regex=True).astype(int)
    d["documents_has_techpassport"] = docs_str.str.contains("техн|техпаспорт", regex=True).astype(int)

    mort_str = d["mortgage"].fillna("").astype(str).str.lower() if "mortgage" in d.columns else pd.Series("", index=d.index)
    d["mortgage_flag"] = mort_str.isin(["1", "1.0", "true", "да", "yes"]).astype(int)

    # Engineered domain features
    d["floor_ratio"] = d["floor"] / np.maximum(d["total_floors"], 1.0)
    d["is_first_floor"] = (d["floor"] == 1).astype(int)
    d["is_top_floor"] = (d["floor"] == d["total_floors"]).astype(int)
    d["area_per_room"] = d["area"] / np.maximum(d["rooms"], 1.0)
    d["building_age"] = np.clip(2026.0 - d["year_built"], 0.0, 90.0)
    d["dist_alatoo_km"] = [
        haversine_km(la, lo, ALATOO_LAT, ALATOO_LON) for la, lo in zip(d["latitude"], d["longitude"])
    ]
    d["dist_magistral_km"] = [
        haversine_km(la, lo, MAGISTRAL_LAT, MAGISTRAL_LON) for la, lo in zip(d["latitude"], d["longitude"])
    ]

    num_cols = [
        "rooms", "area", "floor", "total_floors", "floor_ratio",
        "is_first_floor", "is_top_floor", "area_per_room", "year_built",
        "building_age", "latitude", "longitude", "dist_alatoo_km", "dist_magistral_km",
        "documents_has_ddu", "documents_has_redbook", "documents_has_techpassport", "mortgage_flag",
    ]
    cat_cols = ["district", "house_type", "building_series", "condition", "heating", "jk_class"]

    if is_train:
        meta["num_cols"] = num_cols
        meta["cat_cols"] = cat_cols
        meta["cat_categories"] = {}
        for c in cat_cols:
            cats = sorted(d[c].unique().tolist())
            if "other" not in cats:
                cats.append("other")
            meta["cat_categories"][c] = cats

    out = d[num_cols + cat_cols].copy()
    for c in cat_cols:
        cats = meta["cat_categories"][c]
        out[c] = out[c].where(out[c].isin(cats), "other")
        out[c] = pd.Categorical(out[c], categories=cats)

    return out, meta


def eval_regression(y_true_usd: np.ndarray, y_pred_usd: np.ndarray) -> Dict[str, float]:
    return {
        "mae_usd": round(float(mean_absolute_error(y_true_usd, y_pred_usd)), 1),
        "rmse_usd": round(float(root_mean_squared_error(y_true_usd, y_pred_usd)), 1),
        "mape_pct": round(float(mean_absolute_percentage_error(y_true_usd, y_pred_usd) * 100), 2),
        "r2": round(float(r2_score(y_true_usd, y_pred_usd)), 4),
    }


def train_and_save_ml(csv_path: str, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    raw = pd.read_csv(csv_path)
    total_raw = len(raw)

    # Drop PII columns immediately
    raw = raw.drop(columns=[c for c in PII_AND_DROP_COLS if c in raw.columns], errors="ignore")

    # Clean target & area outliers (removes scraped monthly rents / typos like $1,350)
    raw["price_usd"] = pd.to_numeric(raw["price_usd"], errors="coerce")
    raw["area"] = pd.to_numeric(raw["area"], errors="coerce")
    raw = raw[
        raw["price_usd"].between(15000, 900000) &
        raw["area"].between(12, 450)
    ].copy()
    raw["ppm2_check"] = raw["price_usd"] / raw["area"]
    raw = raw[raw["ppm2_check"].between(450, 4500)].drop(columns=["ppm2_check"]).reset_index(drop=True)

    # Train/test split stratified by price deciles (fixed seed = 42)
    price_bins = pd.qcut(raw["price_usd"], q=10, labels=False, duplicates="drop")
    train_df, test_df = train_test_split(raw, test_size=0.20, random_state=SEED, stratify=price_bins)

    X_train, meta = engineer_features(train_df, meta=None)
    X_test, _ = engineer_features(test_df, meta=meta)
    y_train_usd = train_df["price_usd"].values
    y_test_usd = test_df["price_usd"].values
    y_train_log = np.log1p(y_train_usd)

    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]

    # 1. Baseline 1: Global Median
    global_med = float(np.median(y_train_usd))
    m_median = eval_regression(y_test_usd, np.full_like(y_test_usd, global_med))

    # 2. Baseline 2: Ridge Regression
    ridge_pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ])),
        ("reg", Ridge(alpha=10.0, random_state=SEED)),
    ])
    ridge_pipe.fit(X_train, y_train_log)
    pred_ridge = np.expm1(ridge_pipe.predict(X_test))
    m_ridge = eval_regression(y_test_usd, pred_ridge)

    # 3. LightGBM Regressor
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_train, y_train_log, test_size=0.15, random_state=SEED
    )
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        subsample_for_bin=50000,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_samples=15,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=SEED,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_sub, y_tr_sub,
        eval_set=[(X_val_sub, y_val_sub)],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    pred_lgb_log = lgb_model.predict(X_test)
    pred_lgb = np.expm1(pred_lgb_log)
    m_lgb = eval_regression(y_test_usd, pred_lgb)

    # 4. CatBoost Regressor
    X_tr_cb = X_tr_sub.copy()
    X_val_cb = X_val_sub.copy()
    X_test_cb = X_test.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_val_cb[c] = X_val_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)

    cb_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=SEED,
        verbose=False,
    )
    cb_model.fit(X_tr_cb, y_tr_sub, cat_features=cat_cols, eval_set=(X_val_cb, y_val_sub), early_stopping_rounds=60)
    pred_cb_log = cb_model.predict(X_test_cb)
    pred_cb = np.expm1(pred_cb_log)
    m_cb = eval_regression(y_test_usd, pred_cb)

    # 5. Blend Ensemble (50% LightGBM + 50% CatBoost in log space)
    pred_ens_log = 0.50 * pred_lgb_log + 0.50 * pred_cb_log
    pred_ens = np.expm1(pred_ens_log)
    m_ens = eval_regression(y_test_usd, pred_ens)

    # Build SHAP TreeExplainer on LightGBM (fast & exact!)
    explainer = shap.TreeExplainer(lgb_model)

    # Save top district options for UI dropdowns
    meta["ui_districts"] = [d for d in meta["cat_vocabs"]["district"] if d not in ("unknown", "other")]
    meta["ui_series"] = [s for s in meta["cat_vocabs"]["building_series"] if s not in ("unknown", "other")]
    meta["ui_heating"] = [h for h in meta["cat_vocabs"]["heating"] if h not in ("unknown", "other")]
    meta["ui_jk_class"] = [k for k in meta["cat_vocabs"]["jk_class"] if k not in ("unknown", "other")]
    meta["train_rows"] = int(len(train_df))
    meta["test_rows"] = int(len(test_df))
    meta["total_raw_rows"] = int(total_raw)
    meta["clean_rows"] = int(len(raw))
    meta["median_price_usd"] = round(global_med, 0)

    metrics = {
        "seed": SEED,
        "total_raw_rows": total_raw,
        "clean_rows": len(raw),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "baselines": {"global_median": m_median, "ridge_regression": m_ridge},
        "models": {"lightgbm": m_lgb, "catboost": m_cb, "ensemble_lgb_cb": m_ens},
    }

    artifact = {
        "lgb_model": lgb_model,
        "cb_model": cb_model,
        "explainer": explainer,
        "meta": meta,
        "metrics": metrics,
    }
    joblib.dump(artifact, os.path.join(out_dir, "ml_valuation_bundle.joblib"))
    with open(os.path.join(out_dir, "ml_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


class BishkekPricePredictor:
    def __init__(self, bundle_path: str):
        bundle = joblib.load(bundle_path)
        self.lgb_model = bundle["lgb_model"]
        self.cb_model = bundle["cb_model"]
        self.explainer = bundle["explainer"]
        self.meta = bundle["meta"]
        self.metrics = bundle["metrics"]

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ok, errors, warnings_list = validate_apartment_input(payload)
        if not ok:
            return {"ok": False, "errors": errors, "warnings": warnings_list}

        row = {
            "rooms": float(payload["rooms"]),
            "area": float(payload["area"]),
            "floor": float(payload["floor"]),
            "total_floors": float(payload["total_floors"]),
            "year_built": float(payload.get("year_built") or 2020),
            "district": payload.get("district") or "магистраль",
            "house_type": payload.get("house_type") or "монолит",
            "building_series": payload.get("building_series") or "элитка",
            "condition": payload.get("condition") or "под самоотделку (псо)",
            "heating": payload.get("heating") or "центральное",
            "jk_class": payload.get("jk_class") or "комфорт",
            "documents": payload.get("documents") or "",
            "mortgage": payload.get("mortgage") or "",
            "latitude": payload.get("latitude", np.nan),
            "longitude": payload.get("longitude", np.nan),
        }
        df_in = pd.DataFrame([row])
        X_in, _ = engineer_features(df_in, meta=self.meta)

        pred_lgb_log = float(self.lgb_model.predict(X_in)[0])
        X_cb = X_in.copy()
        for c in self.meta["cat_cols"]:
            X_cb[c] = X_cb[c].astype(str)
        pred_cb_log = float(self.cb_model.predict(X_cb)[0])

        pred_log = 0.50 * pred_lgb_log + 0.50 * pred_cb_log
        fair_usd = float(np.clip(np.expm1(pred_log), 12000.0, 1500000.0))
        area = float(payload["area"])
        fair_ppm2 = fair_usd / area

        # SHAP explanation (convert log-space SHAP into intuitive USD contributions)
        shap_vals = self.explainer.shap_values(X_in)[0]
        base_log = float(np.atleast_1d(self.explainer.expected_value)[0])
        base_usd = float(np.expm1(base_log))
        total_delta_usd = fair_usd - base_usd
        shap_sum = float(np.sum(shap_vals))
        scale = (total_delta_usd / shap_sum) if abs(shap_sum) > 1e-6 else fair_usd

        drivers = []
        for col, sv in zip(X_in.columns, shap_vals):
            usd_impact = float(sv * scale)
            raw_val = X_in[col].iloc[0]
            if isinstance(raw_val, (float, np.floating)):
                val_str = f"{raw_val:.1f}"
            else:
                val_str = str(raw_val)
            drivers.append({
                "feature": col,
                "label": FEATURE_READABLE.get(col, col),
                "value": val_str,
                "usd_impact": round(usd_impact, 0),
            })
        drivers.sort(key=lambda x: abs(x["usd_impact"]), reverse=True)

        asking = payload.get("asking_price_usd")
        overprice_pct = None
        price_verdict = None
        if asking is not None and asking != "":
            asking_f = float(asking)
            overprice_pct = round((asking_f - fair_usd) / fair_usd * 100.0, 1)
            if overprice_pct > 12:
                price_verdict = f"Overpriced by +{overprice_pct:.1f}% vs. Bishkek market estimate"
            elif overprice_pct < -10:
                price_verdict = f"Below market by {abs(overprice_pct):.1f}% — check documents carefully"
            else:
                price_verdict = f"Fair market range ({overprice_pct:+.1f}% vs. estimate)"

        return {
            "ok": True,
            "errors": [],
            "warnings": warnings_list,
            "fair_price_usd": round(fair_usd, 0),
            "fair_price_per_m2": round(fair_ppm2, 0),
            "base_market_usd": round(base_usd, 0),
            "asking_price_usd": float(asking) if asking not in (None, "") else None,
            "overprice_pct": overprice_pct,
            "price_verdict": price_verdict,
            "top_drivers": drivers[:6],
        }


if __name__ == "__main__":
    base = r"C:\Users\user\.gemini\antigravity\scratch\uycheck-kg"
    metrics = train_and_save_ml(
        os.path.join(base, "data", "bishkek_apartments.csv"),
        os.path.join(base, "models"),
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
