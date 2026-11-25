# UyCheck KG — Product Description (Client Day Handover)

**Team:** Nurlanov Azizkhan, Zhanybekov Nurdos, Abdilazizov Nurmukhamed  
**Domain:** Residential Real Estate in Bishkek, Kyrgyzstan  

## What UyCheck KG Does
**UyCheck KG** helps homebuyers in Bishkek evaluate apartment listings before negotiating or paying a deposit by combining three AI modules in one Streamlit web app:

1. **Fair Price Valuation (ML):** Predicts fair market price (`USD` and `$/m²`) for a Bishkek apartment based on 8,821 real `house.kg` listings (CC BY 4.0), estimates overpricing/underpricing %, and explains *why* using SHAP feature impacts.
2. **Listing Text Analyzer (NLP):** Reads pasted apartment descriptions in Russian, Kyrgyz, or English (trained on 7,023 labeled Bishkek ads), classifies the renovation condition across 5 levels (`евроремонт`, `ПСО`, `хорошее`, `среднее`, `не достроено`), extracts deal signals (`ДДУ`, `Красная книга`, `техпаспорт`, `срочно`, `ипотека`), and warns **Low Confidence** on vague, contradictory, or sarcastic inputs.
3. **Kyrgyz Republic Legal & GIK Mortgage Assistant (RAG):** Answers questions strictly from 17 official Kyrgyz Republic documents (GIK KR «Менин үйүм», Law No. 153 on property registration, Housing Code No. 117, GU «Кадастр» guides, DDU checklists, and KR Tax Code rules) with exact source citations, and honestly says **"I don't know"** when information is outside the documents.

## How to Use the App
- **Tab 1 — Integrated Apartment Check (ML + NLP + RAG):** Pick a quick preset or enter your own Bishkek apartment details (district, area, rooms, floor, total floors, year, material, asking price), paste a listing description, and ask a question in Russian, Kyrgyz, or English. Click **Analyze Apartment & Answer** to get a unified valuation + NLP check + cited legal guidance.
- **Tab 2 — KR Legal & GIK Q&A (RAG Chat):** Ask standalone questions about GIK «Менин үйүм» rates/eligibility, GU «Кадастр» registration, Technical Passport vs. «Красная книга», DDU risks, or property taxes in Kyrgyzstan.
- **Tab 3 — Model Quality & Data Card:** Inspect benchmark metrics (baselines vs. final ensemble), NLP scores, and dataset sources.

*Disclaimer: UyCheck KG is an informational helper only and does not replace a licensed notary, official GU «Кадастр» encumbrance extract, or certified appraiser.*
