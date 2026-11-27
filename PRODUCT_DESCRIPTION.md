# UyCheck KG — Product Description

**Team:** Nurlanov Azizkhan, Zhanybekov Nurdos, Abdilazizov Nurmukhamed  
**Domain:** Residential Real Estate in Bishkek, Kyrgyzstan  
**Live App:** https://uycheck-kg.streamlit.app/  
**GitHub:** https://github.com/Hanchik8/uycheck-kg  

## Overview

UyCheck KG helps apartment buyers in Bishkek check whether an asking price matches the market, flag ambiguous listing descriptions, and verify Kyrgyz Republic legal and mortgage rules before paying a deposit.

1. **Market Price Valuation (ML):** Predicts fair market price (`USD` and `$/m²`) from 8,821 Bishkek listings (`raimbekovm/bishkek-real-estate`, CC BY 4.0) and breaks down the main price drivers via SHAP.
2. **Listing Text Check (NLP):** Classifies renovation state from pasted Russian, Kyrgyz, or English descriptions (7,023 training texts), extracts deal signals (DDU, PSO, Tech Passport, Red Book, GIK/mortgage), and flags low-confidence or sarcastic text.
3. **Kyrgyz Republic Legal & GIK Q&A (RAG on LM Studio):** Uses local LM Studio (`http://localhost:1234/v1`) for both embeddings and answer generation over 17 verified KR documents (GIK «Менин үйүм», Law No. 153, Housing Code No. 117, GU «Кадастр», DDU checklist, KR Tax Code), naming exact sources and refusing out-of-scope questions.

## Tabs

- **Apartment Check:** Fill apartment parameters or choose a sample listing, paste ad text, ask a question, and review the unified valuation, SHAP chart, text signals, and cited legal notes.
- **Laws & GIK Q&A:** Ask standalone questions in Russian, Kyrgyz, or English, or ask why the latest apartment estimate came out at that price.
- **System & Data:** Inspect active LLM/embedding runtime (LM Studio or Cloud), benchmark metrics, and dataset license.

*Informational assistant only — does not replace a licensed notary, official GU «Кадастр» title extract, or certified appraiser.*
