# Dataset Card — UyCheck KG (Bishkek Real Estate Dataset)

**Team:** Nurlanov Azizkhan, Zhanybekov Nurdos, Abdilazizov Nurmukhamed  
**Course:** Generative AI (Fall 2026) · Ala-Too International University  

---

## 1. Source & License
- **Primary Dataset (ML + NLP):** `raimbekovm/bishkek-real-estate` (Hugging Face)  
  **Link:** https://huggingface.co/datasets/raimbekovm/bishkek-real-estate  
  **Original Origin:** Public apartment sales listings in Bishkek, Kyrgyzstan (`house.kg`).  
  **License:** **CC BY 4.0** (Creative Commons Attribution 4.0 International).
- **RAG Knowledge Base (17 official Kyrgyz Republic documents):**
  - OJSC State Mortgage Company («ГИК» КР — Program «Менин үйүм»): https://gik.kg
  - Law of KR No. 153 *On State Registration of Rights to Immovable Property* & Housing Code of KR No. 117: https://cbd.minjust.gov.kg
  - State Agency for Land Resources, Cadastre, Geodesy & Cartography (GU «Кадастр»): https://gosreg.gov.kg
  - State Tax Service of KR (Tax Code Art. 409 & 1-year capital gains rule): https://sti.gov.kg

## 2. Size, Structure & Target Columns
- **Raw ML Table:** **8,821** apartment records across 60 columns (`12.08 MB` CSV — fits easily in free Google Colab).
- **Cleaned Modeling Set:** **8,758** residential sales rows (removed 63 scraped rental/typo outliers outside `$15k–$900k` or `$450–$4,500/m²`), split 80/20 with fixed `seed = 42` (**7,006 train / 1,752 test**).
- **ML Target Column:** `price_usd` (continuous USD price; median **$98,000**, median **$1,557/m²**, median area **69.0 m²**).
- **Key Engineered Features:** `area`, `rooms`, `floor`, `total_floors`, `floor_ratio`, `area_per_room`, `year_built`, `building_age`, `district`, `house_type` (`монолит` 4,787 / `кирпич` 2,737 / `панель` 1,264), `building_series`, `condition`, `heating`, `jk_class`, Haversine distance to Ala-Too Square & South Magistral, and document flags (`ДДУ`, `Красная книга`, `техпаспорт`).
- **NLP Text Subset:** **7,023** Russian/Kyrgyz apartment descriptions (`> 20` chars) with verified `condition` labels across 5 classes: `евроремонт` (3,020), `под самоотделку (псо)` (2,519), `хорошее` (1,044), `среднее` (300), `не достроено` (140).

## 3. Privacy & Anonymization
- **Zero PII:** Listing IDs (`id`, `listing_id`), original listing URLs (`url`), phone flags (`has_phone`), exact street addresses (`address`), and seller metadata are dropped prior to training. Only anonymous apartment characteristics, district names, and approximate coordinates are retained.

## 4. Limitations, Risks & Known Biases
1. **Asking Prices vs. Closing Prices:** Listings reflect *asking prices* posted by sellers/agencies on `house.kg`, not final negotiated transaction prices (real sales in Bishkek often close 3–8% lower after negotiation).
2. **District Representation Skew:** High-construction districts are heavily over-represented (e.g., `Магистраль`: 2,012 listings, ~23% of the dataset), while smaller neighborhoods have far fewer examples — valuation variance is higher in sparse districts.
3. **Geographic & Temporal Scope:** Data reflects Bishkek residential apartments only and cannot be generalized to Osh, regional towns, commercial properties, or detached private houses. Prices reflect market conditions at scrape time and shift with USD/KGS exchange rates and construction supply.
4. **NLP Class Imbalance & Self-Reported Labels:** Sellers self-report `condition` (e.g., some call average flats «евроремонт»), and minority classes (`не достроено`: 140; `среднее`: 300) are much smaller than `евроремонт` (3,020). Our app mitigates overclaiming via a calibrated low-confidence threshold (`0.55`) and OOD/sarcasm guard.
5. **Legal Disclaimer:** RAG outputs summarize public Kyrgyz Republic laws and GIK rules for orientation only and do not constitute formal legal advice or replace a notary or official GU «Кадастр» encumbrance check.
