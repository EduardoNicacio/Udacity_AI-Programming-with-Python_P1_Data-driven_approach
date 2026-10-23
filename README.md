# Analyzing Ecommerce Large Datasets for Machine Learning

> An ETL-and-visualization pipeline over the **Amazon Reviews 2023** dataset (McAuley-Lab, Hugging Face) — streams raw electronics metadata and reviews, filters them into a top-performing product table, exports ML-ready `.csv`/`.parquet` artifacts, and answers seven stakeholder questions for the machine learning team. Built as the final composite PCEP portfolio project for Udacity's *AI Programming with Python* program.

**Verdict: the electronics catalog is quality-rich but price-skewed, and price does not buy rating.** Of the **100** streamed metadata records, **21 (21.00%)** survive the top-product filter (rating ≥ 4.5, a valid title, and a price) and **12 (57.14% of those 21)** clear the **4.7** dashboard cutoff. Prices are heavily right-skewed — median **$24.63** against a max of **$909.99** (one 5.0-rated ANNKE DVR pulled the mean of all 21 up to **$81.51**, and to **$127.94** within the ≥ 4.7 tier) — and the rating-vs-price scatter shows no linear relationship. All numbers below are taken directly from the executed notebook.

---

## Results

### Filtering funnel: 100 streamed records → top products

| Stage | Rows | Survival | Notes |
| :--- | ---: | ---: | :--- |
| Streamed `raw_meta_Electronics` (limit 100) | 100 | — | Valid, priced, titled subset of the full metadata frame |
| ✅ Top products (rating ≥ **4.5**) | **21** | **21.00% of 100** | `top_products_df` — exported to `.csv`/`.parquet` |
| ✅ Dashboard tier (rating ≥ **4.7**) | **12** | **57.14% of 21** | Displayed in the final pandas dashboard |

### All 21 top products (sorted by rating, then price)

| Title | Rating | Price |
| :--- | ---: | ---: |
| ANNKE 4K 16CH Security DVR Recorder (16× 8MP cameras) | **5.0** | **$909.99** |
| BIPRA U3 2.5" USB 3.0 NTFS Portable External Hard Drive (250GB) | **5.0** | **$98.49** |
| Monsoon G1/4" to 1/2" ID Compression Fitting (4-Pack) | **5.0** | **$25.71** |
| MHYALUDO Silicone Print Case Cover for Apple AirPods Pro | **5.0** | **$10.99** |
| Polaroid Originals Now Viewfinder i-Type Instant Camera Bundle | **4.8** | **$179.99** |
| CoBak Case for Kindle Paperwhite — PU Leather Cover | **4.8** | **$18.95** |
| Centon Electronics Flash Memory Card (S1-SDHU1-32G) | **4.8** | **$8.99** |
| PNY 1TB PRO Elite U3 V30 microSDXC Flash Memory Card | **4.7** | **$109.99** |
| JBL FLIP 5 — Waterproof Portable Bluetooth Speaker (Renewed) | **4.7** | **$79.90** |
| HP Elitebook 8560W Replacement LCD Screen (LED HD Matte) | **4.7** | **$63.33** |
| Wenlaty Case for iPad 9th/8th/7th Generation (10.2") | **4.7** | **$19.99** |
| Novelty Cute Cartoon USB 2.0 Flash Drive (16GB, Camera Black) | **4.7** | **$8.98** |
| 13–15.6" Laptop Sleeve Case Shoulder Bag Briefcase | **4.6** | **$14.88** |
| Nurse Stickers for Water Bottles and Laptop (108 Designs) | **4.6** | **$8.99** |
| 8x8ft Pink Rose Flower Theme Photography Backdrops | **4.5** | **$38.90** |
| SUPCASE UB Pro Series Case for iPad Pro 12.9" (2020) | **4.5** | **$27.99** |
| May Chen MacBook Pro 16" Case (A2141, Abstract Leaves) | **4.5** | **$26.99** |
| Adafruit FTDI Serial TTL-232 USB Cable [ADA70] | **4.5** | **$24.63** |
| Digi-Tatoo Decal Skin for MacBook Pro 13" [Fresh Marble] | **4.5** | **$19.99** |
| NotoCity Vivoactive 4 Quick-Release Silicone Band (22mm) | **4.5** | **$9.99** |
| Outer Space Planets Stickers (50 Pcs, Waterproof Vinyl) | **4.5** | **$3.99** |

**Price summary (all 21):** min **$3.99** · median **$24.63** · max **$909.99** · mean **$81.51** (mean in the ≥ 4.7 tier: **$127.94**).

The full untruncated titles, both `pandas` scatter plots, the rating histogram, the ratio dashboard, and the seven stakeholder Q&As are embedded in the executed notebook's cells.

---

## Pipeline

```txt
Hugging Face: McAuley-Lab/Amazon-Reviews-2023 (streaming, trust_remote_code=True)
   ├─ raw_review_Electronics (limit 100)  ──► reviews_df (100 × 10)
   ├─ raw_meta_Electronics  (limit 100)   ──► meta_df (100 × 16)
   │     ├─ rename main_category, drop features/description/images
   │     ├─ referential integrity: reviews ↔ metadata merged on title
   │     └─ filter: rating ≥ 4.5 + valid title + price
   └─ ► top_products_df (21 × 3: title, average_rating, price)
          ├─ .to_csv(...) + .to_parquet(...)          → top_products.csv / .parquet
          ├─ get_product() lookup over 21 rows        → exact/missing price answers
          ├─ % of products ≥ 4.5 and ≥ 4.7            → 21.00% / 57.14%
          ├─ seaborn histplot + scatterplot           → figures embedded in notebook
          ├─ top_rated_titles (rating ≥ 4.7, iterrows) → 12 of 21 rows
          └─ pandas dashboard: sort rating desc, .describe(), .info(), display()
```

| Choice | Setting |
| :--- | :--- |
| Source | `McAuley-Lab/Amazon-Reviews-2023` via `load_dataset(..., streaming=True, trust_remote_code=True)`, 100 records per split |
| Top-product filter | `average_rating ≥ 4.5` with valid title and price → 21 rows |
| Dashboard cutoff | `rating ≥ 4.7` → 12 rows (both `top_rated_titles` and the final dashboard tier) |
| Price stats | min 3.99 · median 24.63 · max 909.99 · mean 81.51 (all 21) · 127.94 (≥ 4.7 tier) |
| Exports | `top_products.csv`, `top_products.parquet` (21 × 3) |
| Display | `pd.set_option(...)` wide display; `Items_To_Print = 10` for the top-N printout |
| Plots | `seaborn` rating histogram + price-vs-rating scatterplot (`matplotlib`) |
| Lookup behavior | `get_product()`: title not found → "❌ Product title not found"; found but unpriced → "❌ No valid price listed" |

---

## Getting Started

**Requirements:** Python 3.10+, `datasets` **2.21.0**, pandas, matplotlib, seaborn, pyarrow, jupyter (developed on Python 3.14.7, datasets 2.21.0, pandas 3.0.6, seaborn 0.13.2, matplotlib 3.11.2, pyarrow 23.0.1).

```bash
pip install "datasets<4" pandas matplotlib seaborn pyarrow jupyter
jupyter notebook "Udacity - PCEP Project.ipynb"
```

> **Version pin — important:** use `datasets==2.21.0`. In `datasets` 3.x (including 3.0.1), `Sequence(dict)` typed features are transposed to `struct<field: list>` in `get_nested_type()` during schema building, while the Amazon 2023 JSONL records are list-of-dict — so `raw_meta_Electronics` fails with `ArrowNotImplementedError: Unsupported cast from list<item: struct<...>> to struct` on load. Version 2.21.0 (which also satisfies the notebook's `datasets<4` pip cell) loads cleanly.

Run it headless (data streams from Hugging Face, so an internet connection is required):

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  Udacity_-_PCEP_Project.ipynb
```

`nbconvert` writes `Udacity_-_PCEP_Project.nbconvert.ipynb`, which should be moved over the deliverable. Re-running the notebook rewrites `user_inputs.txt` and the `top_products.*` artifacts; on Windows, `user_inputs.txt` is written in the platform's default encoding and should be re-saved as UTF-8 if non-ASCII characters look wrong.

---

## Project Structure

```txt
.
├── "Udacity - PCEP Project.ipynb"        # filled template (all code stubs completed)
├── Udacity_-_PCEP_Project.ipynb          # executed deliverable — 41/41 cells, 0 errors
├── user_inputs.txt                       # stakeholder Q&A transcript (7 questions + name)
├── top_products.csv                      # 21 × 3 top-product table
├── top_products.parquet                  # same table, Apache Parquet
├── docs/
│   ├── 1. Project Overview.md            # Transformational AI scenario
│   ├── 2. Instructions.md                # step-by-step task outline
│   ├── 3. Environments.md                # environment setup
│   ├── 4. Rubric.md                      # grading criteria
│   └── 5. Candidate Notes.md             # engineering decisions and fixes
└── LICENSE.md                            # MIT
```

> **Data notice:** the Amazon Reviews 2023 datasets (`McAuley-Lab/Amazon-Reviews-2023`) are streamed directly from Hugging Face and are not redistributed by this project. The pipeline reads only 100 records per split as a task-defined limit.

---

## Recommendation (tl;dr)

For the machine learning team's fine-tuning set, start from the **12 products at rating ≥ 4.7** (57.14% of the 21 top products): the cluster is led by three 5.0-rated items — the ANNKE 4K DVR ($909.99), the BIPRA 250GB external drive ($98.49), and the Monsoon compression fitting ($25.71) — plus the 4.8 trio (Polaroid bundle $179.99, CoBak Kindle case $18.95, Centon flash card $8.99) and five 4.7-rated accessories from $8.98 to $109.99. The catalog is not a price-rating market: the scatter shows no linear price↔rating relationship, and right-skew means any aggregate (mean $81.51 vs median $24.63) is dragged by a single $909.99 outlier, so use medians when reporting to stakeholders. Note that the stakeholder lookup handles edge cases explicitly: `get_product("FS-1051 FATSHARK TELEPORTER V3 HEADSET")` finds the record in the metadata stream (rated 3.5 — it never passes the ≥ 4.5 filter) but returns "❌ No valid price listed" because its price is genuinely `None`, while an unknown title returns "❌ Product title not found". Full transcripts, statistics, and dashboard output are in the executed notebook; the engineering log behind them is in [docs/5. Candidate Notes.md](docs/5.%20Candidate%20Notes.md).

---

## License

MIT — see [LICENSE.md](LICENSE.md).
