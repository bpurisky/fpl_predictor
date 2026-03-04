# FPL Predictor

A production-quality Fantasy Premier League points prediction system using temporal machine learning. Built with strict no-leakage methodology, reproducible experiments, and modular architecture.

---

## Project Status

| Stage | Status | Files |
|---|---|---|
| Data Ingestion | ✅ Complete | `src/io/fpl_api.py` |
| Dataset Building | ✅ Complete | `src/data/build_dataset.py` |
| Understat Integration | ✅ Complete | `src/io/understat_scraper.py`, `src/data/understat_features.py` |
| GBM Baseline | ✅ Complete | `src/models/gbm.py`, `src/trainers/gbm_trainer.py` |
| Transformer Model | 🔲 Next | `src/models/transformer.py` |
| Calibration | 🔲 Planned | — |
| Ensemble | 🔲 Planned | — |
| Prediction Pipeline | 🔲 Planned | `src/predict/` |

---

## Repository Structure

```
fpl-predictor/
│
├── config/
│   ├── ingestion.yaml          # API + cache settings
│   ├── gbm.yaml                # GBM hyperparameters + training settings
│   └── config.yaml             # Global config
│
├── data/
│   ├── raw/                    # Raw API responses (versioned + _latest JSON)
│   │   └── understat/          # Understat player + league cache
│   ├── interim/                # Intermediate processed files
│   └── processed/              # Final train/val parquet files
│
├── models/
│   ├── artifacts/              # Saved model .pkl files (versioned by timestamp)
│   └── experiments/            # JSON experiment records per training run
│
├── src/
│   ├── io/
│   │   ├── fpl_api.py          # FPL REST API client
│   │   ├── understat_scraper.py # Understat HTML scraper
│   │   └── understat_cache.py  # Disk cache with TTL logic
│   │
│   ├── data/
│   │   ├── build_dataset.py    # Feature engineering + train/val split
│   │   └── understat_features.py # Shot-level feature aggregation
│   │
│   ├── models/
│   │   └── gbm.py              # FPLGBMModel (HistGradientBoosting + calibration)
│   │
│   ├── trainers/
│   │   └── gbm_trainer.py      # Training orchestrator + experiment logging
│   │
│   ├── predict/                # (next: inference pipeline)
│   └── utils/
│
├── scripts/
│   ├── ingest.py               # CLI: run FPL data ingestion
│   ├── train_gbm.py            # CLI: train GBM baseline
│   └── generate_player_mapping.py  # One-time: FPL ↔ Understat ID mapping
│
├── tests/
│   ├── test_fpl_api.py
│   ├── test_build_dataset.py
│   ├── test_understat.py
│   └── test_gbm.py
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/bpurisky/fpl_predictor.git
cd fpl_predictor
pip install -r requirements.txt
```

**requirements.txt**
```
requests>=2.31.0
urllib3>=2.0.0
pyyaml>=6.0
pandas
numpy
scikit-learn
pyarrow
pytest>=8.0
rapidfuzz       # optional — only needed for generate_player_mapping.py
understat
understatapi
```

---

## Quick Start

### 1. Run FPL Ingestion

```bash
# Bootstrap + fixtures only (fast, ~5 seconds)
python scripts/ingest.py --bootstrap-only

# Full ingestion — all ~600 players (~10 minutes)
python scripts/ingest.py

# Single player debug
python scripts/ingest.py --player-id 427
```

### 2. Build Feature Matrix

```python
from src.io.fpl_api import load_latest
from src.data.build_dataset import build_dataset

bootstrap         = load_latest("bootstrap")
fixtures          = load_latest("fixtures")
element_summaries = {pid: load_latest(f"player_{pid}") for pid in player_ids}

result = build_dataset(
    bootstrap=bootstrap,
    fixtures=fixtures,
    element_summaries=element_summaries,
    val_start_gw=33,
    processed_dir="data/processed",
    version="v1",
)
train_df     = result["train_df"]
val_df       = result["val_df"]
feature_cols = result["feature_cols"]
```

### 3. Train GBM

```bash
python scripts/train_gbm.py

# Skip quantile models (faster)
python scripts/train_gbm.py --no-quantiles

# Custom config or dataset version
python scripts/train_gbm.py --config config/gbm.yaml --version v2
```

### 4. Understat Setup (optional — home network required)

Understat.com may be blocked on corporate/university networks. Run from home wifi:

```bash
pip install rapidfuzz
python scripts/generate_player_mapping.py
# → review data/understat_player_mapping_candidates.csv
# → save verified version as data/understat_player_mapping.csv
```

The GBM baseline works without Understat — `us_*` columns will be `NaN` and are handled natively by `HistGradientBoostingRegressor`.

---

## Data Sources

### FPL Official API

No authentication required.

| Endpoint | Purpose |
|---|---|
| `/bootstrap-static/` | Player list, teams, gameweek metadata |
| `/fixtures/` | All fixtures with difficulty ratings |
| `/element-summary/{id}/` | Per-player match history + upcoming fixtures |

Raw responses are saved to `data/raw/` as both a timestamped version (`bootstrap_20260303T151849Z.json`) and a `_latest.json` convenience copy. Always read from `_latest.json`.

### Understat (optional enrichment)

No public API — data is scraped from embedded JSON in page HTML.

| Page | Purpose |
|---|---|
| `understat.com/league/EPL/{year}` | Season player list + Understat IDs |
| `understat.com/player/{id}` | Per-player match history + every shot |

Requires home network or VPN — commonly blocked by corporate firewalls.

---

## Feature Engineering

### Temporal Correctness Rules

Enforced throughout and tested explicitly:

1. Features for GW *t* may only use match data from GWs `< t`
2. All rolling windows use `shift(1)` before windowing — no target-GW data ever included
3. Train/val split is strictly time-based — **never shuffled**
4. Upcoming fixture features (difficulty, home/away) are pre-kickoff public info only
5. Understat rolling features are merged from past matches only — raw match stats excluded

### FPL Features

**Rolling means (windows 3, 5) + EWM (spans 3, 5)**
- Points, minutes, goals, assists, bonus, BPS, ICT index, clean sheets, goals conceded, xG, xA, xGI

**Minutes reliability**
- `started`, `minutes_pct`, `started_roll3/5`, `sub_risk`

**Fixture context** (pre-kickoff)
- `fixture_difficulty`, `opponent_difficulty`, `is_double_gw`, `n_fixtures`

**Metadata**
- `cost` (£m), `selected`, `was_home`, position dummies (`pos_1–4`), team strength ratings

### Understat Features (prefix: `us_`)

All prefixed `us_` to avoid column collisions with FPL fields.

**Rolling match aggregates:** `us_xG`, `us_xA`, `us_npxG`, `us_shots`, `us_key_passes`, `us_xG_overperformance`

**Shot quality:** `us_xG_per_shot`, `us_danger_zone_pct`, `us_big_chances`, `us_open_play_pct`

---

## GBM Model

`FPLGBMModel` wraps `HistGradientBoostingRegressor` with:

- **`loss="absolute_error"`** — robust to FPL's heavy-tailed point distribution
- **Native NaN support** — `us_*` columns are NaN for unmapped players; no imputation needed
- **Isotonic calibration** — corrects systematic position-level bias without disrupting rank order
- **Quantile models** (P10/P50/P90) — separate estimators for risk-aware selection
  - High P10 = reliable floor player
  - High spread (P90−P10) = boom-or-bust captaincy candidate

**Output per player per GW:**
`predicted_points`, `p10`, `p50`, `p90`, `value_per_m`, `spread`

**Evaluation metrics:**
- Overall MAE and RMSE
- MAE by position (GKP / DEF / MID / FWD)
- Top-15 and Top-30 recall
- Per-GW MAE breakdown

Each training run saves a versioned `.pkl` artifact and a JSON experiment record to `models/experiments/`.

---

## Cache Strategy

| Data | Location | TTL | Notes |
|---|---|---|---|
| FPL bootstrap | `data/raw/bootstrap_latest.json` | 6 hours | Updates on transfer deadlines |
| FPL fixtures | `data/raw/fixtures_latest.json` | 6 hours | Updates as matches complete |
| FPL player summary | `data/raw/element_{id}_latest.json` | 24 hours | Updated after each GW |
| Understat league | `data/raw/understat/league_{year}.json` | 7 days | Player IDs rarely change |
| Understat player (historical) | `data/raw/understat/player_{id}.json` | 365 days | Past seasons never change |
| Understat player (current) | `data/raw/understat/player_{id}.json` | 1 day | New matches added weekly |

---

## Known Issues & Conventions

### Import Path Convention
All scripts add `src/io` and `src/data` to `sys.path` explicitly. **Never use `from io.X import`** — this conflicts with Python's built-in `io` standard library module.

```python
# CORRECT
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src" / "io"))
sys.path.insert(0, str(_ROOT / "src" / "data"))
sys.path.insert(0, str(_ROOT / "src"))
from fpl_api import ...

# BROKEN — conflicts with stdlib
from io.fpl_api import ...
```

### Encoding Convention
All file reads and writes must specify `encoding="utf-8"` explicitly. Windows defaults to `cp1252` which cannot handle accented player names (Traoré, Martinelli etc.).

```python
# CORRECT
path.read_text(encoding="utf-8")
path.write_text(content, encoding="utf-8")
json.dumps(data, ensure_ascii=False)

# BROKEN on Windows
path.read_text()
```

### File Naming Convention
Raw API files are saved in two forms:
- `bootstrap_20260303T151849Z.json` — timestamped archive (never overwritten)
- `bootstrap_latest.json` — always the most recent fetch

Always read from `_latest.json`. Never hardcode `bootstrap.json`.

### Understat Access
Understat.com is blocked on many corporate and university networks. Run Understat scraping from a home network or VPN. The full pipeline works without it — `us_*` features degrade gracefully to `NaN`.

---

## Running Tests

```bash
pytest tests/ -v

pytest tests/test_fpl_api.py -v
pytest tests/test_build_dataset.py -v
pytest tests/test_understat.py -v
pytest tests/test_gbm.py -v
```

### Key Leakage Tests

- `test_rolling_features_no_leakage_first_row` — GW1 rolling must be NaN
- `test_rolling_features_second_row_uses_only_first` — GW2 roll = GW1 value only
- `test_temporal_split_no_overlap` — train and val GW sets are disjoint
- `test_understat_rolling_gw1_is_nan` — first Understat match roll is NaN
- `test_merge_does_not_add_raw_understat_cols` — only `us_*` columns enter the merge
- `test_temporal_correctness_no_future_leakage` — val MAE ≥ train MAE

---

## Planned: Next Stages

### Transformer Model (`src/models/transformer.py`)
- Sequence input: last *K* matches per player
- Masked padding for players with < *K* appearances
- Positional embeddings + `TransformerEncoder` + masked mean pooling + MLP head
- Huber loss, AdamW, gradient clipping

### Ensemble (`src/models/ensemble.py`)
- Weighted average of GBM + Transformer by inverse validation MAE
- Optional stacking meta-model

### Prediction Pipeline (`src/predict/`)
- Weekly script: pull latest data → build features → predict → ranked output
- Columns: `player_name`, `team`, `position`, `predicted_points`, `cost`, `value_per_m`, `risk`

---

## Continuing Development

To continue in a new session:

> *"Here is my FPL predictor project: https://github.com/bpurisky/fpl_predictor — read the README first, then help me build the Transformer model in src/models/transformer.py"*

---

## License

MIT