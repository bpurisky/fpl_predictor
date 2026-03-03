[README.md](https://github.com/user-attachments/files/25717124/README.md)
# FPL Predictor

A production-quality Fantasy Premier League points prediction system using temporal machine learning. Built with strict no-leakage methodology, reproducible experiments, and modular architecture.

---

## Project Goal

Predict expected FPL points for the upcoming gameweek per player, combining:

- **FPL API** — official match history, fixture difficulty, player metadata
- **Understat** — shot-level xG, xA, danger zone %, shot quality metrics

**Primary target:** Expected FPL points for GW *t+1*

**Planned extensions:** Multi-GW horizon, minutes probability, quantile risk predictions, points-per-£ value ranking

---

## Current Status

| Stage | Status | Files |
|---|---|---|
| Data Ingestion | ✅ Complete | `src/io/fpl_api.py` |
| Dataset Building | ✅ Complete | `src/data/build_dataset.py` |
| Understat Integration | ✅ Complete | `src/io/understat_scraper.py`, `src/data/understat_features.py` |
| GBM Baseline | 🔲 Next | `src/models/gbm.py` |
| Transformer Model | 🔲 Planned | `src/models/transformer.py` |
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
│   └── config.yaml             # Global config
│
├── data/
│   ├── raw/                    # Raw API + scrape responses (versioned JSON)
│   │   └── understat/          # Understat player + league cache
│   ├── interim/                # Intermediate processed files
│   └── processed/              # Final train/val parquet files
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
│   ├── models/                 # (next: gbm.py, transformer.py)
│   ├── trainers/               # (next: training loops)
│   ├── predict/                # (next: inference pipeline)
│   └── utils/
│
├── scripts/
│   ├── ingest.py               # CLI: run FPL data ingestion
│   └── generate_player_mapping.py  # One-time: FPL ↔ Understat ID mapping
│
├── tests/
│   ├── test_fpl_api.py
│   ├── test_build_dataset.py
│   └── test_understat.py
│
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-repo/fpl-predictor.git
cd fpl-predictor
pip install -r requirements.txt

# For Understat player mapping generation only
pip install rapidfuzz
```

**requirements.txt**
```
requests>=2.31.0
urllib3>=2.0.0
pyyaml>=6.0
pytest>=8.0
pandas
numpy
scikit-learn
```

---

## Data Sources

### FPL Official API

No authentication required. Three endpoints are used:

| Endpoint | Purpose |
|---|---|
| `/bootstrap-static/` | Player list, teams, gameweek metadata |
| `/fixtures/` | All fixtures with difficulty ratings |
| `/element-summary/{id}/` | Per-player match history + upcoming fixtures |

Raw responses are saved to `data/raw/` with versioned timestamps (`bootstrap_20241001T120000Z.json`) and a `_latest.json` convenience copy.

### Understat

No public API. Data is embedded as `JSON.parse('...')` blocks in page HTML. Two pages are scraped:

| Page | Purpose |
|---|---|
| `understat.com/league/EPL/{year}` | Season player list + Understat IDs |
| `understat.com/player/{id}` | Per-player match history + every shot |

Scraped data is cached to `data/raw/understat/` with TTL logic (historical seasons: 365 days, current season: 1 day).

---

## Quick Start

### 1. Run FPL Ingestion

```bash
# Full ingestion (~600 players, ~10 minutes)
python scripts/ingest.py

# Bootstrap + fixtures only (fast connectivity test)
python scripts/ingest.py --bootstrap-only

# Single player debug
python scripts/ingest.py --player-id 427
```

### 2. Set Up Understat Player Mapping (one-time)

The FPL and Understat player databases use different IDs and name formats. A manually verified mapping file is required.

```bash
# Step 1: Generate fuzzy-matched candidates
python scripts/generate_player_mapping.py
# → writes data/understat_player_mapping_candidates.csv

# Step 2: Open the CSV and review all rows where verified=False
# Fix any wrong understat_id values, set verified=True

# Step 3: Save as the final mapping
cp data/understat_player_mapping_candidates.csv data/understat_player_mapping.csv
```

The script auto-verifies matches with fuzzy score ≥ 95. Typically ~50–100 players need manual review (~20 minutes, one-time effort).

### 3. Build the Feature Matrix

```python
from src.io.fpl_api import load_latest
from src.data.build_dataset import build_dataset

bootstrap = load_latest("bootstrap")
fixtures  = load_latest("fixtures")
# element_summaries: dict mapping player_id → load_latest(f"player_{id}")

result = build_dataset(
    bootstrap=bootstrap,
    fixtures=fixtures,
    element_summaries=element_summaries,
    val_start_gw=33,                     # last 5 GWs as validation
    processed_dir="data/processed",
    version="v1",
)

train_df      = result["train_df"]
val_df        = result["val_df"]
feature_cols  = result["feature_cols"]
```

---

## Feature Engineering

### Temporal Correctness Rules

These rules are enforced throughout and tested explicitly:

1. Features for GW *t* may only use match data from GWs `< t`
2. All rolling windows use `shift(1)` before the window — no target-GW data is ever included
3. Train/val split is strictly time-based — **never shuffled**
4. Upcoming fixture features (difficulty, home/away) are pre-kickoff public information only
5. Understat features are merged as rolling averages of past matches only — raw match stats are excluded from the merge

### FPL Features

**Rolling means (windows: 3, 5 matches)**
- `total_points_roll3/5`, `minutes_roll3/5`
- `goals_scored_roll3/5`, `assists_roll3/5`
- `bonus_roll3/5`, `bps_roll3/5`
- `influence_roll3/5`, `creativity_roll3/5`, `threat_roll3/5`, `ict_index_roll3/5`
- `clean_sheets_roll3/5`, `goals_conceded_roll3/5`
- `expected_goals_roll3/5`, `expected_assists_roll3/5`, `expected_goal_involvements_roll3/5`

**Exponentially weighted means (spans: 3, 5)**
- Same columns as above with `_ewm3/5` suffix

**Minutes reliability features**
- `started` — played ≥ 45 min (binary)
- `minutes_pct` — minutes / 90
- `started_roll3/5` — starting consistency
- `minutes_pct_roll3/5`
- `sub_risk` — proportion of last 5 games below 45 min

**Fixture context** (pre-kickoff public data)
- `fixture_difficulty` — FPL difficulty rating for this team
- `opponent_difficulty` — FPL difficulty rating for opponent
- `is_double_gw` — player has two fixtures this GW
- `n_fixtures` — number of fixtures this GW

**Player / team metadata**
- `cost` — player price in £m at time of match
- `selected` — ownership count
- `was_home`
- `pos_1/2/3/4` — one-hot position encoding (GKP/DEF/MID/FWD)
- `strength_overall/attack/defence_home/away` — FPL team strength ratings

### Understat Features (prefix: `us_`)

All Understat features are rolling averages of past matches. Raw match stats are never merged directly.

**Match-level aggregates (rolling 3, 5 + EWM 3, 5)**
- `us_xG_roll3/5/ewm3/5` — expected goals
- `us_xA_roll3/5/ewm3/5` — expected assists
- `us_npxG_roll3/5/ewm3/5` — non-penalty xG
- `us_shots_roll3/5/ewm3/5`
- `us_key_passes_roll3/5/ewm3/5`
- `us_xG_overperformance_roll3/5/ewm3/5` — goals minus xG (finishing trend)

**Shot quality features (rolling 3, 5 + EWM 3, 5)**
- `us_xG_per_shot_roll3/5/ewm3/5` — average shot quality
- `us_danger_zone_pct_roll3/5/ewm3/5` — % shots from central high-danger zone (X > 0.83, 0.3 < Y < 0.7)
- `us_big_chances_roll3/5/ewm3/5` — shots with xG ≥ 0.30
- `us_open_play_pct_roll3/5/ewm3/5` — % shots from open play

Players without Understat coverage (pre-2022, unmapped) receive `NaN` for all `us_*` columns. GBM handles this natively; the Transformer requires masking.

---

## Cache Strategy

| Data | Cache Location | TTL | Rationale |
|---|---|---|---|
| FPL bootstrap | `data/raw/bootstrap.json` | 6 hours | Changes on transfer deadlines |
| FPL fixtures | `data/raw/fixtures.json` | 6 hours | Updates as matches complete |
| FPL player summary | `data/raw/element_{id}.json` | 24 hours | Updated after each GW |
| Understat league page | `data/raw/understat/league_{year}.json` | 7 days | Player IDs rarely change |
| Understat player (historical) | `data/raw/understat/player_{id}.json` | 365 days | Past seasons never change |
| Understat player (current) | `data/raw/understat/player_{id}.json` | 1 day | New matches added weekly |

Each cache file has a `.meta.json` sidecar recording `fetched_at` timestamp. Weekly ingestion runs only hit the network for stale entries.

---

## Running Tests

```bash
pytest tests/ -v

# Individual test modules
pytest tests/test_fpl_api.py -v
pytest tests/test_build_dataset.py -v
pytest tests/test_understat.py -v
```

### Key Tests

**Temporal leakage tests** (`test_build_dataset.py`)
- `test_rolling_features_no_leakage_first_row` — GW 1 rolling features must be NaN
- `test_rolling_features_second_row_uses_only_first` — GW 2 roll equals GW 1 value only
- `test_rolling_features_value_never_equals_current_gw_target` — rolling ≠ current target
- `test_rolling_features_players_are_independent` — windows don't bleed across players
- `test_temporal_split_no_overlap` — train and val GW sets are disjoint

**Understat leakage tests** (`test_understat.py`)
- `test_understat_rolling_gw1_is_nan` — first match `us_xG_roll3` must be NaN
- `test_understat_rolling_second_match_uses_first_only`
- `test_merge_does_not_add_raw_understat_cols` — only `us_*` columns enter the merge

---

## Configuration

`config/ingestion.yaml` controls all API and cache behaviour:

```yaml
api:
  base_url: "https://fantasy.premierleague.com/api"
  timeout_seconds: 30
  rate_limit_pause: 0.5

paths:
  raw_dir: "data/raw"
  processed_dir: "data/processed"

ingestion:
  save_raw: true
  skip_unavailable_players: true   # status == 'u'
```

---

## Architectural Decisions

**Why a manual player mapping file?**
Fuzzy name matching has a ~3–5% error rate on non-ASCII names (e.g. Traoré, Martinelli, Diogo Jota variants). A single wrong join silently corrupts training data for that player's entire history. The mapping file is a 20-minute one-time cost that eliminates this failure mode permanently.

**Why `shift(1)` before every rolling window?**
`pandas` rolling on a sorted series includes the current row by default. Without `shift(1)`, the feature at row *t* would include the value at row *t* — direct target leakage. The shift ensures the window strictly covers rows `0..t-1`.

**Why prefix Understat columns with `us_`?**
Both FPL and Understat expose an `xG` field. Without a namespace prefix, merging the two DataFrames would silently overwrite one with the other. All Understat-derived columns are named `us_*` throughout.

**Why cache Understat data so aggressively?**
Understat has no rate limit documentation. Scraping 600 players at 2s intervals takes ~20 minutes. Re-doing this on every weekly run is wasteful and risks IP blocks. Historical season data (2022–24) never changes — it is scraped once and cached for a year.

**Why time-based train/val split only?**
Random splitting of time-series data allows future information to leak into training through the rolling features of adjacent rows. A GW 35 row's `roll3` feature includes GWs 32, 33, 34 — if GW 35 is in train and GW 33 is in val, the val set has leaked into train. Strict temporal splitting avoids this entirely.

---

## Planned: Next Stages

### GBM Baseline (`src/models/gbm.py`)
- `HistGradientBoostingRegressor` with absolute error loss
- Native NaN handling (no imputation needed for `us_*` columns)
- Residual isotonic regression calibration
- Quantile regression variants (P10, P50, P90)

### Transformer Model (`src/models/transformer.py`)
- Sequence input: last *K* matches per player
- Masked padding for players with fewer than *K* appearances
- Positional embeddings
- `TransformerEncoder` → masked mean pooling → MLP head
- Huber loss, AdamW, gradient clipping

### Ensemble
- Weighted average by inverse validation MAE
- Optional stacking meta-model

### Prediction Pipeline (`src/predict/`)
- Pull latest data → build features → predict → output ranked DataFrame
- Columns: `player_name`, `team`, `position`, `predicted_points`, `cost`, `value_per_m`, `risk`

---

## Continuing Development

To pick up from the current state in a new session:

> *"Here is my FPL predictor project: https://github.com/your-repo. Read the README and existing src/ files, then help me build the GBM baseline in src/models/gbm.py."*

---

## License

MIT
