# Expected Decision Regret (xDR) — Build Specification

**Read this file completely before writing code.** It is the contract for the
project. Where it says MUST, deviation is a defect. Where it says SHOULD, deviation
is allowed if you record the reason in `docs/DECISIONS.md`.

Sections marked **[VERIFY]** contain claims that were not confirmable at drafting
time. Confirm each against the live source before building on it. Do not silently
assume they are correct.

Sections marked **[OPEN]** require a decision from the repository owner. Defaults
are provided so you are never blocked; flag in `docs/DECISIONS.md` when you use one.

---

## 1. Thesis and success criteria

### 1.1 What this project claims

Every on-ball football action has alternatives the player declined. xDR enumerates
those alternatives from freeze-frame tracking data, scores each with the same
possession-value model that scores the action actually taken, and reports the gap.

The counterfactual layer is the visible half. The **defensible** half is underneath:
a model that ranks actions correctly can still emit probabilities whose magnitudes
mean nothing. This project measures calibration, and measures what happens to
calibration when the model is transferred to a competition it never trained on.

### 1.2 Definition of done

The build is complete when all of the following hold:

1. `make reproduce` runs end to end from an empty `data/` directory and writes every
   number quoted in `README.md`.
2. The README contains no `TBD` placeholders.
3. `pytest` passes with ≥85% line coverage on `api/src/xdr/`.
4. `npm run build` and `npm run typecheck` pass in `web/`.
5. The dashboard renders a real scored moment, a real reliability diagram, and real
   transfer numbers against a locally running API.
6. `docs/DECISIONS.md` records every SHOULD that was deviated from and every default
   taken on an `[OPEN]` item.

### 1.3 What "good" looks like numerically

These are not targets to hit; they are ranges that indicate the pipeline is wired
correctly rather than producing garbage. If a result falls far outside, suspect a bug
before suspecting a discovery.

| Quantity | Plausible range | Interpretation if outside |
|---|---|---|
| Scoring-model Brier (in-competition) | 0.005 – 0.04 | Label leakage if far below; broken features if far above |
| ECE, in-competition, post-calibration | < 0.01 | Calibration layer not fitted or fitted on train |
| ECE, transfer, pre-recalibration | 1.5× – 5× in-competition | No degradation at all means the splits are contaminated |
| Mean xDR per 90 | 0.05 – 0.60 xG-equivalents | Above ~1.0 means the counterfactual head is unbounded |
| Fraction of options returned unscored | 15% – 50% | Near 0% means the support gate is not working |

**A result of "no significant degradation on transfer" is a valid finding and MUST be
reported as such.** Do not tune until degradation appears.

---

## 2. Non-goals

Explicitly out of scope. Do not build these, do not scaffold for them.

- Player ratings, rankings, or leaderboards. xDR is a per-action measurement; aggregating
  it into a player ranking invites causal claims the data does not support.
- Live or in-match inference. The corpus is fixed and historical.
- User accounts, authentication, persistence of user state.
- Any data source other than StatsBomb open data.
- Multi-tournament training pools in v1. One train competition, one target competition.
- Mobile-native anything. The dashboard is responsive; that is the extent of it.

---

## 3. Coordinate systems — read this before touching any geometry

This is the single most likely source of silent, plausible-looking wrong output.

Three coordinate systems appear in this project:

| System | Extent | Origin | y direction | Where it appears |
|---|---|---|---|---|
| StatsBomb raw | 120 × 80 | top-left | increases downward | `events` JSON, `three-sixty` JSON |
| SPADL | 105 × 68 | bottom-left | increases upward | `socceraction` output, all internal maths |
| SVG viewport | 105 × 68 | top-left | increases downward | `web/components/Pitch.tsx` |

**Contract:**

- Conversion from StatsBomb raw to SPADL MUST happen exactly once, in
  `xdr/data/spadl.py`, at ingest. Nothing downstream may see raw coordinates.
- Freeze-frame player locations arrive in StatsBomb raw units and MUST be converted
  with the same transform as event locations, in the same pass. **[VERIFY]** Confirm
  that `three-sixty` locations use the identical 120 × 80 frame as `events`; the
  StatsBomb documentation in `open-data/doc/` is authoritative.
- Attacking direction MUST be normalized left-to-right per possessing team. Without
  this, the same action appears at opposite ends of the pitch in different halves and
  the model learns which half it is.
- The API MUST emit SPADL coordinates only. The single y-flip into SVG space happens in
  the frontend, in `Pitch.tsx`, and nowhere else.
- Write a property test asserting round-trip conversion is idempotent and that a known
  landmark (penalty spot, 11 m from goal line, centred) maps correctly in both systems.

---

## 4. Repository structure

```
decision-regret/
├── README.md                     Findings first, stack last. No TBDs at completion.
├── SPEC.md                       This file.
├── LICENSE                       MIT.
├── Makefile                      Every pipeline stage is a target. `reproduce` chains them.
├── .gitignore
├── .github/workflows/ci.yml      api job (ruff + pytest), web job (typecheck + build)
│
├── configs/
│   ├── base.yaml                 Seeds, paths, model hyperparameters, gate thresholds
│   ├── transfer_euro24_to_weuro25.yaml   Primary transfer study
│   ├── transfer_euro20_to_euro24.yaml    Secondary: temporal, same population
│   └── ablation_no_encoder.yaml  IdentityEncoder swapped in, all else fixed
│
├── api/
│   ├── pyproject.toml            Package metadata, pinned deps, ruff + pytest config
│   ├── Dockerfile                CPU torch in its own layer, then app code
│   ├── src/xdr/
│   │   ├── config.py             Pydantic settings; loads YAML, resolves `extends`
│   │   ├── data/
│   │   │   ├── ingest.py         statsbombpy → raw Parquet. CLI entrypoint.
│   │   │   ├── spadl.py          Raw → SPADL. THE ONLY coordinate conversion site.
│   │   │   └── frames.py         three-sixty JSON → padded point-set tensors
│   │   ├── features/
│   │   │   ├── actions.py        SPADL action features (see §7.1)
│   │   │   ├── labels.py         VAEP scores/concedes labels over an n-action horizon
│   │   │   └── build.py          Orchestrates the above; writes feature Parquet. CLI.
│   │   ├── models/
│   │   │   ├── encoder.py        DeepSetsEncoder + IdentityEncoder  [IMPLEMENTED]
│   │   │   ├── heads.py          Scoring/conceding heads; joint module
│   │   │   ├── dataset.py        torch Dataset over the feature Parquet
│   │   │   ├── train.py          Training loop, early stopping, artifact write. CLI.
│   │   │   ├── calibrate.py      Isotonic fit + reliability write. CLI.
│   │   │   └── support.py        Out-of-distribution support scoring (see §9)
│   │   ├── counterfactual/
│   │   │   ├── options.py        Enumeration + regret arithmetic  [IMPLEMENTED]
│   │   │   └── score.py          Builds feature rows for hypothetical actions
│   │   ├── evaluation/
│   │   │   ├── metrics.py        Brier decomposition, ECE, bootstrap  [IMPLEMENTED]
│   │   │   ├── report.py         In-competition evaluation. CLI.
│   │   │   ├── transfer.py       Cross-competition study. CLI.
│   │   │   └── ablation.py       Encoder-contribution delta. CLI.
│   │   └── serve/
│   │       ├── app.py            FastAPI routes  [IMPLEMENTED]
│   │       ├── schemas.py        Pydantic response models  [IMPLEMENTED]
│   │       └── store.py          DuckDB read layer  [PARTIAL — see §11.3]
│   └── tests/
│       ├── test_encoder.py       Permutation + padding invariance  [IMPLEMENTED]
│       ├── test_metrics.py       Brier identity, ECE monotonicity  [IMPLEMENTED]
│       ├── test_counterfactual.py Enumeration + null-regret rules  [IMPLEMENTED]
│       ├── test_coordinates.py   Round-trip + landmark. WRITE THIS FIRST.
│       ├── test_labels.py        No future leakage across possession boundaries
│       ├── test_support.py       Gate fires on synthetic OOD points
│       └── test_api.py           Route contracts via httpx TestClient
│
├── web/
│   ├── package.json              Next 15, React 19, TypeScript. No CSS framework.
│   ├── next.config.mjs           Rewrites /api/* → API_ORIGIN
│   ├── tsconfig.json             strict: true
│   ├── vercel.json               Framework, region, security headers
│   ├── .env.example              API_ORIGIN
│   ├── app/
│   │   ├── layout.tsx            Fonts, metadata
│   │   ├── page.tsx              Dashboard  [IMPLEMENTED]
│   │   └── globals.css           Design tokens  [IMPLEMENTED]
│   ├── components/
│   │   ├── Pitch.tsx             The option fan. Signature element.  [IMPLEMENTED]
│   │   ├── OptionLedger.tsx      Ranked option table  [IMPLEMENTED]
│   │   ├── Reliability.tsx       Reliability diagram  [IMPLEMENTED]
│   │   └── MomentPicker.tsx      Client-side moment selector. TO BUILD.
│   └── lib/
│       ├── types.ts              Mirrors schemas.py  [IMPLEMENTED]
│       └── api.ts                Typed fetch wrappers  [IMPLEMENTED]
│
├── docs/
│   ├── ARCHITECTURE.md           Hosting split, storage choice  [IMPLEMENTED]
│   ├── DECISIONS.md              Append-only log. CREATE THIS.
│   └── DATA.md                   Competition IDs, coverage counts, provenance. CREATE.
│
├── data/                         gitignored
│   ├── raw/                      statsbombpy dumps
│   └── parquet/                  Normalized SPADL + features
├── artifacts/                    gitignored — model weights, xdr.duckdb, run.json
└── runs/                         gitignored — per-run metrics + config hash
```

**Naming conventions.** Python: `snake_case` modules, `PascalCase` classes, verbs for
CLI entrypoints (`build`, `train`, `report`). TypeScript: `PascalCase` components in
`components/`, `camelCase` functions in `lib/`. Config files named
`{purpose}_{source}_to_{target}.yaml`. Artifacts named `{stage}_{split}.{ext}`.

---

## 5. Tech stack and the tradeoffs behind it

| Layer | Choice | Rejected alternative | Why |
|---|---|---|---|
| Storage | DuckDB + Parquet | Postgres | Read-only analytical queries over a fixed corpus. No writers, no transactions. DuckDB is a file baked into the image — removes a network hop, a pool, and a service. Switch to Postgres the moment users write anything. |
| Action schema | SPADL via `socceraction` | Custom schema | Makes the baseline comparable to published VAEP numbers. A bespoke schema makes every result unfalsifiable. |
| Baseline value model | LightGBM | XGBoost / CatBoost | Fastest to fit on tabular features at this scale; native categorical handling. Difference from XGBoost here is noise. |
| Frame encoder | DeepSets (PyTorch) | Flattened MLP, CNN over rasterized pitch, GNN | The frame is an unordered variable-length set. Flattening imposes an ordering the data lacks. Rasterizing discards precision and inflates parameters. A GNN needs an edge definition that would be invented rather than observed. DeepSets is the minimal architecture with the correct inductive bias. |
| Calibration | Isotonic regression | Platt scaling | Non-parametric; makes no sigmoid-shape assumption. Needs more data than Platt — acceptable at ~10^5 actions. **If the calibration split is under ~2,000 rows, use Platt instead and note it.** |
| Uncertainty | k-NN distance in embedding space | Deep ensembles, MC dropout, conformal | Ensembles cost 5× training time for a 48-hour budget. k-NN support is cheap, explainable in one sentence, and sufficient to gate. See §9 for the upgrade path. |
| API | FastAPI | Flask, Django | Pydantic models are the same objects that define the frontend contract. Free OpenAPI schema. |
| Frontend | Next.js 15 App Router | Streamlit, Vite SPA | Streamlit reads as a shipped notebook and undercuts the engineering claim. App Router server components let the page fetch on the server, so the API URL never enters the client bundle. |
| Styling | Plain CSS + custom properties | Tailwind, CSS-in-JS | ~200 lines of tokens beats a build-step dependency for a five-component surface. Also produces a distinctive result rather than a utility-class default. |
| Charts | Hand-written SVG | Recharts, D3, Chart.js | Two charts, both bespoke. A charting library is 60 kB to render 20 circles and a polyline. |
| Frontend host | Vercel | Same host as API | See §12. |
| API host | Railway / Fly / Render | Vercel | Vercel's Python runtime cannot carry a torch install, and a per-cold-start model load makes the pitch view unusable. |

**Pinning.** All Python deps pinned to minor versions in `pyproject.toml`. `web/` uses
an exact-version `package-lock.json` committed to the repo. CI MUST use `npm ci`, not
`npm install`.

---

## 6. Data layer

### 6.1 Source

StatsBomb open data, accessed through `statsbombpy`. Data is provided under StatsBomb's
open data terms; attribution MUST appear in the README and in the dashboard footer.

### 6.2 Competition selection — verified IDs

The following pairs were confirmed present in `open-data/data/competitions.json` with
non-null `match_available_360`:

| Competition | `competition_id` | `season_id` | 360 |
|---|---|---|---|
| FIFA World Cup 2022 | 43 | 106 | yes |
| UEFA Euro 2024 | 55 | 282 | yes |
| UEFA Euro 2020 | 55 | 43 | yes |
| UEFA Women's Euro 2025 | 53 | 315 | yes |
| La Liga 2020/2021 | 11 | 90 | yes |
| Ligue 1 2022/2023 | 7 | 235 | yes |
| Ligue 1 2021/2022 | 7 | 108 | yes |
| MLS 2023 | 44 | 107 | yes |
| 1. Bundesliga 2023/2024 | 9 | 281 | yes |
| African Cup of Nations 2023 | 1267 | 107 | yes |

**Primary transfer study:** UEFA Euro 2024 (55/282) → UEFA Women's Euro 2025 (53/315).
Both confirmed 360, both international tournaments, roughly one year apart. The shift
under test is the playing population, with tournament structure and era held roughly
fixed.

**Secondary transfer study:** UEFA Euro 2020 (55/43) → UEFA Euro 2024 (55/282).
Temporal shift within the same population. Serves as a control: degradation here should
be smaller than in the primary study. If it is larger, the primary result is not about
population.

### 6.3 The 360-availability trap — MUST READ

Many entries in `competitions.json` carry a non-null `match_updated_360` while
`match_available_360` is `null`. These competitions have **no 360 data**. Filtering on
the wrong field silently yields an empty freeze-frame set and a pipeline that appears
to run while training on nothing.

```python
# CORRECT
available = comps[comps["match_available_360"].notna()]

# WRONG — matches dozens of competitions that have no 360 files
available = comps[comps["match_updated_360"].notna()]
```

Assert non-empty frame coverage immediately after ingest and fail loudly otherwise.

### 6.4 Ingest contract

`xdr/data/ingest.py`:

1. Fetch `competitions()`, filter on `match_available_360.notna()`, intersect with the
   configured competition list. Fail with a clear message if a configured pair is absent.
2. For each match: fetch `events`, `lineups`, `frames`. Write raw Parquet under
   `data/raw/{competition_id}_{season_id}/`.
3. Rate-limit and cache. `statsbombpy` reads from GitHub; a re-run MUST NOT refetch.
   Key the cache on `(competition_id, season_id, match_id)`.
4. Write `docs/DATA.md` with per-competition match counts, event counts, frame counts,
   and the fraction of events carrying a freeze frame.

**[VERIFY]** `statsbombpy`'s function for freeze frames has been named both `frames()`
and `three_sixty()` across versions. Confirm against the installed version's `__all__`
rather than assuming.

### 6.5 Storage schemas

Parquet, partitioned by `competition_id/season_id/match_id`.

```
actions/          # SPADL, one row per action
  action_id, match_id, period_id, time_seconds, team_id, player_id,
  start_x, start_y, end_x, end_y, type_id, result_id, bodypart_id

frames/           # one row per action that has a freeze frame
  action_id, match_id, n_players, points  (list<struct<x,y,teammate,actor>>)

features/         # model input, one row per action
  action_id, <action features §7.1>, frame_ref, label_scores, label_concedes

moments/          # scored output, one row per action with a counterfactual set
  moment_id, match_id, minute, second, team, player_name,
  ball_x, ball_y, chosen_json, best_json, regret, options_json, unscored_count
```

`moments` is the only table the API reads. It is denormalized on purpose: the serving
path does no joins, so a cold container answers in one query.

---

## 7. Features

### 7.1 Action features (deterministic branch)

Per action, plus the same block for the previous `action_window` actions (default 3):

- `start_x, start_y, end_x, end_y` (SPADL units)
- `dx, dy`, Euclidean length, angle
- distance to goal and angle to goal, from both start and end
- `type_id`, `result_id`, `bodypart_id` — categorical
- `time_seconds`, `period_id`, seconds remaining in period
- goal difference from the acting team's perspective
- `same_team_as_previous` flag, time delta from previous action

Use `socceraction.vaep.features` for the standard blocks rather than reimplementing.
**[VERIFY]** The module path has moved between `socceraction` versions
(`socceraction.vaep.features` vs `socceraction.spadl` re-exports). Confirm against the
installed version.

### 7.2 Frame features (learned branch)

Per action with a freeze frame, a padded tensor of shape `(max_players, 4)`:

- `x, y` in SPADL units, normalized to `[0, 1]` by dividing by 105 and 68
- `teammate` flag, `actor` flag

Plus a `(max_players,)` boolean mask. `max_players` default 22.

**Padding MUST be masked in the encoder.** Unmasked padding lets the model learn frame
cardinality, which tracks camera coverage rather than football. `test_encoder.py`
already asserts this; do not weaken that test.

Actions without a freeze frame are excluded from the 360 branch entirely. Do not
impute. Record the excluded count in `docs/DATA.md`.

### 7.3 Labels

VAEP formulation. For each action *a*:

- `label_scores` = 1 if the acting team scores within the next `horizon` actions
- `label_concedes` = 1 if the acting team concedes within the next `horizon` actions

Default `horizon` = 10. Labels MUST NOT cross match boundaries. `test_labels.py` MUST
assert that the last `horizon` actions of every match have correctly truncated
lookahead, and that no label depends on an action from a different match.

Class imbalance is severe (roughly 1–2% positive). Do not resample. Resampling
destroys calibration, which is the point of the project. Use the natural base rate and
let isotonic regression handle the rest.

---

## 8. Model

### 8.1 Architecture

```
action features ──────────────────────────┐
                                          ├─→ concat ─→ MLP head ─→ P(scores)
freeze frame ─→ DeepSetsEncoder ─→ (32,) ─┘                     ─→ P(concedes)
```

Two heads share the trunk. Train jointly with summed binary cross-entropy.

`DeepSetsEncoder` is implemented. Do not modify it without a failing test justifying
the change; its two invariance properties are the reason it was chosen.

### 8.2 Baseline

Fit LightGBM on action features alone, no encoder. This is the number the deep model
must beat, and it is reported in the README whether or not it does.

### 8.3 Training

- Split by **match**, never by action. Actions within a match are dependent; an
  action-level split leaks and produces implausibly good numbers.
- Train / validation / calibration / test = 60 / 15 / 10 / 15 by match.
- The calibration split MUST be disjoint from both training and test.
- Early stopping on validation loss, patience 5.
- AdamW, lr 1e-3, batch 512, max 30 epochs.
- Seed everything: Python `random`, NumPy, torch, and `torch.use_deterministic_algorithms(True)`.
- Write `runs/{run_id}/` containing the resolved config, its SHA-256, git commit,
  metrics JSON, and weights.

---

## 9. Support scoring — the honest-extrapolation mechanism

This is the most technically delicate component and the one most likely to be built
wrong. It is what separates this project from a demo that confidently scores fantasy.

### 9.1 The problem

The value model is fitted on actions that occurred. Counterfactual scoring asks it about
actions that did not. Some of those hypotheticals lie in regions of feature space the
training data never covered — a 40-metre through ball into a configuration no player
attempted. The model will still emit a number. That number is meaningless.

### 9.2 v1 mechanism

For a candidate action's feature vector *v*:

1. Compute mean Euclidean distance from *v* to its k = 20 nearest neighbours in the
   **training set**, in the concatenated feature space (action features standardized,
   plus the frame embedding).
2. Convert to a percentile rank against the distribution of that same statistic computed
   over the **validation set**. Validation, not training — training-set self-distances
   are biased low.
3. `support = 1 − percentile_rank`, so 1.0 means densely covered and 0.0 means far from
   anything observed.
4. Options with `support < min_support` (default 0.35) are returned with `value = null`
   and `scored = false`.

Use `faiss` or `sklearn.neighbors.NearestNeighbors` with a fitted index; do not compute
pairwise distances at request time.

### 9.3 Validating the gate

`test_support.py` MUST assert that synthetic clearly-OOD points — a shot from inside the
defending penalty area, a pass to a coordinate off the pitch — receive support below the
floor. If the gate does not fire on those, it is not working, and every regret number
downstream is unsupported.

Also report in the README: the fraction of enumerated options that were gated. Near 0%
means the gate is inert. Near 100% means the threshold is wrong.

### 9.4 Upgrade path

If time remains after the primary result: fit a 5-member deep ensemble and use
prediction variance as a second support signal. Report agreement between the two
signals. Do not attempt this before the transfer study is complete.

**[OPEN]** The 0.35 floor is a guess. Default to it, but the correct procedure is to
choose it so the gated fraction lands in 15–50%, then report the chosen value and the
resulting fraction. Record the calibration procedure in `docs/DECISIONS.md`.

---

## 10. Counterfactual enumeration

`options.py` is implemented. `score.py` is not. The missing piece is constructing a
feature row for a hypothetical action.

For each candidate option, build the feature vector as if that action had been taken:

- `start_x, start_y` = the real ball position (unchanged)
- `end_x, end_y` = the candidate's destination
- `type_id` = pass / carry / shot as appropriate
- `bodypart_id` = **[OPEN]** foot by default; the actual body part is unknowable for an
  action that did not happen. Default to foot and note the assumption.
- `result_id` = **this is the subtlety.** A counterfactual has no observed outcome. Score
  the *expected* value marginalizing over success and failure, weighted by a
  completion-probability model, rather than assuming success. Assuming success inflates
  every alternative and manufactures regret everywhere.
- Preceding-action context = unchanged from the real sequence
- Frame embedding = unchanged (the frame is the state before the action)

**[OPEN]** The completion-probability model is an additional component not in the
original scope. Two paths: (a) train a small pass-completion model on the same corpus —
correct, costs perhaps four hours; (b) skip it in v1, assume success, and state
prominently in the README and dashboard that regret is an **upper bound** on forgone
value. Default to (b) for the first working end-to-end pass, then implement (a) if time
allows. Do not silently do (b) and describe it as (a).

Rules already encoded in `options.py`, do not weaken:

- Only *visible* teammates are pass targets. An off-camera teammate is not a declined
  option; counting it manufactures regret out of camera framing.
- Regret is clipped at zero. The model cannot claim a player beat every alternative it
  could enumerate.
- Regret is `null` when either side of the subtraction is unsupported.

---

## 11. Evaluation

### 11.1 Metrics

Implemented in `metrics.py`: Murphy three-term Brier decomposition, ECE,
count-weighted reliability curve, percentile bootstrap.

**The bootstrap resampling unit MUST be the match, not the action.** Actions within a
match are correlated; action-level resampling produces intervals several times too
narrow. This is the single most common statistical error in this genre of project.

### 11.2 Required reports

1. **In-competition** (`report.py`): Brier decomposition, ECE, reliability curve on the
   held-out test matches. Deep model and LightGBM baseline, side by side.
2. **Transfer** (`transfer.py`): the same metrics on the target competition, before and
   after isotonic recalibration on a held-out 20% of the target. Report ECE degradation
   and recovered fraction.
3. **Ablation** (`ablation.py`): `IdentityEncoder` swapped in, every other component
   fixed and the same seed. The delta is the encoder's measured contribution. **Report
   this number whether or not it favours the encoder.**
4. **xDR distribution**: mean per 90 with a match-level bootstrap CI, plus the
   distribution's shape. A long right tail is expected; a symmetric distribution
   suggests the counterfactual head is not discriminating.

### 11.3 `store.py` completion

Two functions currently raise `NotImplementedError`: `list_moments` and `get_moment`.
Both read the denormalized `moments` table and deserialize the JSON columns into the
Pydantic schemas. `get_moment("latest")` MUST resolve to the highest-regret scored
moment so the dashboard has a sensible default.

---

## 12. API contract

FastAPI, read-only, no authentication. Routes are implemented; behaviour is specified
here.

| Route | Returns | Notes |
|---|---|---|
| `GET /health` | status, model version, run id | Must work with no artifacts present |
| `GET /matches` | match summaries, ordered by total regret | |
| `GET /matches/{id}/moments` | moment list, filterable by `min_regret` | 404 if none scored |
| `GET /moments/{id}` | full moment with frame, options, regret | `"latest"` is a valid id |
| `GET /calibration?split=` | reliability report | 404 on unknown split |
| `GET /transfer` | source vs target vs recalibrated | |

Response models live in `schemas.py` and are mirrored in `web/lib/types.ts`. **Drift
between those two files is a defect the type checker cannot catch.** When you change
one, change the other in the same commit.

`value: null` and `regret: null` are meaningful states, not errors. Never coerce them
to zero.

---

## 13. Frontend

Implemented: `page.tsx`, `Pitch.tsx`, `OptionLedger.tsx`, `Reliability.tsx`,
`globals.css`, `lib/`.

To build: `MomentPicker.tsx` — a client component listing moments for a match, sorted by
regret, updating the `?moment=` search param. Keep it a client component; everything else
stays a server component so the API URL never enters the client bundle.

**Design constraints that are not negotiable:**

- Unscored options render as dashed outlines. Never hide them. Hiding them makes the fan
  look more confident than the model is, which inverts the project's thesis.
- The `null` regret state renders as "gap unsupported", not as `0.0000`.
- Reliability diagram marker area encodes bin count. Equal-sized markers over unequal
  bins is the standard way this chart misleads.
- Palette is diverging because the quantity is a signed gap. Amber is the action taken,
  teal is the best available. Do not add a third accent.
- Responsive to 380 px. Visible keyboard focus. `prefers-reduced-motion` respected.

---

## 14. Deployment

```
Browser
  │  same-origin /api/*
  ▼
Vercel (Next.js)  ──rewrite──▶  Container host (FastAPI + torch + DuckDB)
```

Vercel hosts the frontend only. `next.config.mjs` rewrites `/api/*` to `API_ORIGIN`.
Consequences: no CORS preflight on the interactive path, the backend URL stays out of the
client bundle, the API can move without a frontend redeploy.

The API image bakes `artifacts/xdr.duckdb` and the model weights. It is stateless and
scales to zero.

**[VERIFY]** Confirm the CPU-only torch wheel index URL in the Dockerfile is current, and
confirm the resulting image size against the chosen host's limit before relying on it.
A CPU torch install is on the order of several hundred megabytes.

**[OPEN]** Container host not chosen. Railway is the default assumption in the config
files. Fly and Render are equivalent for this workload.

---

## 15. Testing requirements

Coverage floor 85% on `api/src/xdr/`. Beyond coverage, these specific properties MUST be
asserted:

| Test | Property | Why it matters |
|---|---|---|
| `test_coordinates.py` | Round-trip idempotence; penalty spot lands correctly in both systems | Silent geometry errors produce plausible wrong output |
| `test_labels.py` | No lookahead across match boundaries | Leakage here makes every metric meaningless |
| `test_encoder.py` | Permutation invariance; padding invisibility | The architectural justification |
| `test_support.py` | Gate fires on synthetic OOD points | The honesty mechanism |
| `test_counterfactual.py` | Regret never negative; null propagates | Already implemented |
| `test_metrics.py` | Brier identity holds; ECE monotone under inflation | Already implemented |
| `test_api.py` | Every route's response validates against its schema | Catches schema drift |

**Write `test_coordinates.py` and `test_labels.py` before their implementations.** Those
two failure modes are silent, and by the time they surface downstream you will have built
on top of them.

---

## 16. Build order

Each milestone has an acceptance criterion. Do not proceed past a failing one.

**M1 — Data spine.** Ingest, SPADL conversion, coordinate tests.
*Accept:* `docs/DATA.md` shows non-zero frame coverage for both configured competitions;
`test_coordinates.py` passes.

**M2 — Features and labels.** Action features, frame tensors, VAEP labels.
*Accept:* `test_labels.py` passes; positive class rate is between 0.5% and 3%.

**M3 — Baseline.** LightGBM on action features, match-level split, Brier reported.
*Accept:* test Brier falls in the §1.3 range. **This alone is a defensible artifact.
If everything after this fails, the project still stands.**

**M4 — Calibration.** Isotonic fit, reliability diagram, decomposition.
*Accept:* post-calibration ECE < 0.01 in-competition.

**M5 — Transfer study.** Train on Euro 2024, evaluate on Women's Euro 2025, recalibrate.
*Accept:* `artifacts/transfer.json` written; degradation and recovered fraction computed.
**This is the resume line. Everything after it is upside.**

**M6 — Encoder.** DeepSets branch, joint training, ablation against M3.
*Accept:* ablation delta reported with a CI, in whichever direction it falls.

**M7 — Support gate.** k-NN scoring, threshold calibration, gate tests.
*Accept:* gated fraction between 15% and 50%; `test_support.py` passes.

**M8 — Counterfactuals.** Option feature construction, scoring, `moments` table.
*Accept:* a real moment renders end to end in the dashboard.

**M9 — Polish.** `MomentPicker`, README numbers, deployment, CI green.

If time runs short, ship M1–M5 and say so. A completed calibration and transfer study
with no counterfactual layer is a stronger artifact than a counterfactual demo with no
evidence its numbers mean anything.

---

## 17. Consolidated verification list

Confirm each before depending on it. Record findings in `docs/DECISIONS.md`.

1. **[VERIFY]** `three-sixty` freeze-frame coordinates use the same 120 × 80 frame as
   `events`. Source: `open-data/doc/` in the StatsBomb repository.
2. **[VERIFY]** `statsbombpy`'s freeze-frame accessor name in the installed version.
3. **[VERIFY]** `socceraction` module paths for features and labels in the installed
   version; these have moved across releases.
4. **[VERIFY]** FIFA Women's World Cup 2023 competition and season IDs, **if** you choose
   it over the Women's Euro 2025 pair. It was not visible in the portion of
   `competitions.json` confirmed at drafting time. The Euro 2024 → Women's Euro 2025 pair
   above **is** confirmed; prefer it.
5. **[VERIFY]** Every configured `(competition_id, season_id)` resolves and has non-null
   `match_available_360` at build time. Coverage changes between repository updates.
6. **[VERIFY]** CPU torch wheel index and resulting image size against the host limit.
7. **[VERIFY]** Whether `socceraction`'s bundled VAEP feature set already includes any
   freeze-frame-derived column. If it does, the encoder ablation is confounded and the
   overlapping column must be removed from the deterministic branch.

## 18. Decisions required from the repository owner

Defaults are given for every item; none of these block the build.

1. **Transfer pair.** Default: Euro 2024 → Women's Euro 2025, both confirmed 360.
   Alternative: World Cup 2022 → Women's World Cup 2023, contingent on item 4 above.
2. **Counterfactual outcome handling.** Default: assume success in v1, label regret an
   upper bound prominently. Upgrade: train a completion-probability model. This is the
   largest single quality decision in the project.
3. **Support threshold.** Default 0.35, then tuned so the gated fraction lands in
   15–50%.
4. **Container host.** Default Railway. Fly and Render are equivalent here.
5. **Scope floor.** Confirm that shipping M1–M5 without the counterfactual layer is
   acceptable if time runs out. The spec assumes yes.
6. **Public repository.** Assumed public with StatsBomb attribution. Confirm before the
   first push, since the README's findings are the point of the repository.
