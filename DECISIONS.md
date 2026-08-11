# Decision log

Append-only. One entry per deviation from `SPEC.md`, per `[OPEN]` default taken, and
per `[VERIFY]` item resolved. Newest last.

Format:

```
## YYYY-MM-DD — short title
**Context:** what prompted the decision
**Decision:** what was chosen
**Rejected:** what was not, and why
**Reversibility:** cheap / moderate / expensive
```

---

## 2026-08-07 — Repository scaffolded from SPEC.md
**Context:** Initial commit. Encoder, metrics, counterfactual arithmetic, API surface,
and dashboard implemented; data, feature, and training entrypoints stubbed.
**Decision:** Stub the pipeline entrypoints rather than write them against an assumed
data shape.
**Rejected:** Writing ingest and feature code before inspecting real freeze-frame JSON —
that code would be rewritten on first contact.
**Reversibility:** cheap

---

## 2026-08-08 — First live end-to-end pass: scope, and what M1-M4 actually shipped

**Context:** Nothing beyond README/SPEC/DECISIONS/one config file existed on disk at the
start of this session — despite the note above, no encoder/metrics/API/dashboard code
had actually been committed. Building the *entire* SPEC (M1-M9: real ingest, both feature
branches, calibration, transfer study, support gate, counterfactual enumeration, full
frontend, 85% coverage, CI, Docker, Vercel) in one pass was not realistic. Scoped this
pass to: real StatsBomb Euro 2024 data, action-features-only LightGBM baseline (no
DeepSets encoder), isotonic calibration, a real FastAPI + Next.js dashboard serving
genuine model output. DeepSets encoder (M6), support gate (M7), counterfactual
enumeration (M8), and the cross-tournament transfer study (M5) are explicitly deferred,
not silently skipped -- every route and schema is already shaped for them (see below).
**Decision:** Ship a smaller, fully real, fully verified vertical slice rather than a
wider one built partly on assumption.
**Rejected:** (a) Attempting all of M1-M9 in one pass — would not have finished any
stage properly. (b) Building all pipeline stages against synthetic data for speed —
would not have surfaced the real bugs this pass actually caught (see entries below).
**Reversibility:** cheap — each deferred milestone extends the existing schema/routes
rather than replacing them.

---

## 2026-08-08 — Toolchain: Python 3.12 venv and Node.js, not the machine's default Python 3.13

**Context:** The machine had Python 3.13 and no Node.js. `socceraction` pins
`numpy<2,pandas<2`, neither of which ship a Python 3.13 wheel, and there is no C
compiler on this machine to build them from source (`numpy` build failed looking for
`cl`/`gcc`/`clang`). Node.js was required to run the Next.js dashboard at all.
**Decision:** Installed Node.js LTS and Python 3.12 via `winget` (both approved by the
user before installing), and built the API venv on 3.12 instead of the machine-default
3.13.
**Rejected:** Building a compiler toolchain to compile numpy/pandas from source under
3.13 — far more invasive than installing an additional Python version.
**Reversibility:** cheap (both are additive installs; nothing about the existing system
Python 3.13 was touched).

---

## 2026-08-08 — Use socceraction's own StatsBomb loader + SPADL converter, not statsbombpy directly

**Context:** SPEC.md §17 flagged two [VERIFY] items: whether `statsbombpy`'s freeze-frame
accessor is named `frames()` or `three_sixty()`, and whether `socceraction`'s feature/label
module paths had moved. Live testing (statsbombpy 1.22.0, socceraction 1.5.3) found a
better answer than resolving either question: `socceraction.data.statsbomb.StatsBombLoader`
wraps the raw StatsBomb API itself and returns events already in the exact schema
`socceraction.spadl.statsbomb.convert_to_actions` expects, including a `freeze_frame_360`
column with per-event 360 data (not just shots). This sidesteps the naming question
entirely rather than answering it.
**Decision:** `xdr/data/ingest.py` uses `StatsBombLoader` for games/events/frames, and
`statsbombpy.sb.competitions()` only for the `match_available_360` coverage check (which
needs the wider, cross-competition competitions table).
**Rejected:** Wiring up `statsbombpy` events/frames directly and hand-mapping columns into
what `convert_to_actions` expects — confirmed harder and more failure-prone once both
paths were tried live (see the Parquet caching bug below, which happened precisely because
of a manual dtype-preserving mapping step).
**Reversibility:** moderate — ingest.py would need rewriting, but nothing downstream
depends on which loader produced the actions/frames Parquet.

---

## 2026-08-08 — Coordinate conversion: cell-offset correction, verified against socceraction's own formula

**Context:** SPEC.md §3 names this "the single most likely source of silent, plausible-
looking wrong output." The first version of `sb_to_spadl` used a plain linear scale
(`x * 105/120`). Reading `socceraction.spadl.statsbomb._convert_locations` before relying
on the naive version showed StatsBomb coordinates address grid *cells* (1-yard or, at
"high fidelity," 0.1-yard), not points — the cell's centre must be subtracted before
scaling, or every point lands about half a cell (~0.4m) off. This matters here because
freeze-frame points are converted by `xdr`'s own code (360 data isn't part of SPADL, so
`socceraction`'s converter never touches it) and must land in the same coordinate frame as
the actions `socceraction` converts, or player dots visually misalign with the action they
surround.
**Decision:** `sb_to_spadl`/`spadl_to_sb` replicate socceraction's cell-aware formula
exactly, parameterized by `fidelity_version` (default 2, i.e. 0.1-yard cells, matching
what `convert_to_actions` inferred for real Euro 2024 matches). `test_coordinates.py`
asserts the correction is actually present (`test_cell_offset_is_applied_not_a_plain_linear_scale`),
not just that some plausible-looking numbers come out.
**Rejected:** Keeping the plain linear scale — it produced plausible-looking numbers
(penalty spot at x=94.5 instead of the correct ~94.46) that would have passed a
loose visual check.
**Reversibility:** cheap — pure function, isolated, already covered by property tests.

---

## 2026-08-08 — Attacking-direction normalization: home/away mirror, not period-aware — verified empirically, not assumed

**Context:** SPEC.md §3 warns that without left-to-right normalization "the same action
appears at opposite ends of the pitch in different halves and the model learns which half
it is." `socceraction`'s own `_fix_direction_of_play` only mirrors by team (home vs away),
never by period — which looks wrong for a sport where teams swap ends at half-time, if
StatsBomb's raw coordinates were pitch-fixed the way broadcast tracking data usually is.
Rather than assume this is a bug in a library SPEC.md picked specifically for its
comparability to published numbers, checked it against real data first: computed mean
shot `start_x` for the home and away side of Portugal vs Czech Republic (Euro 2024,
game_id 3930166), split by period. Home team shots clustered near x=89-93 in *both*
period 1 and period 2; away team shots clustered near x=15 in *both* periods. Direction
was already consistent across periods without any period-aware correction.
**Decision:** Use `socceraction`'s home/away-only mirror as-is, for both actions (via
`convert_to_actions`) and freeze-frame points (via `xdr.data.spadl.normalize_attacking_direction`,
called with `attacks_right = team_id == home_team_id`, no period term). Documented in
`docs/DATA.md`-adjacent code comments rather than left as an unstated assumption.
**Rejected:** Adding a custom period-aware flip pre-emptively "to be safe" — would have
double-mirrored data that was already correct, silently reintroducing the exact bug
SPEC.md warns about, in the opposite direction.
**Reversibility:** cheap — the empirical check is repeatable against any new competition
before trusting this holds there too.

---

## 2026-08-08 — Raw event cache: pickle, not Parquet (deviates from SPEC.md §6.4)

**Context:** SPEC.md §6.4 says "write raw Parquet." The first ingest run produced 109,985
actions; after adding `socceraction.spadl.add_names` (needed for `type_name`/`result_name`/
`bodypart_name`, which the VAEP feature and label functions require), a second run against
the *same* cached raw events produced 140,430 actions from the identical 51 matches —
a ~28% jump with no code change to the conversion step itself. Root cause: the raw event
cache round-tripped StatsBomb's `location` column (and other list/dict columns) through
Parquet, which silently turned Python lists into numpy arrays on reload.
`convert_to_actions`'s interception/dribble-insertion logic branches on
`isinstance(loc, list)`, so the *same* match produced a different action count depending
on whether its events came from a fresh fetch or a Parquet-cached reload — a correctness
bug that would have silently propagated into every downstream number.
**Decision:** Cache raw events per match as a pickle (`events.pkl`), which preserves
Python object identity exactly, instead of Parquet. Verified the fix by clearing the cache
and re-ingesting: action count returned to 109,985, matching the original (correct,
before `add_names`) run.
**Rejected:** Keeping Parquet and JSON-encoding every list/dict-valued column by name —
tried this first (encoding four columns); it missed `location` itself, which isn't
obviously a "nested" column at the pandas dtype level (`dtype=object` either way), and is
exactly the kind of column that's easy to not think to check.
**Reversibility:** cheap — cache format is an implementation detail; deleting
`data/raw/**/events.pkl` and re-running `make ingest` regenerates it.

---

## 2026-08-08 — Match-level bootstrap: resampled frame keeps its original index

**Context:** `evaluation/metrics.match_level_bootstrap` originally concatenated resampled
matches with `ignore_index=True`. `evaluation/report.py`'s `metric_fn` looks calibrated
predictions up by row index (computed once, outside the bootstrap loop, for speed) — with
indices renumbered on every draw, that lookup raised `KeyError` on the very first real
`make reproduce` run of `evaluation/report.py` against actual test-split data.
**Decision:** Resampled frames keep the original row index (duplicated, not renumbered,
when a match is drawn more than once). `test_match_level_bootstrap_widens_interval_vs_action_level`
still passes unchanged since it reads columns, not the index.
**Rejected:** Re-deriving predictions inside the bootstrap loop instead of indexing into a
precomputed Series — correct either way, but ~2000x more prediction calls per report.
**Reversibility:** cheap.

---

## 2026-08-08 — `moments` table ships now with `best`/`regret`/`options` genuinely null, not deferred silently

**Context:** SPEC.md §12/§6.5 specifies a `moments` table and API schema built around the
counterfactual layer (`best`, `regret`, `options`, `unscored_count`), which is M8 work not
built in this pass. `GET /matches` is specified as "ordered by total regret."
**Decision:** Shipped the full schema now, with `best: null`, `regret: null`,
`options: []`, `unscored_count: 0` for every moment — not omitted fields, not a
placeholder `0.0`. `/matches` and moment ranking use mean/predicted action value as the
closest honest stand-in for "total regret" until M8 exists, documented inline in
`serve/store.py` and surfaced in the dashboard copy ("regret ranking arrives with the
counterfactual layer") rather than left for a user to discover by reading source.
`GET /matches/{id}/moments` also gained a `limit` param (default 100): with ~2,000 actions
per match and no counterfactual layer yet to naturally narrow "moments" down to ones with
an interesting declined alternative, the unfiltered list is every action in the match,
which is real data but not a usable picker.
**Rejected:** Leaving the counterfactual fields out of the schema entirely until M8 —
would mean an API-breaking change later instead of an additive one (nulls becoming real
values).
**Reversibility:** cheap — M8 fills in real values for fields that already exist and are
already rendered correctly (dashed/unsupported states) end to end.

---

## 2026-08-08 — Known gaps carried forward from this pass

Recorded together rather than as separate entries since none required a design decision
so much as ran out of session time:

- **Test coverage is ~59% on `api/src/xdr/`, not the ≥85% SPEC.md §1.2 requires.**
  `data/ingest.py`, `evaluation/report.py`, and parts of `models/calibrate.py` and
  `serve/store.py` are exercised end-to-end against real data (see the numbers below) but
  not by the pytest suite, because they're thin orchestration over network/CLI calls.
  `test_coordinates.py`, `test_labels.py`, `test_metrics.py`, `test_api.py`,
  `test_config.py`, `test_features.py`, and `test_store.py`'s read-path tests do exist and
  pass (41 passed, 1 skipped).
- **One test is skipped, not fixed:** `lightgbm` 4.7.0's pandas-DataFrame Dataset
  construction (its Arrow C Data Interface path) segfaults natively on this Windows/
  Python 3.12 environment for some small single/narrow-column synthetic DataFrames,
  reproduced with a bare `lgb.Dataset(pd.DataFrame(...), ...)` outside any xdr code. The
  identical code path trained successfully on the real 109,985-row/173-column Euro 2024
  feature table (this is the run `artifacts/` currently holds). Not root-caused further;
  flagged here rather than worked around in `train.py` with no time left to re-verify the
  real training path afterward.
- **`make` is not installed on this machine.** The `Makefile` targets are correct and
  were exercised by running their underlying commands directly; `make reproduce` itself
  was not run as a literal command.
- **M5 (transfer study), M6 (DeepSets encoder), M7 (support gate), M8 (counterfactual
  enumeration) are not built.** `configs/transfer_euro24_to_weuro25.yaml`,
  `xdr/models/encoder.py`, `xdr/models/support.py`, and `xdr/counterfactual/` remain to be
  written; nothing in this pass assumes they won't be.
- **Container/CI/Docker/Vercel deployment (M9, SPEC.md §14) not attempted.**

**What this pass did verify, live, against real data (UEFA Euro 2024, 55/282):** 51
matches ingested, 109,985 actions, 86.8% carry a freeze frame; `label_scores` positive
rate 0.86% (SPEC.md §1.3 target: 0.5-3%); in-competition test-split Brier 0.00611
(calibrated), ECE 0.00235 (target: <0.01); FastAPI serving real scored actions and a real
reliability diagram; Next.js dashboard rendering all of it live in a browser, including
the interactive moment picker and pitch view.

---

## 2026-08-08 — M5: transfer study reuses the source model rather than retraining

**Context:** `configs/transfer_euro24_to_weuro25.yaml`'s `data.competitions.train` list is
identical to `base.yaml`'s (same competition, same seed), so a match-level split computed
from either config produces the same partition of the same matches.
**Decision:** `evaluation/transfer.py` does not retrain. It loads the LightGBM boosters and
source calibrators `artifacts/` already holds from the ordinary `train`/`calibrate` run,
applies them to Women's Euro 2025 (ingested and feature-built the same way as the source
competition), and only fits something new for the recalibration step: an isotonic (or
Platt, if under 2,000 rows) calibrator on a held-out 20% of the target, evaluated on the
disjoint remaining 80% (`test_transfer.py` asserts the split is disjoint).
**Rejected:** A separate training run under the transfer config — would have produced an
identical model at the cost of a second multi-minute training pass, and invited the two
configs' splits to silently drift apart in the future.
**Reversibility:** cheap.
**Result:** `label_scores` ECE degrades 3.21x on transfer (SPEC.md §1.3's plausible range
is 1.5-5x — inside it) and recalibration recovers 45.8% of the gap. `label_concedes`
degrades a smaller 1.41x and recalibration recovers all of it and then some (206.9%,
i.e. the recalibrated target ECE ends up below the source's own in-competition figure).
Both are reported as genuine findings per SPEC.md §1.3 ("no significant degradation is a
valid finding"), not tuned toward a preferred outcome.

---

## 2026-08-08 — M7: support gate omits the frame-embedding half of SPEC.md §9.2's space

**Context:** SPEC.md §9.2 specifies the k-NN reference space as "action features
standardized, plus the frame embedding." The frame embedding comes from the DeepSets
encoder (M6), which remains deferred (see below).
**Decision:** Fit the gate on the standardized numeric action-feature space alone (155
columns — every `run_meta["feature_columns"]` entry that isn't one of socceraction's raw
categorical columns, keeping the already-numeric `*_onehot` counterparts instead).
`numeric_feature_columns` takes whatever columns it's given, so adding a frame embedding
later is a call-site change, not a rewrite.
**Rejected:** Blocking the support gate on M6 — would have meant no counterfactual layer
at all this pass, for a partial-space gate that still catches the OOD case SPEC.md §9.3
requires (`test_support.py`: a point far outside the training cluster in every action-
feature dimension scores 0.0).
**Reversibility:** cheap, but the gap is real, not silent: a candidate that's bizarre only
in its player configuration (e.g. a pass into a crowd of covering defenders that geometry
alone doesn't flag) won't be caught until M6 lands.
**Result on real data:** k=20, 155 numeric columns. Real historical (in-distribution)
test-split actions gate at 36.06%, matching the ~35% (`min_support`) the percentile-rank
construction predicts for same-distribution data by itself — confirmed by direct
calculation, not just observation, and documented in `support.py`'s own inline comment so
a future reader doesn't mistake this for "the gate is too aggressive." (An earlier version
of this same comment claimed the opposite — "expect near 0%" — before working through why
that's mathematically wrong for a percentile-rank gate; caught before it shipped.)

---

## 2026-08-08 — M8: counterfactual outcome handling takes SPEC.md §10's sanctioned default (b)

**Context:** SPEC.md §10 flags this as "the largest single quality decision in the
project": score every hypothetical action assuming it succeeds (default b, an explicit
upper bound), or train a completion-probability model to marginalize over (default a,
"correct, costs perhaps four hours").
**Decision:** Default (b) for this pass. Every candidate's `result_id` is set to `success`;
`value`/`regret` are upper bounds on forgone value, not point estimates. Stated in
`counterfactual/score.py`'s module docstring, in this entry, and in the README — SPEC.md
§10 is explicit that silently doing (b) and describing it as (a) is not acceptable, and it
would have been easy to let "regret" read as a point estimate without the caveat attached
everywhere it's surfaced.
**Rejected:** Training a pass-completion model — the stated four-hour upgrade path,
deferred alongside M6 for the same reason (session time), tracked as follow-up work below.
**Reversibility:** moderate — every consumer of `value`/`regret` (the dashboard, this
README, any future analysis) would need to stop treating the number as an upper bound once
(a) lands; not a schema change, but a meaning change.

---

## 2026-08-08 — M8: counterfactual feature rows reuse the real feature pipeline, not a reimplementation

**Context:** Scoring a hypothetical action requires the exact same feature vector shape
the model was trained on. Reimplementing socceraction's feature functions by hand for a
synthetic action risked silently diverging from the real computation (the exact failure
mode SPEC.md §3 warns about for coordinates, generalized to features).
**Decision:** `counterfactual/score.py` builds the same `[a0, a1, a2, a3]` gamestates
structure `features/actions.py` feeds socceraction's `XFNS` list, with `a0` (the action
itself) replaced by N candidate rows and `a1`/`a2`/`a3` (real preceding actions) broadcast
unchanged across all N. The same feature functions then produce feature rows that are
bit-for-bit consistent with training, by construction rather than by care.
**Rejected:** Hand-computing each of the ~155 feature columns for a synthetic action —
would have needed independently re-deriving what every socceraction feature function does
and staying in sync with it if socceraction's implementation ever changes.
**Reversibility:** cheap — the reused functions are exactly the ones `features/actions.py`
already depends on.

---

## 2026-08-08 — M8: counterfactuals computed for the served subset, not the full action corpus

**Context:** `list_moments` only ever serves the top 100 actions per match by value; a full
counterfactual set (enumeration + feature construction + prediction + support gating) for
all ~110,000 ingested Euro 2024 actions would cost far more runtime than the dashboard
displaying any of it requires.
**Decision:** `add_counterfactuals` computes real `best`/`regret`/`options`/
`unscored_count` for exactly the top 100 pass/dribble/shot actions per match by value
(5,100 moments across 51 matches) — matching `list_moments`'s default `limit` exactly, so
every moment served under default query params has genuine counterfactual data, not a
subset of a subset. Every other ingested action keeps the same honest `NULL`/`[]` this
project shipped before M8 existed, now for a documented runtime reason rather than because
the layer didn't exist yet.
**Rejected:** (a) Computing it for all ~110k actions — correct but unnecessary for a
working dashboard, and untested at that scale this pass. (b) Restricting to *all* action
types rather than pass/dribble/shot — "declined a pass/carry/shot instead" is not a
coherent framing for a duel, foul, or interception; SPEC.md §10 doesn't rule this out
explicitly but doesn't require it either.
**Reversibility:** cheap — raising `COUNTERFACTUAL_TOP_K` or dropping the `DECISION_TYPES`
filter changes what gets processed, not the schema or the arithmetic. SPEC.md §11.2 item
4's full-corpus xDR distribution (mean per 90 across every action) is follow-up work,
listed below.
**Result:** 114,971 candidates enumerated, 51.0% gated below the support floor (SPEC.md's
target band is 15-50% — just above it, not retuned further this pass). Of the 5,100
processed moments, 2,794 (54.8%) have a real non-null regret; mean regret across those is
0.0341 (95% CI [0.0271, 0.0419], match-level bootstrap, 2,000 draws), median 0.0221, 39.9%
exactly zero (the player's actual choice already matched or beat every scored
alternative), max 0.969.

---

## 2026-08-08 — Windows-specific: `lightgbm`/`scikit-learn` import order caused a native segfault on small predict batches

**Context:** `Booster.predict()` on a small batch (tens to low-thousands of rows — exactly
the size of one moment's counterfactual candidate set) crashed the Python process natively
(`OSError: exception: access violation reading 0x0000000000000000`) the first time M8
called it, despite the identical code path working fine on the ~16,000-row test split in
`evaluation/report.py`. Bisected to import order: the crash reproduces if `lightgbm` is
imported before any of scikit-learn's compiled extensions, and disappears if
`sklearn.neighbors` (or similar) is imported first — an OpenMP-runtime initialization-
order conflict between the two libraries' bundled native runtimes, specific to this
Windows build.
**Decision:** `import sklearn.neighbors` at the top of `xdr/__init__.py`, which every real
entrypoint runs before any `xdr.*` submodule gets a chance to import `lightgbm` on its own.
Verified directly: removing the line reproduces the crash on the same predict call;
restoring it, the identical call succeeds every time.
**Rejected:** Reordering imports individually in every module that happens to import both
libraries — fragile (depends on import order at every call site, forever) versus a single
process-wide fix at the package root.
**Reversibility:** cheap, and worth revisiting if the machine or library versions change —
this is a workaround for a specific native runtime conflict, not a design decision.

---

## 2026-08-08 — Known gaps carried forward from this pass (M5, M7, M8 landed)

- **M6 (DeepSets frame encoder) and its ablation against the LightGBM baseline are still
  not built.** `xdr/models/encoder.py` and `xdr/evaluation/ablation.py` remain to be
  written. The support gate (M7) therefore still runs on the action-feature branch alone
  (see above); the counterfactual layer does not use frame information beyond visible-
  teammate pass targets.
- **The completion-probability model (SPEC.md §10 default (a)) is not built.** Regret is
  an upper bound, not a point estimate, until it is.
- **The counterfactual layer covers ~5,100 of 109,985 ingested actions**, not the full
  corpus — a runtime scope cut (see above), not a claim about the rest.
- **Docker, CI, and Vercel/host deployment (SPEC.md §14, M9) are in progress**, picked up
  immediately after this entry.
- **Test coverage is 54% on `api/src/xdr/`**, down from the prior pass's 59% in
  percentage terms only -- M5/M7/M8 added ~400 lines of new orchestration code
  (`evaluation/transfer.py`, `serve/store.py`'s `add_counterfactuals`) that, like
  `data/ingest.py` and `evaluation/report.py` before them, are exercised end-to-end
  against real data (see the results throughout this log) but not by narrow pytest unit
  tests. The new pure-logic modules do have direct unit tests: `test_counterfactual.py`,
  `test_support.py`, `test_transfer.py`, all passing (55 passed, 1 skipped total).
  Still short of SPEC.md §1.2's ≥85% target.

