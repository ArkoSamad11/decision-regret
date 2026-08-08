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


