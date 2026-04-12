# Changes in this fork

Per GPLv3 §5.a, this file lists all modifications introduced by this fork
with their dates. Modifications are tracked per phase (see `README.md`).

## Fork base

Branched from
[claradepaolis/CAFA-evaluator-PK](https://github.com/claradepaolis/CAFA-evaluator-PK)
at commit `16a6a6d` ("de-duplicate edges before top sort (#2)").

---

## 2026-04-12 — Fork bootstrap

- Branched from CAFA-evaluator-PK.
- Preserved upstream `README.md` verbatim as `README_upstream.md`.
- Added fork-level `README.md` with attribution chain, modification policy,
  and validation guarantees.
- Added `NOTICE` with the full fork chain, copyright statements, and
  speedup-idea acknowledgement.
- Added `CITATION.cff` extending the upstream citation.
- Added this `CHANGES.md`.
- Added synthetic corpus generator (`bench/corpus.py`), oracle freezer
  (`bench/freeze_oracle.py`, `bench/oracle_record.py`) and parity diff
  test suite (`tests/diff/`) capable of asserting bit-exactness
  (Phase A) or numerical equivalence with `rtol=1e-6, atol=1e-9`
  (Phase B) against a frozen unmodified-upstream oracle.

## 2026-04-12 — Phase A: cherry-pick T0chka speedups

Cherry-picked with upstream authorship preserved from
[T0chka/CAFA-evaluator-PK-speedup](https://github.com/T0chka/CAFA-evaluator-PK-speedup)
(`speedup-local` branch, author: Antonina Dolgorukova):

- `ebb352c` — `evaluation.py`: `weighted_only` fast path; skip the
  unweighted metric pass when the caller only wants `_w` columns.
- `c5623d2` — timing instrumentation (later replaced by structured
  logging, see the cleanup commit below).
- `588d7b6` — `evaluation.compute_metrics`: parallelise the threshold
  sweep via `multiprocessing.get_context("fork").Pool` with
  `initializer=_cm_init` and module-level globals, so the `g/p/toi/n_gt`
  arrays are shared across workers once instead of pickled per chunk.
- `ca24bc7` — `graph.propagate`: cached per-term children list,
  fill-mode restricted to rows where the current term is zero, and
  optional shared-memory (`spawn`) multi-process propagation gated by
  a work-size threshold.
- `d69b5b1` — `parser.pred_parser`: incremental `row_nnz` counter,
  precomputed `term_index`, buffered file reads, single dictionary
  lookup per predicted term.

All five commits are bit-exact (`atol=0, rtol=0`) against the frozen
upstream oracle on the `tiny`, `medium`, and `large` synthetic corpora.

## 2026-04-12 — Cleanup on top of the cherry-picks

- `graph.py`: removed the dead unreachable first `propagate()`
  definition left over from the cherry-pick sequence; factored out
  `_children_cache` and `_propagate_serial`.
- `evaluation.py`: extended the `_cm_init` / `_cm_worker` fork-pool
  initializer pattern from the NK/LK branch of `compute_metrics` to
  the PK branch (`gt_exclude is not None`) via `_cme_init` /
  `_cme_worker`, so PK evaluation now shares the same parallel shape
  as NK/LK and no longer pays the pickle cost of
  `Pool.starmap(compute_confusion_matrix_exclude, arg_lists)`.
- All three modules: replaced `print()` timing instrumentation with
  structured stdlib `logging` under the `cafaeval.parser`,
  `cafaeval.propagate`, `cafaeval.metrics`, and `cafaeval.eval`
  loggers. Extras are passed via `logger.info(..., extra={...})` so
  consumers can extract machine-readable payloads without parsing log
  strings. No `print()` remains in the library.
- Parity suite re-run after cleanup: still 6/6 bit-exact.

## Planned (not yet applied)

### Phase B — sparse + numba rewrite
Gate: every commit must pass `tests/diff/test_oracle_parity.py` with
`rtol=1e-6, atol=1e-9` on all validation corpora.

- B1 — sparse representation keyed on predicted terms ∪ propagated GT,
  replacing full ontology-sized dense matrices.
- B2 — numba kernels for propagation and per-threshold metric aggregation.
- B3 — `evaluation.compute_metrics` rewritten over the sparse layout.
- B4 — removal of `multiprocessing.Pool` in favour of numba `prange`
  parallelism (eliminates fork-related hangs observed in CI).
