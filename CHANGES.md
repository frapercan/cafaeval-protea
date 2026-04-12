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

## 2026-04-12 — Phase B1: sparse NK/LK confusion matrix kernel

- `evaluation.compute_confusion_matrix_sparse`: new kernel that replaces
  the per-threshold dense mask scan (`O(n_tau · n_prot · n_toi)`) with a
  single scatter pass over non-zero predictions followed by one
  right-to-left cumulative sum (`O(nnz + n_prot · n_tau)`). Each
  prediction is bucketed into the highest tau index at which its score
  is still active (`searchsorted(tau_arr, score, side='right') - 1`) via
  `np.bincount` on the flattened `(row, bin)` index, which avoids the
  Python-level hot loop of `np.add.at`.
- `compute_metrics`: NK/LK branch now calls the sparse kernel directly
  (no `multiprocessing.Pool` — the kernel is already faster than the
  fork overhead of a Pool on real corpora). Gated by the
  `CAFAEVAL_SPARSE` env var (default on; set to `0` to fall back to the
  pooled dense path for A/B comparison or debugging).
- The PK branch (`gt_exclude is not None`) keeps the dense
  per-protein kernel — `toi_perprotein` varies per protein and doesn't
  map cleanly onto a single flat scatter; sparsifying it is a separate
  step.
- Parity: bit-exact (`atol=0, rtol=0`) on the `tiny`, `medium`, `large`
  synthetic oracle corpora. Against unmodified upstream on a 2.3M-row
  real prediction file, max column divergence was 1.9e-14 (float
  summation order), well inside the Phase B tolerance of
  `rtol=1e-6, atol=1e-9`.

## 2026-04-12 — Phase B4: sparse push-up propagation kernel

- `graph._ancestors_csr`: new lazy per-`Graph` cache. Walks the DAG in
  topological order (leaves → roots → leaves) to compute the transitive
  ancestor set of every term, flattened into a CSR-style
  `(indptr, indices)` pair so ancestor lookups are constant-time index
  slicing. The parents adjacency is built via a single
  `np.nonzero(dag)` + `argsort` + `bincount` pass (one scan of the
  matrix) instead of one `np.flatnonzero` call per term.
- `graph._propagate_sparse_pushup`: new kernel that replaces the
  per-term dense sweep (`_propagate_serial`, `O(n_terms · n_prot ·
  avg_children)`) with a single scatter over input non-zeros:
    1. collect input non-zeros `(row, col, score)`;
    2. gather the flat ancestor list for every `col` via vectorised
       `np.repeat` offsets (no Python loop);
    3. encode `(row, ancestor)` as a single int64 flat key, stable-sort
       once, and reduce per-group with `np.maximum.reduceat`;
    4. write the group-maxes back in place.
  Cost is `O(nnz · avg_ancestors + R log R)` with `R` the expanded
  triple count — on sparse CAFA inputs this skips over every term that
  has no predictions, unlike the dense sweep which pays for all of
  them.
- `graph.propagate`: sparse path is gated by the same `CAFAEVAL_SPARSE`
  env var as the confusion matrix kernel (default on). The
  `_children_cache(ont)` call was moved out of the unconditional top
  of `propagate()` into the dense-fallback branch, so the sparse path
  no longer pays the per-term children materialisation cost. The
  `spawn` shared-memory worker branch now computes its own
  `children_by_term` from the cache helper (latent fix to support
  dense fallback under parallelism).
- `mode='fill'` semantics are preserved: originally non-zero cells
  keep their input value; only zero cells are overwritten by
  propagated scores (restored after the group-max by writing back the
  snapshotted input non-zeros).
- Parity: bit-exact (`atol=0, rtol=0`) on the `tiny`, `medium`,
  `large` synthetic oracle corpora (Phase A). On a 4.45M-row real
  prediction file the sparse path agrees with the dense fallback to
  `3.6e-15` (floating-point summation order), well inside the Phase B
  tolerance of `rtol=1e-6, atol=1e-9`.

### Planned (not yet applied)

- B2 — sparse PK kernel (`compute_confusion_matrix_exclude`).
- B3 — numba JIT for the scatter + aggregation loop, if profiling
  shows it is worth the optional build dependency.
