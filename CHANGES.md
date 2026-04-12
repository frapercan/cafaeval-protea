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

## 2026-04-13 — Phase B2: sparse PK confusion matrix kernel

- `evaluation.compute_confusion_matrix_exclude_sparse`: new kernel
  that extends the Phase B1 scatter + right-to-left cumsum strategy
  to the PK setting where each protein has its own valid column set.
  Filter is expressed as two boolean ANDs over dense
  ``(n_prot, n_terms)`` matrices:
    1. ``toi_mask[None, :]`` — global terms of interest.
    2. ``~excluded_mask`` — per-protein exclude set from
       ``gt_exclude.matrix[proteins_with_gt, :]``.
  The scatter then sees only surviving non-zeros, so the per-protein
  Python list of toi arrays (``toi_perprotein``, ``gt_perprotein``)
  is never materialised. Cost drops from
  ``O(n_tau · Σ_p |toi_p|)`` to ``O(nnz_valid + n_prot · n_tau)``.
- `compute_metrics` (PK branch): when ``CAFAEVAL_SPARSE=1`` (default)
  the sparse kernel is called directly with ``pred_sub``,
  ``gt_with_annots``, ``toi_mask``, ``excluded_mask`` and a
  vectorised ``n_gt``. The dense fallback retains the exact
  list-comprehension summation order used upstream so
  ``test_norm_metric`` inside the dense kernel does not trip on ULP
  noise on real corpora.
- Parity is now **Phase B only** (``rtol=1e-6, atol=1e-9``). The PK
  sparse kernel reorders the per-protein inner sums relative to the
  dense kernel, so the bit-exact Phase A tolerance no longer holds:
  on the oracle corpora we observed divergences of ``4.4e-16``
  (single ULP) in ``pr``. On a 4.45M-row real PK corpus
  (6 953 ground-truth rows, `pk_known_terms.tsv` exclude set) the
  sparse and dense kernels agree to ``2.1e-14``, well inside the
  Phase B tolerance. All 6 oracle parity tests pass under
  ``CAFAEVAL_PARITY_PHASE=B``.
- Benchmark on the same real PK corpus at n_cpu=1:
  dense 2.73s → sparse 1.45s (1.88× speedup). At n_cpu=4 dense drops
  to 1.94s while sparse stays at 1.45s — the sparse kernel already
  beats a 4-way fork pool of the dense kernel without any
  parallelism of its own.

## 2026-04-13 — Phase B3: vectorised prediction-file parser

Phase B profiling on the real 4.45M-row PK corpus showed that after
B1/B2/B4 the sparse confusion matrix kernels had dropped to <15 ms
total and ``propagate`` to ~42 ms, while ``pred_parser`` was still
eating ~60% of wall time (1.0-1.4s) sitting inside the Python line
loop — ``str.split``, ``dict.get``, ``str.strip`` took 4.7M calls
each. A numba JIT on the scatter kernel (the original B3 plan) would
have saved a few milliseconds on top of an already vectorised hot
spot, so Phase B3 was redirected to rewriting ``pred_parser`` itself.

- `parser._pred_parser_vectorised`: new PyArrow-backed bulk parser.
  ``pyarrow.csv.read_csv`` ingests the whole prediction file at
  native speed. ``pid`` and ``tid`` are dictionary-encoded once so
  the Python-level ``ns_dict`` / ``gts[ns].ids`` / ``term_index``
  lookups run over the (small) unique-value sets instead of over
  every one of the 4.45M rows. The resulting numpy int code arrays
  are filtered per namespace with vectorised comparisons. Per-namespace
  reduction uses the same sort + ``np.maximum.reduceat`` group-max +
  scatter pattern as the Phase B4 propagation kernel. Duplicate
  ``(protein, term)`` predictions collapse to the max, matching the
  legacy loop's "store the max if higher" semantics bit-exactly.
- Alt-id expansion (rare on CAFA inputs) is flagged during the
  per-unique-tid column lookup (column value ``-2``) and handled by
  a short Python loop over just the affected rows, so the common
  fast path never walks the full alt dictionary.
- `parser._pred_parser_legacy`: the original per-line loop is
  preserved and routed to for:
    1. ``max_terms``-capped runs (the top-k cap is order-sensitive so
       the vectorised path cannot reproduce it), and
    2. any pathological input that makes the fast path raise (format
       errors, missing columns). The caller clears any partial state
       written by the fast path before falling back.
- `parser.pred_parser`: gated by the new ``CAFAEVAL_FAST_PARSER`` env
  var (default on). Set to ``0`` for A/B comparison against the
  legacy loop.
- `pyproject.toml`: declared ``pyarrow>=12`` as an optional
  ``[fast]`` install extra. The package imports pyarrow lazily
  inside the fast path only, so the hard dependency set stays at
  ``numpy + pandas + matplotlib``.
- Parity: 6/6 on Phase B oracle gate. On the 4.45M-row real corpus
  the fast and legacy parsers agree **bit-exactly** (``max |Δ| = 0``)
  for both NK and PK outputs — the vectorised reduction order
  happens to match the legacy accumulation because every
  ``(protein, term)`` pair's maximum is order-independent.

Benchmark on the 4.45M-row real corpus:

  NK  legacy 1.22s → fast 0.45s  (2.72×)
  PK  legacy 1.44s → fast 0.63s  (2.28×)

### Planned (not yet applied)

- Optional: numba-JIT fallback for environments without pyarrow, if
  profiling on a legacy-only install warrants it.
