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
- No source files under `src/cafaeval/` have been modified at this point.

## Planned (not yet applied)

### Phase A — bit-exact optimizations
Gate: every commit must pass `tests/diff/test_oracle_parity.py` with
`atol=0, rtol=0` on all validation corpora.

- A1 — `evaluation.py`: skip the unweighted metric pass when an IA file
  is provided (weighted-only fast path).
- A2 — `parser.py`: replace full-row `np.count_nonzero` scans with an
  incremental counter updated only on 0→non-zero transitions.
- A3 — `graph.py`: cache the children list of each term at propagation
  setup time instead of recomputing via `np.where(dag[:, i])` inside the
  loop.
- A4 — `graph.py`: fill-mode propagation restricted to rows where the
  current term is zero; replace sum-based zero checks with boolean
  existence checks; avoid array element deletion in favour of slicing.

### Phase B — sparse + numba rewrite
Gate: every commit must pass `tests/diff/test_oracle_parity.py` with
`rtol=1e-6, atol=1e-9` on all validation corpora.

- B1 — sparse representation keyed on predicted terms ∪ propagated GT,
  replacing full ontology-sized dense matrices.
- B2 — numba kernels for propagation and per-threshold metric aggregation.
- B3 — `evaluation.compute_metrics` rewritten over the sparse layout.
- B4 — removal of `multiprocessing.Pool` in favour of numba `prange`
  parallelism (eliminates fork-related hangs observed in CI).

### Cross-cutting
- Structured logging via stdlib `logging` under the `cafaeval.*` hierarchy.
  No `print()` calls, no `basicConfig()` inside the library. Extra fields
  are passed via `logger.info(..., extra={...})`.
