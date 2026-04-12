# cafaeval (PROTEA fork)

> **This is a modified fork.** See [`CHANGES.md`](CHANGES.md) for the full list
> of modifications and their dates (required by GPLv3 §5.a).

This repository is a fork of
[**CAFA-evaluator-PK**](https://github.com/claradepaolis/CAFA-evaluator-PK) by
Clara De Paolis, which is itself a fork of
[**CAFA-evaluator**](https://github.com/BioComputingUP/CAFA-evaluator) by the
BioComputing UP group at the University of Padua (Piovesan et al., 2024).

The fork exists **only** to provide a faster local evaluator for iterative
work in the PROTEA thesis project. All scoring semantics — Fmax, Smin,
weighted variants, Partial-Knowledge (PK) evaluation, information accretion
weighting — are preserved and validated against the upstream output before
any new result is trusted.

The installable package name remains `cafaeval` (identical import path) so
that existing downstream code using `from cafaeval.evaluation import cafa_eval`
keeps working unchanged.

---

## Attribution

### Upstream authors (primary — always cite)

The original evaluator and all of its scoring logic are the work of:

> **CAFA-evaluator: A Python Tool for Benchmarking Ontological Classification Methods**
> D. Piovesan, D. Zago, P. Joshi, M. C. De Paolis Kaluza, M. Mehdiabadi,
> R. Ramola, A. M. Monzon, W. Reade, I. Friedberg, P. Radivojac, S. C. E. Tosatto.
> *Bioinformatics Advances*, 2024.
> DOI: [10.1093/bioadv/vbae043](https://doi.org/10.1093/bioadv/vbae043)

Upstream repository:
[BioComputingUP/CAFA-evaluator](https://github.com/BioComputingUP/CAFA-evaluator).
Copyright © 2022 Damiano Piovesan. Licensed under GPLv3.

### Direct parent (PK variant)

The Partial-Knowledge evaluation extensions were contributed by
**Clara De Paolis** in
[claradepaolis/CAFA-evaluator-PK](https://github.com/claradepaolis/CAFA-evaluator-PK).
This fork is branched directly from that repository and inherits its
semantics for `-known` annotations and terms-of-interest filtering.

### Speedup ideas and cherry-picked commits

The Phase A algorithmic speedups in this fork (weighted-only fast path,
cached per-term children lists, fill-mode restricted to zero rows,
incremental non-zero counter in the prediction parser, shared-memory
parallel DAG propagation, fork-pool `initializer` pattern for the
threshold sweep) come from
**Antonina Dolgorukova (`T0chka`)** and her public fork
[T0chka/CAFA-evaluator-PK-speedup](https://github.com/T0chka/CAFA-evaluator-PK-speedup),
which she shared in the CAFA 6 Kaggle discussion
[*"Speeding up cafaeval"*](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/discussion/664359)
(post #664359).

The five substantive commits from her `speedup-local` branch were
cherry-picked into this fork with authorship preserved (see `git log`).
On top of that, we added: dead-code removal, structured `cafaeval.*`
logging, extension of the fork-pool `initializer` pattern from the NK/LK
branch to the PK (`gt_exclude`) branch of `compute_metrics`, and the
parity harness under `bench/` and `tests/diff/`.

If you use the speedups in published work please acknowledge the
upstream paper above, Clara De Paolis' PK fork, **and** Antonina
Dolgorukova's speedup work.

---

## Scope of modifications

This fork modifies the following parts of the upstream:

| Area | Upstream module | Planned change | Validation |
|---|---|---|---|
| Parser | `src/cafaeval/parser.py` | Incremental non-zero counter; early filtering | bit-exact (A) |
| Propagation | `src/cafaeval/graph.py` | Cached children lists; fill-mode restricted to zero rows; sparse rewrite with numba kernels | bit-exact in A, `rtol=1e-6` in B |
| Metrics | `src/cafaeval/evaluation.py` | Weighted-only fast path when IA file is present; sparse metric computation | bit-exact in A |
| Logging | (new) | Structured stdlib `logging` at module granularity — see *Logging* below | n/a |
| Orchestrator | `src/cafaeval/__main__.py` | Thin reshuffling only; no semantic change | bit-exact |

Detailed per-commit diff against the upstream is maintained in
[`CHANGES.md`](CHANGES.md).

### Validation policy

No optimization lands in this fork without a passing parity test against a
frozen upstream oracle. The oracle is built in `bench/` by running the
**unmodified upstream** against a set of deterministic synthetic corpora
(tiny / medium / large) and serializing the full output — Fmax, Smin,
weighted Fmax, weighted Smin, precision-recall curves, optimal thresholds
— into `bench/oracle/*.pkl`. The diff tests under `tests/diff/` reload
that oracle and compare the fork's output:

- **Phase A** (safe optimizations — parser, cached children, weighted-only,
  zero-row propagation): `atol=0, rtol=0`. A single-bit divergence is a bug.
- **Phase B** (sparse rewrite + numba kernels): `rtol=1e-6, atol=1e-9`.
  Divergence at this level is attributed to float summation order.

No result from this fork is used in the PROTEA thesis until the relevant
corpus passes its diff test.

---

## Logging

The upstream evaluator is silent at the library boundary, which makes
long-running calls inside other pipelines opaque. This fork adds structured
logging using the stdlib `logging` module (no new dependencies), organized
as a proper logger hierarchy:

```
cafaeval                    # root logger
├── cafaeval.parser         # parsing predictions / ground truth
├── cafaeval.propagate      # DAG propagation
├── cafaeval.metrics        # compute_metrics
└── cafaeval.eval           # orchestrator
```

Conventions:

- **INFO**: high-level events with timing, e.g.
  `"parser: parsed 12345 proteins in 3.21s"`.
- **DEBUG**: per-namespace, per-threshold detail, matrix shapes.
- **WARNING**: non-fatal anomalies (terms missing from the ontology,
  proteins without ground truth).
- **No `print()` calls**, no `basicConfig()` inside the library. Handler
  configuration is always the consumer's responsibility.
- Structured fields are passed via `logger.info(..., extra={...})` so that
  downstream consumers can extract machine-readable payloads without
  parsing log strings.

A downstream project that wants to capture these logs only needs:

```python
import logging
logging.getLogger("cafaeval").setLevel(logging.INFO)
# attach your own handler here
```

---

## Usage

The CLI and library interfaces are unchanged from the upstream. See
[`README_upstream.md`](README_upstream.md) for the full input-file formats,
command-line flags, and output layout. The `cafaeval` console script and
the `cafa_eval(...)` Python entry point continue to work exactly as
documented there.

---

## License

GNU General Public License v3 (GPLv3), inherited unchanged from the
upstream. See [`LICENCE.md`](LICENCE.md).

Per GPLv3 §5.a, modifications introduced by this fork are documented in
[`CHANGES.md`](CHANGES.md) with their dates. The original upstream
copyright notice (© 2022 Damiano Piovesan) is preserved verbatim in
`LICENCE.md` and is not superseded by the existence of this fork.
