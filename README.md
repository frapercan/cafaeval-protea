# cafaeval-protea

> **This is a modified fork.** See [`CHANGES.md`](CHANGES.md) for the full list
> of modifications and their dates (required by GPLv3 §5.a).

`cafaeval-protea` is a fork of
[**CAFA-evaluator-PK**](https://github.com/claradepaolis/CAFA-evaluator-PK) by
Clara De Paolis, which is itself a fork of
[**CAFA-evaluator**](https://github.com/BioComputingUP/CAFA-evaluator) by the
BioComputing UP group at the University of Padua (Piovesan et al., 2024).

The fork exists to provide a faster evaluator for iterative work, without
changing any scoring semantics. Fmax, Smin, weighted variants,
Partial-Knowledge (PK) evaluation and information accretion weighting are
all preserved and validated against the upstream output before any
optimization lands.

The Python import path remains `cafaeval` (identical API) so that existing
downstream code using `from cafaeval.evaluation import cafa_eval` keeps
working unchanged. Only the PyPI **distribution** name differs:

```bash
pip install cafaeval-protea          # installs the fork
python -c "from cafaeval.evaluation import cafa_eval"  # same import
```

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

| Area | Upstream module | Change | Status | Validation |
|---|---|---|---|---|
| Parser | `src/cafaeval/parser.py` | (A) incremental non-zero counter, buffered reads, single dict lookup per term; (B3) PyArrow-backed vectorised parser with dictionary-encoded `pid`/`tid` and sort-based per-namespace group-max | done | bit-exact on real corpora |
| Propagation | `src/cafaeval/graph.py` | (A) cached per-term children lists, fill-mode restricted to zero rows, shared-memory spawn worker; (B4) sparse push-up kernel with flat ancestor CSR and `np.maximum.reduceat` group-max over input non-zeros | done | bit-exact in A, `rtol=1e-6` in B |
| NK/LK metric | `src/cafaeval/evaluation.py` | (A) weighted-only fast path, fork-pool `initializer` for threshold sweep; (B1) sparse confusion-matrix kernel via `np.bincount` scatter + right-to-left cumsum | done | bit-exact |
| PK metric | `src/cafaeval/evaluation.py` | (A) fork-pool `initializer` pattern extended to the `gt_exclude` branch; (B2) sparse PK kernel with boolean-mask filter `(pred != 0) & toi_mask & ~excluded_mask` | done | `rtol=1e-6` in B (ULP reorder) |
| Logging | (new) | Structured stdlib `logging` at module granularity — see *Logging* below | done | n/a |
| Orchestrator | `src/cafaeval/__main__.py` | Thin reshuffling only; no semantic change | done | bit-exact |

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

- **Phase A** (parser cherry-picks, cached children, weighted-only,
  zero-row propagation, NK sparse kernel, sparse propagate): `atol=0,
  rtol=0`. A single-bit divergence is a bug.
- **Phase B** (sparse PK confusion matrix): `rtol=1e-6, atol=1e-9`. The
  PK sparse kernel reorders per-protein inner sums, so a ULP-level
  (`~4e-16`) divergence in `pr` is expected and tolerated.

The active phase is controlled by the `CAFAEVAL_PARITY_PHASE` env var
(`A` or `B`). The default flipped from `A` to `B` when the Phase B2 PK
kernel landed. On a 4.45M-row real corpus the fork agrees with
unmodified upstream to `2.1e-14` in PK and `1.9e-14` in NK, well inside
the Phase B tolerance.

No result from this fork is trusted until the relevant corpus passes its
diff test.

---

## Performance

End-to-end wall time on a 4.45M-row real CAFA prediction file
(`cafa_eval(...)` from OBO load through metric assembly), measured on a
single workstation. "Upstream" is unmodified
[`claradepaolis/CAFA-evaluator-PK`](https://github.com/claradepaolis/CAFA-evaluator-PK)
at commit `16a6a6d`.

| Mode | Upstream (n_cpu=1) | Fork (n_cpu=1) | Upstream (n_cpu=4) | Fork (n_cpu=4) | Speedup |
|---|---|---|---|---|---|
| NK | 1.39 s | **0.45 s** | 1.45 s | **0.42 s** | 3.1× – 3.5× |
| PK | 2.96 s | **0.65 s** | 2.61 s | **0.63 s** | 4.1× – 4.6× |

Single-threaded fork beats upstream-at-4-cores for both modes. After the
sparse kernels landed, confusion-matrix compute is `<15 ms` total,
propagate is `~40 ms`, and the prediction parser is now the dominant
term at `~400–500 ms` — hence the PyArrow rewrite (Phase B3).

## Runtime knobs

All optimizations are gated by environment variables so the legacy path
is always available for A/B comparison or debugging.

| Var | Default | Effect |
|---|---|---|
| `CAFAEVAL_SPARSE` | `1` | Sparse NK + PK confusion-matrix kernels and sparse push-up propagation. Set to `0` to fall back to the dense/pool path. |
| `CAFAEVAL_FAST_PARSER` | `1` | PyArrow-backed vectorised `pred_parser`. Set to `0` to force the legacy per-line loop. Also falls back automatically when `max_terms` is set or the fast path raises. |
| `CAFAEVAL_PARITY_PHASE` | `B` | Tolerance used by `tests/diff/test_oracle_parity.py` — `A` for bit-exact, `B` for `rtol=1e-6, atol=1e-9`. |

## Install

```bash
pip install cafaeval-protea               # core install (numpy + pandas + matplotlib)
pip install "cafaeval-protea[fast]"       # enables the PyArrow parser fast path
```

The hard dependency set is kept at `numpy + pandas + matplotlib`;
`pyarrow>=12` is an optional `[fast]` extra. Without it, `pred_parser`
automatically falls back to the legacy loop.

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
