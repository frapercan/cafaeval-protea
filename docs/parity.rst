Parity harness
==============

Every optimisation in this fork is gated by a parity harness. The
harness answers a single question: given a fixed synthetic corpus,
does the fork produce the same numerical output as pristine upstream?

Why it exists
-------------

Rewriting confusion matrices and propagation kernels in terms of
sparse scatter operations reorders per-protein and per-threshold
sums. The reordering is safe under float associativity *up to ULP
noise*, but it is not bit-exact. Without a harness, a subtle indexing
bug would silently corrupt the output and still agree with upstream to
six decimal places.

The harness freezes upstream output once against a set of synthetic
corpora and then asserts the fork matches that frozen snapshot
within a declared tolerance.

Tolerance phases
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Phase
     - Tolerance
     - When it is active
   * - A
     - ``atol=0, rtol=0`` (bit-exact)
     - Phase A cherry-picks only. Any single-bit divergence indicates
       a cherry-pick bug.
   * - B
     - ``rtol=1e-6, atol=1e-9``
     - Default. From Phase B2 onward the fork reorders inner
       per-protein sums and can no longer match upstream bit-for-bit
       (observed ULP-level ``4.4e-16`` divergence in ``pr``).

The active phase is controlled by the ``CAFAEVAL_PARITY_PHASE``
environment variable::

    CAFAEVAL_PARITY_PHASE=B pytest tests/diff/   # default
    CAFAEVAL_PARITY_PHASE=A pytest tests/diff/   # bit-exact gate

Phase C (the memory work — sparse DAG and sparse prediction) introduces
no new sum reordering, so it is verified **bit-exact** (``atol=0,
rtol=0``) against the *pre-C fork* rather than upstream: the sparse DAG
reproduces the dense ``dag.sum``/``np.nonzero`` results exactly, and the
CSR prediction feeds the same Phase B kernels with the same non-zeros.
It therefore keeps the existing Phase B oracle tolerance unchanged. The
equivalence was additionally checked end-to-end on a 24 000-term ×
30 000-protein synthetic corpus (identical output TSVs) across the
sparse / dense / no-orphans / IA-weighted / NK / PK scenarios.

Branches covered
----------------

``cafa_eval`` has two structurally different code paths depending on
whether ``exclude=`` is passed:

* ``gt_exclude is None`` — the **NK / LK** path. ``compute_metrics``
  calls ``compute_confusion_matrix_sparse`` with ``g`` / ``p``
  directly and skips the per-protein exclude mask entirely.
* ``gt_exclude is not None`` — the **PK** path.
  ``compute_confusion_matrix_exclude_sparse`` walks surviving
  non-zeros after a per-protein exclude AND.

Each corpus is therefore frozen **twice** — once with the PK exclude
file (``<name>.pk.pkl``) and once without (``<name>.nk.pkl``). The
parametrised fixture in ``tests/diff/conftest.py`` produces one
fixture instance per ``(corpus, variant)`` pair, so the harness runs
12 oracle tests (3 corpora × 2 variants × 2 metric scopes).

NK and LK share the same code path in both upstream and the fork, so
the NK oracle also gates LK.

Semantic divergence on PK
-------------------------

Starting with commit ``cec8ccd`` (2026-04-23), the fork carries a
deliberate semantic divergence from upstream on the PK branch. Upstream
counts ``metrics['n']`` (the row count used for coverage and, under
``normalization='cafa'``, the precision denominator) over every row of
``proteins_with_gt`` (any GT in the term-of-interest set, pre-exclusion),
while the matching denominator ``ne`` from ``_count_proteins_in_toi``
drops proteins whose TOI annotations were fully contained in the
per-protein exclude set. The asymmetry can push ``coverage = n / ne``
above 1 (observed at 1.3–1.9 on a real GOA-derived benchmark).

The fix in ``compute_confusion_matrix_exclude_sparse`` masks the row
count to the same post-exclusion eligibility set, so ``n`` and ``ne``
live in the same population. TP, FP, FN and recall are unaffected;
precision under ``normalization='cafa'`` tightens to the correct value
and coverage is bounded in ``[0, 1]``.

Effect on the parity harness:

* PK oracle tests xfail on purpose. The helper ``_maybe_xfail_pk`` in
  ``tests/diff/conftest.py`` skips the comparison for PK variants of
  ``test_main_df_matches_oracle`` and ``test_best_metrics_match_oracle``.
* NK and LK oracle tests continue to enforce strict parity (Phase B
  tolerance).
* A regression test ``tests/test_pk_coverage_bug.py`` pins the bounded
  ``n ≤ ne`` invariant after the fix.

The change is documented end-to-end in ``CHANGES.md``.

Self-parity (sparse vs dense)
-----------------------------

In addition to the frozen-oracle tests, the harness includes an
**in-fork self-parity** test (``tests/diff/test_self_parity_nk_lk.py``):
it runs the same synthetic corpus through ``cafa_eval`` twice, once
with ``CAFAEVAL_SPARSE=1`` and once with ``CAFAEVAL_SPARSE=0``, and
asserts both paths agree within Phase B tolerance.

This is a belt-and-suspenders check. It catches sparse-vs-dense
divergence on the NK / LK branch without needing an upstream install,
and flags regressions in the dense fallback path that would otherwise
be invisible to the frozen oracle.

Running the harness
-------------------

Against the checked-in oracle::

    pytest tests/diff/ -v

Expected output: 12 oracle tests + 1 self-parity test, all passing
under Phase B tolerance.

Re-freezing the oracle
----------------------

The oracle pickles in ``bench/oracle/`` are keyed by a corpus
fingerprint. If the synthetic corpus generator (``bench/corpus.py``)
changes, or upstream changes, the oracle must be re-frozen against a
**pristine** upstream install.

The recipe used to produce the current oracle::

    # 1. Clone pristine upstream at the fork base commit.
    git clone --depth 50 \
        https://github.com/claradepaolis/CAFA-evaluator-PK.git \
        /tmp/cafa-upstream
    git -C /tmp/cafa-upstream checkout 16a6a6d

    # 2. Run the freezer against it, importing upstream from source.
    cd /path/to/cafaeval-protea
    PYTHONPATH=/tmp/cafa-upstream/src:. \
        python -m bench.freeze_oracle

The freezer produces six files::

    bench/oracle/tiny.pk.pkl
    bench/oracle/tiny.nk.pkl
    bench/oracle/medium.pk.pkl
    bench/oracle/medium.nk.pkl
    bench/oracle/large.pk.pkl
    bench/oracle/large.nk.pkl

Each record contains the corpus fingerprint, the exact ``cafa_eval``
call kwargs, and pickled ``df`` / ``dfs_best`` DataFrames. The
fingerprint binds the oracle to a specific corpus contents — if the
fixture fingerprint drifts, the test fails loudly instead of silently
comparing against the wrong snapshot.

Synthetic corpora
-----------------

Three corpora, all generated deterministically by ``bench.corpus``:

``tiny``
    Small smoke-test corpus. Used for fast iteration during
    development.
``medium``
    Mid-size corpus that exercises every metric column.
``large``
    Large synthetic corpus that stresses the sparse kernels. Closer
    to real CAFA workloads in density and shape.

None of the corpora include real biological data; they are seeded
random DAGs and random prediction scores chosen to cover all code
paths.
