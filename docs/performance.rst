Performance
===========

All benchmarks below are on a real GOA-derived benchmark corpus
(8 712 BP / 4 992 MF / 5 125 CC ground-truth proteins, ~4.45 M
prediction rows) with ``th_step=0.01`` and ``n_cpu=1``. Upstream is
pristine ``claradepaolis/CAFA-evaluator-PK 16a6a6d``.

Headline
--------

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 20

   * - Variant
     - Upstream
     - Fork (B1–B7)
     - Speedup
   * - NK / LK
     - 92.96 s
     - 4.08 s
     - **22.8×**
   * - PK
     - 418.53 s
     - 10.33 s
     - **40.5×**

NK and LK share the same code path (``gt_exclude is None``); the fork
speeds up both equally. The PK path (``gt_exclude is not None``) gets
a larger win because its per-protein Python loops were heavier to
begin with.

Phase-by-phase breakdown
------------------------

Phase A — cherry-picked ideas from `T0chka/CAFA-evaluator-PK-speedup`_
(Antonina Dolgorukova). Bit-exact (``atol=0, rtol=0``) against the
oracle.

.. _T0chka/CAFA-evaluator-PK-speedup: https://github.com/T0chka/CAFA-evaluator-PK-speedup

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Phase
     - Change
   * - A.1
     - ``weighted_only`` fast path — skip the unweighted metric pass when
       the caller only needs ``_w`` columns.
   * - A.2
     - ``compute_metrics``: ``fork``-pool with shared-state initializer,
       eliminates per-chunk pickling of ``g/p/toi/n_gt``.
   * - A.3
     - ``graph.propagate``: cached children adjacency, fill-mode
       restricted to zero rows, optional ``spawn`` shared-memory parallel
       propagation.
   * - A.4
     - ``parser.pred_parser``: precomputed ``term_index``, buffered file
       reads, single dict lookup per term.

Phase B — sparse + vectorised rewrites. Parity gate loosened to
``rtol=1e-6, atol=1e-9`` because the rewrites reorder inner per-protein
sums.

.. list-table::
   :header-rows: 1
   :widths: 10 15 75

   * - Phase
     - Area
     - Change
   * - B1
     - NK kernel
     - Sparse scatter + right-to-left cumsum. Replaces per-threshold
       dense mask scan (``O(n_tau · n_prot · n_toi)``) with
       ``O(nnz + n_prot · n_tau)``.
   * - B2
     - PK kernel
     - Sparse scatter extended with per-protein exclude AND — the
       per-protein ``toi_perprotein`` / ``gt_perprotein`` Python lists
       are never materialised.
   * - B3
     - Parser
     - PyArrow ``read_csv`` + dictionary-encoded ``(pid, tid)`` +
       vectorised per-namespace group-max. 2.3-2.7× faster than the
       per-line loop on multi-million-row inputs.
   * - B4
     - Propagation
     - Sparse push-up kernel. Flat CSR ancestor cache + scatter over
       input non-zeros + ``np.maximum.reduceat`` for per-group max.
       Skips every term that has no predictions.
   * - B6
     - gt_parser
     - Skip dead dense scans in ``propagate`` when the caller is on the
       sparse path; accept a ``triples=`` passthrough so
       ``_propagate_sparse_pushup`` reuses the caller's scatter
       coordinates instead of calling ``np.nonzero`` on the full
       ``(n_prot, n_terms)`` bool matrix.
   * - B7
     - Prep
     - Trim ``compute_metrics`` and ``evaluate_prediction`` prep work:
       detect ``toi_is_full`` once and collapse column slices to
       no-ops, defer ``g``/``p``/``pred_sub`` materialisation to the
       branch that actually uses them, replace the per-protein
       ``setdiff1d`` loop with a vectorised
       ``_count_proteins_in_toi`` helper.

Phase B7 drilldown
------------------

Before B7, the PK prep block dominated BP namespace cost:

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Line
     - Wall time
   * - ``gt_matrix[:, toi].sum(1) > 0``
     - 1.15 s
   * - ``g = gt_with_annots[:, toi]``
     - 1.07 s
   * - ``p = pred[proteins_has_gt, :][:, toi]``
     - 2.29 s
   * - ``excluded_mask``, ``valid_gt_mask``, ``n_gt``
     - 0.59 s
   * - ``pred_sub = pred[proteins_has_gt, :]``
     - 0.61 s

After B7, end-to-end on the same corpus:

.. list-table::
   :header-rows: 1
   :widths: 55 25 25

   * - Measurement
     - Before B7
     - After B7
   * - NK end-to-end
     - 6.68 s
     - 4.08 s
   * - PK end-to-end
     - 28.73 s
     - 10.33 s
   * - ``compute_metrics`` PK BP prep
     - 5.443 s
     - 0.599 s
   * - ``compute_metrics`` PK BP kernel
     - 1.805 s
     - 0.746 s
   * - ``evaluate_prediction`` PK total
     - 5.68 s
     - 2.05 s

Micro-benchmarks
----------------

Individual hot spots, isolated:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Kernel
     - Corpus
     - Before
     - After
   * - ``pred_parser`` (NK)
     - 4.45M rows
     - 1.22 s
     - 0.45 s (2.72×)
   * - ``pred_parser`` (PK)
     - 4.45M rows
     - 1.44 s
     - 0.63 s (2.28×)
   * - ``compute_confusion_matrix_exclude`` (PK, n_cpu=1)
     - real PK
     - 2.73 s
     - 1.45 s (1.88×)
   * - ``gt_parser`` (NK, cold cache)
     - real corpus
     - 1.71 s
     - 1.46 s
   * - ``gt_parser`` (PK, cold cache)
     - real corpus
     - 2.21 s
     - 0.64 s
   * - ``gt_parser`` (PK, hot ancestor cache)
     - real corpus
     - 2.21 s
     - 0.16 s

What is left on the table
-------------------------

* Phase B5 — optional numba kernel on the per-namespace parser
  reduction. Not scheduled; the current PyArrow path already brings
  parser time below the confusion-matrix cost.
* A numba-JIT legacy parser for environments where ``pyarrow`` is
  unavailable. Only worth building if profiling on a legacy-only
  install warrants it.
