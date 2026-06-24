Architecture
============

This page documents the fork-specific kernels. For the upstream
algorithm and its mathematical definitions, see the `CAFA-evaluator
paper`_.

.. _CAFA-evaluator paper: https://academic.oup.com/bioinformatics/article/40/8/btae476/7721017

Sparse confusion matrix kernel
------------------------------

Upstream evaluates one confusion matrix per threshold by scanning
a dense ``(n_prot, n_toi)`` mask of active predictions. For ``n_tau``
thresholds this is ``O(n_tau · n_prot · n_toi)`` cell visits. At
``th_step=0.01`` with ~25 000 terms and ~8 000 proteins that is 20 B
visits per namespace.

The fork replaces the scan with a single scatter pass followed by one
right-to-left cumulative sum:

1. Walk non-zero predictions once. For each prediction ``(i, j, s)``,
   compute its **highest active threshold bin** ::

       bin = np.searchsorted(tau_arr, s, side="right") - 1

2. Scatter ``(row, bin)`` into a flat
   ``(n_prot · n_tau)``-sized ``np.bincount``. This gives, for each
   protein and each threshold, the number of predictions whose score
   sits *exactly* in that bin.

3. Run ``np.cumsum`` from right to left along the threshold axis.
   A prediction active at bin ``b`` is also active at every lower
   bin, so the right-to-left cumsum converts "count in bin ``b``" to
   "count of active predictions at threshold ``tau[b]`` or lower".

4. Combine with the ground-truth vector to produce TP / FP / FN in
   ``O(n_prot · n_tau)`` arithmetic.

Cost drops from ``O(n_tau · n_prot · n_toi)`` to
``O(nnz + n_prot · n_tau)``. On real CAFA inputs that is a ~30×
reduction in arithmetic volume.

The PK variant (``compute_confusion_matrix_exclude_sparse``) extends
the same idea with two dense bool filters expressed as a single AND:

.. code-block:: python

    toi_mask                 # (n_terms,)  global terms of interest
    excluded_mask            # (n_prot, n_terms)  per-protein exclude

The scatter only sees non-zeros that survive both masks, so
``toi_perprotein`` / ``gt_perprotein`` — the upstream per-protein
Python lists of valid columns — are never materialised.

Sparse push-up propagation
--------------------------

Upstream propagates scores along the DAG by sweeping terms in a
topological order: for each term, walk its children and take the
per-protein max. This is ``O(n_terms · n_prot · avg_children)``
cell updates, and it pays for every term in the ontology even if only
a handful have predictions.

The fork walks the DAG differently:

1. **Flat ancestor cache.** ``graph._ancestors_csr`` precomputes the
   transitive ancestor set of every term once per ``Graph``
   instance, flattened into a CSR-style ``(indptr, indices)`` pair.
   Ancestor lookups are then constant-time index slicing.

2. **Scatter input non-zeros.** Collect all input non-zeros
   ``(row, col, score)``. For each non-zero, gather the flat ancestor
   list for its ``col`` via vectorised ``np.repeat`` offsets — no
   Python loop — producing an expanded ``(row, ancestor)`` triple
   stream.

3. **Stable sort + reduceat.** Encode ``(row, ancestor)`` as a single
   ``int64`` key, stable-sort once, and reduce per-group with
   ``np.maximum.reduceat``. This replaces the per-term sweep with a
   single pass over ``R`` expanded triples.

4. **Write group-maxes back in place.** ``mode='fill'`` semantics are
   preserved by snapshotting the input non-zeros and writing them
   back on top after the group-max.

Cost is ``O(nnz · avg_ancestors + R log R)``. On a sparse CAFA input
where ``nnz`` is a tiny fraction of ``n_terms · n_prot``, this skips
over every term that has no predictions.

The Phase B6 ``triples=`` passthrough lets the caller (``gt_parser``
or the prediction parser) pass the non-zero coordinates it already
has, avoiding an ``np.nonzero`` scan of the full ``(n_prot, n_terms)``
bool matrix on the hot path. On the BP ontology that scan was 488 M
cells to find ~27 000 non-zeros.

Sparse DAG adjacency
--------------------

Upstream stores the ontology DAG as a dense ``(n_terms, n_terms)``
boolean adjacency matrix (``Graph.dag``). For the BP namespace
(~25 000 terms) that is a ~600 MB allocation whose every cell is read
by ``dag.sum(axis=0)`` (in-degrees for the topological sort),
``dag.sum(axis=1)`` (terms-of-interest), and ``np.nonzero(dag)``
(parent lists for the ancestor cache) — all ``O(n_terms²)`` scans of
a matrix that is >99.9 % zeros.

The fork drops the dense matrix entirely. Each term already keeps its
parent set (``adj``) and child set (``children``); from these, a CSR
parent/child index pair plus the in/out-degree vectors are built once
in ``Graph._build_sparse_dag``. Every former consumer reads the
sparse form:

* in-degrees for the Kahn topological sort come from the child-count
  vector (``_child_count``), not ``dag.sum(axis=0)``;
* ``toi`` comes from ``_parent_count > 0``;
* ``_children_cache`` / ``_ancestors_csr`` slice the CSR index arrays
  instead of calling ``np.flatnonzero(dag[:, t])`` / ``np.nonzero(dag)``.

The counts and CSR slices reproduce the dense reductions exactly
(``_parent_count[i] == dag.sum(axis=1)[i]``; CSR slices are ascending,
matching ``np.nonzero``), so the topological order, ``toi`` and
ancestor sets are bit-identical. The Kahn queue also moves from a
Python ``list`` (``pop(0)`` is ``O(n)`` → ``O(n²)``) to
``collections.deque`` (``popleft`` is ``O(1)``); the FIFO order, and
therefore ``Graph.order``, is unchanged.

Result: ``Graph`` construction on the full GO drops peak adjacency
memory from ~600 MB to a few MB, and builds ~3× faster (no
``O(n_terms²)`` allocation or scans).

Sparse prediction storage (no dense ``(n_prot, n_terms)``)
----------------------------------------------------------

Upstream — and the fork through Phase B — materialise the prediction
scores as a dense ``(n_prot, n_terms)`` ``float64`` matrix. At
full-GO scale (~25 000 terms × tens of thousands of proteins) that is
the single largest allocation in an evaluation — tens of GB — yet it
is almost entirely zero: each protein predicts a few hundred terms.

The fork never builds it. The vectorised parser already computes the
prediction non-zeros as COO ``(rows, cols, scores)`` for its
group-max; instead of scattering them into a dense matrix it keeps the
COO, propagates it through ``graph.propagate_to_coo`` (a sparse-native
``mode='max'`` push-up that returns propagated COO, bit-identical to
the dense ``propagate``), and stores the result as a
``scipy.sparse.csr_matrix`` on ``Prediction.matrix``.

The sparse confusion-matrix kernels already consume only the
non-zeros, so they read the CSR directly via its ``data`` / ``indices``
/ ``indptr`` arrays (not ``scipy.sparse.find``, which re-runs
``sum_duplicates`` — a full argsort of every non-zero — on each call).
Ground truth stays dense, so the TP gather ``g[nz_rows, nz_cols]`` is
unchanged. The dense fallback path (``CAFAEVAL_SPARSE=0``) densifies
only its own slice.

On a 25 000-term × 30 000-protein corpus this cuts peak RSS from
**11.9 GB to 7.0 GB (−41 %)** while *also* shaving wall-clock (the
kernel no longer scans, and ``find`` is gone), bit-exact against the
dense path on every NK / PK / weighted / no-orphans scenario.

Vectorised prediction parser
----------------------------

``parser._pred_parser_vectorised`` replaces the upstream per-line
loop with PyArrow ``read_csv``. The high-level shape is:

1. ``pyarrow.csv.read_csv`` reads the full prediction file at native
   speed into three columns: ``pid``, ``tid``, ``score``.

2. ``pid`` and ``tid`` are **dictionary-encoded**. The Python-level
   lookups into ``ns_dict`` / ``gts[ns].ids`` / ``term_index`` run
   once per *unique* value instead of once per row. On a 4.45M-row
   file with ~20 000 unique terms and ~8 000 unique proteins, the
   lookup cost drops by three orders of magnitude.

3. Per-namespace filtering uses vectorised comparisons against the
   integer code arrays. No Python-level masking.

4. Duplicate ``(protein, term)`` rows are collapsed to their max via
   the same sort + ``np.maximum.reduceat`` pattern as the propagation
   kernel. This matches the upstream loop's "store the max if higher"
   semantics bit-exactly because the maximum is order-independent.

5. Alt-id expansion is rare on CAFA inputs and would break the fast
   path's vectorised structure. Unique terms whose column lookup
   resolves to the sentinel ``-2`` are handed off to a short Python
   loop over just the affected rows.

Gate: ``CAFAEVAL_FAST_PARSER=1`` (default). Falls back to the legacy
loop if PyArrow is not installed or the fast path raises. The
``max_terms`` cap is order-sensitive and also routes to the legacy
loop.

toi_is_full short-circuit
-------------------------

``compute_metrics`` and ``evaluate_prediction`` frequently run on
``toi = np.arange(n_terms)`` — i.e. the caller wants metrics over the
entire namespace. In that case column slices of the form
``gt_matrix[:, toi]`` are no-ops that copy the full matrix, and
``toi_mask`` can be built as a single ``np.ones``.

The fork detects this path once per ``compute_metrics`` call:

.. code-block:: python

    toi_is_full = (
        isinstance(toi, np.ndarray)
        and toi.ndim == 1
        and toi.size == n_terms
        and bool(np.array_equal(toi, np.arange(n_terms)))
    )

When ``toi_is_full`` is true:

* ``(gt_matrix[:, toi] != 0).any(axis=1)`` is replaced by
  ``(gt_matrix != 0).any(axis=1)``.
* ``g = gt_with_annots[:, toi]`` / ``p = pred[...][:, toi]`` are
  skipped entirely on the sparse path — those variables are dead
  when the sparse kernel takes the non-zeros directly.
* ``toi_mask`` is built as ``np.ones(n_terms, dtype=bool)`` instead
  of ``np.zeros`` + scatter.

On the benchmark corpus this alone removed ~3.4 s of BP-namespace
prep cost per ``compute_metrics`` call.

Triples passthrough API
-----------------------

Several kernels in the fork accept an optional
``triples=(nz_rows, nz_cols, nz_scores)`` keyword argument. This lets
a caller that already has non-zero coordinates (e.g. because it just
scattered them to build the matrix) forward those coordinates through
instead of forcing the callee to rediscover them with
``np.nonzero``.

The convention is:

* Callers pass ``triples=...`` as an **internal** (underscore-prefixed)
  kwarg so the public API shape stays stable.
* Callees treat ``triples`` as a *hint*: they are free to ignore it
  on dense fallbacks, or use it to skip an expensive rediscovery.
* The fork's dense fallback paths are never triples-aware — they stay
  as close to upstream as possible so the self-parity test remains a
  meaningful check.

Current users of the passthrough:

* ``gt_parser`` → ``graph.propagate`` → ``_propagate_sparse_pushup``
  (Phase B6).

Logging
-------

All ``print()`` calls in the upstream library are replaced by
structured ``logging`` calls under four loggers:

* ``cafaeval.parser``
* ``cafaeval.propagate``
* ``cafaeval.metrics``
* ``cafaeval.eval``

Extras are passed via ``logger.info(..., extra={...})`` so machine
consumers can read structured timing payloads without regex-ing log
strings.
