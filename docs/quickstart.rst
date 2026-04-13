Quickstart
==========

CLI
---

The fork keeps the upstream command-line interface. A minimal call::

    cafaeval \
        path/to/go.obo \
        path/to/predictions/ \
        path/to/ground_truth.tsv \
        --ia path/to/ia.tsv \
        --toi_file path/to/toi.tsv \
        --th_step 0.01 \
        --n_cpu 1

For partial-knowledge evaluation, add the ``--exclude`` flag pointing
at the known-terms file::

    cafaeval ... --exclude path/to/known_terms.tsv

Python API
----------

.. code-block:: python

    from cafaeval.evaluation import cafa_eval

    df, dfs_best = cafa_eval(
        "path/to/go.obo",
        "path/to/predictions/",
        "path/to/ground_truth.tsv",
        ia="path/to/ia.tsv",
        exclude="path/to/known_terms.tsv",  # PK; omit for NK/LK
        toi_file="path/to/toi.tsv",
        norm="cafa",
        prop="max",
        th_step=0.01,
        n_cpu=1,
    )

Returns
^^^^^^^

``df``
    ``pandas.DataFrame`` with one row per (namespace, threshold),
    containing all metrics (``pr``, ``rc``, ``f``, ``s``, and their
    weighted variants).

``dfs_best``
    ``dict`` keyed by metric name (``f``, ``f_w``, ``s``, ``f_micro``,
    ``f_micro_w``) mapping to the row of ``df`` at which each metric
    hits its optimum.

What is different from upstream
-------------------------------

By default the fork runs with:

* ``CAFAEVAL_SPARSE=1`` — sparse confusion matrix and sparse push-up
  propagation kernels (see :doc:`architecture`).
* ``CAFAEVAL_FAST_PARSER=1`` — PyArrow-backed vectorised
  prediction-file parser (no effect without the ``[fast]`` extra
  installed).

Both flags are safe to leave on; the slow paths exist only for A/B
comparison and fallback. Outputs agree with upstream within
``rtol=1e-6, atol=1e-9`` (see :doc:`parity`).
