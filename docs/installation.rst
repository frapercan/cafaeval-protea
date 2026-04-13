Installation
============

From source (recommended while the fork is pre-release)::

    git clone https://github.com/frapercan/cafaeval-protea.git
    cd cafaeval-protea
    pip install -e .

To enable the vectorised prediction-file parser (Phase B3, roughly
2.5× faster on multi-million-row inputs), install the optional
``[fast]`` extra::

    pip install -e '.[fast]'

This pulls in ``pyarrow>=12``. The import is lazy — without the extra,
``pred_parser`` transparently falls back to the legacy line-by-line
loop.

Runtime dependencies
--------------------

* Python ≥ 3.9
* ``numpy``
* ``pandas``
* ``matplotlib``

Optional
^^^^^^^^

* ``pyarrow>=12`` — vectorised parser (``[fast]`` extra)

Development
-----------

The parity harness needs a second environment with pristine upstream
installed to re-freeze the oracle pickles. See :doc:`parity` for the
workflow.

To run the parity tests against the checked-in oracle::

    pip install -e '.[fast]'
    pip install pytest
    pytest tests/diff/ -v

Environment variables
---------------------

``CAFAEVAL_SPARSE``
    Default ``1``. Set to ``0`` to force the dense multiprocessing
    fallback for confusion matrix and propagation kernels. Used for
    A/B comparisons and the in-fork self-parity test.

``CAFAEVAL_FAST_PARSER``
    Default ``1``. Set to ``0`` to force the legacy line-by-line
    prediction parser.

``CAFAEVAL_PARITY_PHASE``
    Default ``B`` (``rtol=1e-6, atol=1e-9``). Set to ``A`` to run the
    parity harness under bit-exact tolerance (``atol=0, rtol=0``);
    only the Phase A cherry-picks pass under this setting.
