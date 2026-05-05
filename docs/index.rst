cafaeval-protea
===============

A speedup fork of `cafaeval`_ (`CAFA-evaluator-PK`_) with a bit-exact
parity harness against pristine upstream.

.. _cafaeval: https://github.com/BioComputingUP/CAFA-evaluator
.. _CAFA-evaluator-PK: https://github.com/claradepaolis/CAFA-evaluator-PK

At a glance
-----------

End-to-end speedup versus upstream ``CAFA-evaluator-PK 16a6a6d`` on a
real GOA-derived benchmark corpus, measured at ``th_step=0.01`` (the
CAFA default), ``n_cpu=1``:

+---------+-----------+--------------+-----------+
| Variant | Upstream  | Fork (B1–B7) | Speedup   |
+=========+===========+==============+===========+
| NK/LK   | 92.96 s   | 4.08 s       | **22.8×** |
+---------+-----------+--------------+-----------+
| PK      | 418.53 s  | 10.33 s      | **40.5×** |
+---------+-----------+--------------+-----------+

Every optimisation in this fork is gated by a parity harness that runs
the fork against a frozen snapshot of unmodified upstream on three
synthetic corpora, under ``rtol=1e-6, atol=1e-9``. See :doc:`parity`.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Technical

   performance
   parity
   architecture

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/evaluation
   api/graph
   api/parser

Attribution
-----------

This fork descends from

1. `BioComputingUP/CAFA-evaluator`_ (original upstream, Piovesan et al.)
2. `claradepaolis/CAFA-evaluator-PK`_ at commit ``16a6a6d`` (partial
   knowledge variant, de Paolis Kaluza et al.)
3. `T0chka/CAFA-evaluator-PK-speedup`_ (cherry-picked Phase A speedups,
   Dolgorukova)

See ``NOTICE`` for the full chain and ``CHANGES.md`` for the per-phase
modification log, as required by GPLv3 §5.a.

.. _BioComputingUP/CAFA-evaluator: https://github.com/BioComputingUP/CAFA-evaluator
.. _claradepaolis/CAFA-evaluator-PK: https://github.com/claradepaolis/CAFA-evaluator-PK
.. _T0chka/CAFA-evaluator-PK-speedup: https://github.com/T0chka/CAFA-evaluator-PK-speedup

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
