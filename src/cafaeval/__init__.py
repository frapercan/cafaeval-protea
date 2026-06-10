"""cafaeval: a parity-preserving speedup fork of CAFA-evaluator-PK.

The public API is unchanged from upstream. The main entry point is
:func:`cafaeval.evaluation.cafa_eval`, which loads an ontology, a ground
truth file and a directory of prediction files and returns the
precision-recall curves plus the per-metric optimal rows (Fmax, Smin and
their weighted and micro-averaged variants).

Every optimisation in this fork is gated by a parity harness against a
frozen snapshot of pristine upstream; see the ``tests/diff`` suite and
the project documentation for details.
"""

__version__ = "0.1.0"
