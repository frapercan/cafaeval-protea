"""Sparse vs dense parity for ``mode='fill'`` propagation.

``prop='fill'`` is the production default in the PROTEA stack, and the metric
numbers (f_micro_w, Fmax) depend on it. The sparse path (``CAFAEVAL_SPARSE=1``)
and the dense serial sweep (``CAFAEVAL_SPARSE=0``) must produce a BIT-IDENTICAL
propagated matrix, both matching the canonical upstream cafaeval ``fill``.

The original sparse pushup scattered every input non-zero straight into every
ancestor and group-maxed. That is correct for ``mode='max'`` but WRONG for
``fill``: ``fill`` is a stepwise recurrence where a non-zero intermediate node
"blocks" deeper descendants (it keeps its own value and that value, not the
descendant's, flows up to the parent). The pushup ignored those blockers and
overshot, e.g. setting a parent to a deep leaf's score even though a
lower-scored non-zero node sat between them. ``_propagate_sparse_fill`` walks
the topological order instead and is bit-identical to the dense sweep.
"""
from __future__ import annotations

import os

import numpy as np

from cafaeval.graph import Graph, propagate


def _make_graph(term_ids, parent_edges):
    """Build a minimal single-namespace Graph.

    ``term_ids`` fixes the term -> index order; ``parent_edges`` is a list of
    ``(child_id, parent_id)`` (child ``is_a`` parent).
    """
    children_parents = {t: [] for t in term_ids}
    for child, parent in parent_edges:
        children_parents[child].append(parent)
    terms_dict = {
        t: {"name": t, "namespace": "test", "def": "",
            "alt_id": [], "rel": children_parents[t]}
        for t in term_ids
    }
    return Graph("test", terms_dict)


def _upstream_fill(matrix, ont):
    """Verbatim canonical upstream cafaeval ``fill`` (BioComputingUP /
    claradepaolis CAFA-evaluator-PK), kept here as an independent oracle so the
    test does not merely compare the fork against itself.
    """
    m = matrix.copy()
    order = np.asarray(ont.order)
    n = ont.idxs
    # dense parent adjacency: dag[i, j] == 1 iff j is a parent of i
    dag = np.zeros((n, n), dtype=bool)
    for t in range(n):
        for p in ont._par_idx[ont._par_indptr[t]:ont._par_indptr[t + 1]]:
            dag[t, p] = True
    deepest = np.where(np.sum(m[:, order], axis=0) > 0)[0][0]
    order_ = np.delete(order, list(range(0, deepest)))
    for i in order_:
        children = np.where(dag[:, i] != 0)[0]
        if children.size > 0:
            cols = np.concatenate((children, [i]))
            rows = np.where(m[:, i] == 0)[0]
            if rows.size:
                idx = np.ix_(rows, cols)
                m[rows, i] = m[idx].max(axis=1)
    return m


def _propagate_fill(matrix, ont, sparse):
    os.environ["CAFAEVAL_SPARSE"] = "1" if sparse else "0"
    for attr in ("_ancestors_csr", "_children_by_term"):
        if hasattr(ont, attr):
            delattr(ont, attr)
    out = matrix.copy()
    propagate(out, ont, ont.order, mode="fill", parallel=0)
    return out


def _run_all(ont, base):
    dense = _propagate_fill(base, ont, sparse=False)
    sparse = _propagate_fill(base, ont, sparse=True)
    upstream = _upstream_fill(base, ont)
    return dense, sparse, upstream


def test_fill_blocked_intermediate():
    """The exact counter-example the buggy pushup got wrong.

    Chain leaf -> mid -> root with mid originally non-zero and *lower* than the
    leaf. ``fill`` must give root = mid's value (0.5), not the leaf's (1.0):
    the non-zero mid blocks the leaf. The pushup produced 1.0.
    """
    ont = _make_graph(["leaf", "mid", "root"],
                      [("leaf", "mid"), ("mid", "root")])
    idx = {t: ont.terms_dict[t]["index"] for t in ("leaf", "mid", "root")}
    base = np.zeros((1, ont.idxs))
    base[0, idx["leaf"]] = 1.0
    base[0, idx["mid"]] = 0.5

    dense, sparse, upstream = _run_all(ont, base)

    expected = base.copy()
    expected[0, idx["root"]] = 0.5  # mid (0.5) flows up, leaf (1.0) is blocked

    np.testing.assert_array_equal(dense, expected)
    np.testing.assert_array_equal(sparse, expected)
    np.testing.assert_array_equal(upstream, expected)
    np.testing.assert_array_equal(sparse, dense)


def test_fill_shared_ancestor_multi_term():
    """Multi-term row with a shared ancestor and a non-zero intermediate.

    Two leaves feed a mid (originally 0.6); mid feeds the root. The root must
    take the mid's blocked value (0.6), not the higher leaf (0.9).
    """
    ont = _make_graph(["l1", "l2", "mid", "root"],
                      [("l1", "mid"), ("l2", "mid"), ("mid", "root")])
    idx = {t: ont.terms_dict[t]["index"] for t in ("l1", "l2", "mid", "root")}
    base = np.zeros((1, ont.idxs))
    base[0, idx["l1"]] = 0.9
    base[0, idx["l2"]] = 0.3
    base[0, idx["mid"]] = 0.6

    dense, sparse, upstream = _run_all(ont, base)

    expected = base.copy()
    expected[0, idx["root"]] = 0.6

    np.testing.assert_array_equal(sparse, expected)
    np.testing.assert_array_equal(dense, expected)
    np.testing.assert_array_equal(upstream, expected)


def test_fill_pure_chain_no_block():
    """No intermediate blocker: a single leaf value fills the whole chain."""
    ont = _make_graph(["leaf", "mid", "root"],
                      [("leaf", "mid"), ("mid", "root")])
    idx = {t: ont.terms_dict[t]["index"] for t in ("leaf", "mid", "root")}
    base = np.zeros((1, ont.idxs))
    base[0, idx["leaf"]] = 0.8

    dense, sparse, upstream = _run_all(ont, base)

    expected = base.copy()
    expected[0, idx["mid"]] = 0.8
    expected[0, idx["root"]] = 0.8

    np.testing.assert_array_equal(sparse, expected)
    np.testing.assert_array_equal(dense, expected)
    np.testing.assert_array_equal(upstream, expected)


def test_fill_two_level_block():
    """Blocker two levels below the root still caps everything above it."""
    ont = _make_graph(["leaf", "m1", "m2", "root"],
                      [("leaf", "m1"), ("m1", "m2"), ("m2", "root")])
    idx = {t: ont.terms_dict[t]["index"] for t in ("leaf", "m1", "m2", "root")}
    base = np.zeros((1, ont.idxs))
    base[0, idx["leaf"]] = 1.0
    base[0, idx["m1"]] = 0.5

    dense, sparse, upstream = _run_all(ont, base)

    expected = base.copy()
    expected[0, idx["m2"]] = 0.5
    expected[0, idx["root"]] = 0.5

    np.testing.assert_array_equal(sparse, expected)
    np.testing.assert_array_equal(dense, expected)
    np.testing.assert_array_equal(upstream, expected)


def test_fill_high_tau_bpo_shaped():
    """A wide, deep, multi-term row shaped like the high-tau biological_process
    rows where the divergence was originally observed: many leaves, several
    shared ancestors, a few non-zero intermediate blockers at high score.
    """
    # 3 leaves -> 2 mids -> 1 hub -> root. Cross edges make mids share leaves.
    term_ids = ["la", "lb", "lc", "ma", "mb", "hub", "root"]
    edges = [
        ("la", "ma"), ("lb", "ma"), ("lb", "mb"), ("lc", "mb"),
        ("ma", "hub"), ("mb", "hub"), ("hub", "root"),
    ]
    ont = _make_graph(term_ids, edges)
    idx = {t: ont.terms_dict[t]["index"] for t in term_ids}
    base = np.zeros((1, ont.idxs))
    # leaves at high tau-ish scores
    base[0, idx["la"]] = 0.99
    base[0, idx["lb"]] = 0.80
    base[0, idx["lc"]] = 0.75
    # mb is a non-zero blocker LOWER than its descendant lb (0.80)
    base[0, idx["mb"]] = 0.50

    dense, sparse, upstream = _run_all(ont, base)

    # ma: zero -> max(la=0.99, lb=0.80) = 0.99
    # mb: blocked at 0.50 (lb/lc do not overwrite it)
    # hub: zero -> max(ma=0.99, mb=0.50) = 0.99
    # root: zero -> hub = 0.99
    expected = base.copy()
    expected[0, idx["ma"]] = 0.99
    expected[0, idx["hub"]] = 0.99
    expected[0, idx["root"]] = 0.99

    np.testing.assert_array_equal(sparse, expected)
    np.testing.assert_array_equal(dense, expected)
    np.testing.assert_array_equal(upstream, expected)
    # The headline invariant: sparse and dense agree bit-for-bit.
    np.testing.assert_array_equal(sparse, dense)


def test_fill_randomized_dag_stress():
    """Random DAGs + random multi-row sparse inputs: sparse, dense and the
    upstream oracle must agree bit-for-bit on every trial.
    """
    rng = np.random.default_rng(20240625)
    for _ in range(200):
        n = int(rng.integers(5, 16))
        term_ids = [f"T{i}" for i in range(n)]
        # i < j edge guarantees acyclicity; index order respects topology
        edges = []
        for i in range(n):
            candidates = list(range(i + 1, n))
            rng.shuffle(candidates)
            for j in candidates[: int(rng.integers(0, 3))]:
                edges.append((term_ids[i], term_ids[j]))
        ont = _make_graph(term_ids, edges)

        n_rows = int(rng.integers(1, 6))
        base = np.zeros((n_rows, ont.idxs))
        for r in range(n_rows):
            n_cells = int(rng.integers(1, n))
            for c in rng.choice(n, size=n_cells, replace=False):
                base[r, c] = round(float(rng.uniform(0.01, 1.0)), 4)
        if base.sum() == 0:  # propagate raises on an all-zero matrix
            continue

        dense, sparse, upstream = _run_all(ont, base)
        np.testing.assert_array_equal(sparse, dense)
        np.testing.assert_array_equal(sparse, upstream)
