"""The per-protein vectors must agree with the aggregate they were reduced into.

The kernels build ``tp_at_tau`` and ``pred_at_tau`` per protein and collapse them
with ``.sum(axis=0)``. Emitting the uncollapsed arrays lets a caller stratify a
score by a per-protein property, which is the whole reason the sink exists.

Two properties matter and neither is obvious from reading the code.

First, passing no sink must change nothing at all. An optional feature that
perturbs the default path is worse than no feature, because every number already
published would have to be re-derived to find out whether it moved.

Second, the emitted arrays must reproduce the published totals exactly. Two
routes to the same quantity drift unless something checks, and a stratified
table whose cells do not sum to the headline is worse than no stratified table.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from cafaeval.evaluation import (
    PerProteinSink,
    compute_confusion_matrix_sparse,
)


def _toy():
    """Five proteins, four terms, a handful of thresholds."""
    rng = np.random.default_rng(0)
    n_prot, n_terms = 5, 4
    pred = csr_matrix(np.round(rng.random((n_prot, n_terms)), 2))
    gt = (rng.random((n_prot, n_terms)) > 0.5).astype(np.float64)
    toi = np.arange(n_terms)
    n_gt = gt[:, toi].sum(axis=1)
    tau = np.round(np.arange(0.1, 1.0, 0.1), 2)
    return tau, gt, pred, toi, n_gt


class TestTheDefaultPathIsUntouched:
    def test_no_sink_gives_the_same_metrics_as_before(self) -> None:
        tau, gt, pred, toi, n_gt = _toy()
        a = compute_confusion_matrix_sparse(tau, gt, pred, toi, n_gt)
        b = compute_confusion_matrix_sparse(tau, gt, pred, toi, n_gt, None, None)
        np.testing.assert_array_equal(a, b)

    def test_passing_a_sink_does_not_change_the_metrics(self) -> None:
        """The sink observes; it must not participate."""
        tau, gt, pred, toi, n_gt = _toy()
        without = compute_confusion_matrix_sparse(tau, gt, pred, toi, n_gt)
        sink = PerProteinSink()
        with_sink = compute_confusion_matrix_sparse(
            tau, gt, pred, toi, n_gt, per_protein_sink=sink
        )
        np.testing.assert_array_equal(without, with_sink)


class TestTheEmittedArraysReproduceTheAggregate:
    def test_the_sink_receives_one_record(self) -> None:
        tau, gt, pred, toi, n_gt = _toy()
        sink = PerProteinSink()
        compute_confusion_matrix_sparse(tau, gt, pred, toi, n_gt, per_protein_sink=sink)
        assert len(sink.records) == 1

    def test_the_shape_is_one_row_per_protein(self) -> None:
        tau, gt, pred, toi, n_gt = _toy()
        sink = PerProteinSink()
        compute_confusion_matrix_sparse(tau, gt, pred, toi, n_gt, per_protein_sink=sink)
        rec = sink.records[0]
        assert rec["tp_at_tau"].shape == (gt.shape[0], len(tau))
        assert rec["pred_at_tau"].shape == (gt.shape[0], len(tau))

    def test_the_column_sums_are_the_published_totals(self) -> None:
        """The property the whole feature rests on."""
        tau, gt, pred, toi, n_gt = _toy()
        sink = PerProteinSink()
        metrics = compute_confusion_matrix_sparse(
            tau, gt, pred, toi, n_gt, per_protein_sink=sink
        )
        rec = sink.records[0]
        tp_totals = rec["tp_at_tau"].sum(axis=0)
        pred_totals = rec["pred_at_tau"].sum(axis=0)
        np.testing.assert_allclose(metrics[:, 1], tp_totals, rtol=0, atol=1e-12)
        np.testing.assert_allclose(metrics[:, 2], pred_totals - tp_totals, rtol=0, atol=1e-12)

    def test_the_coverage_column_is_the_count_of_predicting_proteins(self) -> None:
        tau, gt, pred, toi, n_gt = _toy()
        sink = PerProteinSink()
        metrics = compute_confusion_matrix_sparse(
            tau, gt, pred, toi, n_gt, per_protein_sink=sink
        )
        rec = sink.records[0]
        np.testing.assert_array_equal(
            metrics[:, 0], (rec["pred_at_tau"] > 0).sum(axis=0)
        )


class TestTheSinkOwnsItsData:
    def test_the_arrays_are_copies_not_views(self) -> None:
        """The kernels reuse buffers; a view would change under the caller."""
        tau, gt, pred, toi, n_gt = _toy()
        sink = PerProteinSink()
        compute_confusion_matrix_sparse(tau, gt, pred, toi, n_gt, per_protein_sink=sink)
        rec = sink.records[0]
        before = rec["tp_at_tau"].copy()
        compute_confusion_matrix_sparse(tau, gt, pred, toi, n_gt, per_protein_sink=sink)
        np.testing.assert_array_equal(rec["tp_at_tau"], before)


class TestTheSinkIsReachableFromTheRealEntryPoint:
    """The kernels are not what consumers call.

    PROTEA calls ``cafa_eval``, which calls ``evaluate_prediction``, which calls
    ``compute_metrics``, which calls a kernel. A sink wired only into the kernel
    is a feature that exists and cannot be used, which is worse than one that
    does not exist: it reads as done.
    """

    def test_evaluate_prediction_forwards_the_sink(self) -> None:
        import inspect

        from cafaeval.evaluation import cafa_eval, evaluate_prediction

        assert "per_protein_sink" in inspect.signature(evaluate_prediction).parameters
        assert "per_protein_sink" in inspect.signature(cafa_eval).parameters

    def test_each_record_says_which_namespace_and_variant_it_is(self) -> None:
        """A caller holding several records must be able to tell them apart.

        The weighted variant is the one carrying the headline metric, so a
        record that does not say which variant it is cannot be used for the
        thing the sink exists for.
        """
        tau, gt, pred, toi, n_gt = _toy()
        sink = PerProteinSink()
        compute_confusion_matrix_sparse(
            tau, gt, pred, toi, n_gt,
            per_protein_sink=sink, sink_ns="biological_process", sink_variant="weighted",
        )
        rec = sink.records[0]
        assert rec["ns"] == "biological_process"
        assert rec["variant"] == "weighted"

    def test_context_defaults_to_none_rather_than_to_a_guess(self) -> None:
        tau, gt, pred, toi, n_gt = _toy()
        sink = PerProteinSink()
        compute_confusion_matrix_sparse(tau, gt, pred, toi, n_gt, per_protein_sink=sink)
        assert sink.records[0]["ns"] is None
        assert sink.records[0]["variant"] is None


class TestTheRowsCanBeNamed:
    """Anonymous arrays are useless for the purpose the sink exists for.

    Every axis the sink is meant to support, sequence length, identity to the
    nearest donor, taxonomic relation, is keyed by protein. Arrays indexed by an
    opaque row number cannot be joined to any of them, so the sink has to carry
    enough to recover which protein each row is.

    Two numberings are in play. The NK/LK kernel sees every row, so array row i
    is matrix row i. The PK kernel can be handed a subset, and then the two
    disagree; ``row_index`` is what reconciles them, and getting this wrong
    would silently attribute one protein's score to another, which is the exact
    failure this project spent a week correcting elsewhere.
    """

    def test_the_record_carries_the_id_map_and_the_row_index(self) -> None:
        tau, gt, pred, toi, n_gt = _toy()
        ids = {f"P{i:05d}": i for i in range(gt.shape[0])}
        sink = PerProteinSink()
        compute_confusion_matrix_sparse(
            tau, gt, pred, toi, n_gt, per_protein_sink=sink,
            sink_row_index=np.arange(gt.shape[0]), sink_ids=ids,
        )
        rec = sink.records[0]
        assert rec["ids"] == ids
        np.testing.assert_array_equal(rec["row_index"], np.arange(gt.shape[0]))

    def test_row_index_has_one_entry_per_array_row(self) -> None:
        """The property that makes the join sound rather than plausible."""
        tau, gt, pred, toi, n_gt = _toy()
        sink = PerProteinSink()
        compute_confusion_matrix_sparse(
            tau, gt, pred, toi, n_gt, per_protein_sink=sink,
            sink_row_index=np.arange(gt.shape[0]),
        )
        rec = sink.records[0]
        assert len(rec["row_index"]) == rec["tp_at_tau"].shape[0]

    def test_the_id_map_is_copied_not_aliased(self) -> None:
        tau, gt, pred, toi, n_gt = _toy()
        ids = {"P1": 0}
        sink = PerProteinSink()
        compute_confusion_matrix_sparse(
            tau, gt, pred, toi, n_gt, per_protein_sink=sink, sink_ids=ids,
        )
        ids["P2"] = 1
        assert "P2" not in sink.records[0]["ids"]

    def test_identity_is_absent_rather_than_guessed_when_not_supplied(self) -> None:
        tau, gt, pred, toi, n_gt = _toy()
        sink = PerProteinSink()
        compute_confusion_matrix_sparse(tau, gt, pred, toi, n_gt, per_protein_sink=sink)
        assert sink.records[0]["ids"] is None
        assert sink.records[0]["row_index"] is None
