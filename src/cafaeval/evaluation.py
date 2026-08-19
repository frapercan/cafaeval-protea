import os

import time
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import multiprocessing as mp
from cafaeval.parser import obo_parser, gt_parser, pred_parser, gt_exclude_parser, update_toi
from cafaeval.tests import test_norm_metric, test_intersection
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
_metrics_logger = logging.getLogger("cafaeval.metrics")
_metrics_logger.addHandler(logging.NullHandler())
_eval_logger = logging.getLogger("cafaeval.eval")
_eval_logger.addHandler(logging.NullHandler())

# Return a mask for all the predictions (matrix) >= tau
class PerProteinSink:
    """Optional side channel for the per-protein vectors the kernels reduce away.

    The confusion-matrix kernels build ``tp_at_tau`` and ``pred_at_tau`` with one
    row per protein and then collapse them with ``.sum(axis=0)``. Everything
    downstream sees only the collapsed number, so a caller wanting to report a
    score stratified by a per-protein property (length, homology to the donor
    bank, taxonomic distance) has nothing to stratify.

    This is a sink rather than a return value on purpose. Changing the return
    shape of the kernels would ripple through compute_metrics,
    evaluate_prediction and cafa_eval, and every existing caller of each. An
    optional keyword that defaults to None leaves all four signatures and all
    four return types exactly as they were: when nobody passes a sink, nothing
    in this module behaves differently.

    The arrays handed over are the ones the aggregate was computed from, not a
    recomputation, so a caller can verify that their column sums reproduce the
    published totals. ``test_per_protein_sink`` does exactly that, because two
    numbers describing the same quantity by different routes will drift unless
    something checks.
    """

    __slots__ = ("records",)

    def __init__(self):
        self.records = []

    def record(self, *, tp_at_tau, pred_at_tau, n_gt, protein_rows=None,
               ns=None, variant=None, row_index=None, ids=None):
        """Store one kernel's per-protein arrays.

        Copies are taken: the kernels reuse and mutate their buffers, and a sink
        holding a view would silently change under the caller.
        """
        self.records.append(
            {
                "tp_at_tau": np.asarray(tp_at_tau, dtype=np.float64).copy(),
                "pred_at_tau": np.asarray(pred_at_tau, dtype=np.float64).copy(),
                "n_gt": np.asarray(n_gt).copy(),
                "protein_rows": None if protein_rows is None else np.asarray(protein_rows).copy(),
                # Which namespace and which of the two metric variants produced
                # this record. Without them a caller holding several records
                # cannot tell the weighted BPO arrays from the unweighted CCO
                # ones, and the weighted variant is the one that carries the
                # headline metric.
                "ns": ns,
                "variant": variant,
                # Which protein each array row is. Without these the arrays are
                # anonymous and cannot be joined to anything, which defeats the
                # purpose: the axes they exist to support (length, identity to
                # the nearest donor, taxonomic relation) are all keyed by
                # protein. ``row_index`` maps array row to the index used in
                # ``ids``, because the PK path hands the kernel a subset of rows
                # and the two numberings then differ.
                "row_index": None if row_index is None else np.asarray(row_index).copy(),
                "ids": None if ids is None else dict(ids),
            }
        )


def solidify_prediction(pred, tau):
    return pred >= tau


def _csr_nonzeros(mat):
    """(rows, cols, vals) of a CSR's stored non-zeros.

    Reads the CSR arrays directly (``data``/``indices``/``indptr``) instead of
    ``scipy.sparse.find``, which re-runs ``sum_duplicates`` (a full argsort of
    every non-zero) on each call. Our prediction CSR is already canonical, and
    the kernel's downstream bincount/searchsorted are order-independent, so no
    sort is needed. Explicit zeros are dropped to match ``np.nonzero`` exactly.
    """
    mat = mat.tocsr()
    data = mat.data
    cols = mat.indices.astype(np.int64, copy=False)
    rows = np.repeat(np.arange(mat.shape[0], dtype=np.int64), np.diff(mat.indptr))
    if data.size and not data.all():
        keep = data != 0
        return rows[keep], cols[keep], data[keep]
    return rows, cols, data


# computes the f metric for each precision and recall in the input arrays
def compute_f(pr, rc):
    n = 2 * pr * rc
    d = pr + rc
    return np.divide(n, d, out=np.zeros_like(n, dtype=float), where=d != 0)


def compute_s(ru, mi):
    return np.sqrt(ru**2 + mi**2)


def compute_confusion_matrix(tau_arr, g, pred_matrix, toi, n_gt, ic_arr=None):
    """
    Perform the evaluation at the matrix level for all tau thresholds
    The calculation is
    """
    # n, tp, fp, fn, pr, rc (fp = misinformation, fn = remaining uncertainty)
    metrics = np.zeros((len(tau_arr), 6), dtype='float')

    for i, tau in enumerate(tau_arr):

        # Filter predictions based on tau threshold
        p = solidify_prediction(pred_matrix, tau)

        # Terms subsets
        intersection = np.logical_and(p, g)  # TP
        mis = np.logical_and(p, np.logical_not(g))  # FP, predicted but not in the ground truth
        remaining = np.logical_and(np.logical_not(p), g)  # FN, not predicted but in the ground truth

        # Weighted evaluation
        if ic_arr is not None:
            p = p * ic_arr[toi]
            intersection = intersection * ic_arr[toi]  # TP
            mis = mis * ic_arr[toi]  # FP, predicted but not in the ground truth
            remaining = remaining * ic_arr[toi]  # FN, not predicted but in the ground truth

        n_pred = p.sum(axis=1)  # TP + FP (number of terms predicted in each protein)
        n_intersection = intersection.sum(axis=1)  # TP (number of TP terms per protein)
        # Number of proteins with at least one term predicted with score >= tau
        metrics[i, 0] = (p.sum(axis=1) > 0).sum()

        # Sum of confusion matrices
        metrics[i, 1] = n_intersection.sum()  # TP (total terms)
        metrics[i, 2] = mis.sum(axis=1).sum()  # FP
        metrics[i, 3] = remaining.sum(axis=1).sum()  # FN

        # Macro-averaging
        metrics[i, 4] = np.divide(n_intersection, n_pred, out=np.zeros_like(n_intersection, dtype='float'), where=n_pred > 0).sum()  # Precision
        metrics[i, 5] = np.divide(n_intersection, n_gt, out=np.zeros_like(n_gt, dtype='float'), where=n_gt > 0).sum()  # Recall

    return metrics


def compute_confusion_matrix_exclude(tau_arr, g_perprotein, pred_matrix, toi_perprotein, n_gt, ic_arr=None):
    """
    Perform the evaluation at the matrix level for all tau thresholds
    The calculation is

    Here, g is the full ground truth matrix without filtering terms of interest (toi).
    Instead,
    """
    # n, tp, fp, fn, pr, rc (fp = misinformation, fn = remaining uncertainty)
    metrics = np.zeros((len(tau_arr), 6), dtype='float')

    for i, tau in enumerate(tau_arr):

        # Filter predictions based on tau threshold
        p_perprotein = [solidify_prediction(pred_matrix[p_idx, tois], tau) for p_idx, tois in enumerate(toi_perprotein)]

        # Terms subsets
        intersection = [np.logical_and(p_i, g_i) for p_i, g_i in zip(p_perprotein, g_perprotein)]  # TP
        mis = [np.logical_and(p_i, np.logical_not(g_i)) for p_i, g_i in zip(p_perprotein, g_perprotein)]  # FP, predicted but not in the ground truth
        remaining = [np.logical_and(np.logical_not(p_i), g_i) for p_i, g_i in zip(p_perprotein, g_perprotein)]  # FN, not predicted but in the ground truth

        # Weighted evaluation
        if ic_arr is not None:
            p_perprotein = [p_i * ic_arr[tois] for p_i, tois in zip(p_perprotein, toi_perprotein)]
            intersection = [inter * ic_arr[tois] for inter, tois in zip(intersection, toi_perprotein)]  # TP
            mis = [misinf * ic_arr[tois] for misinf, tois in zip(mis, toi_perprotein)]  # FP, predicted but not in the ground truth
            remaining = [rem * ic_arr[tois] for rem, tois in zip(remaining, toi_perprotein)]  # FN, not predicted but in the ground truth

        n_pred = np.array([p_i.sum() for p_i in p_perprotein])  # TP + FP
        n_intersection = np.array([inter.sum() for inter in intersection])  # TP
        precision = np.divide(n_intersection, n_pred, out=np.zeros_like(n_intersection, dtype='float'), where=n_pred > 0)
        recall = np.divide(n_intersection, n_gt, out=np.zeros_like(n_gt, dtype='float'), where=n_gt > 0)

        # metrics tests
        test_norm_metric(precision, name='precision')
        test_norm_metric(recall, name='recall')
        test_intersection(n_intersection, n_pred, n_gt)


        # Number of proteins with at least one term predicted with score >= tau
        metrics[i, 0] = (n_pred > 0).sum()

        # Sum of confusion matrices
        metrics[i, 1] = n_intersection.sum()  # TP
        metrics[i, 2] = np.sum([m.sum() for m in mis])  # FP
        metrics[i, 3] = np.sum([r.sum() for r in remaining])  # FN

        # Macro-averaging
        metrics[i, 4] = precision.sum()  # Precision
        metrics[i, 5] = recall.sum()  # Recall

    return metrics

def compute_confusion_matrix_sparse(tau_arr, g, pred_matrix, toi, n_gt, ic_arr=None,
                                    per_protein_sink=None, sink_ns=None,
                                    sink_variant=None, sink_row_index=None,
                                    sink_ids=None):
    """Sparse alternative to :func:`compute_confusion_matrix`.

    Scatters each non-zero prediction into the highest tau bin at which it is
    still active, then recovers per-threshold per-protein sums via a single
    right-to-left cumulative sum. Total cost is O(nnz + n_prot * n_tau)
    instead of O(n_tau * n_prot * n_toi) for the dense scan, which is the
    dominant win on real-world corpora where the prediction matrix is wide
    and mostly zero above typical thresholds.

    Accepts the same inputs as ``compute_confusion_matrix`` and returns a
    ``(n_tau, 6)`` array with the same column order (n, tp, fp, fn, pr, rc).
    ``tau_arr`` must be sorted in ascending order, which is already the case
    for all call sites (``np.arange(th_step, 1, th_step)``).
    """
    n_prot, n_toi = pred_matrix.shape
    n_tau = len(tau_arr)
    metrics = np.zeros((n_tau, 6), dtype='float')

    # Weight vector over the TOI columns. None → unweighted (all ones, cheap).
    if ic_arr is None:
        w_toi = None
    else:
        w_toi = ic_arr[toi].astype(np.float64, copy=False)

    # Total gt weight (per protein) is already in n_gt for both modes;
    # we only need its sum to fill the FN column.
    total_gt = float(n_gt.sum())

    if issparse(pred_matrix):
        # CSR prediction: pull (row, col, score) straight from the stored
        # non-zeros — no dense (n_prot x n_terms) is ever materialised.
        nz_rows, nz_cols, nz_scores = _csr_nonzeros(pred_matrix)
    else:
        nz_rows, nz_cols = np.nonzero(pred_matrix)
        nz_scores = pred_matrix[nz_rows, nz_cols]
    if nz_rows.size == 0:
        metrics[:, 3] = total_gt
        return metrics

    # g (ground truth) stays dense, so the TP gather is a plain fancy index.
    # The downstream bincount/searchsorted are order-independent, so the
    # row/col order returned by find() does not affect the result.
    nz_is_tp = np.asarray(g[nz_rows, nz_cols]).ravel().astype(np.float64, copy=False)

    # Highest tau index at which a pred with this score is still active.
    # A pred is active at tau_arr[k] iff k <= last_idx.
    last_idx = np.searchsorted(tau_arr, nz_scores, side='right') - 1
    if last_idx.min() < 0:
        keep = last_idx >= 0
        last_idx = last_idx[keep]
        nz_rows = nz_rows[keep]
        nz_cols = nz_cols[keep]
        nz_is_tp = nz_is_tp[keep]
        if last_idx.size == 0:
            metrics[:, 3] = total_gt
            return metrics

    # Flat scatter via bincount (much faster than np.add.at for large nnz).
    flat = nz_rows.astype(np.int64, copy=False) * n_tau + last_idx.astype(np.int64, copy=False)
    length = n_prot * n_tau

    if w_toi is None:
        w_pred = np.ones_like(nz_is_tp)
    else:
        w_pred = w_toi[nz_cols]
    w_tp = w_pred * nz_is_tp

    delta_pred_w = np.bincount(flat, weights=w_pred, minlength=length).reshape(n_prot, n_tau)
    delta_tp_w = np.bincount(flat, weights=w_tp, minlength=length).reshape(n_prot, n_tau)

    # Right-to-left cumulative sum: active weight at tau index k is the sum of
    # contributions of all bins >= k (a pred dies out at the first k > last_idx).
    pred_at_tau = np.cumsum(delta_pred_w[:, ::-1], axis=1)[:, ::-1]
    tp_at_tau = np.cumsum(delta_tp_w[:, ::-1], axis=1)[:, ::-1]

    # Per-threshold aggregation. Matches the column order of the dense kernel.
    tp_totals = tp_at_tau.sum(axis=0)
    pred_totals = pred_at_tau.sum(axis=0)

    metrics[:, 0] = (pred_at_tau > 0).sum(axis=0)
    metrics[:, 1] = tp_totals
    metrics[:, 2] = pred_totals - tp_totals
    metrics[:, 3] = total_gt - tp_totals

    # Macro precision / recall. Guard divides by zero to match the dense kernel
    # (which uses np.divide with where=denom > 0, returning 0 elsewhere).
    safe_pred = np.where(pred_at_tau > 0, pred_at_tau, 1.0)
    precision = np.where(pred_at_tau > 0, tp_at_tau / safe_pred, 0.0)
    metrics[:, 4] = precision.sum(axis=0)

    if np.any(n_gt > 0):
        n_gt_col = n_gt.astype(np.float64, copy=False)[:, None]
        safe_gt = np.where(n_gt_col > 0, n_gt_col, 1.0)
        recall = np.where(n_gt_col > 0, tp_at_tau / safe_gt, 0.0)
        metrics[:, 5] = recall.sum(axis=0)

    # Handed over BEFORE returning, and after the aggregate has been computed
    # from these same arrays, so a caller can check the two agree.
    if per_protein_sink is not None:
        per_protein_sink.record(tp_at_tau=tp_at_tau, pred_at_tau=pred_at_tau, n_gt=n_gt,
                                ns=sink_ns, variant=sink_variant,
                                row_index=sink_row_index, ids=sink_ids)

    return metrics


def compute_confusion_matrix_exclude_sparse(
    tau_arr, pred_sub, gt_sub, toi_mask, excluded_mask, n_gt, ic_arr=None,
    per_protein_sink=None, sink_ns=None, sink_variant=None,
    sink_row_index=None, sink_ids=None,
):
    """Sparse alternative to :func:`compute_confusion_matrix_exclude`.

    Same single-pass scatter + right-to-left cumsum strategy as
    :func:`compute_confusion_matrix_sparse`, extended to the PK setting where
    the valid column set is protein-specific. A prediction at ``(row, col)``
    contributes to protein ``row`` only if:

    1. ``col`` belongs to the global toi (``toi_mask``), and
    2. ``col`` is not in this protein's exclude set (``~excluded_mask``).

    Because the filter is expressed as a boolean AND over dense matrices,
    the sparse scatter sees only the surviving non-zeros — we never
    materialise the Python list of per-protein toi arrays. Cost drops from
    ``O(n_tau * sum_p |toi_p|)`` to ``O(nnz_valid + n_prot * n_tau)``.

    Parameters
    ----------
    pred_sub:
        ``(n_prot, n_terms)`` float — predictions restricted to proteins
        with at least one GT annotation in the toi.
    gt_sub:
        ``(n_prot, n_terms)`` bool/int — ground truth for the same proteins.
    toi_mask:
        ``(n_terms,) bool`` — global terms-of-interest flag.
    excluded_mask:
        ``(n_prot, n_terms) bool`` — per-protein exclude flag.
    n_gt:
        ``(n_prot,) float`` — pre-computed weight of valid GT annotations
        per protein (dense sum over ``gt_sub & toi_mask & ~excluded_mask``,
        optionally multiplied by ``ic_arr``).
    ic_arr:
        ``(n_terms,) float`` or ``None`` — optional per-term weight.
    """
    n_prot, n_terms = pred_sub.shape
    n_tau = len(tau_arr)
    metrics = np.zeros((n_tau, 6), dtype='float')

    total_gt = float(n_gt.sum())

    # Only predictions at valid columns contribute. Pull the prediction
    # non-zeros (from CSR storage when sparse, no dense materialisation), then
    # apply the PK exclusion (column in TOI and not excluded) on the flat
    # coordinate arrays — equivalent to the old dense
    # ``(pred_sub != 0) & toi_mask[None,:] & ~excluded_mask`` then np.nonzero.
    if issparse(pred_sub):
        p_rows, p_cols, p_scores = _csr_nonzeros(pred_sub)
    else:
        p_rows, p_cols = np.nonzero(pred_sub)
        p_scores = pred_sub[p_rows, p_cols]
    keep = toi_mask[p_cols] & (~excluded_mask[p_rows, p_cols])
    nz_rows = p_rows[keep]
    nz_cols = p_cols[keep]
    nz_scores = p_scores[keep]
    if nz_rows.size == 0:
        metrics[:, 3] = total_gt
        return metrics

    nz_is_tp = np.asarray(gt_sub[nz_rows, nz_cols]).ravel().astype(np.float64, copy=False)

    last_idx = np.searchsorted(tau_arr, nz_scores, side='right') - 1
    if last_idx.min() < 0:
        keep = last_idx >= 0
        last_idx = last_idx[keep]
        nz_rows = nz_rows[keep]
        nz_cols = nz_cols[keep]
        nz_is_tp = nz_is_tp[keep]
        if last_idx.size == 0:
            metrics[:, 3] = total_gt
            return metrics

    flat = nz_rows.astype(np.int64, copy=False) * n_tau + last_idx.astype(np.int64, copy=False)
    length = n_prot * n_tau

    if ic_arr is None:
        w_pred = np.ones_like(nz_is_tp)
    else:
        w_pred = ic_arr[nz_cols].astype(np.float64, copy=False)
    w_tp = w_pred * nz_is_tp

    delta_pred_w = np.bincount(flat, weights=w_pred, minlength=length).reshape(n_prot, n_tau)
    delta_tp_w = np.bincount(flat, weights=w_tp, minlength=length).reshape(n_prot, n_tau)

    pred_at_tau = np.cumsum(delta_pred_w[:, ::-1], axis=1)[:, ::-1]
    tp_at_tau = np.cumsum(delta_tp_w[:, ::-1], axis=1)[:, ::-1]

    tp_totals = tp_at_tau.sum(axis=0)
    pred_totals = pred_at_tau.sum(axis=0)

    # For coverage (`n` column) to match the denominator used by normalize()
    # — which is the post-exclusion protein count from _count_proteins_in_toi
    # — restrict the row count to proteins that still have ≥1 GT annotation
    # in TOI after the per-protein exclude mask. Without this, `n` counts
    # proteins whose TOI annotations were all already known in t0, while the
    # denominator drops them, producing coverage > 1 in PK.
    eligible_rows = ((gt_sub != 0) & toi_mask[None, :] & (~excluded_mask)).any(axis=1)

    metrics[:, 0] = ((pred_at_tau > 0) & eligible_rows[:, None]).sum(axis=0)
    metrics[:, 1] = tp_totals
    metrics[:, 2] = pred_totals - tp_totals
    metrics[:, 3] = total_gt - tp_totals

    safe_pred = np.where(pred_at_tau > 0, pred_at_tau, 1.0)
    precision = np.where(pred_at_tau > 0, tp_at_tau / safe_pred, 0.0)
    metrics[:, 4] = precision.sum(axis=0)

    if np.any(n_gt > 0):
        n_gt_col = n_gt.astype(np.float64, copy=False)[:, None]
        safe_gt = np.where(n_gt_col > 0, n_gt_col, 1.0)
        recall = np.where(n_gt_col > 0, tp_at_tau / safe_gt, 0.0)
        metrics[:, 5] = recall.sum(axis=0)

    # PK also hands over which rows were eligible: this kernel can be given a
    # protein subset, so a caller stratifying by protein needs to know which
    # rows the arrays refer to rather than assuming they are the whole set.
    if per_protein_sink is not None:
        per_protein_sink.record(
            tp_at_tau=tp_at_tau, pred_at_tau=pred_at_tau, n_gt=n_gt,
            protein_rows=eligible_rows, ns=sink_ns, variant=sink_variant,
            row_index=sink_row_index, ids=sink_ids,
        )

    return metrics


_CM_G = None
_CM_P = None
_CM_TOI = None
_CM_N_GT = None
_CM_IC = None


def _cm_init(g, p, toi, n_gt, ic_arr):
    global _CM_G, _CM_P, _CM_TOI, _CM_N_GT, _CM_IC
    _CM_G = g
    _CM_P = p
    _CM_TOI = toi
    _CM_N_GT = n_gt
    _CM_IC = ic_arr


def _cm_worker(tau_chunk):
    return compute_confusion_matrix(
        tau_chunk, _CM_G, _CM_P, _CM_TOI, _CM_N_GT, _CM_IC
    )


_CME_G = None
_CME_P = None
_CME_TOI = None
_CME_N_GT = None
_CME_IC = None


def _cme_init(g_perprotein, p, toi_perprotein, n_gt, ic_arr):
    global _CME_G, _CME_P, _CME_TOI, _CME_N_GT, _CME_IC
    _CME_G = g_perprotein
    _CME_P = p
    _CME_TOI = toi_perprotein
    _CME_N_GT = n_gt
    _CME_IC = ic_arr


def _cme_worker(tau_chunk):
    return compute_confusion_matrix_exclude(
        tau_chunk, _CME_G, _CME_P, _CME_TOI, _CME_N_GT, _CME_IC
    )

def compute_metrics(pred, gt_matrix, tau_arr, toi, gt_exclude=None, ic_arr=None, n_cpu=0,
                    per_protein_sink=None, sink_ns=None, sink_variant=None,
                    sink_ids=None):
    """
    Takes the prediction and the ground truth and for each threshold in tau_arr
    calculates the confusion matrix and returns the coverage,
    precision, recall, remaining uncertainty and misinformation.
    Toi is the list of terms (indexes) to be considered
    """
    t0 = time.time()
    # Parallelization
    if n_cpu == 0:
        n_cpu = mp.cpu_count()

    mode = "pk" if gt_exclude is not None else "nk_lk"
    _metrics_logger.debug(
        "compute_metrics start",
        extra={"n_cpu": n_cpu, "n_tau": int(len(tau_arr)), "mode": mode},
    )

    columns = ["n", "tp", "fp", "fn", "pr", "rc"]
    use_sparse = os.environ.get("CAFAEVAL_SPARSE", "1") not in ("0", "false", "False")

    n_terms = gt_matrix.shape[1]
    toi_is_full = (
        isinstance(toi, np.ndarray)
        and toi.ndim == 1
        and toi.size == n_terms
    )
    # Detecting the common "toi == arange(n_terms)" case lets us skip a full
    # ``(n_prot, n_terms)`` column copy on every cafa_eval call. We only pay
    # the cheap stride check when toi already has the right shape; the
    # equality probe is bounded by a small constant via ``array_equal``.
    if toi_is_full:
        toi_is_full = bool(np.array_equal(toi, np.arange(n_terms)))

    # Filter out proteins with no annotations in Terms-Of-Interest. ``any``
    # short-circuits per row on the bool result so it beats ``sum > 0`` on
    # large dense matrices, especially when toi spans the whole ontology.
    if toi_is_full:
        proteins_has_gt = (gt_matrix != 0).any(axis=1)
    else:
        proteins_has_gt = (gt_matrix[:, toi] != 0).any(axis=1)
    proteins_with_gt = np.where(proteins_has_gt)[0]

    # Heavy slices ``g``/``p``/``gt_with_annots``/``pred_sub`` are deferred to
    # the branch that actually consumes them. The PK sparse kernel only needs
    # ``pred_sub`` + ``gt_with_annots`` + masks; the NK sparse kernel only
    # needs ``g`` + ``p``. Materialising both up front cost ~3 s per call on
    # the BP namespace.
    g = None
    p = None
    gt_with_annots = None
    toi_mask = None
    excluded_mask = None
    toi_perprotein = None
    gt_perprotein = None

    if gt_exclude is not None:
        if proteins_has_gt.all():
            gt_with_annots = gt_matrix
        else:
            gt_with_annots = gt_matrix[proteins_with_gt, :]
        if use_sparse:
            # Dense boolean masks replace the per-protein Python list of toi
            # arrays. Each protein's effective toi is the global toi minus
            # the terms flagged in ``gt_exclude``; per-protein GT weight is
            # the sum over that intersection.
            if toi_is_full:
                toi_mask = np.ones(n_terms, dtype=bool)
            else:
                toi_mask = np.zeros(n_terms, dtype=bool)
                toi_mask[toi] = True
            # gt_exclude.matrix is already bool (built by gt_parser /
            # gt_exclude_parser via ``np.zeros_like(gt.matrix)`` where
            # ``gt.matrix.dtype == bool``), so the legacy ``!= 0`` was a
            # redundant elementwise compare on top of the fancy-index copy.
            excluded_mask = gt_exclude.matrix[proteins_with_gt, :]
            # ``n_gt`` only needs the count of GT cells that survive the
            # ``toi_mask & ~excluded_mask`` filter. Materialising the full
            # ``valid_gt_mask`` (three dense bool ops on
            # ``(n_prot, n_terms)``) costs ~300 ms on the BP namespace
            # — almost as much as the kernel itself. Instead we walk the
            # GT non-zero coordinates once and accumulate via ``bincount``.
            gt_nz_rows, gt_nz_cols = np.nonzero(gt_with_annots)
            if gt_nz_rows.size:
                if not toi_is_full:
                    keep = toi_mask[gt_nz_cols]
                    gt_nz_rows = gt_nz_rows[keep]
                    gt_nz_cols = gt_nz_cols[keep]
                keep = ~excluded_mask[gt_nz_rows, gt_nz_cols]
                gt_nz_rows = gt_nz_rows[keep]
                gt_nz_cols = gt_nz_cols[keep]
            n_prot_with_gt = gt_with_annots.shape[0]
            if ic_arr is None:
                n_gt = np.bincount(
                    gt_nz_rows, minlength=n_prot_with_gt,
                ).astype(np.float64)
            else:
                n_gt = np.bincount(
                    gt_nz_rows, weights=ic_arr[gt_nz_cols],
                    minlength=n_prot_with_gt,
                ).astype(np.float64)
        else:
            # Dense fallback: preserve the exact list-based summation order
            # used by upstream so test_norm_metric inside the dense kernel
            # does not trip on ULP noise.
            toi_perprotein = [
                np.setdiff1d(toi, gt_exclude.matrix[prot, :].nonzero()[0],
                             assume_unique=True)
                for prot in proteins_with_gt
            ]
            gt_perprotein = [
                gt_with_annots[p_idx, tois]
                for p_idx, tois in enumerate(toi_perprotein)
            ]
            n_gt = np.array([gpp.sum().item() for gpp in gt_perprotein])
            if ic_arr is not None:
                n_gt = np.array([(gpp * ic_arr[tois]).sum().item()
                                 for gpp, tois in zip(gt_perprotein, toi_perprotein)])
        n_empty = int(np.count_nonzero(n_gt == 0))
        if n_empty:
            _metrics_logger.warning(
                f"proteins with no annotations in TOI: {n_empty}",
                extra={"count": n_empty},
            )
    else:
        # NK / LK path — needs g and p.
        if proteins_has_gt.all():
            gt_with_annots = gt_matrix
            pred_filtered = pred
        else:
            gt_with_annots = gt_matrix[proteins_with_gt, :]
            pred_filtered = pred[proteins_has_gt, :]
        if toi_is_full:
            g = gt_with_annots
            p = pred_filtered
        else:
            g = gt_with_annots[:, toi]
            p = pred_filtered[:, toi]
        # Simple metrics: number of terms annotated in each protein
        if ic_arr is None:
            n_gt = g.sum(axis=1)
        # Weighted metrics
        else:
            n_gt = (g * ic_arr[toi]).sum(axis=1)

    # The dense fallback fans tau chunks over a fork pool, so the per-protein
    # arrays live in worker processes and never come back. Refuse rather than
    # return an empty sink: a caller that asked for stratification and silently
    # got none would publish a stratified table with no data behind it.
    if per_protein_sink is not None and not use_sparse:
        raise NotImplementedError(
            "per_protein_sink is only supported by the sparse kernels; unset "
            "CAFAEVAL_SPARSE=0 to use it."
        )

    t1 = time.time()
    if gt_exclude is None and use_sparse:
        # Sparse kernel: single-pass scatter + cumsum. No pool needed — the
        # kernel is fast enough that fork overhead would dominate.
        # NK/LK path passes every row through, so array row i is matrix row i.
        metrics = compute_confusion_matrix_sparse(
            tau_arr, g, p, toi, n_gt, ic_arr, per_protein_sink=per_protein_sink,
            sink_ns=sink_ns, sink_variant=sink_variant,
            sink_row_index=(np.arange(p.shape[0]) if per_protein_sink is not None else None),
            sink_ids=sink_ids,
        )
    elif gt_exclude is None:
        # Dense fallback kernel (opt-in, CAFAEVAL_SPARSE=0) needs a dense
        # prediction; densify the CSR just for this path.
        p_dense = p.toarray() if issparse(p) else p
        tau_chunks = np.array_split(tau_arr, n_cpu)
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_cpu, initializer=_cm_init,
                      initargs=(g, p_dense, toi, n_gt, ic_arr)) as pool:
            metrics = np.concatenate(pool.map(_cm_worker, tau_chunks), axis=0)
    elif use_sparse:
        # PK sparse kernel: boolean-masked single-pass scatter. The toi and
        # exclude masks live as dense ``(n_prot, n_terms)`` bools, so the
        # scatter never needs a Python-level per-protein loop.
        if proteins_has_gt.all():
            pred_sub = pred
        else:
            pred_sub = pred[proteins_has_gt, :]
        # PK path may hand the kernel a subset, so the array row numbering and
        # the ``ids`` numbering differ. Record which original rows went in.
        metrics = compute_confusion_matrix_exclude_sparse(
            tau_arr, pred_sub, gt_with_annots, toi_mask, excluded_mask, n_gt, ic_arr,
            per_protein_sink=per_protein_sink, sink_ns=sink_ns, sink_variant=sink_variant,
            sink_row_index=(
                np.flatnonzero(proteins_has_gt) if per_protein_sink is not None else None
            ),
            sink_ids=sink_ids,
        )
    else:
        # Dense fallback: fan out tau chunks over a fork pool using the
        # per-protein toi / gt lists we prepared above. Gated by
        # ``CAFAEVAL_SPARSE=0`` for A/B comparison against the sparse kernel.
        tau_chunks = np.array_split(tau_arr, n_cpu)
        if proteins_has_gt.all():
            pred_sub = pred
        else:
            pred_sub = pred[proteins_has_gt, :]
        if issparse(pred_sub):
            pred_sub = pred_sub.toarray()
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_cpu, initializer=_cme_init,
                      initargs=(gt_perprotein, pred_sub, toi_perprotein, n_gt, ic_arr)) as pool:
            metrics = np.concatenate(pool.map(_cme_worker, tau_chunks), axis=0)

    t2 = time.time()
    kernel = "sparse" if use_sparse else f"pool({n_cpu})"
    _metrics_logger.info(
        f"compute_metrics {mode} {kernel} n_tau={len(tau_arr)} "
        f"prep={t1 - t0:.3f}s kernel={t2 - t1:.3f}s total={t2 - t0:.3f}s",
        extra={"mode": mode, "n_cpu": n_cpu, "n_tau": int(len(tau_arr)),
               "kernel": kernel,
               "prep_seconds": round(t1 - t0, 3),
               "pool_seconds": round(t2 - t1, 3),
               "total_seconds": round(t2 - t0, 3)},
    )
    return pd.DataFrame(metrics, columns=columns)


def normalize(metrics, ns, tau_arr, ne, normalization):

    # Normalize columns
    for column in metrics.columns:
        if column != "n":
            # By default normalize by gt
            denominator = ne
            # Otherwise normalize by pred
            if normalization == 'pred' or (normalization == 'cafa' and column == "pr"):
                denominator = metrics["n"]
            metrics[column] = np.divide(metrics[column], denominator,
                                        out=np.zeros_like(metrics[column], dtype='float'),
                                        where=denominator > 0)

    metrics['ns'] = [ns] * len(tau_arr)
    metrics['tau'] = tau_arr
    metrics['cov'] = metrics['n'] / ne
    metrics['mi'] = metrics['fp']
    metrics['ru'] = metrics['fn']

    metrics['f'] = compute_f(metrics['pr'], metrics['rc'])
    metrics['s'] = compute_s(metrics['ru'], metrics['mi'])

    # Micro-average, calculation is based on the average of the confusion matrices
    metrics['pr_micro'] = np.divide(metrics['tp'], metrics['tp'] + metrics['fp'],
                                    out=np.zeros_like(metrics['tp'], dtype='float'),
                                    where=(metrics['tp'] + metrics['fp']) > 0)
    metrics['rc_micro'] = np.divide(metrics['tp'], metrics['tp'] + metrics['fn'],
                                    out=np.zeros_like(metrics['tp'], dtype='float'),
                                    where=(metrics['tp'] + metrics['fn']) > 0)
    metrics['f_micro'] = compute_f(metrics['pr_micro'], metrics['rc_micro'])

    return metrics

def _count_proteins_in_toi(gt_matrix, toi, exclude_matrix):
    """Number of proteins with at least one GT annotation in the toi.

    Vectorised replacement for the legacy ``sum(...)`` over a Python list
    comprehension that called ``setdiff1d`` per protein. On the BP namespace
    of a real PK corpus this drops the ``evaluate_prediction`` per-namespace
    overhead from ~1 s to a few ms.

    When ``exclude_matrix`` is ``None`` (NK/LK) the count is just the number
    of rows that have any non-zero in the toi columns. When it is provided
    (PK) the count is the number of rows that still have a surviving
    annotation after the per-protein exclude set is removed.
    """
    n_terms = gt_matrix.shape[1]
    toi_is_full = (
        isinstance(toi, np.ndarray)
        and toi.ndim == 1
        and toi.size == n_terms
        and bool(np.array_equal(toi, np.arange(n_terms)))
    )

    if exclude_matrix is None:
        if toi_is_full:
            has_any = (gt_matrix != 0).any(axis=1)
        else:
            has_any = (gt_matrix[:, toi] != 0).any(axis=1)
        return int(has_any.sum())

    if toi_is_full:
        toi_mask = np.ones(n_terms, dtype=bool)
    else:
        toi_mask = np.zeros(n_terms, dtype=bool)
        toi_mask[toi] = True
    valid = (gt_matrix != 0) & toi_mask[None, :] & (exclude_matrix == 0)
    return int(valid.any(axis=1).sum())


def evaluate_prediction(prediction, gt, ontologies, tau_arr, gt_exclude=None, normalization='cafa', n_cpu=0, weighted_only=False,
                        per_protein_sink=None):
    t0 = time.time()
    _eval_logger.debug(
        "evaluate_prediction start",
        extra={"n_namespaces": len(prediction), "weighted_only": weighted_only,
               "normalization": normalization, "n_cpu": n_cpu},
    )

    dfs = []
    dfs_w = []

    # Unweighted metrics
    for ns in prediction:
        t_ns = time.time()
        # Number of proteins with positive annotations in the TOI. The PK
        # branch (gt_exclude not None) further restricts to proteins that
        # still have a surviving annotation after the per-protein exclude
        # set is removed. Both shapes are computed via a single vectorised
        # mask AND — the legacy code did this with a Python list
        # comprehension over ``proteins_with_gt``, which alone cost ~1 s on
        # the BP namespace.
        num_annot_prots = _count_proteins_in_toi(
            gt[ns].matrix, ontologies[ns].toi,
            gt_exclude[ns].matrix if gt_exclude is not None else None,
        )
        exclude = gt_exclude[ns] if gt_exclude is not None else None
        ne = np.full(len(tau_arr), num_annot_prots)

        if not weighted_only:
            t_metrics = time.time()
            metrics_df = compute_metrics(
                prediction[ns].matrix, gt[ns].matrix, tau_arr, ontologies[ns].toi, exclude, None, n_cpu,
                per_protein_sink=per_protein_sink, sink_ns=ns, sink_variant="unweighted",
                sink_ids=getattr(gt[ns], "ids", None))
            t_norm = time.time()
            dfs.append(normalize(metrics_df, ns, tau_arr, ne, normalization))
            _eval_logger.info(
                f"evaluate {ns:>18s} unweighted  metrics={t_norm - t_metrics:.3f}s "
                f"norm={time.time() - t_norm:.3f}s",
                extra={"ns": ns, "variant": "unweighted",
                       "metrics_seconds": round(t_norm - t_metrics, 3),
                       "normalize_seconds": round(time.time() - t_norm, 3),
                       "total_seconds": round(time.time() - t_ns, 3)},
            )

        # Weighted metrics
        if ontologies[ns].ia is not None:
            t_w = time.time()
            num_annot_prots = _count_proteins_in_toi(
                gt[ns].matrix, ontologies[ns].toi_ia,
                gt_exclude[ns].matrix if gt_exclude is not None else None,
            )
            exclude = gt_exclude[ns] if gt_exclude is not None else None
            ne = np.full(len(tau_arr), num_annot_prots)

            t_metrics_w = time.time()
            metrics_df_w = compute_metrics(
                prediction[ns].matrix, gt[ns].matrix, tau_arr, ontologies[ns].toi_ia, exclude, ontologies[ns].ia, n_cpu,
                per_protein_sink=per_protein_sink, sink_ns=ns, sink_variant="weighted",
                sink_ids=getattr(gt[ns], "ids", None))
            t_norm_w = time.time()
            dfs_w.append(normalize(metrics_df_w, ns, tau_arr, ne, normalization))
            _eval_logger.info(
                f"evaluate {ns:>18s} weighted    metrics={t_norm_w - t_metrics_w:.3f}s "
                f"norm={time.time() - t_norm_w:.3f}s",
                extra={"ns": ns, "variant": "weighted",
                       "metrics_seconds": round(t_norm_w - t_metrics_w, 3),
                       "normalize_seconds": round(time.time() - t_norm_w, 3),
                       "total_seconds": round(time.time() - t_w, 3)},
            )

    t_merge = time.time()

    if weighted_only:
        dfs_w = pd.concat(dfs_w)
        base_cols = ("ns", "tau")
        metric_cols = [c for c in dfs_w.columns if c not in base_cols]
        for c in metric_cols:
            dfs_w[f"{c}_w"] = dfs_w[c]
        _eval_logger.info(
            f"evaluate_prediction done (weighted_only) in {time.time() - t0:.2f}s",
            extra={"total_seconds": round(time.time() - t0, 3)},
        )
        return dfs_w

    dfs = pd.concat(dfs)

    # Merge weighted and unweighted dataframes
    if dfs_w:
        dfs_w = pd.concat(dfs_w)
        dfs = pd.merge(dfs, dfs_w, on=['ns', 'tau'], suffixes=('', '_w'))

    _eval_logger.info(
        f"evaluate_prediction done in {time.time() - t0:.2f}s "
        f"(merge {time.time() - t_merge:.3f}s)",
        extra={"total_seconds": round(time.time() - t0, 3),
               "merge_seconds": round(time.time() - t_merge, 3)},
    )
    return dfs


def cafa_eval(obo_file, pred_dir, gt_file, ia=None, no_orphans=False, norm='cafa', prop='max',
              exclude=None, toi_file=None, max_terms=None, th_step=0.01, n_cpu=1, weighted_only=False,
              per_protein_sink=None):
    t_total = time.time()
    _eval_logger.info(
        f"cafa_eval start: norm={norm} prop={prop} th_step={th_step} "
        f"n_cpu={n_cpu} weighted_only={weighted_only}",
        extra={"norm": norm, "prop": prop, "th_step": th_step, "n_cpu": n_cpu,
               "weighted_only": weighted_only, "obo_file": obo_file,
               "pred_dir": pred_dir, "gt_file": gt_file, "exclude": exclude,
               "toi_file": toi_file, "max_terms": max_terms, "ia": ia},
    )

    # Tau array, used to compute metrics at different score thresholds
    tau_arr = np.arange(th_step, 1, th_step)

    # Parse the OBO file and creates a different graphs for each namespace
    t_obo = time.time()
    ontologies = obo_parser(obo_file, ("is_a", "part_of"), ia, not no_orphans)
    if toi_file is not None:
        ontologies = update_toi(ontologies, toi_file)
    _eval_logger.info(
        f"obo parsed: {len(ontologies)} namespaces, n_tau={len(tau_arr)} "
        f"({time.time() - t_obo:.2f}s)",
        extra={"seconds": round(time.time() - t_obo, 3),
               "namespaces": list(ontologies.keys()),
               "n_tau": int(len(tau_arr))},
    )

    # Parse ground truth file
    t_gt = time.time()
    gt = gt_parser(gt_file, ontologies)
    if exclude is not None:
        gt_exclude = gt_exclude_parser(exclude, gt, ontologies)
    else:
        gt_exclude = None
    gt_stats = ", ".join(f"{ns[:2]}={len(g.ids)}" for ns, g in gt.items())
    _eval_logger.info(
        f"ground truth parsed: {gt_stats}, exclude={'yes' if gt_exclude else 'no'} "
        f"({time.time() - t_gt:.2f}s)",
        extra={"seconds": round(time.time() - t_gt, 3),
               "has_exclude": gt_exclude is not None},
    )

    # Set prediction files looking recursively in the prediction folder
    pred_folder = os.path.normpath(pred_dir) + "/"  # add the tailing "/"
    pred_files = []
    for root, dirs, files in os.walk(pred_folder):
        for file in files:
            pred_files.append(os.path.join(root, file))
    logger.debug("Prediction paths {}".format(pred_files))
    _eval_logger.info(
        f"prediction files discovered: {len(pred_files)} file(s) under {pred_dir}",
        extra={"count": len(pred_files), "pred_dir": pred_dir},
    )

    # Parse prediction files and perform evaluation
    dfs = []
    for idx, file_name in enumerate(pred_files, 1):
        basename = os.path.basename(file_name)
        t_file = time.time()
        t_parse = time.time()
        prediction = pred_parser(file_name, ontologies, gt, prop, max_terms, n_cpu)
        if not prediction:
            logger.warning("Prediction: {}, not evaluated".format(file_name))
            _eval_logger.warning(
                f"prediction skipped: {basename} [{idx}/{len(pred_files)}]",
                extra={"file": file_name, "index": idx, "total": len(pred_files)},
            )
            continue

        _eval_logger.info(
            f"prediction parsed: {basename} [{idx}/{len(pred_files)}] "
            f"ns={list(prediction.keys())} ({time.time() - t_parse:.2f}s)",
            extra={"file": file_name, "index": idx, "total": len(pred_files),
                   "parse_seconds": round(time.time() - t_parse, 3),
                   "namespaces": list(prediction.keys())},
        )
        t_eval = time.time()
        df_pred = evaluate_prediction(prediction, gt, ontologies, tau_arr, gt_exclude,
                                      normalization=norm, n_cpu=n_cpu, weighted_only=weighted_only,
                                      per_protein_sink=per_protein_sink)
        df_pred['filename'] = file_name.replace(pred_folder, '').replace('/', '_')
        dfs.append(df_pred)
        _eval_logger.info(
            f"prediction evaluated: {basename} eval={time.time() - t_eval:.2f}s "
            f"total={time.time() - t_file:.2f}s",
            extra={"file": file_name, "index": idx, "total": len(pred_files),
                   "eval_seconds": round(time.time() - t_eval, 3),
                   "file_seconds": round(time.time() - t_file, 3)},
        )
        logger.info("Prediction: {}, evaluated".format(file_name))

    # Concatenate all dataframes and save them
    t_final = time.time()
    df = None
    dfs_best = {}
    if dfs:
        df = pd.concat(dfs)

        # Remove rows with no coverage
        df = df[df['cov'] > 0].reset_index(drop=True)
        df.set_index(['filename', 'ns', 'tau'], inplace=True)

        # Calculate the best index for each namespace and each evaluation metric
        for metric, cols in [('f', ['rc', 'pr']), ('f_w', ['rc_w', 'pr_w']), ('s', ['ru', 'mi']), ('f_micro', ['rc_micro', 'pr_micro']), ('f_micro_w', ['rc_micro_w', 'pr_micro_w'])]:
            if metric in df.columns:
                index_best = df.groupby(level=['filename', 'ns'])[metric].idxmax() if metric in ['f', 'f_w', 'f_micro', 'f_micro_w'] else df.groupby(['filename', 'ns'])[metric].idxmin()
                df_best = df.loc[index_best]
                if metric[-2:] != '_w':
                    df_best['cov_max'] = df.reset_index('tau').loc[[ele[:-1] for ele in index_best]].groupby(level=['filename', 'ns'])['cov'].max()
                else:
                    df_best['cov_max'] = df.reset_index('tau').loc[[ele[:-1] for ele in index_best]].groupby(level=['filename', 'ns'])['cov_w'].max()
                dfs_best[metric] = df_best
        _eval_logger.info(
            f"cafa_eval aggregation done: {len(dfs)} result(s), "
            f"best=[{', '.join(dfs_best.keys())}] ({time.time() - t_final:.2f}s)",
            extra={"seconds": round(time.time() - t_final, 3),
                   "n_results": len(dfs),
                   "best_metrics": list(dfs_best.keys())},
        )
    else:
        logger.info("No predictions evaluated")
        _eval_logger.warning("cafa_eval no predictions evaluated")

    _eval_logger.info(
        f"cafa_eval done in {time.time() - t_total:.2f}s",
        extra={"total_seconds": round(time.time() - t_total, 3)},
    )
    return df, dfs_best


def write_results(df, dfs_best, out_dir='results', th_step=0.01):

    # Create output folder here in order to store the log file
    out_folder = os.path.normpath(out_dir) + "/"
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # Set the number of decimals to write in the output files based on the threshold step size
    decimals = int(np.ceil(-np.log10(th_step))) + 1

    df.to_csv('{}/evaluation_all.tsv'.format(out_folder), float_format="%.{}f".format(decimals), sep="\t")

    for metric in dfs_best:
        dfs_best[metric].to_csv('{}/evaluation_best_{}.tsv'.format(out_folder, metric), float_format="%.{}f".format(decimals), sep="\t")
