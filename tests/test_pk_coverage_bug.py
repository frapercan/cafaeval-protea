"""Regression test for the PK coverage > 1 bug.

Scenario: PK evaluation where some proteins have ground-truth annotations
that are ALL already known in t0 (i.e., zero novel annotations after the
per-protein exclude mask). Before the fix, `metrics['n']` counted predictions
for those proteins, while the denominator `ne` dropped them — producing
coverage > 1. After the fix, `n` is restricted to the same eligible
population as `ne` and coverage stays in [0, 1].
"""
from __future__ import annotations

import numpy as np

from cafaeval.evaluation import (
    compute_confusion_matrix_exclude_sparse,
    _count_proteins_in_toi,
)


def _make_pk_scenario():
    """Return (pred_sub, gt_sub, toi_mask, excluded_mask, n_gt) with 3 proteins:

    - protein 0: has novel GT (term 2 is new), excluded {0, 1}. Eligible.
    - protein 1: all GT annotations excluded (no novel). NOT eligible.
    - protein 2: has novel GT (term 3). Eligible.

    All 3 receive high-confidence predictions for NON-excluded terms.
    """
    n_prot = 3
    n_terms = 4
    tau_arr = np.array([0.1, 0.5, 0.9], dtype=np.float64)

    # Ground truth: protein 0 has terms {0,1,2}; protein 1 has {0,1}; protein 2 has {0,3}
    gt_sub = np.zeros((n_prot, n_terms), dtype=bool)
    gt_sub[0, [0, 1, 2]] = True
    gt_sub[1, [0, 1]] = True
    gt_sub[2, [0, 3]] = True

    # Per-protein exclude (what was in t0):
    # protein 0: {0, 1} excluded → term 2 survives as novel
    # protein 1: {0, 1} excluded → nothing survives (ineligible for PK)
    # protein 2: {0}    excluded → term 3 survives as novel
    excluded_mask = np.zeros((n_prot, n_terms), dtype=bool)
    excluded_mask[0, [0, 1]] = True
    excluded_mask[1, [0, 1]] = True
    excluded_mask[2, [0]] = True

    # Predictions: every protein predicts term 2 and term 3 at high confidence
    pred_sub = np.zeros((n_prot, n_terms), dtype=np.float32)
    pred_sub[:, [2, 3]] = 0.95

    # TOI = all terms
    toi_mask = np.ones(n_terms, dtype=bool)

    # n_gt per row (count of GT terms in TOI, pre-exclusion view from the caller)
    n_gt = gt_sub.astype(np.float64).sum(axis=1)

    return pred_sub, gt_sub, toi_mask, excluded_mask, n_gt, tau_arr


def test_pk_coverage_never_exceeds_one():
    pred_sub, gt_sub, toi_mask, excluded_mask, n_gt, tau_arr = _make_pk_scenario()

    metrics = compute_confusion_matrix_exclude_sparse(
        tau_arr, pred_sub, gt_sub, toi_mask, excluded_mask, n_gt
    )
    n_at_tau = metrics[:, 0]

    # ne = number of proteins with ≥1 NOVEL GT in TOI → should be 2 (protein 0 and 2)
    gt_matrix_full = gt_sub  # shape matches gt_matrix for the helper here
    ne = _count_proteins_in_toi(gt_matrix_full, np.arange(gt_sub.shape[1]), excluded_mask)
    assert ne == 2, f"helper returned {ne}, expected 2"

    # n at any active tau should never exceed ne (else coverage > 1)
    assert (n_at_tau <= ne).all(), (
        f"PK coverage would be >1: n={n_at_tau.tolist()} ne={ne}. "
        f"This is the regression the fix addresses."
    )

    # Specifically, at low tau where all predictions fire, n should equal 2
    # (only the 2 eligible proteins count, even though predictor emits for all 3)
    assert n_at_tau[0] == 2, f"expected n[tau=0.1]=2, got {n_at_tau[0]}"


def test_pk_precision_recall_unchanged_by_fix():
    """Sanity: the fix only touches the `n` column. TP/FP/FN and derived
    P/R/F should be identical to pre-fix behaviour."""
    pred_sub, gt_sub, toi_mask, excluded_mask, n_gt, tau_arr = _make_pk_scenario()

    metrics = compute_confusion_matrix_exclude_sparse(
        tau_arr, pred_sub, gt_sub, toi_mask, excluded_mask, n_gt
    )
    # At tau=0.1 (all preds fire, all get counted):
    # TP: proteins 0+2 each score term 2 / 3 respectively as TP → but only
    # protein 0's term 2 is both novel and GT. Protein 2's term 3 is novel & GT.
    # Protein 1 has excluded predictions → masked out by ~excluded_mask in line 266.
    # So TP at low tau: protein 0 term 2 (TP, weight 1) + protein 2 term 3 (TP, weight 1) = 2
    tp = metrics[0, 1]
    assert tp == 2.0, f"expected TP=2 at low tau, got {tp}"
