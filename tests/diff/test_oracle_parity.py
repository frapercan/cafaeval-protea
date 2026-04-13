"""Parity test: fork output vs frozen upstream oracle.

Every optimization in this fork must keep this test passing. The tolerance
is tightened or relaxed depending on the phase:

- **Phase A** (bit-exact optimizations): ``atol=0, rtol=0``. Any single-bit
  divergence indicates a bug in the optimization.
- **Phase B** (sparse + numba rewrite): ``rtol=1e-6, atol=1e-9``. Divergence
  at this level is attributed to float summation order.

The active tolerance is controlled via the ``CAFAEVAL_PARITY_PHASE``
environment variable (``A`` or ``B``). Default is ``B`` — from the Phase
B2 sparse PK kernel onwards the fork reorders inner per-protein sums and
can no longer match upstream bit-for-bit (observed ULP-level ``4.4e-16``
divergence in ``pr``). Set the env var to ``A`` explicitly to re-run the
bit-exact gate over the Phase A cherry-picks only.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

PHASE = os.environ.get("CAFAEVAL_PARITY_PHASE", "B").upper()

if PHASE == "A":
    ATOL = 0.0
    RTOL = 0.0
elif PHASE == "B":
    ATOL = 1e-9
    RTOL = 1e-6
else:
    raise RuntimeError(f"CAFAEVAL_PARITY_PHASE must be A or B, got {PHASE!r}")


def _run_fork(corpus, call_kwargs):
    from cafaeval.evaluation import cafa_eval

    # ``_variant_with_exclude`` is injected by the conftest fixture to
    # route the call to the PK or the NK / LK branch. Strip it before
    # forwarding to ``cafa_eval`` itself.
    kwargs = dict(call_kwargs)
    with_exclude = bool(kwargs.pop("_variant_with_exclude", True))
    return cafa_eval(
        str(corpus.obo_path),
        str(corpus.pred_dir),
        str(corpus.gt_path),
        ia=str(corpus.ia_path),
        exclude=str(corpus.exclude_path) if with_exclude else None,
        toi_file=str(corpus.toi_path),
        **kwargs,
    )


def _assert_df_equal(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    assert actual is not None, f"{label}: fork returned None, oracle has {len(expected)} rows"
    assert list(actual.columns) == list(expected.columns), (
        f"{label}: column mismatch\nfork={list(actual.columns)}\noracle={list(expected.columns)}"
    )
    assert actual.index.equals(expected.index), (
        f"{label}: index mismatch ({len(actual)} vs {len(expected)} rows)"
    )
    for col in expected.columns:
        a = actual[col].to_numpy()
        e = expected[col].to_numpy()
        if np.issubdtype(e.dtype, np.number):
            if not np.allclose(a, e, atol=ATOL, rtol=RTOL, equal_nan=True):
                diff = np.abs(a - e)
                idx = int(np.nanargmax(diff))
                raise AssertionError(
                    f"{label}: column {col!r} diverges "
                    f"(max |Δ|={diff[idx]:.3e} at row {idx}, "
                    f"actual={a[idx]}, expected={e[idx]}, phase={PHASE})"
                )
        else:
            assert (a == e).all(), f"{label}: non-numeric column {col!r} differs"


def test_main_df_matches_oracle(oracle_and_corpus):
    oracle, corpus = oracle_and_corpus
    df_fork, _ = _run_fork(corpus, oracle.call_kwargs)
    df_oracle = pickle.loads(oracle.df_pickle)
    _assert_df_equal(df_fork, df_oracle, label=f"{oracle.corpus_name}.df")


def test_best_metrics_match_oracle(oracle_and_corpus):
    oracle, corpus = oracle_and_corpus
    _, dfs_best_fork = _run_fork(corpus, oracle.call_kwargs)
    dfs_best_oracle = pickle.loads(oracle.dfs_best_pickle)

    assert set(dfs_best_fork.keys()) == set(dfs_best_oracle.keys()), (
        f"best-metric set diverges: fork={sorted(dfs_best_fork)} "
        f"oracle={sorted(dfs_best_oracle)}"
    )
    for metric_key in dfs_best_oracle:
        _assert_df_equal(
            dfs_best_fork[metric_key],
            dfs_best_oracle[metric_key],
            label=f"{oracle.corpus_name}.dfs_best[{metric_key!r}]",
        )
