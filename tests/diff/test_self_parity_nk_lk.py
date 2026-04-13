"""Self-parity test: fork sparse path vs fork dense path, no PK exclude.

The frozen upstream oracle (``test_oracle_parity.py``) always invokes
``cafa_eval`` with ``exclude=...`` not None, so it only exercises the
PK code path of ``compute_metrics`` and ``evaluate_prediction``. The
NK / LK path (``gt_exclude is None``) is structurally different — it
calls ``compute_confusion_matrix_sparse`` with ``g`` / ``p`` directly
and skips the per-protein exclude mask entirely.

This module covers the NK / LK path by running the same synthetic
corpus through ``cafa_eval`` twice — once with ``CAFAEVAL_SPARSE=1``
(default sparse kernel) and once with ``CAFAEVAL_SPARSE=0`` (dense
fallback / pool path) — and asserting both runs produce the same
DataFrame within Phase B tolerance.

This is a *self*-parity gate: it does not validate the fork against
upstream, but it does ensure every Phase B refactor of
``compute_metrics`` keeps the sparse and dense paths in agreement, and
that those refactors do not silently break the NK / LK branch while
the upstream oracle (PK only) keeps passing.
"""
from __future__ import annotations

import os
import importlib

import numpy as np
import pandas as pd
import pytest

ATOL = 1e-9
RTOL = 1e-6


def _run_with_sparse_flag(corpus, sparse: str):
    os.environ["CAFAEVAL_SPARSE"] = sparse
    import cafaeval.evaluation as ev
    importlib.reload(ev)
    return ev.cafa_eval(
        str(corpus.obo_path),
        str(corpus.pred_dir),
        str(corpus.gt_path),
        ia=str(corpus.ia_path),
        toi_file=str(corpus.toi_path),
        norm="cafa",
        prop="max",
        th_step=0.05,
        n_cpu=1,
    )


def _assert_df_close(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    assert list(actual.columns) == list(expected.columns), (
        f"{label}: column mismatch\nsparse={list(actual.columns)}\ndense={list(expected.columns)}"
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
                    f"sparse={a[idx]}, dense={e[idx]})"
                )
        else:
            assert (a == e).all(), f"{label}: non-numeric column {col!r} differs"


@pytest.fixture
def nk_corpus(tmp_path):
    from bench.corpus import LARGE, build_corpus
    return build_corpus(LARGE, tmp_path / "corpora")


def test_nk_sparse_matches_dense(nk_corpus):
    """NK / LK path: sparse kernel agrees with dense pool fallback."""
    prev = os.environ.get("CAFAEVAL_SPARSE")
    try:
        df_sparse, best_sparse = _run_with_sparse_flag(nk_corpus, "1")
        df_dense, best_dense = _run_with_sparse_flag(nk_corpus, "0")
    finally:
        if prev is None:
            os.environ.pop("CAFAEVAL_SPARSE", None)
        else:
            os.environ["CAFAEVAL_SPARSE"] = prev
        import cafaeval.evaluation as ev
        importlib.reload(ev)

    _assert_df_close(df_sparse, df_dense, label="NK.df")
    assert set(best_sparse.keys()) == set(best_dense.keys())
    for key in best_dense:
        _assert_df_close(best_sparse[key], best_dense[key], label=f"NK.dfs_best[{key!r}]")
