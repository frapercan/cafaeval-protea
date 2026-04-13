"""Shared fixtures for the parity diff test suite."""
from __future__ import annotations

import pathlib
import pickle

import pytest

from bench.corpus import ALL_SPECS, build_corpus
from bench.oracle_record import OracleRecord

ORACLE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "bench" / "oracle"
CORPORA_ROOT = pathlib.Path(__file__).resolve().parents[2] / "bench" / "corpora"

# Each corpus is frozen twice — once with the PK exclude file (exercises
# the ``gt_exclude is not None`` branch of ``compute_metrics`` /
# ``evaluate_prediction``) and once without (exercises the NK / LK
# branch). The legacy single-pickle layout (``<corpus>.pkl``) is kept as
# a fallback for oracles frozen before the variant split landed.
VARIANTS = ("pk", "nk")


def _load_oracle(name: str, variant: str) -> OracleRecord:
    variant_path = ORACLE_ROOT / f"{name}.{variant}.pkl"
    if variant_path.exists():
        return pickle.loads(variant_path.read_bytes())
    if variant == "pk":
        legacy_path = ORACLE_ROOT / f"{name}.pkl"
        if legacy_path.exists():
            return pickle.loads(legacy_path.read_bytes())
    pytest.skip(
        f"oracle {variant_path} not found. Run "
        f"`python -m bench.freeze_oracle --only {name}` against upstream first."
    )


def _ids(param):
    name, variant = param
    return f"corpus-{name}-{variant}"


@pytest.fixture(
    scope="session",
    params=[(s.name, v) for s in ALL_SPECS for v in VARIANTS],
    ids=_ids,
)
def oracle_and_corpus(request):
    """Build the corpus on disk and pair it with its frozen oracle.

    Yields one fixture instance per (corpus, variant) pair so the
    parity test runs over both the PK and the NK / LK code paths.
    """
    name, variant = request.param
    spec = next(s for s in ALL_SPECS if s.name == name)
    corpus = build_corpus(spec, CORPORA_ROOT)
    oracle = _load_oracle(name, variant)
    if oracle.corpus_fingerprint != corpus.fingerprint:
        pytest.fail(
            f"corpus fingerprint drift for {name}.{variant}: "
            f"oracle={oracle.corpus_fingerprint} current={corpus.fingerprint}. "
            f"Regenerate the oracle against unmodified upstream."
        )
    # Strip the synthetic ``_with_exclude`` marker before forwarding the
    # call kwargs to ``cafa_eval`` — that flag lives only in the oracle
    # record so the parity test can route the corpus to the correct
    # branch. ``cafa_eval`` itself receives ``exclude=...`` explicitly.
    with_exclude = bool(oracle.call_kwargs.get("_with_exclude", True))
    cleaned_kwargs = {k: v for k, v in oracle.call_kwargs.items()
                      if not k.startswith("_")}
    oracle.call_kwargs = cleaned_kwargs
    oracle.call_kwargs.setdefault("_variant_with_exclude", with_exclude)
    return oracle, corpus
