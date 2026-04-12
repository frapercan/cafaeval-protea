"""Shared fixtures for the parity diff test suite."""
from __future__ import annotations

import pathlib
import pickle

import pytest

from bench.corpus import ALL_SPECS, build_corpus
from bench.oracle_record import OracleRecord

ORACLE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "bench" / "oracle"
CORPORA_ROOT = pathlib.Path(__file__).resolve().parents[2] / "bench" / "corpora"


def _load_oracle(name: str) -> OracleRecord:
    path = ORACLE_ROOT / f"{name}.pkl"
    if not path.exists():
        pytest.skip(
            f"oracle {path} not found. Run "
            f"`python -m bench.freeze_oracle --only {name}` against upstream first."
        )
    return pickle.loads(path.read_bytes())


@pytest.fixture(scope="session", params=[s.name for s in ALL_SPECS], ids=lambda n: f"corpus-{n}")
def oracle_and_corpus(request):
    """Build the corpus on disk and pair it with its frozen oracle."""
    spec = next(s for s in ALL_SPECS if s.name == request.param)
    corpus = build_corpus(spec, CORPORA_ROOT)
    oracle = _load_oracle(spec.name)
    if oracle.corpus_fingerprint != corpus.fingerprint:
        pytest.fail(
            f"corpus fingerprint drift for {spec.name}: "
            f"oracle={oracle.corpus_fingerprint} current={corpus.fingerprint}. "
            f"Regenerate the oracle against unmodified upstream."
        )
    return oracle, corpus
