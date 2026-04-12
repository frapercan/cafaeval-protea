"""Dataclass used to serialize frozen upstream cafa_eval output.

Kept in its own module so that pickling is stable regardless of whether
``bench.freeze_oracle`` is imported or executed as ``python -m``. If
``OracleRecord`` lived in ``bench.freeze_oracle``, running that file as
``__main__`` would pickle instances under ``__main__.OracleRecord`` and
break loading from other contexts.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict


@dataclasses.dataclass
class OracleRecord:
    """Everything needed to verify parity against a frozen upstream run."""

    corpus_name: str
    corpus_fingerprint: str
    call_kwargs: Dict[str, Any]
    df_pickle: bytes
    dfs_best_pickle: bytes
    python_version: str
    platform: str
    elapsed_seconds: float
    upstream_commit_hint: str
