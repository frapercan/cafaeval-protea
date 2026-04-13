"""Freeze the unmodified upstream cafaeval output as a parity oracle.

This script is meant to be run **once**, with an environment that has the
unmodified upstream cafaeval installed (or the fork checked out at the
pristine base commit ``16a6a6d``). It runs ``cafa_eval`` against every
synthetic corpus produced by ``bench.corpus`` and serializes the full
output dict-of-DataFrames into ``bench/oracle/<corpus>.pkl``.

The oracle covers the full metric surface the parity test checks:

- the main ``df`` DataFrame (all metrics × all thresholds × all namespaces),
- the per-metric ``dfs_best`` dict (f, f_w, s, f_micro, f_micro_w),
- the exact ``cafa_eval`` call kwargs and the corpus fingerprint.

The fingerprint binds the oracle to a specific corpus contents. If either
the corpus generator or the upstream code changes, the oracle must be
regenerated.

Usage::

    python -m bench.freeze_oracle --only tiny
    python -m bench.freeze_oracle             # all corpora
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import pickle
import platform
import sys
import time
from typing import Any, Dict

from bench.corpus import ALL_SPECS, CorpusSpec, build_corpus
from bench.oracle_record import OracleRecord

LOG = logging.getLogger("bench.freeze_oracle")

EVAL_KWARGS = dict(
    norm="cafa",
    prop="max",
    th_step=0.05,
    n_cpu=1,
)


def _run_upstream(corpus, call_kwargs: Dict[str, Any], *, with_exclude: bool):
    from cafaeval.evaluation import cafa_eval

    t0 = time.perf_counter()
    df, dfs_best = cafa_eval(
        str(corpus.obo_path),
        str(corpus.pred_dir),
        str(corpus.gt_path),
        ia=str(corpus.ia_path),
        exclude=str(corpus.exclude_path) if with_exclude else None,
        toi_file=str(corpus.toi_path),
        **call_kwargs,
    )
    elapsed = time.perf_counter() - t0
    return df, dfs_best, elapsed


def freeze(spec: CorpusSpec, corpora_root: pathlib.Path, oracle_root: pathlib.Path) -> list[pathlib.Path]:
    """Freeze both the PK (with exclude) and the NK/LK (without exclude)
    code paths. The resulting oracles cover both branches of
    ``compute_metrics`` and ``evaluate_prediction`` so a parity test
    cannot regress either branch silently.
    """
    corpus = build_corpus(spec, corpora_root)
    LOG.info("corpus %s fingerprint=%s", spec.name, corpus.fingerprint)
    oracle_root.mkdir(parents=True, exist_ok=True)
    out_paths: list[pathlib.Path] = []

    for variant, with_exclude in (("pk", True), ("nk", False)):
        df, dfs_best, elapsed = _run_upstream(
            corpus, EVAL_KWARGS, with_exclude=with_exclude
        )
        LOG.info(
            "upstream cafa_eval[%s] finished in %.2fs (rows=%d)",
            variant, elapsed, 0 if df is None else len(df),
        )

        kwargs = dict(EVAL_KWARGS)
        kwargs["_with_exclude"] = with_exclude
        record = OracleRecord(
            corpus_name=f"{spec.name}.{variant}",
            corpus_fingerprint=corpus.fingerprint,
            call_kwargs=kwargs,
            df_pickle=pickle.dumps(df),
            dfs_best_pickle=pickle.dumps(dfs_best),
            python_version=sys.version,
            platform=platform.platform(),
            elapsed_seconds=elapsed,
            upstream_commit_hint="16a6a6d",
        )

        out_path = oracle_root / f"{spec.name}.{variant}.pkl"
        out_path.write_bytes(pickle.dumps(record))
        LOG.info("wrote oracle to %s", out_path)
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("cafaeval").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Freeze upstream cafa_eval output as parity oracle")
    parser.add_argument("--corpora-root", default="bench/corpora", type=pathlib.Path)
    parser.add_argument("--oracle-root", default="bench/oracle", type=pathlib.Path)
    parser.add_argument("--only", choices=[s.name for s in ALL_SPECS], default=None)
    args = parser.parse_args()

    specs = [s for s in ALL_SPECS if args.only is None or s.name == args.only]
    for spec in specs:
        freeze(spec, args.corpora_root, args.oracle_root)


if __name__ == "__main__":
    main()
