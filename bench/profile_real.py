"""Profile cafa_eval on a real-world corpus.

Reads OBO / ground-truth / prediction / IA files from paths passed on the
command line, runs ``cafa_eval`` under ``cProfile``, and prints:

1. a short cumulative-time table (top N) so it is obvious where the
   time is spent,
2. total wall-clock seconds and approximate memory high-water mark,
3. the path to the raw .prof file so it can be opened with
   ``snakeviz`` for a visual breakdown.

Usage example::

    python bench/profile_real.py \\
        --obo /abs/path/go-basic.obo \\
        --pred-dir /abs/path/predictions_NK \\
        --gt /abs/path/gt_NK.tsv \\
        --ia /abs/path/IA_cafa6.tsv \\
        --n-cpu 4

The script does not copy or vendor the input files — it reads them in
place, so the fork repository stays clean.
"""
from __future__ import annotations

import argparse
import cProfile
import logging
import pathlib
import pstats
import resource
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obo", type=pathlib.Path, required=True)
    parser.add_argument("--pred-dir", type=pathlib.Path, required=True)
    parser.add_argument("--gt", type=pathlib.Path, required=True)
    parser.add_argument("--ia", type=pathlib.Path, default=None)
    parser.add_argument("--exclude", type=pathlib.Path, default=None,
                        help="Optional PK exclude file (known_terms.tsv).")
    parser.add_argument("--n-cpu", type=int, default=1)
    parser.add_argument("--th-step", type=float, default=0.05)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("bench/profile_real.prof"))
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from cafaeval.evaluation import cafa_eval

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    df, dfs_best = cafa_eval(
        str(args.obo),
        str(args.pred_dir),
        str(args.gt),
        ia=str(args.ia) if args.ia else None,
        exclude=str(args.exclude) if args.exclude else None,
        norm="cafa",
        prop="max",
        th_step=args.th_step,
        n_cpu=args.n_cpu,
    )
    pr.disable()
    elapsed = time.perf_counter() - t0

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("", file=sys.stderr)
    print(f"[profile] wall-clock: {elapsed:.2f}s", file=sys.stderr)
    print(f"[profile] RSS high-water: {rss_kb / 1024:.1f} MB", file=sys.stderr)
    if df is not None:
        print(f"[profile] result df: {len(df)} rows", file=sys.stderr)
        print(f"[profile] best metrics: {list(dfs_best.keys())}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pr.dump_stats(str(args.out))
    print(f"[profile] raw stats: {args.out}", file=sys.stderr)

    ps = pstats.Stats(pr, stream=sys.stdout).sort_stats("cumulative")
    print("")
    print(f"[profile] top {args.top} by cumulative time")
    ps.print_stats(args.top)

    ps2 = pstats.Stats(pr, stream=sys.stdout).sort_stats("tottime")
    print("")
    print(f"[profile] top {args.top} by total self time")
    ps2.print_stats(args.top)


if __name__ == "__main__":
    main()
