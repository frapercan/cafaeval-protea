"""Synthetic validation corpora for the cafaeval parity harness.

Generates deterministic OBO + ground truth + prediction + IA + exclude files
that exercise every code path of ``cafa_eval``: propagation, Fmax, Smin,
weighted variants, Partial-Knowledge exclusion, and terms-of-interest
filtering. Three tiered sizes are defined so the parity test can run in
seconds on the tiny corpus during iteration and on the large corpus before
committing a change.

No external data source and no PROTEA database are required: the entire
corpus is produced from a seeded ``random.Random`` instance.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import pathlib
import random
from typing import Dict, List, Tuple

NAMESPACES = (
    ("biological_process", "BP"),
    ("molecular_function", "MF"),
    ("cellular_component", "CC"),
)


@dataclasses.dataclass(frozen=True)
class CorpusSpec:
    """Size parameters for a synthetic corpus."""

    name: str
    terms_per_ns: int
    proteins: int
    max_children: int
    pred_terms_per_protein: int
    gt_terms_per_protein: int
    exclude_terms_per_protein: int
    seed: int


TINY = CorpusSpec(
    name="tiny",
    terms_per_ns=15,
    proteins=20,
    max_children=3,
    pred_terms_per_protein=6,
    gt_terms_per_protein=4,
    exclude_terms_per_protein=1,
    seed=1,
)

MEDIUM = CorpusSpec(
    name="medium",
    terms_per_ns=120,
    proteins=200,
    max_children=4,
    pred_terms_per_protein=30,
    gt_terms_per_protein=12,
    exclude_terms_per_protein=3,
    seed=2,
)

LARGE = CorpusSpec(
    name="large",
    terms_per_ns=600,
    proteins=1000,
    max_children=5,
    pred_terms_per_protein=100,
    gt_terms_per_protein=25,
    exclude_terms_per_protein=5,
    seed=3,
)

ALL_SPECS: Tuple[CorpusSpec, ...] = (TINY, MEDIUM, LARGE)


@dataclasses.dataclass
class Corpus:
    """Materialized corpus on disk."""

    spec: CorpusSpec
    root: pathlib.Path
    obo_path: pathlib.Path
    gt_path: pathlib.Path
    pred_dir: pathlib.Path
    ia_path: pathlib.Path
    exclude_path: pathlib.Path
    toi_path: pathlib.Path
    fingerprint: str


def _term_id(ns_idx: int, i: int) -> str:
    return f"GO:{(ns_idx + 1) * 10_000_000 + i:07d}"


def _build_dag(spec: CorpusSpec, rng: random.Random, ns_idx: int) -> List[Tuple[str, List[str]]]:
    """Build a random DAG. Returns [(term_id, parents), ...] in topo order.

    Term 0 is the root (no parents). Every non-root term has at least one
    is_a parent drawn from lower indices, plus zero or more extra parents
    to create a non-tree DAG shape that exercises propagation with
    multiple paths.
    """
    terms: List[Tuple[str, List[str]]] = []
    for i in range(spec.terms_per_ns):
        tid = _term_id(ns_idx, i)
        if i == 0:
            terms.append((tid, []))
            continue
        parents = [_term_id(ns_idx, rng.randrange(i))]
        for _ in range(rng.randint(0, 1)):
            extra = _term_id(ns_idx, rng.randrange(i))
            if extra not in parents:
                parents.append(extra)
        terms.append((tid, parents))
    return terms


def _write_obo(path: pathlib.Path, dags: Dict[str, List[Tuple[str, List[str]]]]) -> None:
    lines: List[str] = [
        "format-version: 1.2",
        "data-version: cafaeval-protea-synthetic/1.0",
        "ontology: go",
        "",
    ]
    for ns_full, terms in dags.items():
        for i, (tid, parents) in enumerate(terms):
            lines.append("[Term]")
            lines.append(f"id: {tid}")
            lines.append(f"name: synthetic_{ns_full}_{i}")
            lines.append(f"namespace: {ns_full}")
            for p in parents:
                lines.append(f"is_a: {p} ! parent")
            lines.append("")
    path.write_text("\n".join(lines))


def _write_gt(path: pathlib.Path, rows: List[Tuple[str, str]]) -> None:
    with path.open("w") as f:
        for protein, term in rows:
            f.write(f"{protein}\t{term}\n")


def _write_predictions(path: pathlib.Path, rows: List[Tuple[str, str, float]]) -> None:
    with path.open("w") as f:
        for protein, term, score in rows:
            f.write(f"{protein}\t{term}\t{score:.6f}\n")


def _write_ia(path: pathlib.Path, ia: Dict[str, float]) -> None:
    with path.open("w") as f:
        for term, val in ia.items():
            f.write(f"{term}\t{val:.6f}\n")


def _write_exclude(path: pathlib.Path, rows: List[Tuple[str, str]]) -> None:
    with path.open("w") as f:
        for protein, term in rows:
            f.write(f"{protein}\t{term}\n")


def _write_toi(path: pathlib.Path, terms: List[str]) -> None:
    with path.open("w") as f:
        for t in terms:
            f.write(f"{t}\n")


def _fingerprint(paths: List[pathlib.Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def build_corpus(spec: CorpusSpec, root: pathlib.Path) -> Corpus:
    """Materialize ``spec`` under ``root``. Deterministic for a given seed."""

    root = root / spec.name
    root.mkdir(parents=True, exist_ok=True)
    pred_dir = root / "predictions"
    pred_dir.mkdir(exist_ok=True)

    rng = random.Random(spec.seed)

    dags: Dict[str, List[Tuple[str, List[str]]]] = {}
    all_terms: List[str] = []
    for ns_idx, (ns_full, _ns_short) in enumerate(NAMESPACES):
        dag = _build_dag(spec, rng, ns_idx)
        dags[ns_full] = dag
        all_terms.extend(tid for tid, _ in dag)

    obo_path = root / "ontology.obo"
    _write_obo(obo_path, dags)

    proteins = [f"SYNTH{i:05d}" for i in range(spec.proteins)]

    gt_rows: List[Tuple[str, str]] = []
    for protein in proteins:
        chosen = rng.sample(all_terms, spec.gt_terms_per_protein)
        for term in chosen:
            gt_rows.append((protein, term))
    gt_path = root / "ground_truth.tsv"
    _write_gt(gt_path, gt_rows)

    pred_rows: List[Tuple[str, str, float]] = []
    for protein in proteins:
        chosen = rng.sample(all_terms, spec.pred_terms_per_protein)
        for term in chosen:
            score = round(rng.uniform(0.01, 0.99), 4)
            pred_rows.append((protein, term, score))
    pred_path = pred_dir / "method_synthetic.tsv"
    _write_predictions(pred_path, pred_rows)

    ia = {t: round(rng.uniform(0.1, 5.0), 4) for t in all_terms}
    ia_path = root / "ia.tsv"
    _write_ia(ia_path, ia)

    exclude_rows: List[Tuple[str, str]] = []
    if spec.exclude_terms_per_protein > 0:
        for protein in proteins:
            chosen = rng.sample(all_terms, spec.exclude_terms_per_protein)
            for term in chosen:
                exclude_rows.append((protein, term))
    exclude_path = root / "exclude.tsv"
    _write_exclude(exclude_path, exclude_rows)

    toi_count = max(1, math.ceil(len(all_terms) * 0.8))
    toi_terms = rng.sample(all_terms, toi_count)
    toi_path = root / "terms_of_interest.tsv"
    _write_toi(toi_path, toi_terms)

    fingerprint = _fingerprint([obo_path, gt_path, pred_path, ia_path, exclude_path, toi_path])

    return Corpus(
        spec=spec,
        root=root,
        obo_path=obo_path,
        gt_path=gt_path,
        pred_dir=pred_dir,
        ia_path=ia_path,
        exclude_path=exclude_path,
        toi_path=toi_path,
        fingerprint=fingerprint,
    )


def build_all(root: pathlib.Path) -> List[Corpus]:
    root.mkdir(parents=True, exist_ok=True)
    return [build_corpus(spec, root) for spec in ALL_SPECS]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic cafaeval corpora")
    parser.add_argument("--root", default="bench/corpora", type=pathlib.Path)
    parser.add_argument("--only", choices=[s.name for s in ALL_SPECS], default=None)
    args = parser.parse_args()

    if args.only is None:
        corpora = build_all(args.root)
    else:
        spec = next(s for s in ALL_SPECS if s.name == args.only)
        corpora = [build_corpus(spec, args.root)]

    for c in corpora:
        print(f"{c.spec.name:8s}  fingerprint={c.fingerprint}  root={c.root}")
