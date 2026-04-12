import os
import time
from cafaeval.graph import Graph, Prediction, GroundTruth, propagate
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
_parser_logger = logging.getLogger("cafaeval.parser")
_parser_logger.addHandler(logging.NullHandler())
# import xml.etree.ElementTree as ET


def obo_parser(obo_file, valid_rel=("is_a", "part_of"), ia_file=None, orphans=True):
    """
    Parse a OBO file and returns a list of ontologies, one for each namespace.
    Obsolete terms are excluded as well as external namespaces.
    """
    # Parse the OBO file and creates a different graph for each namespace
    term_dict = {}
    term_id = None
    namespace = None
    name = None
    term_def = None
    alt_id = []
    rel = []
    obsolete = True
    with open(obo_file) as f:
        for line in f:
            line = line.strip().split(": ")
            if line and len(line) > 1:
                k = line[0]
                v = ": ".join(line[1:])
                if k == "id":
                    # Populate the dictionary with the previous entry
                    if term_id is not None and obsolete is False and namespace is not None:
                        term_dict.setdefault(namespace, {})[term_id] = {'name': name,
                                                                       'namespace': namespace,
                                                                       'def': term_def,
                                                                       'alt_id': alt_id,
                                                                       'rel': rel}
                    # Assign current term ID
                    term_id = v

                    # Reset optional fields
                    alt_id = []
                    rel = []
                    obsolete = False
                    namespace = None

                elif k == "alt_id":
                    alt_id.append(v)
                elif k == "name":
                    name = v
                elif k == "namespace" and v != 'external':
                    namespace = v
                elif k == "def":
                    term_def = v
                elif k == 'is_obsolete':
                    obsolete = True
                elif k == "is_a" and k in valid_rel:
                    s = v.split('!')[0].strip()
                    rel.append(s)
                elif k == "relationship" and v.startswith("part_of") and "part_of" in valid_rel:
                    s = v.split()[1].strip()
                    rel.append(s)

        # Last record
        if obsolete is False and namespace is not None:
            term_dict.setdefault(namespace, {})[term_id] = {'name': name,
                                                          'namespace': namespace,
                                                          'def': term_def,
                                                          'alt_id': alt_id,
                                                          'rel': rel}

    # Parse IA file
    ia_dict = None
    if ia_file is not None:
        ia_dict = ia_parser(ia_file)

    ontologies = {}
    for ns, ont_dict in term_dict.items():
        ontologies[ns] = Graph(ns, ont_dict, ia_dict, orphans)

    return ontologies


def update_toi(ontologies, toi_file):
    """
    Remove terms not of interest from evaluation, eg for terms obsoleted since ontology was created
    :param ontologies: dict returned from obo_parser
    :param term_file: file with GO IDs to include in the terms of interest
    :return: copy of ontologies with updated toi
    """
    # load file of terms
    new_toi = {ns: [] for ns in ontologies.keys()}
    with open(toi_file) as f:
        for line in f:
            line = line.strip().split()
            if line:
                term = line[0]
                for ns in ontologies.keys():
                    if term in ontologies[ns].terms_dict.keys():
                        new_toi[ns].append(ontologies[ns].terms_dict[term]['index'])

                    # catch alt IDs if used
                    elif term in ontologies[ns].terms_dict_alt.keys():
                        alt_ids = ontologies[ns].terms_dict_alt[term]
                        for alt_id in alt_ids:
                            new_toi[ns].append(ontologies[ns].terms_dict[alt_id]['index'])

    # take intersection to make sure roots are excluded if needed
    for ns in ontologies.keys():
        ontologies[ns].toi = np.array(list(set(new_toi[ns]).intersection(ontologies[ns].toi)))

    # toi_ia is the non-zero IA terms. We need to remove any terms not in the TOI file from there too
    for ns in ontologies.keys():
        if ontologies[ns].toi_ia is not None:
            ontologies[ns].toi_ia = np.array(list(set(new_toi[ns]).intersection(ontologies[ns].toi_ia)))

    return ontologies


def gt_parser(gt_file, ontologies):
    """
    Parse ground truth file. Discard terms not included in the ontology.
    """
    gt_dict = {}
    replaced = {}
    with open(gt_file) as f:
        for line in f:
            line = line.strip().split()
            if line:
                p_id, term_id = line[:2]
                for ns in ontologies:
                    if term_id in ontologies[ns].terms_dict:
                        gt_dict.setdefault(ns, {}).setdefault(p_id, []).append(term_id)
                        break
                    # Replace alternative ids with canonical ids
                    elif term_id in ontologies[ns].terms_dict_alt:
                        for t_id in ontologies[ns].terms_dict_alt[term_id]:
                            gt_dict.setdefault(ns, {}).setdefault(p_id, []).append(t_id)
                            replaced.setdefault(ns, 0)
                            replaced[ns] += 1
                        break

    gts = {}
    for ns in ontologies:
        if gt_dict.get(ns):
            matrix = np.zeros((len(gt_dict[ns]), ontologies[ns].idxs), dtype='bool')
            ids = {}
            for i, p_id in enumerate(gt_dict[ns]):
                ids[p_id] = i
                for term_id in gt_dict[ns][p_id]:
                    matrix[i, ontologies[ns].terms_dict[term_id]['index']] = 1
            logger.debug("gt matrix {} {} ".format(ns, matrix))

            propagate(matrix, ontologies[ns], ontologies[ns].order, mode='max')
            logger.debug("gt matrix propagated {} {} ".format(ns, matrix))
            gts[ns] = GroundTruth(ids, matrix, ns)
            logger.info('Ground truth: {}, proteins {}, annotations {}, replaced alt. ids {}'.format(ns, len(ids),
                                                                                np.count_nonzero(matrix), replaced.get(ns, 0)))

    return gts


def gt_exclude_parser(exclude_file, gt, ontologies):
    """
    Process terms that should be excluded from evaluation.
    """
    # Propagate exclude terms and parse alternative IDs
    exclude_gt = gt_parser(exclude_file, ontologies)

    # reindex exclusion matrices to match ground truth
    exclude = {}
    for ns in gt:
        exclude_matrix = np.zeros_like(gt[ns].matrix)
        for protein, gt_index in gt[ns].ids.items():
            # Keep row corresponding to gt proteins
            if protein in exclude_gt[ns].ids:
                exclude_matrix[gt_index, :] = exclude_gt[ns].matrix[exclude_gt[ns].ids[protein], :]
        exclude[ns] = GroundTruth(gt[ns].ids, exclude_matrix, ns)
    return exclude


def _pred_parser_legacy(pred_file, ontologies, gts, ns_dict, term_index, ids,
                        matrix, row_nnz, replaced, max_terms):
    """Original line-by-line parser.

    Retained as the fallback for ``max_terms``-capped parses (the cap is
    order-sensitive so the vectorised path cannot reproduce it) and for
    any pathological input the fast path refuses to handle.
    """
    with open(pred_file, buffering=1024 * 1024) as f:
        for line in f:
            line = line.strip().split()
            if line and len(line) > 2:
                p_id, term_id, prob = line[:3]
                ns = ns_dict.get(term_id)
                if ns in gts and p_id in gts[ns].ids:
                    i = gts[ns].ids[p_id]
                    term_ids = [term_id]
                    if term_id in ontologies[ns].terms_dict_alt:
                        term_ids = ontologies[ns].terms_dict_alt[term_id]
                        replaced.setdefault(ns, 0)
                        replaced[ns] += len(term_ids)
                    for term_id in term_ids:
                        j = term_index[ns].get(term_id)
                        old = matrix[ns][i, j]
                        if max_terms is not None and old == 0.0 and row_nnz[ns][i] > max_terms:
                            continue
                        prob_f = float(prob)
                        if prob_f > old:
                            ids[ns][p_id] = i
                            matrix[ns][i, j] = prob_f
                            if old == 0.0:
                                row_nnz[ns][i] += 1


def _pred_parser_vectorised(pred_file, ontologies, gts, ns_dict, term_index,
                            ids, matrix, replaced):
    """PyArrow-backed bulk parser for the common case (no ``max_terms`` cap).

    Strategy:

    1. ``pyarrow.csv.read_csv`` reads the whole file at native speed
       (~10x pandas C engine on CAFA-shaped predictions).
    2. ``pid`` / ``tid`` string columns are dictionary-encoded once, so
       Python-level dict lookups run over the (small) unique-value sets
       instead of over every one of the millions of rows.
    3. The resulting numpy int code arrays are filtered per namespace
       with vectorised comparisons; per-namespace reductions use the same
       sort + ``np.maximum.reduceat`` group-max + scatter as the sparse
       propagation kernel.

    Raises on any format surprise (non-tsv, wrong column count) so the
    caller can fall back to :func:`_pred_parser_legacy`.
    """
    import pyarrow as pa  # lazy: pyarrow is an optional speed dependency
    import pyarrow.csv as pc

    tbl = pc.read_csv(
        pred_file,
        read_options=pc.ReadOptions(column_names=["pid", "tid", "prob"]),
        parse_options=pc.ParseOptions(delimiter="\t"),
        convert_options=pc.ConvertOptions(column_types={
            "pid": pa.string(),
            "tid": pa.string(),
            "prob": pa.float64(),
        }),
    )
    if tbl.num_rows == 0:
        return

    pids_dict = tbl.column("pid").combine_chunks().dictionary_encode()
    tids_dict = tbl.column("tid").combine_chunks().dictionary_encode()
    probs_arr = tbl.column("prob").combine_chunks().to_numpy(zero_copy_only=True)

    pid_unique = pids_dict.dictionary.to_pylist()
    tid_unique = tids_dict.dictionary.to_pylist()
    pid_codes = pids_dict.indices.to_numpy(zero_copy_only=False)
    tid_codes = tids_dict.indices.to_numpy(zero_copy_only=False)

    ns_list = list(gts.keys())
    ns_to_idx = {ns: i for i, ns in enumerate(ns_list)}

    # Per unique tid: which namespace does it belong to (-1 if none).
    tid_ns_code = np.full(len(tid_unique), -1, dtype=np.int8)
    for code, tid in enumerate(tid_unique):
        ns = ns_dict.get(tid)
        if ns is not None and ns in ns_to_idx:
            tid_ns_code[code] = ns_to_idx[ns]

    # Per unique tid, per namespace: canonical column index (-1 if absent).
    # Also flag whether the tid is an alt id in that namespace (stored as
    # -2 so alt handling can be triggered without another Python lookup).
    tid_col_per_ns = []
    alt_tid_lookup_per_ns = []
    for ns in ns_list:
        tmap = term_index[ns]
        alt_dict = ontologies[ns].terms_dict_alt
        col_arr = np.full(len(tid_unique), -1, dtype=np.int64)
        alt_codes = []
        for code, tid in enumerate(tid_unique):
            canon_col = tmap.get(tid)
            if canon_col is not None:
                col_arr[code] = canon_col
            elif alt_dict and tid in alt_dict:
                col_arr[code] = -2
                alt_codes.append(code)
        tid_col_per_ns.append(col_arr)
        alt_tid_lookup_per_ns.append(alt_codes)

    # Per unique pid, per namespace: row index (-1 if not in this GT set).
    pid_row_per_ns = []
    for ns in ns_list:
        gt_ids = gts[ns].ids
        row_arr = np.full(len(pid_unique), -1, dtype=np.int64)
        for code, pid in enumerate(pid_unique):
            v = gt_ids.get(pid)
            if v is not None:
                row_arr[code] = v
        pid_row_per_ns.append(row_arr)

    row_ns = tid_ns_code[tid_codes]  # int8, same length as the full table

    for ns_idx, ns in enumerate(ns_list):
        ns_mask = row_ns == ns_idx
        if not ns_mask.any():
            continue

        pid_codes_ns = pid_codes[ns_mask]
        tid_codes_ns = tid_codes[ns_mask]
        probs_ns = probs_arr[ns_mask]

        # Protein filter: only keep rows whose pid is in this GT.
        p_idx_all = pid_row_per_ns[ns_idx][pid_codes_ns]
        in_gt = p_idx_all >= 0
        if not in_gt.any():
            continue
        p_idx_all = p_idx_all[in_gt]
        tid_codes_ns = tid_codes_ns[in_gt]
        probs_ns = probs_ns[in_gt]

        # Column resolution: canonical terms, alt ids (value -2), misses (-1).
        col_lookup = tid_col_per_ns[ns_idx]
        col_all = col_lookup[tid_codes_ns]
        canonical_mask = col_all >= 0
        alt_mask = col_all == -2

        p_idx_parts = [p_idx_all[canonical_mask]]
        t_idx_parts = [col_all[canonical_mask]]
        v_parts = [probs_ns[canonical_mask]]

        if alt_mask.any():
            # Alt-id expansion: each alt id may map to a set of canonical
            # term ids. Loop over the (small) alt subset only.
            alt_dict = ontologies[ns].terms_dict_alt
            ns_term_index = term_index[ns]
            alt_pos = np.flatnonzero(alt_mask)
            exp_p = []
            exp_t = []
            exp_v = []
            for k in alt_pos:
                tid_str = tid_unique[tid_codes_ns[k]]
                canon_set = alt_dict.get(tid_str)
                if not canon_set:
                    continue
                replaced[ns] = replaced.get(ns, 0) + len(canon_set)
                for canon in canon_set:
                    col = ns_term_index.get(canon)
                    if col is None:
                        continue
                    exp_p.append(int(p_idx_all[k]))
                    exp_t.append(col)
                    exp_v.append(float(probs_ns[k]))
            if exp_p:
                p_idx_parts.append(np.asarray(exp_p, dtype=np.int64))
                t_idx_parts.append(np.asarray(exp_t, dtype=np.int64))
                v_parts.append(np.asarray(exp_v, dtype=np.float64))

        p_final = np.concatenate(p_idx_parts) if len(p_idx_parts) > 1 else p_idx_parts[0]
        t_final = np.concatenate(t_idx_parts) if len(t_idx_parts) > 1 else t_idx_parts[0]
        v_final = np.concatenate(v_parts) if len(v_parts) > 1 else v_parts[0]
        if p_final.size == 0:
            continue

        # Sort-based group-max over (row, col). Single int64 flat key, one
        # stable argsort, np.maximum.reduceat finishes the reduction.
        n_terms = matrix[ns].shape[1]
        flat = p_final * np.int64(n_terms) + t_final
        order = np.argsort(flat, kind="stable")
        flat_s = flat[order]
        v_s = v_final[order]
        starts = np.empty(flat_s.size, dtype=bool)
        starts[0] = True
        np.not_equal(flat_s[1:], flat_s[:-1], out=starts[1:])
        start_idx = np.flatnonzero(starts)
        max_v = np.maximum.reduceat(v_s, start_idx)
        unique_flat = flat_s[start_idx]
        unique_rows = unique_flat // np.int64(n_terms)
        unique_cols = unique_flat % np.int64(n_terms)

        # Scatter into the namespace matrix. The matrix starts at zero, so
        # direct assignment is equivalent to the legacy
        # "if prob > old: matrix[i, j] = prob" write rule.
        matrix[ns][unique_rows, unique_cols] = max_v

        # Register proteins that contributed at least one non-zero value.
        prot_has_any = np.zeros(matrix[ns].shape[0], dtype=bool)
        prot_has_any[unique_rows[max_v > 0.0]] = True
        surviving_rows = np.flatnonzero(prot_has_any)
        if surviving_rows.size:
            gt_ids = gts[ns].ids
            inv_ids = {int(v): k for k, v in gt_ids.items()}
            for r in surviving_rows.tolist():
                ids[ns][inv_ids[r]] = r


def pred_parser(pred_file, ontologies, gts, prop_mode, max_terms=None, n_cpu=0):
    """
    Parse a prediction file and returns a list of prediction objects, one for each namespace.
    If a predicted is predicted multiple times for the same target, it stores the max.
    This is the slow step if the input file is huge, ca. 1 minute for 5GB input on SSD disk.
    """
    ids = {}
    matrix = {}
    ns_dict = {}  # {namespace: term}
    replaced = {}
    row_nnz = {}
    term_index = {}
    for ns in gts:
        matrix[ns] = np.zeros(gts[ns].matrix.shape, dtype='float')
        row_nnz[ns] = np.zeros(gts[ns].matrix.shape[0], dtype=np.int32)
        ids[ns] = {}
        term_index[ns] = {t: info["index"] for t, info in ontologies[ns].terms_dict.items()}
        for term in ontologies[ns].terms_dict:
            ns_dict[term] = ns
        for term in ontologies[ns].terms_dict_alt:
            ns_dict[term] = ns

    t0 = time.time()
    fast_path = (
        max_terms is None
        and os.environ.get("CAFAEVAL_FAST_PARSER", "1") not in ("0", "false", "False")
    )
    used_fast_path = False
    if fast_path:
        try:
            _pred_parser_vectorised(
                pred_file, ontologies, gts, ns_dict, term_index,
                ids, matrix, replaced,
            )
            used_fast_path = True
        except Exception as exc:
            _parser_logger.warning(
                "pred_parser fast path failed, falling back to legacy loop",
                extra={"file": pred_file, "error": repr(exc)},
            )
            # Reset any partial state the fast path may have written.
            for ns in gts:
                matrix[ns].fill(0)
                ids[ns].clear()
            replaced.clear()

    if not used_fast_path:
        _pred_parser_legacy(
            pred_file, ontologies, gts, ns_dict, term_index,
            ids, matrix, row_nnz, replaced, max_terms,
        )

    t1 = time.time()
    _parser_logger.info(
        "pred_parser prepared",
        extra={"file": pred_file, "seconds": round(t1 - t0, 3),
               "path": "vectorised" if used_fast_path else "legacy"},
    )

    predictions = {}
    tp0 = time.time()
    for ns in ids:
        if ids[ns]:
            logger.debug("pred matrix {} {} ".format(ns, matrix))
            t_ns = time.time()
            propagate(matrix[ns], ontologies[ns], ontologies[ns].order, mode=prop_mode, parallel=n_cpu)
            _parser_logger.info(
                "pred_parser propagated namespace",
                extra={"file": pred_file, "ns": ns,
                       "seconds": round(time.time() - t_ns, 3)},
            )
            logger.debug("pred matrix {} {} ".format(ns, matrix))
            predictions[ns] = Prediction(ids[ns], matrix[ns], ns)
            logger.info("Prediction: {}, {}, proteins {}, annotations {}, replaced alt. ids {}".format(pred_file, ns, len(ids[ns]),
                                                                                np.count_nonzero(matrix[ns]), replaced.get(ns, 0)))
    _parser_logger.info(
        "pred_parser propagated",
        extra={"file": pred_file, "seconds": round(time.time() - tp0, 3)},
    )

    if not predictions:
        logger.warning("Empty prediction! Check format or overlap with ground truth")

    return predictions


def ia_parser(file):
    ia_dict = {}
    with open(file) as f:
        for line in f:
            if line:
                term, ia = line.strip().split()
                ia_dict[term] = float(ia)
    return ia_dict
