import numpy as np
import logging
import os
from collections import deque

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
_propagate_logger = logging.getLogger("cafaeval.propagate")
_propagate_logger.addHandler(logging.NullHandler())


class Graph:
    """
    Ontology class. One ontology == one namespace
    DAG is the adjacence matrix (sparse) which represent a Directed Acyclic Graph where
    DAG(i,j) == 1 means that the go term i is_a (or is part_of) j
    Parents that are in a different namespace are discarded
    """
    def __init__(self, namespace, terms_dict, ia_dict=None, orphans=False):

        """
        terms_dict = {term: {name: , namespace: , def: , alt_id: , rel:}}
        """
        self.namespace = namespace
        self.terms_dict = {}  # {term: {index: , name: , namespace: , def: }  used to assign term indexes in the gt
        self.terms_dict_alt = {}  # {alt_id: set(term, ...) }  alternative ids to canonical ids
        self.terms_list = []  # [{id: term, name:, namespace: , def:, adj: set(), children: set()}, ...]
        self.idxs = None  # Number of terms
        self.order = None
        self.toi = None
        self.toi_ia = None
        self.ia = None

        rel_list = []
        for self.idxs, (term_id, term) in enumerate(terms_dict.items()):
            rel_list.extend([[term_id, rel, term['namespace']] for rel in term['rel']])
            self.terms_list.append({'id': term_id, 'name': term['name'], 'namespace': namespace, 'def': term['def'],
                                 'adj': set(), 'children': set()})
            self.terms_dict[term_id] = {'index': self.idxs, 'name': term['name'], 'namespace': namespace, 'def': term['def']}
            for a_id in term['alt_id']:
                self.terms_dict_alt.setdefault(a_id, set()).add(term_id)

        self.idxs += 1

        # Sparse DAG: ``dag[i, j] == 1`` (i is_a/part_of j) is held as the
        # per-term parent (``adj``) and child sets plus CSR index arrays, not a
        # dense (idxs x idxs) boolean matrix that is almost entirely zeros.
        # id1 term (row, axis 0), id2 parent (column, axis 1)
        for id1, id2, ns in rel_list:
            if self.terms_dict.get(id2):
                i = self.terms_dict[id1]['index']
                j = self.terms_dict[id2]['index']
                # Sets dedupe duplicate edges, exactly like the boolean matrix:
                # a parent-child pair contributes a single edge.
                self.terms_list[i]['adj'].add(j)
                self.terms_list[j]['children'].add(i)
                logging.debug("i,j {},{} {},{}".format(i, j, id1, id2))
            else:
                logging.debug('Skipping branch to external namespace: {}'.format(id2))

        self._build_sparse_dag()
        logging.debug("dag edges {}".format(int(self._par_indptr[-1])))

        # Topological sorting
        self.top_sort()
        logging.debug("order sorted {}".format(self.order))

        if orphans:
            self.toi = np.arange(self.idxs)  # All terms, also those without parents
        else:
            self.toi = np.nonzero(self._parent_count > 0)[0]  # Only terms with parents
        logging.debug("toi {}".format(self.toi))

        if ia_dict is not None:
            self.set_ia(ia_dict)

        # total = terms with >=1 parent; roots = no parents; leaves = no children
        logging.info("Ontology: {}, total {}, roots {}, leaves {}, alternative_ids {}".format(self.namespace,
                                                                int(np.count_nonzero(self._parent_count)),
                                                                int(np.count_nonzero(self._parent_count == 0)),
                                                                int(np.count_nonzero(self._child_count == 0)),
                                                                len(self.terms_dict_alt)))

        return

    def _build_sparse_dag(self):
        """Build CSR parent/child index arrays and degree vectors from the
        per-term ``adj`` (parents) and ``children`` sets, reproducing the old
        dense boolean adjacency exactly:

        * ``self._parent_count[i] == old_dag.sum(axis=1)[i]``  (parents of i)
        * ``self._child_count[j]  == old_dag.sum(axis=0)[j]``  (children of j)
        * ``self._par_idx[self._par_indptr[t]:...]`` are the parents of ``t``,
          ascending, i.e. exactly ``np.nonzero(old_dag[t, :])``.
        * ``self._chi_idx[self._chi_indptr[t]:...]`` are the children of ``t``,
          ascending, i.e. exactly ``np.flatnonzero(old_dag[:, t])``.
        """
        n = self.idxs
        parent_count = np.fromiter((len(t['adj']) for t in self.terms_list),
                                   dtype=np.int64, count=n)
        child_count = np.fromiter((len(t['children']) for t in self.terms_list),
                                  dtype=np.int64, count=n)
        par_indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(parent_count, out=par_indptr[1:])
        chi_indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(child_count, out=chi_indptr[1:])
        par_idx = np.empty(int(par_indptr[-1]), dtype=np.int64)
        chi_idx = np.empty(int(chi_indptr[-1]), dtype=np.int64)
        for t in range(n):
            adj = self.terms_list[t]['adj']
            if adj:
                par_idx[par_indptr[t]:par_indptr[t + 1]] = sorted(adj)
            chl = self.terms_list[t]['children']
            if chl:
                chi_idx[chi_indptr[t]:chi_indptr[t + 1]] = sorted(chl)
        self._parent_count = parent_count
        self._child_count = child_count
        self._par_indptr, self._par_idx = par_indptr, par_idx
        self._chi_indptr, self._chi_idx = chi_indptr, chi_idx

    def top_sort(self):
        """
        Takes a sparse matrix representing a DAG and returns an array with nodes indexes in topological order
        https://en.wikipedia.org/wiki/Topological_sorting
        """
        indexes = []
        visited = 0
        rows = self.idxs

        # create a vector containing the in-degree of each node
        # (number of children per node == old dense dag.sum(axis=0)); copy so
        # the Kahn sweep below can decrement it in place
        in_degree = self._child_count.copy()
        # logging.debug("degree {}".format(in_degree))

        # find the nodes with in-degree 0 (leaves) and add them to the queue.
        # deque gives O(1) popleft; a plain list's pop(0) is O(n) -> O(n^2) total.
        # FIFO order is identical, so self.order is unchanged.
        queue = deque(np.nonzero(in_degree == 0)[0].tolist())
        # logging.debug("queue {}".format(queue))

        # for each element of the queue increment visits, add them to the list of ordered nodes
        # and decrease the in-degree of the neighbor nodes
        # and add them to the queue if they reach in-degree == 0
        while queue:
            visited += 1
            idx = queue.popleft()
            indexes.append(idx)
            in_degree[idx] -= 1
            adj = self.terms_list[idx]['adj']
            if len(adj) > 0:
                for j in adj:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)

        # if visited is equal to the number of nodes in the graph then the sorting is complete
        # otherwise the graph can't be sorted with topological order
        if visited == rows:
            self.order = indexes
        else:
            raise Exception("The sparse matrix doesn't represent an acyclic graph")

    def set_ia(self, ia_dict):
        self.ia = np.zeros(self.idxs, dtype='float')
        for term_id in self.terms_dict:
            if ia_dict.get(term_id):
                self.ia[self.terms_dict[term_id]['index']] = ia_dict.get(term_id)
            else:
                logging.debug('Missing IA for term: {}'.format(term_id))
        # Convert inf to zero
        np.nan_to_num(self.ia, copy=False, nan=0, posinf=0, neginf=0)
        self.toi_ia = np.nonzero(self.ia > 0)[0]


class Prediction:
    """
    The score matrix contains the scores given by the predictor for every node of the ontology
    """
    def __init__(self, ids, matrix, namespace=None):
        self.ids = ids
        self.matrix = matrix  # scores
        self.namespace = namespace

    def __str__(self):
        return "\n".join(["{}\t{}\t{}".format(index, self.matrix[index], self.namespace) for index, _id in enumerate(self.ids)])


class GroundTruth:
    def __init__(self, ids, matrix, namespace=None):
        self.ids = ids
        self.matrix = matrix
        self.namespace = namespace


_PROPAGATE_WORK_THRESHOLD = 800_000_000


def _children_cache(ont):
    children_by_term = getattr(ont, "_children_by_term", None)
    if children_by_term is None:
        # CSR children slices (ascending), identical to the old
        # ``np.flatnonzero(dag[:, term_id])`` per-term scan but without the
        # dense matrix.
        indptr, idx = ont._chi_indptr, ont._chi_idx
        children_by_term = [
            idx[indptr[term_id]:indptr[term_id + 1]]
            for term_id in range(len(indptr) - 1)
        ]
        ont._children_by_term = children_by_term
    return children_by_term


def _propagate_serial(matrix, order_, children_by_term, mode):
    for term_id in order_:
        children = children_by_term[term_id]
        if children.size == 0:
            continue
        if mode == "max":
            child_max = matrix[:, children].max(axis=1)
            matrix[:, term_id] = np.maximum(matrix[:, term_id], child_max)
        elif mode == "fill":
            rows = np.flatnonzero(matrix[:, term_id] == 0)
            if rows.size:
                idx = np.ix_(rows, children)
                matrix[rows, term_id] = matrix[idx].max(axis=1)


def _ancestors_csr(ont):
    """Lazy per-ontology cache of transitive ancestors (self inclusive).

    Returns ``(indptr, indices)`` flat arrays with
    ``ancestors[t] == indices[indptr[t]:indptr[t+1]]``. Each list contains
    term ``t`` itself plus every term reachable by walking DAG parent edges
    up to the roots. Computed once per ``Graph`` instance.
    """
    cached = getattr(ont, "_ancestors_csr", None)
    if cached is not None:
        return cached

    n = int(ont.idxs)

    # Parent CSR is precomputed on the Graph: ``sorted_cols[p_indptr[t]:p_indptr[t+1]]``
    # are the parents of ``t``, ascending. This reproduces exactly what the old
    # ``np.nonzero(dense_dag)`` + group-by-child-row built, without ever
    # materialising the dense matrix.
    p_indptr = ont._par_indptr
    sorted_cols = ont._par_idx

    # ont.order is leaves → roots; reverse so parents are always finished
    # before we look them up.
    top_down = np.asarray(ont.order)[::-1]

    # Use Python sets during construction, flatten to arrays afterwards.
    ancestors = [None] * n
    for t in top_down:
        t_int = int(t)
        s = {t_int}
        parents_slice = sorted_cols[p_indptr[t_int]:p_indptr[t_int + 1]]
        for p in parents_slice:
            s.update(ancestors[int(p)])
        ancestors[t_int] = s

    lens = np.fromiter((len(ancestors[t]) for t in range(n)), dtype=np.int64, count=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=indptr[1:])
    total = int(indptr[-1])
    indices = np.empty(total, dtype=np.int64)
    for t in range(n):
        a = ancestors[t]
        if a:
            indices[indptr[t]:indptr[t + 1]] = np.fromiter(a, dtype=np.int64, count=len(a))

    ont._ancestors_csr = (indptr, indices)
    return indptr, indices


def _propagate_sparse_pushup(matrix, ont, mode, triples=None):
    """Sparse alternative to ``_propagate_serial``.

    Scatters each input non-zero ``(row, col, score)`` into every ancestor of
    ``col`` (self inclusive), reduces by ``(row, ancestor)`` via a stable
    sort + ``np.maximum.reduceat`` group-max, and writes the reduced values
    back in-place. Cost is ``O(nnz * avg_ancestors + R log R)`` where ``R``
    is the expanded triple count — on typical CAFA-shaped inputs this beats
    the per-term dense sweep by 1-2 orders of magnitude because only
    predicted terms contribute work.

    If ``triples`` is ``(nz_rows, nz_cols, nz_scores)`` the caller already
    knows the input non-zero positions (e.g. the parser just scattered them
    into ``matrix``) and we skip the dense ``np.nonzero`` scan. This is a
    large win for ground-truth matrices where the non-zero density is
    ~1e-4 and ``np.nonzero`` would otherwise touch every cell.

    For ``mode='fill'`` the update semantics are "only overwrite cells that
    were originally zero"; this is restored by snapshotting the input
    non-zero positions and writing their original values back at the end.
    """
    n_prot, n_terms = matrix.shape
    if n_prot == 0 or n_terms == 0:
        return

    if triples is not None:
        nz_rows, nz_cols, nz_scores = triples
    else:
        nz_rows, nz_cols = np.nonzero(matrix)
        if nz_rows.size == 0:
            return
        nz_scores = matrix[nz_rows, nz_cols]

    if nz_rows.size == 0:
        return

    indptr, anc_indices = _ancestors_csr(ont)

    n_anc = (indptr[nz_cols + 1] - indptr[nz_cols]).astype(np.int64)
    total = int(n_anc.sum())
    if total == 0:
        return

    # Vectorised gather of the per-non-zero ancestor slices into one flat
    # array. Equivalent to concatenating ``anc_indices[indptr[c]:indptr[c+1]]``
    # for every non-zero column ``c``, but without a Python loop.
    #   block_starts[i]     = indptr[nz_cols[i]]
    #   global_offsets[i]   = sum(n_anc[:i])
    #   local_offset[j]     = j - global_offsets[ block(j) ]
    #   anc_ptr[j]          = block_starts[ block(j) ] + local_offset[j]
    block_starts = indptr[nz_cols]
    global_offsets = np.zeros(nz_rows.size + 1, dtype=np.int64)
    np.cumsum(n_anc, out=global_offsets[1:])
    base_per_j = np.repeat(block_starts, n_anc)
    local_offset = np.arange(total, dtype=np.int64) - np.repeat(global_offsets[:-1], n_anc)
    expanded_cols = anc_indices[base_per_j + local_offset]
    expanded_rows = np.repeat(nz_rows, n_anc)
    expanded_scores = np.repeat(nz_scores, n_anc)

    # Group-max over (row, ancestor). Use a flat key so a single argsort
    # gives the grouping we need.
    flat = expanded_rows.astype(np.int64) * n_terms + expanded_cols
    order_idx = np.argsort(flat, kind='stable')
    flat_s = flat[order_idx]
    scores_s = expanded_scores[order_idx]

    group_starts = np.empty(flat_s.size, dtype=bool)
    group_starts[0] = True
    np.not_equal(flat_s[1:], flat_s[:-1], out=group_starts[1:])
    start_idx = np.flatnonzero(group_starts)
    group_max = np.maximum.reduceat(scores_s, start_idx)
    unique_flat = flat_s[start_idx]

    out_rows = unique_flat // n_terms
    out_cols = unique_flat % n_terms

    # Max against what is currently in the matrix. For mode='max' this is the
    # final answer; for mode='fill' we restore original non-zeros below.
    current = matrix[out_rows, out_cols]
    np.maximum(current, group_max, out=current)
    matrix[out_rows, out_cols] = current

    if mode == "fill":
        # Only zero cells may be filled by propagation; originally non-zero
        # cells keep their input value regardless of what descendants say.
        matrix[nz_rows, nz_cols] = nz_scores


def propagate(matrix, ont, order, mode="max", parallel=0, chunk_rows=65536,
              _shm_name=None, _shape=None, _dtype_str=None,
              _row_start=None, _row_end=None, _deepest=None, _triples=None):
    """
    Update inplace the score matrix (proteins x terms) propagating scores up to
    the root. ``mode='max'`` takes the max of each term and its children;
    ``mode='fill'`` only updates rows where the current term is zero.

    When ``parallel > 1`` and the estimated work is above the threshold the
    matrix is shared across processes via ``shared_memory`` (spawn context)
    and rows are partitioned among workers. Recursive calls re-enter this
    function through the ``_shm_name`` path.
    """
    import multiprocessing as mp
    from multiprocessing import shared_memory

    if _shm_name is None:
        if matrix is None:
            raise TypeError("matrix must not be None")
        if matrix.shape[0] == 0:
            raise Exception("Empty matrix")

        use_sparse = os.environ.get("CAFAEVAL_SPARSE", "1") not in ("0", "false", "False")
        if use_sparse:
            _propagate_logger.debug(
                "propagate sparse pushup",
                extra={"mode": mode, "rows": int(matrix.shape[0])},
            )
            _propagate_sparse_pushup(matrix, ont, mode, triples=_triples)
            return

        # Dense fallback: compute ``deepest`` so the serial sweep can skip
        # the all-zero columns at the front of ``order``. The sparse path
        # does not need this because ``_ancestors_csr`` is order-independent.
        has_any = np.any(matrix[:, order] != 0, axis=0)
        nonzero_idx = np.flatnonzero(has_any)
        if nonzero_idx.size == 0:
            raise Exception("The matrix is empty")
        deepest = int(nonzero_idx[0])
        order_ = order[deepest:]

        # Dense fallback path still needs the per-term children list.
        children_by_term = _children_cache(ont)

        n_proc = int(parallel) if parallel else 0
        if n_proc > 1:
            sum_children = int(sum(children_by_term[t].size for t in order_))
            work = int(matrix.shape[0]) * sum_children
            if work < _PROPAGATE_WORK_THRESHOLD:
                _propagate_logger.info(
                    "propagate serial (below threshold)",
                    extra={"work": work, "threshold": _PROPAGATE_WORK_THRESHOLD,
                           "n_proc_requested": n_proc, "mode": mode},
                )
                n_proc = 0
            else:
                _propagate_logger.info(
                    "propagate parallel",
                    extra={"work": work, "n_proc": n_proc, "mode": mode,
                           "rows": int(matrix.shape[0])},
                )

        if n_proc <= 1:
            _propagate_logger.debug(
                "propagate serial",
                extra={"mode": mode, "rows": int(matrix.shape[0]),
                       "terms": int(len(order_))},
            )
            _propagate_serial(matrix, order_, children_by_term, mode)
            return

        shm = shared_memory.SharedMemory(create=True, size=matrix.nbytes)
        shm_matrix = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm.buf)
        shm_matrix[:] = matrix
        try:
            n_rows = int(matrix.shape[0])
            chunk_rows = int(np.ceil(n_rows / n_proc))
            _propagate_logger.debug(
                "propagate chunking",
                extra={"n_rows": n_rows, "chunk_rows": chunk_rows, "n_proc": n_proc},
            )
            ctx = mp.get_context("spawn")
            procs = []
            for row_start in range(0, n_rows, chunk_rows):
                row_end = min(n_rows, row_start + chunk_rows)
                proc = ctx.Process(
                    target=propagate,
                    args=(None, ont, order),
                    kwargs={
                        "mode": mode,
                        "parallel": 0,
                        "chunk_rows": chunk_rows,
                        "_shm_name": shm.name,
                        "_shape": matrix.shape,
                        "_dtype_str": matrix.dtype.str,
                        "_row_start": row_start,
                        "_row_end": row_end,
                        "_deepest": deepest,
                    },
                )
                proc.start()
                procs.append(proc)
            for proc in procs:
                proc.join()
                if proc.exitcode != 0:
                    raise RuntimeError("Worker failed")
            matrix[:] = shm_matrix
        finally:
            shm.close()
            shm.unlink()
        return

    shm = shared_memory.SharedMemory(name=_shm_name)
    try:
        full = np.ndarray(tuple(_shape), dtype=np.dtype(_dtype_str), buffer=shm.buf)
        row_start = int(_row_start)
        row_end = int(_row_end)
        view = full[row_start:row_end]
        if view.shape[0] == 0:
            return

        deepest = int(_deepest) if _deepest is not None else 0
        order_ = order[deepest:]
        children_by_term = _children_cache(ont)
        _propagate_serial(view, order_, children_by_term, mode)
    finally:
        shm.close()
    return