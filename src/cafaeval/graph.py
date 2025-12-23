import numpy as np
import copy
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())


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
        self.dag = []  # [[], ...] terms (rows, axis 0) x parents (columns, axis 1)
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

        self.dag = np.zeros((self.idxs, self.idxs), dtype='bool')

        # id1 term (row, axis 0), id2 parent (column, axis 1)
        for id1, id2, ns in rel_list:
            if self.terms_dict.get(id2):
                i = self.terms_dict[id1]['index']
                j = self.terms_dict[id2]['index']
                self.dag[i, j] = 1
                # Remove duplicates in adj and children lists
                # This ensures that a parent-child term does not have multiple edges, which could lead to wrong topological sorting
                self.terms_list[i]['adj'].add(j)
                self.terms_list[j]['children'].add(i)
                logging.debug("i,j {},{} {},{}".format(i, j, id1, id2))
            else:
                logging.debug('Skipping branch to external namespace: {}'.format(id2))
        logging.debug("dag {}".format(self.dag))
        
        # Topological sorting
        self.top_sort()
        logging.debug("order sorted {}".format(self.order))

        if orphans:
            self.toi = np.arange(self.dag.shape[0])  # All terms, also those without parents
        else:
            self.toi = np.nonzero(self.dag.sum(axis=1) > 0)[0]  # Only terms with parents
        logging.debug("toi {}".format(self.toi))

        if ia_dict is not None:
            self.set_ia(ia_dict)

        logging.info("Ontology: {}, total {}, roots {}, leaves {}, alternative_ids {}".format(self.namespace,
                                                                len(np.where(self.dag.sum(axis=1) != 0)[0]),
                                                                len(np.where(self.dag.sum(axis=1) == 0)[0]),
                                                                len(np.where(self.dag.sum(axis=0) == 0)[0]),
                                                                len(self.terms_dict_alt)))

        return

    def top_sort(self):
        """
        Takes a sparse matrix representing a DAG and returns an array with nodes indexes in topological order
        https://en.wikipedia.org/wiki/Topological_sorting
        """
        indexes = []
        visited = 0
        (rows, cols) = self.dag.shape

        # create a vector containing the in-degree of each node
        in_degree = self.dag.sum(axis=0)
        # logging.debug("degree {}".format(in_degree))

        # find the nodes with in-degree 0 (leaves) and add them to the queue
        queue = np.nonzero(in_degree == 0)[0].tolist()
        # logging.debug("queue {}".format(queue))

        # for each element of the queue increment visits, add them to the list of ordered nodes
        # and decrease the in-degree of the neighbor nodes
        # and add them to the queue if they reach in-degree == 0
        while queue:
            visited += 1
            idx = queue.pop(0)
            indexes.append(idx)
            in_degree[idx] -= 1
            l = self.terms_list[idx]['adj']
            if len(l) > 0:
                for j in l:
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


def propagate(matrix, ont, order, mode='max'):
    """
    Update inplace the score matrix (proteins x terms) up to the root taking the max between children and parents
    """
    if matrix.shape[0] == 0:
        raise Exception("Empty matrix")
        
    children_by_term = getattr(ont, "_children_by_term", None)
    if children_by_term is None:
        children_by_term = [
            np.flatnonzero(ont.dag[:, i]) for i in range(ont.dag.shape[1])
        ]
        ont._children_by_term = children_by_term

    has_any = np.any(matrix[:, order] != 0, axis=0)
    idx = np.flatnonzero(has_any)
    if idx.size == 0:
        raise Exception("The matrix is empty")
    deepest = int(idx[0])

    # Remove leaves
    order_ = order[deepest:]

    for i in order_:
        # Get direct children
        children = children_by_term[i]
        if children.size > 0:
            # Add current terms to children
            if mode == 'max':
                child_max = matrix[:, children].max(axis=1)
                matrix[:, i] = np.maximum(matrix[:, i], child_max)
            elif mode == 'fill':
                # Select only rows where the current term is 0
                rows = np.flatnonzero(matrix[:, i] == 0)
                if rows.size:
                    idx = np.ix_(rows, children)
                    matrix[rows, i] = matrix[idx].max(axis=1)
    return




def propagate(matrix, ont, order, mode="max", parallel=0, chunk_rows=65536,
              _shm_name=None, _shape=None, _dtype_str=None,
              _row_start=None, _row_end=None, _deepest=None):
    import multiprocessing as mp
    from multiprocessing import shared_memory

    work_threshold = 800_000_000

    children_by_term = getattr(ont, "_children_by_term", None)
    if children_by_term is None:
        children_by_term = [
            np.flatnonzero(ont.dag[:, term_id])
            for term_id in range(ont.dag.shape[1])
        ]
        ont._children_by_term = children_by_term

    if _shm_name is None:
        if matrix is None:
            raise TypeError("matrix must not be None")
        if matrix.shape[0] == 0:
            raise Exception("Empty matrix")

        has_any = np.any(matrix[:, order] != 0, axis=0)
        nonzero_idx = np.flatnonzero(has_any)
        if nonzero_idx.size == 0:
            raise Exception("The matrix is empty")
        deepest = int(nonzero_idx[0])
        order_ = order[deepest:]

        n_proc = int(parallel) if parallel else 0
        if n_proc > 1:
            sum_children = int(sum(children_by_term[t].size for t in order_))
            work = int(matrix.shape[0]) * sum_children
            print(f"[PROPAGATE] work: {work}")
            if work < work_threshold:
                n_proc = 0

        if n_proc <= 1:
            print(f"[PROPAGATE] single process")
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
            return

        shm = shared_memory.SharedMemory(create=True, size=matrix.nbytes)
        shm_matrix = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm.buf)
        shm_matrix[:] = matrix
        print(f"[PROPAGATE] multiprocess")
        try:
            n_rows = int(matrix.shape[0])
            chunk_rows = int(np.ceil(n_rows / n_proc))

            print(f"[PROPAGATE] chunk_rows: {chunk_rows}")
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

        for term_id in order_:
            children = children_by_term[term_id]
            if children.size == 0:
                continue
            if mode == "max":
                child_max = view[:, children].max(axis=1)
                view[:, term_id] = np.maximum(view[:, term_id], child_max)
            elif mode == "fill":
                rows = np.flatnonzero(view[:, term_id] == 0)
                if rows.size:
                    idx = np.ix_(rows, children)
                    view[rows, term_id] = view[idx].max(axis=1)
    finally:
        shm.close()
    return