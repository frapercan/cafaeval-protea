import os

import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from cafaeval.parser import obo_parser, gt_parser, pred_parser, gt_exclude_parser, update_toi
from cafaeval.tests import test_norm_metric, test_intersection
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
_metrics_logger = logging.getLogger("cafaeval.metrics")
_metrics_logger.addHandler(logging.NullHandler())
_eval_logger = logging.getLogger("cafaeval.eval")
_eval_logger.addHandler(logging.NullHandler())

# Return a mask for all the predictions (matrix) >= tau
def solidify_prediction(pred, tau):
    return pred >= tau


# computes the f metric for each precision and recall in the input arrays
def compute_f(pr, rc):
    n = 2 * pr * rc
    d = pr + rc
    return np.divide(n, d, out=np.zeros_like(n, dtype=float), where=d != 0)


def compute_s(ru, mi):
    return np.sqrt(ru**2 + mi**2)
    # return np.where(np.isnan(ru), mi, np.sqrt(ru + np.nan_to_num(mi)))


def compute_confusion_matrix(tau_arr, g, pred_matrix, toi, n_gt, ic_arr=None):
    """
    Perform the evaluation at the matrix level for all tau thresholds
    The calculation is
    """
    # n, tp, fp, fn, pr, rc (fp = misinformation, fn = remaining uncertainty)
    metrics = np.zeros((len(tau_arr), 6), dtype='float')

    for i, tau in enumerate(tau_arr):

        # Filter predictions based on tau threshold
        p = solidify_prediction(pred_matrix, tau)

        # Terms subsets
        intersection = np.logical_and(p, g)  # TP
        mis = np.logical_and(p, np.logical_not(g))  # FP, predicted but not in the ground truth
        remaining = np.logical_and(np.logical_not(p), g)  # FN, not predicted but in the ground truth

        # Weighted evaluation
        if ic_arr is not None:
            p = p * ic_arr[toi]
            intersection = intersection * ic_arr[toi]  # TP
            mis = mis * ic_arr[toi]  # FP, predicted but not in the ground truth
            remaining = remaining * ic_arr[toi]  # FN, not predicted but in the ground truth

        n_pred = p.sum(axis=1)  # TP + FP (number of terms predicted in each protein)
        n_intersection = intersection.sum(axis=1)  # TP (number of TP terms per protein)
        # Number of proteins with at least one term predicted with score >= tau
        metrics[i, 0] = (p.sum(axis=1) > 0).sum()

        # Sum of confusion matrices
        metrics[i, 1] = n_intersection.sum()  # TP (total terms)
        metrics[i, 2] = mis.sum(axis=1).sum()  # FP
        metrics[i, 3] = remaining.sum(axis=1).sum()  # FN

        # Macro-averaging
        metrics[i, 4] = np.divide(n_intersection, n_pred, out=np.zeros_like(n_intersection, dtype='float'), where=n_pred > 0).sum()  # Precision
        metrics[i, 5] = np.divide(n_intersection, n_gt, out=np.zeros_like(n_gt, dtype='float'), where=n_gt > 0).sum()  # Recall

    return metrics


def compute_confusion_matrix_exclude(tau_arr, g_perprotein, pred_matrix, toi_perprotein, n_gt, ic_arr=None):
    """
    Perform the evaluation at the matrix level for all tau thresholds
    The calculation is

    Here, g is the full ground truth matrix without filtering terms of interest (toi).
    Instead,
    """
    # n, tp, fp, fn, pr, rc (fp = misinformation, fn = remaining uncertainty)
    metrics = np.zeros((len(tau_arr), 6), dtype='float')

    for i, tau in enumerate(tau_arr):

        # Filter predictions based on tau threshold
        p_perprotein = [solidify_prediction(pred_matrix[p_idx, tois], tau) for p_idx, tois in enumerate(toi_perprotein)]

        # Terms subsets
        intersection = [np.logical_and(p_i, g_i) for p_i, g_i in zip(p_perprotein, g_perprotein)]  # TP
        mis = [np.logical_and(p_i, np.logical_not(g_i)) for p_i, g_i in zip(p_perprotein, g_perprotein)]  # FP, predicted but not in the ground truth
        remaining = [np.logical_and(np.logical_not(p_i), g_i) for p_i, g_i in zip(p_perprotein, g_perprotein)]  # FN, not predicted but in the ground truth

        # Weighted evaluation
        if ic_arr is not None:
            p_perprotein = [p_i * ic_arr[tois] for p_i, tois in zip(p_perprotein, toi_perprotein)]
            intersection = [inter * ic_arr[tois] for inter, tois in zip(intersection, toi_perprotein)]  # TP
            mis = [misinf * ic_arr[tois] for misinf, tois in zip(mis, toi_perprotein)]  # FP, predicted but not in the ground truth
            remaining = [rem * ic_arr[tois] for rem, tois in zip(remaining, toi_perprotein)]  # FN, not predicted but in the ground truth

        n_pred = np.array([p_i.sum() for p_i in p_perprotein])  # TP + FP
        n_intersection = np.array([inter.sum() for inter in intersection])  # TP
        precision = np.divide(n_intersection, n_pred, out=np.zeros_like(n_intersection, dtype='float'), where=n_pred > 0)
        recall = np.divide(n_intersection, n_gt, out=np.zeros_like(n_gt, dtype='float'), where=n_gt > 0)

        # metrics tests
        test_norm_metric(precision, name='precision')
        test_norm_metric(recall, name='recall')
        test_intersection(n_intersection, n_pred, n_gt)


        # Number of proteins with at least one term predicted with score >= tau
        metrics[i, 0] = (n_pred > 0).sum()

        # Sum of confusion matrices
        metrics[i, 1] = n_intersection.sum()  # TP
        metrics[i, 2] = np.sum([m.sum() for m in mis])  # FP
        metrics[i, 3] = np.sum([r.sum() for r in remaining])  # FN

        # Macro-averaging
        metrics[i, 4] = precision.sum()  # Precision
        metrics[i, 5] = recall.sum()  # Recall

    return metrics

_CM_G = None
_CM_P = None
_CM_TOI = None
_CM_N_GT = None
_CM_IC = None


def _cm_init(g, p, toi, n_gt, ic_arr):
    global _CM_G, _CM_P, _CM_TOI, _CM_N_GT, _CM_IC
    _CM_G = g
    _CM_P = p
    _CM_TOI = toi
    _CM_N_GT = n_gt
    _CM_IC = ic_arr


def _cm_worker(tau_chunk):
    return compute_confusion_matrix(
        tau_chunk, _CM_G, _CM_P, _CM_TOI, _CM_N_GT, _CM_IC
    )


_CME_G = None
_CME_P = None
_CME_TOI = None
_CME_N_GT = None
_CME_IC = None


def _cme_init(g_perprotein, p, toi_perprotein, n_gt, ic_arr):
    global _CME_G, _CME_P, _CME_TOI, _CME_N_GT, _CME_IC
    _CME_G = g_perprotein
    _CME_P = p
    _CME_TOI = toi_perprotein
    _CME_N_GT = n_gt
    _CME_IC = ic_arr


def _cme_worker(tau_chunk):
    return compute_confusion_matrix_exclude(
        tau_chunk, _CME_G, _CME_P, _CME_TOI, _CME_N_GT, _CME_IC
    )

def compute_metrics(pred, gt_matrix, tau_arr, toi, gt_exclude=None, ic_arr=None, n_cpu=0):
    """
    Takes the prediction and the ground truth and for each threshold in tau_arr
    calculates the confusion matrix and returns the coverage,
    precision, recall, remaining uncertainty and misinformation.
    Toi is the list of terms (indexes) to be considered
    """
    t0 = time.time()
    # Parallelization
    if n_cpu == 0:
        n_cpu = mp.cpu_count()

    mode = "pk" if gt_exclude is not None else "nk_lk"
    _metrics_logger.debug(
        "compute_metrics start",
        extra={"n_cpu": n_cpu, "n_tau": int(len(tau_arr)), "mode": mode},
    )

    columns = ["n", "tp", "fp", "fn", "pr", "rc"]
    # filter out proteins with no annotations in Terms-Of-Interest (toi)
    proteins_has_gt = gt_matrix[:, toi].sum(1) > 0
    proteins_with_gt = np.where(proteins_has_gt)[0]
    gt_with_annots = gt_matrix[proteins_with_gt, :]
    g = gt_with_annots[:, toi]
    p = pred[proteins_has_gt, :][:, toi]

    if gt_exclude is not None:
        toi_perprotein = [np.setdiff1d(toi, gt_exclude.matrix[prot, :].nonzero()[0],
                                       assume_unique=True) for prot in
                          proteins_with_gt]
        gt_perprotein = [gt_with_annots[p_idx, tois] for p_idx, tois in enumerate(toi_perprotein)]
        # The number of GT annotations per proteins will change to exclude the set from g_exclude
        n_gt = np.array([gpp.sum().item() for gpp in gt_perprotein])  # number of terms annotated in each protein
        n_empty = int(np.count_nonzero(n_gt == 0))
        if n_empty:
            _metrics_logger.warning(
                "proteins with no annotations in TOI",
                extra={"count": n_empty},
            )
        if ic_arr is not None:
            n_gt = np.array([(gpp * ic_arr[tois]).sum().item() for gpp, tois in zip(gt_perprotein, toi_perprotein)])
    else:
        count_g = g
        # Simple metrics: number of terms annotated in each protein
        if ic_arr is None:
            n_gt = count_g.sum(axis=1)
        # Weighted metrics
        else:
            n_gt = (count_g * ic_arr[toi]).sum(axis=1)

    t1 = time.time()
    tau_chunks = np.array_split(tau_arr, n_cpu)
    if gt_exclude is None:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_cpu, initializer=_cm_init,
                      initargs=(g, p, toi, n_gt, ic_arr)) as pool:
            metrics = np.concatenate(pool.map(_cm_worker, tau_chunks), axis=0)
    else:
        pred_sub = pred[gt_matrix[:, toi].sum(1) > 0, :]
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_cpu, initializer=_cme_init,
                      initargs=(gt_perprotein, pred_sub, toi_perprotein, n_gt, ic_arr)) as pool:
            metrics = np.concatenate(pool.map(_cme_worker, tau_chunks), axis=0)

    t2 = time.time()
    _metrics_logger.info(
        "compute_metrics done",
        extra={"mode": mode, "n_cpu": n_cpu, "n_tau": int(len(tau_arr)),
               "prep_seconds": round(t1 - t0, 3),
               "pool_seconds": round(t2 - t1, 3),
               "total_seconds": round(t2 - t0, 3)},
    )
    return pd.DataFrame(metrics, columns=columns)


def normalize(metrics, ns, tau_arr, ne, normalization):

    # Normalize columns
    for column in metrics.columns:
        if column != "n":
            # By default normalize by gt
            denominator = ne
            # Otherwise normalize by pred
            if normalization == 'pred' or (normalization == 'cafa' and column == "pr"):
                denominator = metrics["n"]
            metrics[column] = np.divide(metrics[column], denominator,
                                        out=np.zeros_like(metrics[column], dtype='float'),
                                        where=denominator > 0)

    metrics['ns'] = [ns] * len(tau_arr)
    metrics['tau'] = tau_arr
    metrics['cov'] = metrics['n'] / ne
    metrics['mi'] = metrics['fp']
    metrics['ru'] = metrics['fn']

    metrics['f'] = compute_f(metrics['pr'], metrics['rc'])
    metrics['s'] = compute_s(metrics['ru'], metrics['mi'])

    # Micro-average, calculation is based on the average of the confusion matrices
    metrics['pr_micro'] = np.divide(metrics['tp'], metrics['tp'] + metrics['fp'],
                                    out=np.zeros_like(metrics['tp'], dtype='float'),
                                    where=(metrics['tp'] + metrics['fp']) > 0)
    metrics['rc_micro'] = np.divide(metrics['tp'], metrics['tp'] + metrics['fn'],
                                    out=np.zeros_like(metrics['tp'], dtype='float'),
                                    where=(metrics['tp'] + metrics['fn']) > 0)
    metrics['f_micro'] = compute_f(metrics['pr_micro'], metrics['rc_micro'])

    return metrics

def evaluate_prediction(prediction, gt, ontologies, tau_arr, gt_exclude=None, normalization='cafa', n_cpu=0, weighted_only=False):
    t0 = time.time()
    _eval_logger.debug(
        "evaluate_prediction start",
        extra={"n_namespaces": len(prediction), "weighted_only": weighted_only,
               "normalization": normalization, "n_cpu": n_cpu},
    )

    dfs = []
    dfs_w = []

    # Unweighted metrics
    for ns in prediction:
        t_ns = time.time()
        # number of proteins with positive annotations
        proteins_has_gt = gt[ns].matrix[:, ontologies[ns].toi].sum(1) > 0
        proteins_with_gt = np.where(proteins_has_gt)[0]
        num_annot_prots = proteins_has_gt.sum()  # number of proteins with positive annotations in TOIs
        if gt_exclude is None:
            exclude = None
        else:
            exclude = gt_exclude[ns]
            toi_perprotein = [
                np.setdiff1d(ontologies[ns].toi, gt_exclude[ns].matrix[p, :].nonzero()[0],
                             assume_unique=True) for p in proteins_with_gt]
            # update the number of proteins with positive annotations, now on protein-specific TOIs
            num_annot_prots = sum([gt[ns].matrix[p, toi_perprotein[p_idx]].sum()>0 for
                                   p_idx, p in enumerate(proteins_with_gt)])

        ne = np.full(len(tau_arr), num_annot_prots)

        if not weighted_only:
            t_metrics = time.time()
            metrics_df = compute_metrics(
                prediction[ns].matrix, gt[ns].matrix, tau_arr, ontologies[ns].toi, exclude, None, n_cpu)
            t_norm = time.time()
            dfs.append(normalize(metrics_df, ns, tau_arr, ne, normalization))
            _eval_logger.debug(
                "evaluate_prediction namespace unweighted",
                extra={"ns": ns,
                       "metrics_seconds": round(t_norm - t_metrics, 3),
                       "normalize_seconds": round(time.time() - t_norm, 3),
                       "total_seconds": round(time.time() - t_ns, 3)},
            )

        # Weighted metrics
        if ontologies[ns].ia is not None:
            t_w = time.time()
            # number of proteins with positive annotations
            proteins_has_gt = gt[ns].matrix[:, ontologies[ns].toi_ia].sum(1) > 0
            num_annot_prots = (proteins_has_gt).sum()

            if gt_exclude is None:
                exclude = None
            else:
                exclude = gt_exclude[ns]
                toi_perprotein_ia = [
                    np.setdiff1d(ontologies[ns].toi_ia, gt_exclude[ns].matrix[p, :].nonzero()[0],
                                 assume_unique=True) for p in proteins_with_gt]
                # update the number of proteins with positive annotations, now on protein-specific TOIs
                num_annot_prots = sum([gt[ns].matrix[p, toi_perprotein_ia[p_idx]].sum() > 0 for
                                       p_idx, p in enumerate(proteins_with_gt)])

            ne = np.full(len(tau_arr), num_annot_prots)

            t_metrics_w = time.time()
            metrics_df_w = compute_metrics(
                prediction[ns].matrix, gt[ns].matrix, tau_arr, ontologies[ns].toi_ia, exclude, ontologies[ns].ia, n_cpu)
            t_norm_w = time.time()
            dfs_w.append(normalize(metrics_df_w, ns, tau_arr, ne, normalization))
            _eval_logger.debug(
                "evaluate_prediction namespace weighted",
                extra={"ns": ns,
                       "metrics_seconds": round(t_norm_w - t_metrics_w, 3),
                       "normalize_seconds": round(time.time() - t_norm_w, 3),
                       "total_seconds": round(time.time() - t_w, 3)},
            )

    t_merge = time.time()

    if weighted_only:
        dfs_w = pd.concat(dfs_w)
        base_cols = ("ns", "tau")
        metric_cols = [c for c in dfs_w.columns if c not in base_cols]
        for c in metric_cols:
            dfs_w[f"{c}_w"] = dfs_w[c]
        _eval_logger.info(
            "evaluate_prediction done (weighted_only)",
            extra={"total_seconds": round(time.time() - t0, 3)},
        )
        return dfs_w

    dfs = pd.concat(dfs)

    # Merge weighted and unweighted dataframes
    if dfs_w:
        dfs_w = pd.concat(dfs_w)
        dfs = pd.merge(dfs, dfs_w, on=['ns', 'tau'], suffixes=('', '_w'))

    _eval_logger.info(
        "evaluate_prediction done",
        extra={"total_seconds": round(time.time() - t0, 3),
               "merge_seconds": round(time.time() - t_merge, 3)},
    )
    return dfs


def cafa_eval(obo_file, pred_dir, gt_file, ia=None, no_orphans=False, norm='cafa', prop='max',
              exclude=None, toi_file=None, max_terms=None, th_step=0.01, n_cpu=1, weighted_only=False):
    t_total = time.time()
    _eval_logger.info(
        "cafa_eval start",
        extra={"norm": norm, "prop": prop, "th_step": th_step, "n_cpu": n_cpu,
               "weighted_only": weighted_only, "obo_file": obo_file,
               "pred_dir": pred_dir, "gt_file": gt_file, "exclude": exclude,
               "toi_file": toi_file, "max_terms": max_terms, "ia": ia},
    )

    # Tau array, used to compute metrics at different score thresholds
    tau_arr = np.arange(th_step, 1, th_step)

    # Parse the OBO file and creates a different graphs for each namespace
    t_obo = time.time()
    ontologies = obo_parser(obo_file, ("is_a", "part_of"), ia, not no_orphans)
    if toi_file is not None:
        ontologies = update_toi(ontologies, toi_file)
    _eval_logger.info(
        "obo parsed",
        extra={"seconds": round(time.time() - t_obo, 3),
               "namespaces": list(ontologies.keys()),
               "n_tau": int(len(tau_arr))},
    )

    # Parse ground truth file
    t_gt = time.time()
    gt = gt_parser(gt_file, ontologies)
    if exclude is not None:
        gt_exclude = gt_exclude_parser(exclude, gt, ontologies)
    else:
        gt_exclude = None
    _eval_logger.info(
        "ground truth parsed",
        extra={"seconds": round(time.time() - t_gt, 3),
               "has_exclude": gt_exclude is not None},
    )

    # Set prediction files looking recursively in the prediction folder
    pred_folder = os.path.normpath(pred_dir) + "/"  # add the tailing "/"
    pred_files = []
    for root, dirs, files in os.walk(pred_folder):
        for file in files:
            pred_files.append(os.path.join(root, file))
    logger.debug("Prediction paths {}".format(pred_files))
    _eval_logger.info(
        "prediction files discovered",
        extra={"count": len(pred_files), "pred_dir": pred_dir},
    )

    # Parse prediction files and perform evaluation
    dfs = []
    for idx, file_name in enumerate(pred_files, 1):
        t_file = time.time()
        t_parse = time.time()
        prediction = pred_parser(file_name, ontologies, gt, prop, max_terms, n_cpu)
        if not prediction:
            logger.warning("Prediction: {}, not evaluated".format(file_name))
            _eval_logger.warning(
                "prediction skipped",
                extra={"file": file_name, "index": idx, "total": len(pred_files)},
            )
            continue

        _eval_logger.info(
            "prediction parsed",
            extra={"file": file_name, "index": idx, "total": len(pred_files),
                   "parse_seconds": round(time.time() - t_parse, 3),
                   "namespaces": list(prediction.keys())},
        )
        t_eval = time.time()
        df_pred = evaluate_prediction(prediction, gt, ontologies, tau_arr, gt_exclude,
                                      normalization=norm, n_cpu=n_cpu, weighted_only=weighted_only)
        df_pred['filename'] = file_name.replace(pred_folder, '').replace('/', '_')
        dfs.append(df_pred)
        _eval_logger.info(
            "prediction evaluated",
            extra={"file": file_name, "index": idx, "total": len(pred_files),
                   "eval_seconds": round(time.time() - t_eval, 3),
                   "file_seconds": round(time.time() - t_file, 3)},
        )
        logger.info("Prediction: {}, evaluated".format(file_name))

    # Concatenate all dataframes and save them
    t_final = time.time()
    df = None
    dfs_best = {}
    if dfs:
        df = pd.concat(dfs)

        # Remove rows with no coverage
        df = df[df['cov'] > 0].reset_index(drop=True)
        df.set_index(['filename', 'ns', 'tau'], inplace=True)

        # Calculate the best index for each namespace and each evaluation metric
        for metric, cols in [('f', ['rc', 'pr']), ('f_w', ['rc_w', 'pr_w']), ('s', ['ru', 'mi']), ('f_micro', ['rc_micro', 'pr_micro']), ('f_micro_w', ['rc_micro_w', 'pr_micro_w'])]:
            if metric in df.columns:
                index_best = df.groupby(level=['filename', 'ns'])[metric].idxmax() if metric in ['f', 'f_w', 'f_micro', 'f_micro_w'] else df.groupby(['filename', 'ns'])[metric].idxmin()
                df_best = df.loc[index_best]
                if metric[-2:] != '_w':
                    df_best['cov_max'] = df.reset_index('tau').loc[[ele[:-1] for ele in index_best]].groupby(level=['filename', 'ns'])['cov'].max()
                else:
                    df_best['cov_max'] = df.reset_index('tau').loc[[ele[:-1] for ele in index_best]].groupby(level=['filename', 'ns'])['cov_w'].max()
                dfs_best[metric] = df_best
        _eval_logger.info(
            "cafa_eval aggregation done",
            extra={"seconds": round(time.time() - t_final, 3),
                   "n_results": len(dfs),
                   "best_metrics": list(dfs_best.keys())},
        )
    else:
        logger.info("No predictions evaluated")
        _eval_logger.warning("cafa_eval no predictions evaluated")

    _eval_logger.info(
        "cafa_eval done",
        extra={"total_seconds": round(time.time() - t_total, 3)},
    )
    return df, dfs_best


def write_results(df, dfs_best, out_dir='results', th_step=0.01):

    # Create output folder here in order to store the log file
    out_folder = os.path.normpath(out_dir) + "/"
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # Set the number of decimals to write in the output files based on the threshold step size
    decimals = int(np.ceil(-np.log10(th_step))) + 1

    df.to_csv('{}/evaluation_all.tsv'.format(out_folder), float_format="%.{}f".format(decimals), sep="\t")

    for metric in dfs_best:
        dfs_best[metric].to_csv('{}/evaluation_best_{}.tsv'.format(out_folder, metric), float_format="%.{}f".format(decimals), sep="\t")
