# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Argoverse 3D object detection evaluation.

Evaluation:

    Precision/Recall

        1. Average Precision: Standard VOC-style average precision calculation
            except a true positive requires a 3D Euclidean center distance of less
            than a predefined threshold.

    True Positive Errors

        All true positive errors accumulate error solely when an object is a true positive match
        to a ground truth detection. The matching criterion is represented by `tp_thresh` in the DetectionCfg class.
        In our challenge, we use a `tp_thresh` of 2.0 meters.

        1. Average Translation Error: The average Euclidean distance (center-based) between a
            detection and its ground truth assignment.
        2. Average Scale Error: The average intersection over union (IoU) after the prediction
            and assigned ground truth's pose has been aligned.
        3. Average Orientation Error: The average angular distance between the detection and
            the assigned ground truth. We choose the smallest angle between the two different
            headings when calculating the error.

    Composite Scores

        1. Composite Detection Score: The ranking metric for the detection leaderboard. This
            is computed as the product of mAP with the sum of the complements of the true positive
            errors (after normalization), i.e.:
                - Average Translation Measure (ATM): ATE / TP_THRESHOLD; 0 <= 1 - ATE / TP_THRESHOLD <= 1.
                - Average Scaling Measure (ASM): 1 - ASE / 1;  0 <= 1 - ASE / 1 <= 1.
                - Average Orientation Measure (AOM): 1 - AOE / PI; 0 <= 1 - AOE / PI <= 1.

            These (as well as AP) are averaged over each detection class to produce:
                - mAP
                - mATM
                - mASM
                - mAOM

            Lastly, the Composite Detection Score is computed as:
                CDS = mAP * (mATE + mASE + mAOE); 0 <= mAP * (mATE + mASE + mAOE) <= 1.

        ** In the case of no true positives under the specified threshold, the true positive measures
            will assume their upper bounds of 1.0. respectively.

Results:

    The results are represented as a (C + 1, P) table, where C + 1 represents the number of evaluation classes
    in addition to the mean statistics average across all classes, and P refers to the number of included statistics,
    e.g. AP, ATE, ASE, AOE, CDS by default.
"""
import logging
from multiprocessing import get_context
from typing import Dict, Final, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from av2.evaluation.detection.constants import NUM_DECIMALS, MetricNames, TruePositiveErrorNames
from av2.evaluation.detection.utils import (
    DetectionCfg,
    accumulate,
    compute_average_precision,
    load_mapped_avm_and_egoposes,
)
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import ORDERED_CUBOID_COL_NAMES
from av2.utils.io import TimestampedCitySE3EgoPoses
from av2.utils.typing import NDArrayBool, NDArrayFloat

TP_ERROR_COLUMNS: Final[Tuple[str, ...]] = tuple(x.value for x in TruePositiveErrorNames)
DTS_COLUMN_NAMES: Final[Tuple[str, ...]] = tuple(ORDERED_CUBOID_COL_NAMES) + ("score",)
GTS_COLUMN_NAMES: Final[Tuple[str, ...]] = tuple(ORDERED_CUBOID_COL_NAMES) + ("num_interior_pts",)
UUID_COLUMN_NAMES: Final[Tuple[str, ...]] = (
    "log_id",
    "timestamp_ns",
    "category",
)

logger = logging.getLogger(__name__)


JobType = List[Tuple[NDArrayFloat, NDArrayFloat, DetectionCfg, Optional[ArgoverseStaticMap], Optional[SE3]]]


def evaluate(
    dts: pd.DataFrame,
    gts: pd.DataFrame,
    cfg: DetectionCfg,
    n_jobs: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a set of detections against the ground truth annotations.

    Each sweep is processed independently, computing assignment between detections and ground truth annotations.

    Args:
        dts: (N,14) Table of detections.
        gts: (M,15) Table of ground truth annotations.
        cfg: Detection configuration.
        n_jobs: Number of jobs running concurrently during evaluation.

    Returns:
        (C+1,K) Table of evaluation metrics where C is the number of classes. Plus a row for their means.
        K refers to the number of evaluation metrics.

    Raises:
        RuntimeError: If accumulation fails.
        ValueError: If ROI pruning is enabled but a dataset directory is not specified.
    """
    if cfg.eval_only_roi_instances and cfg.dataset_dir is None:
        raise ValueError(
            "ROI pruning has been enabled, but the dataset directory has not be specified. "
            "Please set `dataset_directory` to the split root, e.g. av2/sensor/val."
        )

    dts = pl.from_pandas(dts)
    gts = pl.from_pandas(gts)

    expr = pl.col(["category"]).is_in(list(cfg.categories))
    dts = dts.filter(expr).sort(list(UUID_COLUMN_NAMES))
    gts = gts.filter(expr).sort(list(UUID_COLUMN_NAMES))

    uuid_to_dts: Dict[Tuple[str, str, str], pl.DataFrame] = dts.partition_by(
        list(UUID_COLUMN_NAMES), as_dict=True, maintain_order=True
    )
    uuid_to_gts: Dict[Tuple[str, str, str], pl.DataFrame] = gts.partition_by(
        list(UUID_COLUMN_NAMES), as_dict=True, maintain_order=True
    )

    log_id_to_avm: Optional[Dict[str, ArgoverseStaticMap]] = None
    log_id_to_timestamped_poses: Optional[Dict[str, TimestampedCitySE3EgoPoses]] = None

    # Load maps and egoposes if roi-pruning is enabled.
    if cfg.eval_only_roi_instances and cfg.dataset_dir is not None:
        logger.info("Loading maps and egoposes ...")
        log_ids: List[str] = gts["log_id"].unique().to_list()
        log_id_to_avm, log_id_to_timestamped_poses = load_mapped_avm_and_egoposes(log_ids, cfg.dataset_dir)

    args_list: JobType = []
    uuids = sorted(uuid_to_dts.keys() | uuid_to_gts.keys())
    for uuid in uuids:
        log_id, timestamp_ns, _ = uuid
        args: JobType

        sweep_dts: NDArrayFloat = np.zeros((0, 10))
        sweep_gts: NDArrayFloat = np.zeros((0, 10))
        if uuid in uuid_to_dts:
            sweep_dts = uuid_to_dts[uuid][list(DTS_COLUMN_NAMES)].to_numpy()
        if uuid in uuid_to_gts:
            sweep_gts = uuid_to_gts[uuid][list(GTS_COLUMN_NAMES)].to_numpy()

        args = sweep_dts, sweep_gts, cfg, None, None
        if log_id_to_avm is not None and log_id_to_timestamped_poses is not None:
            avm = log_id_to_avm[log_id]
            city_SE3_ego = log_id_to_timestamped_poses[log_id][int(timestamp_ns)]
            args = sweep_dts, sweep_gts, cfg, avm, city_SE3_ego
        args_list.append(args)

    logger.info("Starting evaluation ...")
    with get_context("spawn").Pool(processes=n_jobs) as p:
        outputs: Optional[List[Tuple[NDArrayFloat, NDArrayFloat]]] = p.starmap(accumulate, args_list)

    if outputs is None:
        raise RuntimeError("Accumulation has failed! Please check the integrity of your detections and annotations.")
    dts_list, gts_list = zip(*outputs)

    METRIC_COLUMN_NAMES = tuple(map(str, cfg.affinity_thresholds_m)) + TP_ERROR_COLUMNS + ("is_evaluated",)
    dts_metrics: NDArrayFloat = np.concatenate(dts_list)
    gts_metrics: NDArrayFloat = np.concatenate(gts_list)

    dts[list(METRIC_COLUMN_NAMES)] = dts_metrics
    gts[list(METRIC_COLUMN_NAMES)] = gts_metrics

    # Compute summary metrics.
    metrics = summarize_metrics(dts, gts, cfg)
    means = metrics.mean()
    means[0, "category"] = "AVERAGE_METRICS"
    metrics = metrics.vstack(means)
    return dts.to_pandas(), gts.to_pandas(), metrics.to_pandas().round(NUM_DECIMALS).set_index("category")


def summarize_metrics(
    dts: pl.DataFrame,
    gts: pl.DataFrame,
    cfg: DetectionCfg,
) -> pl.DataFrame:
    """Calculate and print the 3D object detection metrics.

    Args:
        dts: (N,14) Table of detections.
        gts: (M,15) Table of ground truth annotations.
        cfg: Detection configuration.

    Returns:
        The summary metrics.
    """
    # Sample recall values in the [0, 1] interval.
    recall_interpolated: NDArrayFloat = np.linspace(0, 1, cfg.num_recall_samples, endpoint=True)

    default_values_list = []
    average_precisions_list = []
    for category in cfg.categories:
        values = {"category": category} | {s.value: cfg.metrics_defaults[i] for i, s in enumerate(tuple(MetricNames))}
        default_values_list.append(values)

        ap_values = {"category": category} | {str(t): 0.0 for t in cfg.affinity_thresholds_m}
        average_precisions_list.append(ap_values)

    # Initialize the summary metrics.
    summary = pl.DataFrame(default_values_list)
    average_precisions = pl.DataFrame(average_precisions_list)
    for i, category in enumerate(cfg.categories):
        is_valid_dts = (pl.col("category") == category) & (pl.col("is_evaluated").cast(bool))

        # Get valid detections and sort them in descending order.
        category_dts = dts.filter(is_valid_dts).sort(by="score", reverse=True)

        is_valid_gts = (pl.col("category") == category) & (pl.col("is_evaluated").cast(bool))
        num_gts = len(gts.filter(is_valid_gts))

        # Cannot evaluate without ground truth information.
        if num_gts == 0:
            continue

        for affinity_threshold_m in cfg.affinity_thresholds_m:
            true_positives: NDArrayBool = category_dts[str(affinity_threshold_m)].to_numpy().astype(bool)

            # Continue if there aren't any true positives.
            if len(true_positives) == 0:
                continue

            # Compute average precision for the current threshold.
            threshold_average_precision, _ = compute_average_precision(true_positives, recall_interpolated, num_gts)

            # Record the average precision.
            average_precisions[i, str(affinity_threshold_m)] = threshold_average_precision

        mean_average_precisions: NDArrayFloat = average_precisions[i, 1:].to_numpy().mean(axis=-1)

        # Select only the true positives for each instance.
        middle_idx = len(cfg.affinity_thresholds_m) // 2
        middle_threshold = cfg.affinity_thresholds_m[middle_idx]
        is_tp_t = category_dts[str(middle_threshold)].to_numpy().astype(bool)

        # Initialize true positive metrics.
        tp_errors: NDArrayFloat = np.array(cfg.tp_normalization_terms)

        # Check whether any true positives exist under the current threshold.
        has_true_positives = np.any(is_tp_t)

        # If true positives exist, compute the metrics.
        if has_true_positives:
            tp_error_cols = [str(x.value) for x in TruePositiveErrorNames]
            tp_errors = category_dts[is_tp_t, tp_error_cols].to_numpy().mean(axis=0)

        # Convert errors to scores.
        tp_scores = 1 - np.divide(tp_errors, cfg.tp_normalization_terms)

        # Compute Composite Detection Score (CDS).
        cds = mean_average_precisions * np.mean(tp_scores)
        summary[i, "AP"] = mean_average_precisions
        summary[i, "ATE"] = tp_errors[0]
        summary[i, "ASE"] = tp_errors[1]
        summary[i, "AOE"] = tp_errors[2]
        summary[i, "CDS"] = cds

    # Return the summary.
    return summary
