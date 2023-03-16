import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Union, Dict, List, Iterator, Tuple, Optional
from pathlib import Path
import argparse
from tqdm import tqdm


CATEGORY_MAP = {"ANIMAL":0, "ARTICULATED_BUS":1, "BICYCLE":2, "BICYCLIST":3, "BOLLARD":4,
                "BOX_TRUCK":5, "BUS":6, "CONSTRUCTION_BARREL":7, "CONSTRUCTION_CONE":8, "DOG":9,
                "LARGE_VEHICLE":10, "MESSAGE_BOARD_TRAILER":11, "MOBILE_PEDESTRIAN_CROSSING_SIGN":12,
                "MOTORCYCLE":13, "MOTORCYCLIST":14, "OFFICIAL_SIGNALER":15, "PEDESTRIAN":16,
                "RAILED_VEHICLE":17, "REGULAR_VEHICLE":18, "SCHOOL_BUS":19, "SIGN":20,
                "STOP_SIGN":21, "STROLLER":22, "TRAFFIC_LIGHT_TRAILER":23, "TRUCK":24,
                "TRUCK_CAB":25, "VEHICULAR_TRAILER":26, "WHEELCHAIR":27, "WHEELED_DEVICE":28,
                "WHEELED_RIDER":29, "NONE": -1}

BACKGROUND_CATEGORIES = ['BOLLARD', 'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 
                         'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'SIGN', 'STOP_SIGN']
PEDESTRIAN_CATEGORIES = ['PEDESTRIAN', 'STROLLER', 'WHEELCHAIR', 'OFFICIAL_SIGNALER']
SMALL_VEHICLE_CATEGORIES = ['BICYCLE', 'BICYCLIST', 'MOTORCYCLE', 'MOTORCYCLIST',
                            'WHEELED_DEVICE', 'WHEELED_RIDER']
VEHICLE_CATEGORIES = ['ARTICULATED_BUS', 'BOX_TRUCK', 'BUS', 'LARGE_VEHICLE', 'RAILED_VEHICLE',
                      'REGULAR_VEHICLE', 'SCHOOL_BUS', 'TRUCK', 'TRUCK_CAB',
                      'VEHICULAR_TRAILER', 'TRAFFIC_LIGHT_TRAILER', 'MESSAGE_BOARD_TRAILER']
ANIMAL_CATEGORIES = ['ANIMAL', 'DOG']

NO_CLASSES = {'All': [k for k in range(-1, 30)]}
FOREGROUND_BACKGROUND = {'Background': [-1],
                         'Foreground': [CATEGORY_MAP[k] for k in (BACKGROUND_CATEGORIES +
                                                                  PEDESTRIAN_CATEGORIES +
                                                                  SMALL_VEHICLE_CATEGORIES +
                                                                  VEHICLE_CATEGORIES +
                                                                  ANIMAL_CATEGORIES)]}
PED_CYC_VEH_ANI = {'Background': [-1],
                   'Object': [CATEGORY_MAP[k] for k in BACKGROUND_CATEGORIES],
                   'Pedestrian': [CATEGORY_MAP[k] for k in PEDESTRIAN_CATEGORIES],
                   'Small Vehicle': [CATEGORY_MAP[k] for k in SMALL_VEHICLE_CATEGORIES],
                   'Vehicle': [CATEGORY_MAP[k] for k in VEHICLE_CATEGORIES],
                   'Animal': [CATEGORY_MAP[k] for k in ANIMAL_CATEGORIES]}

Array = Union[np.ndarray, torch.Tensor]

def epe(pred, gt):
    return torch.sqrt(torch.sum((pred - gt) ** 2, -1))

def accuracy(pred, gt, threshold):
    l2_norm = torch.sqrt(torch.sum((pred - gt) ** 2, -1))
    gt_norm = torch.sqrt(torch.sum(gt * gt, -1))
    relative_err = l2_norm / (gt_norm + 1e-20)
    error_lt_5 = (l2_norm < threshold).bool()
    relative_err_lt_5 = (relative_err < threshold).bool()
    return  (error_lt_5 | relative_err_lt_5).float()


def accuracy_strict(pred, gt):
    return accuracy(pred, gt, 0.05)


def accuracy_relax(pred, gt):
    return accuracy(pred, gt, 0.10)


def outliers(pred, gt):
    l2_norm = torch.sqrt(torch.sum((pred - gt) ** 2, -1))
    gt_norm = torch.sqrt(torch.sum(gt * gt, -1))
    relative_err = l2_norm / (gt_norm + 1e-20)

    l2_norm_gt_3 = (l2_norm > 0.3).bool()
    relative_err_gt_10 = (relative_err > 0.1).bool()
    return (l2_norm_gt_3 | relative_err_gt_10).float()


def angle_error(pred, gt):
    unit_label = gt / gt.norm(dim=-1, keepdim=True)
    unit_pred = pred / pred.norm(dim=-1, keepdim=True)
    eps = 1e-7
    dot_product = (unit_label * unit_pred).sum(-1).clamp(min=-1+eps, max=1-eps)
    dot_product[dot_product != dot_product] = 0  # Remove NaNs
    return torch.acos(dot_product)


def coutn(pred, gt):
    return torch.ones(len(pred))

def tp(pred, gt):
    return (pred & gt).sum()

def tn(pred, gt):
    return (~pred & ~gt).sum()

def fp(pred, gt):
    return (pred & ~gt).sum()

def fn(pred, gt):
    return (~pred & gt).sum()


FLOW_METRICS = {'EPE': epe, 'Accuracy Strict': accuracy_strict, 'Accuracy Relax': accuracy_relax,
                'Outliers': outliers, 'Angle Error': angle_error}
SEG_METRICS = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def metrics(pred_flow, pred_dynamic, gt, classes, dynamic, close, object_classes):
    results = []
    for cls, class_idxs in object_classes.items():
        class_mask = classes == class_idxs[0]
        for i in class_idxs[1:]:
            class_mask = class_mask | (classes == i)

        for motion, m_mask in [('Dynamic', dynamic), ('Static', ~dynamic)]:
            for distance, d_mask in [('Close', close), ('Far', ~close)]:
                mask = class_mask & m_mask & d_mask
                cnt = mask.sum().item()
                gt_sub = gt[mask]
                pred_sub = pred_flow[mask]
                result = [cls, motion, distance, cnt]
                if cnt > 0:
                    result += [FLOW_METRICS[m](pred_sub, gt_sub).mean().cpu().item() for m in FLOW_METRICS]
                    result += [SEG_METRICS[m](pred_dynamic[mask], dynamic[mask]).cpu().item() for m in SEG_METRICS]
                else:
                    result += [np.nan for m in FLOW_METRICS]
                    result += [0 for m in SEG_METRICS]
                results.append(result)
    return results

def evaluate_iterator(annotations_root: Path, preds: Iterator[Tuple[str, Array, Array]]):
    results = []
    with h5py.File(annotation_file, 'r') as anno:
        for name, pred, pred_dynamic in tqdm(preds):
            gt = anno[name]
            loss_breakdown = metrics(torch.from_numpy(pred),
                                     torch.from_numpy(pred_dynamic),
                                     torch.from_numpy(gt['flow'][()]),
                                     torch.from_numpy(gt['classes'][()]),
                                     torch.from_numpy(gt['dynamic'][()]),
                                     torch.from_numpy(gt['close'][()]),
                                     FOREGROUND_BACKGROUND)
            results.extend([[name] + bd for bd in loss_breakdown])
    df = pd.DataFrame(results, columns=['Example', 'Class', 'Motion', 'Distance', 'Count']
                      + list(FLOW_METRICS) + list(SEG_METRICS))
    return df
        
def submission_iterator(submission_file):
    with h5py.File(submission_file, 'r') as f:
        for ex_name in f.keys():
            yield ex_name, f[ex_name]['flow'][()], f[ex_name]['dynamic'][()]

def results_to_dict(results_dataframe):
    output = {}
    grouped = results_dataframe.groupby(['Class', 'Motion', 'Distance'])
    for m in FLOW_METRICS.keys():
        avg = grouped.apply(lambda x: (x.EPE * x.Count).sum() / x.Count.sum())
        for segment in avg.index:
            if segment[0] == 'Background' and segment[1] == 'Dynamic':
                continue
            name = m + '/' + '/'.join([str(i) for i in segment])
            output[name] = avg[segment]
    grouped = results_dataframe.groupby(['Class', 'Motion'])
    for m in FLOW_METRICS.keys():
        avg = grouped.apply(lambda x: (x.EPE * x.Count).sum() / x.Count.sum())
        for segment in avg.index:
            if segment[0] == 'Background' and segment[1] == 'Dynamic':
                continue
            name = m + '/' + '/'.join([str(i) for i in segment])
            output[name] = avg[segment]
    output['Dynamic IoU'] = results_dataframe.TP.sum() / (results_dataframe.TP.sum()
                                                          + results_dataframe.FP.sum()
                                                          + results_dataframe.FN.sum())
    output['EPE 3-Way Average'] = (output['EPE/Foreground/Dynamic'] +
                                   output['EPE/Foreground/Static'] +
                                   output['EPE/Background/Static']) / 3

    
    return output    

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score.

    Args:

        test_annotation_file: Path to test_annotation_file on the server, for argoverse challenge this is path to the dataset split folder
        user_submission_file: Path to file submitted by the user, for argoverse challenge this is a zip file
        phase_codename: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission metadata.
    """
    output = {}
    if phase_codename == "test_split":
        print("Evaluating for Test Phase")
        results_df = evaluate_iterator(test_annotation_file,
                                       submission_iterator(user_submission_file))
        output['result'] = [{'test_split': results_to_dict(results_df)}]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Test Phase")
    return output
