import glob
import logging
import os

import numpy as np
import copy
import tabulate
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
import numpy as np
np.seterr(divide='ignore',invalid='ignore')

def dump(save_dir, outputs):
    filenames = outputs["filename"]
    batch_size = len(filenames)
    preds = outputs["pred"].cpu().numpy()  # B x 1 x H x W
    masks = outputs["mask"].cpu().numpy()  # B x 1 x H x W
    # heights = outputs["height"].cpu().numpy()
    # widths = outputs["width"].cpu().numpy()
    clsnames = outputs["clsname"]
    for i in range(batch_size):
        file_dir, filename = os.path.split(filenames[i])
        _, subname = os.path.split(file_dir)
        filename = "{}_{}_{}".format(clsnames[i], subname, filename)
        filename, _ = os.path.splitext(filename)
        save_file = os.path.join(save_dir, filename + ".npz")
        np.savez(
            save_file,
            filename=filenames[i],
            pred=preds[i],
            mask=masks[i],
            # height=heights[i],
            # width=widths[i],
            clsname=clsnames[i],
        )

def merge_together(save_dir):
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))
    fileinfos = []
    preds = []
    masks = []
    for npz_file in npz_file_list:
        npz = np.load(npz_file)
        fileinfos.append(
            {
                "filename": str(npz["filename"]),
                # "height": npz["height"],
                # "width": npz["width"],
                "clsname": str(npz["clsname"]),
            }
        )
        preds.append(npz["pred"])
        masks.append(npz["mask"])
    preds = np.concatenate(np.asarray(preds), axis=0)  # N x H x W
    masks = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    return fileinfos, preds, masks


class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )


class EvalDataMeta:
    def __init__(self, preds, masks, file_info):
        self.preds = preds  # N x H x W
        self.masks = masks  # N x H x W
        self.file_info = file_info


class EvalImage:
    def __init__(self, data_meta, **kwargs):
        self.preds = self.encode_pred(data_meta.preds, **kwargs)
        self.masks = self.encode_mask(data_meta.masks)
        self.file_info = data_meta.file_info
        self.preds_good = sorted(self.preds[self.masks == 0], reverse=True)
        self.preds_defe = sorted(self.preds[self.masks == 1], reverse=True)
        self.num_good = len(self.preds_good)
        self.num_defe = len(self.preds_defe)
        self.desc_score_indices = np.argsort(self.preds, kind="mergesort")[::-1]
        self.y_score = self.preds[self.desc_score_indices]
        self.y_true = self.masks == 1
        self.y_true = self.y_true[self.desc_score_indices]
        self.y_true2 = self.y_true[self.desc_score_indices]

    @staticmethod
    def encode_pred(preds):
        raise NotImplementedError

    def encode_mask(self, masks):
        N, _, _ = masks.shape
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(np.int)  # (N, )
        return masks

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


class EvalImageMean(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).mean(axis=1)  # (N, )


class EvalImageStd(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).std(axis=1)  # (N, )


class EvalImageMax(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        for i in range(0, 8):
            preds = (F.avg_pool2d(preds, 8, stride=1))
        preds = preds.cpu().numpy()  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )


class EvalPerPixelAUC:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc

class EvalPerPixelPRO:
    def __init__(self, data_meta):
        self.preds = data_meta.preds
        self.masks = data_meta.masks
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        pro = compute_pro(self.masks, self.preds)
        return pro

class EvalPerPixelAP:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1
    def eval_auc(self):
        ap = average_precision_score(self.masks, self.preds)
        return ap
class EvalImageAP(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        for i in range(0, 8):
            preds = (F.avg_pool2d(preds, 8, stride=1))
        preds = preds.cpu().numpy()  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )
    def eval_auc(self):
        ap = average_precision_score(self.masks, self.preds)
        return ap
class EvalPerPixelF1:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1
    def eval_auc(self):
        precisions, recalls, thresholds = precision_recall_curve(self.masks, self.preds)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        return f1_px
class EvalImageF1(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        for i in range(0, 8):
            preds = (F.avg_pool2d(preds, 8, stride=1))
        preds = preds.cpu().numpy()  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )
    def eval_auc(self):
        precisions, recalls, thresholds = precision_recall_curve(self.masks, self.preds)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
        return f1_sp
class EvalPerPixelAUPR:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1
    def eval_auc(self):
        pr_auc = compute_aupr(self.preds, self.masks)
        return pr_auc
class EvalImageAUPR(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        for i in range(0, 8):
            preds = (F.avg_pool2d(preds, 8, stride=1))
        preds = (F.avg_pool2d(preds, 2, stride=1))
        preds = preds.cpu().numpy()  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )
    def eval_auc(self):
        pr_auc = compute_aupr(self.preds, self.masks)
        return pr_auc

eval_lookup_table = {
    "mean": EvalImageMean,
    "std": EvalImageStd,
    "max": EvalImageMax,
    "pixel": EvalPerPixelAUC,
    "pro" :EvalPerPixelPRO,
    "appx": EvalPerPixelAP,
    "apsp": EvalImageAP,
    "f1px": EvalPerPixelF1,
    "f1sp": EvalImageF1,
    "auprpx": EvalPerPixelAUPR,
    "auprsp": EvalImageAUPR,
}


def performances(fileinfos, preds, masks, config):
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    for clsname in clsnames:
        preds_cls = []
        masks_cls = []
        file_cls = []
        for fileinfo, pred, mask in zip(fileinfos, preds, masks):
            if fileinfo["clsname"] == clsname:
                preds_cls.append(pred[None, ...])
                masks_cls.append(mask[None, ...])
                file_cls.append(fileinfo['filename'])
        preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W
        masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W
        data_meta = EvalDataMeta(preds_cls, masks_cls, file_cls)

        # auc
        if config.get("auc", None):
            for metric in config["auc"]:
                evalname = metric["name"]
                kwargs = metric.get("kwargs", {})
                eval_method = eval_lookup_table[evalname](data_meta, **kwargs)
                auc = eval_method.eval_auc()
                ret_metrics["{}_{}_auc".format(clsname, evalname)] = auc

    if config.get("auc", None):
        for metric in config["auc"]:
            evalname = metric["name"]
            evalvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for clsname in clsnames
            ]
            mean_auc = np.mean(np.array(evalvalues))
            ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc

    return ret_metrics


def log_metrics(ret_metrics, config):
    logger = logging.getLogger("global_logger")
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"])) + ["mean"]

    # auc
    if config.get("auc", None):
        auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
        evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
        record = Report(["clsname"] + evalnames)

        for clsname in clsnames:
            clsvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for evalname in evalnames
            ]
            record.add_one_record([clsname] + clsvalues)

        logger.info(f"\n{record}")


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)
    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1
        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        df_new = pd.DataFrame([{"pro": mean(pros), "fpr": fpr, "threshold": th}])
        df = pd.concat([df, df_new], ignore_index=True)
        # df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()
    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def compute_aupr(
    predicted_masks,
    ground_truth_masks,
    include_optimal_threshold_rates=False,
):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        predicted_masks: [list of np.arrays or np.array] [NxHxW] Contains
                               generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    pred_mask = copy.deepcopy(predicted_masks)
    gt_mask = copy.deepcopy(ground_truth_masks)
    num = 200
    out = {}

    if pred_mask is None or gt_mask is None:
        for key in out:
            out[key].append(float('nan'))
    else:
        fprs, tprs = [], []
        precisions, f1s = [], []
        gt_mask = np.array(gt_mask, np.uint8)

        t = (gt_mask == 1)
        f = ~t
        n_true = t.sum()
        n_false = f.sum()
        th_min = pred_mask.min() - 1e-8
        th_max = pred_mask.max() + 1e-8
        pred_gt = pred_mask[t]
        th_gt_min = pred_gt.min()
        th_gt_max = pred_gt.max()

        '''
        Using scikit learn to compute pixel au_roc results in a memory error since it tries to store the NxHxW float score values.
        To avoid this, we compute the tp, fp, tn, fn at equally spaced thresholds in the range between min of predicted 
        scores and maximum of predicted scores
        '''
        percents = np.linspace(100, 0, num=num // 2)
        th_gt_per = np.percentile(pred_gt, percents)
        th_unif = np.linspace(th_gt_max, th_gt_min, num=num // 2)
        thresholds = np.concatenate([th_gt_per, th_unif, [th_min, th_max]])
        thresholds = np.flip(np.sort(thresholds))

        if n_true == 0 or n_false == 0:
            raise ValueError("gt_submasks must contains at least one normal and anomaly samples")

        for th in thresholds:
            p = (pred_mask > th).astype(np.uint8)
            p = (p == 1)
            fp = (p & f).sum()
            tp = (p & t).sum()

            fpr = fp / n_false
            tpr = tp / n_true
            if tp + fp > 0:
                prec = tp / (tp + fp)
            else:
                prec = 1.0
            if prec > 0. and tpr > 0.:
                f1 = (2 * prec * tpr) / (prec + tpr)
            else:
                f1 = 0.0
            fprs.append(fpr)
            tprs.append(tpr)
            precisions.append(prec)

        pr_auc = metrics.auc(tprs, precisions)
        pr_auc = round(pr_auc, 4)

    return pr_auc