"""
 -  FwIoU, bIoU, Dice


:
1. FwIoU (Frequency Weighted IoU): IoU
   : FwIoU = (t_i * IoU_i) / (t_i)t_ii

2. bIoU (Boundary IoU): IoU
   : bIoU = |B_pred  B_gt| / |B_pred  B_gt|B

3. Dice: DiceF1
   : Dice = 2|AB| / (|A| + |B|)
"""

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion


class ConfusionMatrixAllClass(object):
    """

     : FwIoU, bIoU, Dice
    """
    def __init__(self, num_classes, boundary_width=3, compute_boundary=True):
        """
        Args:
            num_classes:
            boundary_width: bIoU3
            compute_boundary: IoU
        """
        self.num_classes = num_classes
        self.boundary_width = boundary_width
        self.compute_boundary = compute_boundary

        self.mat1 = None
        self.mat2 = None  # /

        #  : bIoU
        self.boundary_intersection = None
        self.boundary_union = None

    def _get_boundary_mask(self, mask_np, width=3):
        """


        Args:
            mask_np: numpy [H, W]
            width:
        Returns:
            per_class_boundary: dict
        """
        per_class_boundary = {}
        struct = ndimage.generate_binary_structure(2, 1)

        for class_id in range(self.num_classes):
            class_mask = (mask_np == class_id)
            if not class_mask.any():
                per_class_boundary[class_id] = np.zeros_like(class_mask, dtype=bool)
                continue

            dilated = binary_dilation(class_mask, struct, iterations=width)
            eroded = binary_erosion(class_mask, struct, iterations=width)

            #  =  XOR   -
            class_boundary = dilated & ~eroded
            per_class_boundary[class_id] = class_boundary

        return per_class_boundary

    def _update_boundary_stats(self, gt_np, pred_np):
        """


        Args:
            gt_np: numpyground truth [H, W]
            pred_np: numpy [H, W]
        """
        gt_boundaries = self._get_boundary_mask(gt_np, self.boundary_width)
        pred_boundaries = self._get_boundary_mask(pred_np, self.boundary_width)

        # IoU
        for class_id in range(self.num_classes):
            gt_boundary = gt_boundaries[class_id]
            pred_boundary = pred_boundaries[class_id]

            intersection = np.sum(gt_boundary & pred_boundary)
            union = np.sum(gt_boundary | pred_boundary)

            self.boundary_intersection[class_id] += intersection
            self.boundary_union[class_id] += union

    def update(self, a, b):
        """


        Args:
            a: ground truth  (tensor)
            b:  (tensor)
        """
        n = self.num_classes

        if self.mat1 is None:
            self.mat1 = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        if self.mat2 is None:
            self.mat2 = torch.zeros((2, 2), dtype=torch.int64, device=a.device)
        if self.boundary_intersection is None:
            self.boundary_intersection = torch.zeros(n, dtype=torch.float64, device=a.device)
        if self.boundary_union is None:
            self.boundary_union = torch.zeros(n, dtype=torch.float64, device=a.device)

        # tensor
        a_original = a.clone()
        b_original = b.clone()

        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k].to(torch.int64)
            self.mat1 += torch.bincount(inds, minlength=n**2).reshape(n, n)

            # ========== : / ==========
            a[a != 0] = 1
            b[b != 0] = 1
            k = (a >= 0) & (a < 2)
            inds = 2 * a[k].to(torch.int64) + b[k].to(torch.int64)
            self.mat2 += torch.bincount(inds, minlength=2**2).reshape(2, 2)

            # ==========  : bIoU ==========
            if self.compute_boundary:
                # numpy
                if a_original.dim() == 2:
                    #  [H, W]
                    gt_np = a_original.cpu().numpy()
                    pred_np = b_original.cpu().numpy()
                    self._update_boundary_stats(gt_np, pred_np)
                elif a_original.dim() == 3:
                    # batch [B, H, W]
                    for i in range(a_original.shape[0]):
                        gt_np = a_original[i].cpu().numpy()
                        pred_np = b_original[i].cpu().numpy()
                        self._update_boundary_stats(gt_np, pred_np)

            del a
            del b

    def reset(self):
        if self.mat1 is not None:
            self.mat1.zero_()
        if self.mat2 is not None:
            self.mat2.zero_()
        if self.boundary_intersection is not None:
            self.boundary_intersection.zero_()
        if self.boundary_union is not None:
            self.boundary_union.zero_()

    def compute(self):
        """


        Returns:
            acc_global:
            acc:
            iu: IoU
            dice: Dice
            dice_bg_fg: /Dice
        """
        h = self.mat1.float()

        acc_global = torch.diag(h).sum() / (h.sum() + 1e-10)

        #  (Recall)
        acc = torch.diag(h) / (h.sum(1) + 1e-10)

        # IoU
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + 1e-10)

        # Dice
        dice = 2 * torch.diag(h) / (h.sum(1) + h.sum(0) + 1e-10)

        # /Dice
        h_bg_fg = self.mat2.float()
        dice_bg_fg = 2 * torch.diag(h_bg_fg) / (h_bg_fg.sum(1) + h_bg_fg.sum(0) + 1e-10)

        return acc_global, acc, iu, dice, dice_bg_fg

    def compute_fwiou(self):
        """
         IoU (Frequency Weighted IoU)

        FwIoU = (t_i * IoU_i) / (t_i)
         t_i iground truth

        Returns:
            fwiou: IoU
            per_class_weighted_iou: IoU
        """
        h = self.mat1.float()

        #  (ground truth)
        freq = h.sum(1)  #  =
        total = freq.sum()

        # IoU
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + 1e-10)

        # NaN ()
        iu = torch.where(torch.isnan(iu), torch.zeros_like(iu), iu)

        # IoU
        fwiou = (freq * iu).sum() / (total + 1e-10)

        per_class_weighted_iou = freq * iu / (total + 1e-10)

        return fwiou, per_class_weighted_iou

    def compute_biou(self):
        """
         IoU (Boundary IoU)

        bIoU = |B_pred  B_gt| / |B_pred  B_gt|


        Returns:
            mean_biou: IoU
            mean_biou_all: IoU
            per_class_biou: IoU
        """
        if not self.compute_boundary:
            device = self.mat1.device if self.mat1 is not None else 'cpu'
            zero = torch.tensor(0.0, device=device)
            return zero, zero, torch.zeros(self.num_classes, device=device)

        intersection = self.boundary_intersection.float()
        union = self.boundary_union.float()

        # IoU
        per_class_biou = intersection / (union + 1e-10)

        # union0NaN
        valid_mask = union > 0

        # IoU
        valid_foreground = valid_mask.clone()
        valid_foreground[-1] = False

        if valid_foreground.sum() > 0:
            mean_biou = per_class_biou[valid_foreground].mean()
        else:
            mean_biou = torch.tensor(0.0, device=per_class_biou.device)

        # IoU
        if valid_mask.sum() > 0:
            mean_biou_all = per_class_biou[valid_mask].mean()
        else:
            mean_biou_all = torch.tensor(0.0, device=per_class_biou.device)

        return mean_biou, mean_biou_all, per_class_biou

    def compute_dice(self):
        """
         Dice

        Dice = 2|AB| / (|A| + |B|) = 2*TP / (2*TP + FP + FN)

        Returns:
            mean_dice: Dice
            mean_dice_all: Dice
            per_class_dice: Dice
        """
        h = self.mat1.float()

        # Dice: 2 * TP / (row_sum + col_sum)
        per_class_dice = 2 * torch.diag(h) / (h.sum(1) + h.sum(0) + 1e-10)

        # NaN
        per_class_dice = torch.where(
            torch.isnan(per_class_dice),
            torch.zeros_like(per_class_dice),
            per_class_dice
        )

        # Dice
        mean_dice = per_class_dice[:-1].mean()

        # Dice
        mean_dice_all = per_class_dice.mean()

        return mean_dice, mean_dice_all, per_class_dice

    def compute_precision_recall(self):
        """
         PrecisionRecall

        Precision = TP / (TP + FP) =  /
        Recall = TP / (TP + FN) =  /

        :
        - h[i,j]: ij
        - TP for class i = h[i,i]
        - FP for class i = sum(h[:,i]) - h[i,i] (i)
        - FN for class i = sum(h[i,:]) - h[i,i] (i)

        Returns:
            mean_precision: Precision
            mean_recall: Recall
            per_class_precision: Precision
            per_class_recall: Recall
        """
        h = self.mat1.float()

        # TP:
        tp = torch.diag(h)

        # FP:  -
        fp = h.sum(0) - tp

        # FN:  -
        fn = h.sum(1) - tp

        # Precision = TP / (TP + FP)
        per_class_precision = tp / (tp + fp + 1e-10)

        # Recall = TP / (TP + FN)
        per_class_recall = tp / (tp + fn + 1e-10)

        # NaN
        per_class_precision = torch.where(
            torch.isnan(per_class_precision),
            torch.zeros_like(per_class_precision),
            per_class_precision
        )
        per_class_recall = torch.where(
            torch.isnan(per_class_recall),
            torch.zeros_like(per_class_recall),
            per_class_recall
        )

        mean_precision = per_class_precision[:-1].mean()
        mean_recall = per_class_recall[:-1].mean()

        return mean_precision, mean_recall, per_class_precision, per_class_recall

    def compute_all_metrics(self):
        """


        Returns:
            metrics_dict:
        """
        acc_global, acc, iu, dice_basic, dice_bg_fg = self.compute()

        # FwIoU
        fwiou, per_class_fwiou = self.compute_fwiou()

        # bIoU
        mean_biou, mean_biou_all, per_class_biou = self.compute_biou()

        # Dice ()
        mean_dice, mean_dice_all, per_class_dice = self.compute_dice()

        #  Precision and Recall
        mean_precision, mean_recall, per_class_precision, per_class_recall = self.compute_precision_recall()

        metrics_dict = {
            'acc_global': acc_global,
            'acc_per_class': acc,
            'iou_per_class': iu,                # IoU
            'miou': iu[:-1].mean(),             # mIoU
            'miou_all': iu.mean(),              # mIoU

            # ==========  FwIoU ==========
            'fwiou': fwiou,                     # IoU
            'fwiou_per_class': per_class_fwiou, # IoU

            # ==========  bIoU ==========
            'biou': mean_biou,                  # IoU
            'biou_all': mean_biou_all,          # IoU
            'biou_per_class': per_class_biou,   # IoU

            # ==========  Dice ==========
            'dice': mean_dice,                  # Dice
            'dice_all': mean_dice_all,          # Dice
            'dice_per_class': per_class_dice,   # Dice

            # ==========  Precision & Recall ==========
            'precision': mean_precision,              # mPrecision
            'recall': mean_recall,                    # mRecall
            'precision_per_class': per_class_precision,  # Precision
            'recall_per_class': per_class_recall,        # Recall

            # ========== / ==========
            'dice_bg_fg': dice_bg_fg,           # /Dice
        }

        return metrics_dict

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat1)
        torch.distributed.all_reduce(self.mat2)
        if self.compute_boundary:
            torch.distributed.all_reduce(self.boundary_intersection)
            torch.distributed.all_reduce(self.boundary_union)

    def __str__(self):
        metrics = self.compute_all_metrics()

        def format_list(tensor):
            return [f'{i:.1f}' for i in (tensor * 100).tolist()]

        output = (
            '\n====================  ====================\n'
            f'  acc_global (Acc): {metrics["acc_global"].item() * 100:.2f}%\n'
            f'  acc_per_class: {format_list(metrics["acc_per_class"])}\n'
            f'  iou_per_class: {format_list(metrics["iou_per_class"])}\n'
            f'  miou: {metrics["miou"].item() * 100:.2f}%\n'
            f'  miou_all: {metrics["miou_all"].item() * 100:.2f}%\n'
            f'  FwIoU: {metrics["fwiou"].item() * 100:.2f}%\n'
            f'  bIoU : {metrics["biou"].item() * 100:.2f}%\n'
            f'  biou_per_class: {format_list(metrics["biou_per_class"])}\n'
            f'  Dice : {metrics["dice"].item() * 100:.2f}%\n'
            f'  Dice_all: {metrics["dice_all"].item() * 100:.2f}%\n'
            f'  Dice_per_class: {format_list(metrics["dice_per_class"])}\n'
            f'  mPrecision: {metrics["precision"].item() * 100:.2f}%\n'
            f'  Precision_per_class: {format_list(metrics["precision_per_class"])}\n'
            f'  mRecall: {metrics["recall"].item() * 100:.2f}%\n'
            f'  Recall_per_class: {format_list(metrics["recall_per_class"])}\n'
            f'  fg/bgDice: {format_list(metrics["dice_bg_fg"])}\n'
            '==================================================\n'
        )

        return output



def compute_fwiou_from_confusion_matrix(confusion_matrix):
    """
    FwIoU

    Args:
        confusion_matrix: [num_classes, num_classes]
    Returns:
        fwiou: IoU (float)
    """
    if isinstance(confusion_matrix, np.ndarray):
        h = torch.from_numpy(confusion_matrix).float()
    else:
        h = confusion_matrix.float()

    freq = h.sum(1)
    total = freq.sum()

    iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + 1e-10)
    iu = torch.where(torch.isnan(iu), torch.zeros_like(iu), iu)

    fwiou = (freq * iu).sum() / (total + 1e-10)

    return fwiou.item()


def compute_dice_score(pred, target, num_classes, smooth=1e-5, ignore_background=True):
    """
    Dice

    Args:
        pred:  [B, H, W]  [H, W]
        target:  [B, H, W]  [H, W]
        num_classes:
        smooth:
        ignore_background:
    Returns:
        mean_dice: Dice
        per_class_dice: Dice
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    dice_per_class = []

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2 * intersection + smooth) / (union + smooth)
        dice_per_class.append(dice)

    dice_per_class = torch.stack(dice_per_class)

    if ignore_background:
        mean_dice = dice_per_class[:-1].mean()
    else:
        mean_dice = dice_per_class.mean()

    return mean_dice, dice_per_class


def compute_boundary_iou(pred, target, num_classes, boundary_width=3, ignore_background=True):
    """
    IoU

    Args:
        pred:  [H, W] (numpytensor)
        target:  [H, W] (numpytensor)
        num_classes:
        boundary_width:
        ignore_background:
    Returns:
        mean_biou: IoU
        per_class_biou: IoU
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.cpu().numpy()
    else:
        pred_np = pred

    if isinstance(target, torch.Tensor):
        target_np = target.cpu().numpy()
    else:
        target_np = target

    struct = ndimage.generate_binary_structure(2, 1)
    biou_per_class = []
    valid_classes = []

    for c in range(num_classes):
        pred_c = (pred_np == c)
        target_c = (target_np == c)

        if not pred_c.any() and not target_c.any():
            biou_per_class.append(0.0)
            continue

        def get_boundary(mask):
            if not mask.any():
                return np.zeros_like(mask, dtype=bool)
            dilated = binary_dilation(mask, struct, iterations=boundary_width)
            eroded = binary_erosion(mask, struct, iterations=boundary_width)
            return dilated & ~eroded

        pred_boundary = get_boundary(pred_c)
        target_boundary = get_boundary(target_c)

        # IoU
        intersection = np.sum(pred_boundary & target_boundary)
        union = np.sum(pred_boundary | target_boundary)

        if union == 0:
            biou = 0.0
        else:
            biou = intersection / union

        biou_per_class.append(biou)
        if union > 0:
            valid_classes.append(c)

    biou_per_class = torch.tensor(biou_per_class)

    if ignore_background:
        valid_foreground = [c for c in valid_classes if c != num_classes - 1]
        if len(valid_foreground) > 0:
            mean_biou = biou_per_class[valid_foreground].mean()
        else:
            mean_biou = torch.tensor(0.0)
    else:
        if len(valid_classes) > 0:
            mean_biou = biou_per_class[valid_classes].mean()
        else:
            mean_biou = torch.tensor(0.0)

    return mean_biou, biou_per_class
